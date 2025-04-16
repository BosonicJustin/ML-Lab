from jax.nn import relu
import jax.numpy as jnp
from jax.lax import conv_general_dilated, conv_transpose
import equinox as eqx
import jax
from jax.lax import reduce_window, max as jax_max


class ImageConvolution(eqx.Module):
    kernel: jnp.ndarray
    stride: tuple = eqx.field(static=True)

    def __init__(self, kernel_height=1, kernel_width=1, stride=(1, 1), im_channels=1, c_out=1, key=None):
        r_key = key if key else jax.random.PRNGKey(0)
        self.kernel = jax.random.normal(r_key, (c_out, im_channels, kernel_height, kernel_width))
        self.stride = stride
    
    def __call__(self, image_batch):
        return conv_general_dilated(lhs=image_batch, rhs=self.kernel, window_strides=self.stride, padding="VALID")


class JointConvLayer(eqx.Module):
    conv1: eqx.Module  = eqx.field()
    conv2: eqx.Module  = eqx.field()

    def __init__(self, n_features, kernel_size, im_channels=1):
        self.conv1 = ImageConvolution(kernel_size, kernel_size, im_channels=im_channels, c_out=n_features)
        self.conv2 = ImageConvolution(kernel_size, kernel_size, im_channels=n_features, c_out=n_features)


    def __call__(self, x):
        x_out = relu(self.conv1(x))

        return relu(self.conv2(x_out))


class ImageUpConvolution(eqx.Module):
    kernel: jnp.ndarray
    stride: tuple = eqx.field(static=True)

    def __init__(self, channels_in, channels_out, kernel_size, key=None):
        r_key = key if key else jax.random.PRNGKey(0)
        # (C_in, C_out, KH, KW)
        self.kernel = jax.random.normal(r_key, (channels_in, channels_out, kernel_size, kernel_size))
        self.stride = (2, 2)


    def __call__(self, x):
        return conv_transpose(lhs=x, rhs=self.kernel, strides=self.stride, padding="VALID", dimension_numbers=("NCHW", "IOHW", "NCHW"))


class UNet(eqx.Module):
    # Lowering dimension
    joint_conv1: eqx.Module  = eqx.field()
    joint_conv2: eqx.Module  = eqx.field()
    joint_conv3: eqx.Module  = eqx.field()
    joint_conv4: eqx.Module  = eqx.field()
    joint_conv_lowest: eqx.Module = eqx.field()

    # Raising dimension
    up_conv1: eqx.Module = eqx.field()
    joint_conv1_up: eqx.Module = eqx.field()

    up_conv2: eqx.Module = eqx.field()
    joint_conv2_up: eqx.Module = eqx.field()

    up_conv3: eqx.Module = eqx.field()
    joint_conv3_up: eqx.Module = eqx.field()

    up_conv4: eqx.Module = eqx.field()
    joint_conv4_up: eqx.Module = eqx.field()

    final_conv: eqx.Module = eqx.field()

    max_pool2: callable = eqx.field(static=True)

    def __init__(self, im_channels=1, num_classes=2):
        # For downsampling
        self.joint_conv1 = JointConvLayer(n_features=64, kernel_size=3, im_channels=im_channels)
        self.joint_conv2 = JointConvLayer(n_features=128, kernel_size=3, im_channels=64)
        self.joint_conv3 = JointConvLayer(n_features=256, kernel_size=3, im_channels=128)
        self.joint_conv4 = JointConvLayer(n_features=512, kernel_size=3, im_channels=256)
        self.joint_conv_lowest = JointConvLayer(n_features=1024, kernel_size=3, im_channels=512)

        # For upsampling
        self.up_conv1 = ImageUpConvolution(channels_in=1024, channels_out=512, kernel_size=2)
        self.joint_conv1_up = JointConvLayer(n_features=512, kernel_size=3, im_channels=1024)

        self.up_conv2 = ImageUpConvolution(channels_in=512, channels_out=256, kernel_size=2)
        self.joint_conv2_up = JointConvLayer(n_features=256, kernel_size=3, im_channels=512)

        self.up_conv3 = ImageUpConvolution(channels_in=256, channels_out=128, kernel_size=2)
        self.joint_conv3_up = JointConvLayer(n_features=128, kernel_size=3, im_channels=256)

        self.up_conv4 = ImageUpConvolution(channels_in=128, channels_out=64, kernel_size=2)
        self.joint_conv4_up = JointConvLayer(n_features=64, kernel_size=3, im_channels=128)

        self.max_pool2 = lambda x: reduce_window(x, -jnp.inf, jax_max, (1, 1, 2, 2), (1, 1, 2, 2), "VALID")
        self.final_conv = ImageConvolution(kernel_height=1, kernel_width=1, stride=(1,1), im_channels=64, c_out=num_classes)

    def __call__(self, image):
        layer_1_output = self.joint_conv1(image)
        layer_2_output = self.joint_conv2(self.max_pool2(layer_1_output))
        layer_3_output = self.joint_conv3(self.max_pool2(layer_2_output))
        layer_4_output = self.joint_conv4(self.max_pool2(layer_3_output))

        lowest_layer = self.joint_conv_lowest(self.max_pool2(layer_4_output))

        up_sampled1 = self.up_conv1(lowest_layer)
        dim = layer_4_output.shape[3]
        parsed = layer_4_output[:,:, 4:dim - 4, 4:dim - 4]
        lift1 = self.joint_conv1_up(jnp.concatenate((parsed, up_sampled1), axis=1))

        up_sampled2 = self.up_conv2(lift1)
        dim = layer_3_output.shape[3]
        parsed = layer_3_output[:,:, 16:dim - 16, 16:dim - 16]
        lift2 = self.joint_conv2_up(jnp.concatenate((parsed, up_sampled2), axis=1))

        up_sampled3 = self.up_conv3(lift2)
        dim = layer_2_output.shape[3]
        parsed = layer_2_output[:,:, 40:dim - 40, 40:dim - 40]
        lift3 = self.joint_conv3_up(jnp.concatenate((parsed, up_sampled3), axis=1))

        up_sampled4 = self.up_conv4(lift3)
        dim = layer_1_output.shape[3]
        parsed = layer_1_output[:,:, 88:dim - 88, 88:dim - 88]
        lift4 = self.joint_conv4_up(jnp.concatenate((parsed, up_sampled4), axis=1))

        return self.final_conv(lift4)


if __name__ == "__main__":
    u = UNet(im_channels=1)

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 1, 572, 572))
    print('U OUTPUT', u(x).shape)