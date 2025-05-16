import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class SigLIPEncoder(nn.Module):
    """
    Non-trainable module that wraps SigLIP for encoding images and text.
    This is a singleton class - only one instance should be created and shared.
    """
    _instance = None

    @classmethod
    def get_instance(cls, model_name='hf-hub:Marqo/marqo-fashionSigLIP'):
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def __init__(self, model_name='hf-hub:Marqo/marqo-fashionSigLIP'):
        if SigLIPEncoder._instance is not None:
            raise Exception("SigLIPEncoder is a singleton. Use get_instance() instead.")
            
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()  # Set to eval mode as we won't train this
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        SigLIPEncoder._instance = self
            
    def encode_images(self, images):
        """
        Encode a batch of images.
        Args:
            images: Tensor of shape [B*N, C, H, W] where B is batch size and N is number of images per product
        Returns:
            embeddings: Tensor of shape [B*N, embedding_dim]
        """
        with torch.no_grad():
            return self.model.encode_image(images)
            
    def encode_text(self, texts):
        """
        Encode a batch of text.
        Args:
            texts: List of strings to encode
        Returns:
            embeddings: Tensor of shape [B, embedding_dim]
        """
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(next(self.model.parameters()).device)
            return self.model.encode_text(tokens)
    
    def get_train_preprocess(self):
        """Returns the preprocessing transform for training images"""
        return self.preprocess
    
    def get_val_preprocess(self):
        """Returns the preprocessing transform for validation images"""
        return self.preprocess
    
    def to(self, device):
        self.model = self.model.to(device)
        return super().to(device)


class SetTransformerFusion(nn.Module):
    """
    This module is used to fuse multiple image embeddings into a single embedding.
    Keep in mind that the input order of embeddings should not matter - it must be permutation in
    """
    def __init__(
        self, 
        embedding_dim=512,      # Dimension of input embeddings from SigLIP
        hidden_dim=512,         # Hidden dimension in transformer
        num_heads=8,            # Number of attention heads
        num_layers=2,           # Number of transformer layers
        dropout=0.1,            # Dropout rate
        pooling_method="attention",  # Options: "mean", "max", "attention"
        hidden_dim_coefficient=4
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pooling_method = pooling_method
        self.hidden_dim_coefficient = hidden_dim_coefficient
        
        # Input projection if dimensions don't match
        self.input_projection = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()
        
        # Transformer encoder layers without positional encodings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * self.hidden_dim_coefficient,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling components (if selected)
        if pooling_method == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, embeddings, mask=None):
        """
        Forward pass through the set transformer fusion module.
        
        Args:
            embeddings: Tensor of shape [batch_size, num_images, embedding_dim]
                        representing the embeddings of product images
            mask: Boolean tensor of shape [batch_size, num_images] where True means
                  the position is valid (not padding), False means it's padding
                  
        Returns:
            fused_embedding: Tensor of shape [batch_size, embedding_dim]
        """        
        # Create attention mask for transformer if mask is provided
        attention_mask = None
        if mask is not None:
            # Convert boolean mask to transformer attention mask
            attention_mask = ~mask  # Invert since transformer uses 1 for masked positions
        
        # Project input if needed
        x = self.input_projection(embeddings)
        
        # Apply transformer layers (no positional encoding)
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Pooling to get a single embedding per product
        if self.pooling_method == "mean":
            # Apply mask if available
            if mask is not None:
                x = x * mask.unsqueeze(-1)
                pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
                
        elif self.pooling_method == "max":
            # Apply mask if available
            if mask is not None:
                # Set padded positions to large negative value
                x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
            pooled = torch.max(x, dim=1)[0]
            
        elif self.pooling_method == "attention":
            # Learn weights for each image
            attn_weights = self.attention_pool(x).squeeze(-1)
            
            # Apply mask if available
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask, -1e9)
                
            attn_weights = F.softmax(attn_weights, dim=1)
            pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
            
        return self.output_projection(pooled)
    
    
class QueryEncoder(nn.Module):
    """
    Encoder for query text using SigLIP.
    """
    def __init__(self):
        super().__init__()
        self.siglip = SigLIPEncoder.get_instance()
        
    def forward(self, queries):
        """
        Encode queries using SigLIP.
        Args:
            queries: List of query strings
        Returns:
            query_embeddings: Tensor of shape [B, embedding_dim]
        """
        return self.siglip.encode_text(queries)
    
    def to(self, device):
        self.siglip.to(device)
        return super().to(device)


class ProductEncoder(nn.Module):
    """
    Complete model for encoding products using both images and text.
    """
    def __init__(
        self, 
        embedding_dim=768,
        image_fusion_hidden_dim=512,
        image_fusion_heads=8, 
        image_fusion_layers=2,
        cross_modal_hidden_dim=1024,
        dropout=0.1
    ):
        super().__init__()
        
        # Get shared SigLIP instance
        self.siglip = SigLIPEncoder.get_instance()
        
        # Image fusion module
        self.image_fusion = SetTransformerFusion(
            embedding_dim=embedding_dim,
            hidden_dim=image_fusion_hidden_dim,
            num_heads=image_fusion_heads,
            num_layers=image_fusion_layers,
            dropout=dropout,
            pooling_method="attention"
        )
        
        # Cross-modal fusion module
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, cross_modal_hidden_dim),
            nn.LayerNorm(cross_modal_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cross_modal_hidden_dim, embedding_dim)
        )
        
        # Final normalization
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, images, description_texts, image_mask=None):
        """
        Forward pass through the product encoder.
        
        Args:
            images: Tensor of shape [B, N, C, H, W]
            description_texts: List of product descriptions
            image_mask: Boolean tensor of shape [B, N]
                       
        Returns:
            product_embedding: Tensor of shape [B, embedding_dim]
        """
        B, N, C, H, W = images.shape
        
        # Process images
        images = images.view(-1, C, H, W)
        image_features = self.siglip.encode_images(images)
        image_features = image_features.view(B, N, -1)
        
        # Process description text
        description_features = self.siglip.encode_text(description_texts)
        
        # Fuse multiple image embeddings
        fused_image_embedding = self.image_fusion(image_features, image_mask)
        
        # Concatenate image and text embeddings
        multimodal_input = torch.cat([fused_image_embedding, description_features], dim=1)
        
        # Apply cross-modal fusion
        product_embedding = self.cross_modal_fusion(multimodal_input)
        
        return self.norm(product_embedding)
    
    def get_train_preprocess(self):
        """Returns the preprocessing transform for training images"""
        return self.siglip.get_train_preprocess()
        
    def get_val_preprocess(self):
        """Returns the preprocessing transform for validation images"""
        return self.siglip.get_val_preprocess()
        
    def to(self, device):
        self.siglip.to(device)
        return super().to(device)


if __name__ == "__main__":
    # Example usage with random data
    batch_size = 4
    max_images = 5  # Maximum number of images per product
    embedding_dim = 768

    # Create random number of images per product (between 1 and max_images)
    num_images_per_product = torch.randint(1, max_images + 1, (batch_size,))
    
    # Create padded image embeddings tensor
    images = torch.randn(batch_size, max_images, 3, 224, 224)
    image_mask = torch.zeros(batch_size, max_images, dtype=torch.bool)
    
    # Fill in mask for each product's actual images
    for i in range(batch_size):
        num_imgs = num_images_per_product[i]
        image_mask[i, :num_imgs] = True
    
    # Create sample texts
    descriptions = ["A beautiful dress"] * batch_size
    queries = ["red dress", "blue shirt", "green pants", "yellow shoes"]
    
    # Initialize models
    product_encoder = ProductEncoder()
    query_encoder = QueryEncoder()
    
    # Get embeddings
    product_embedding = product_encoder(images, descriptions, image_mask)
    query_embedding = query_encoder(queries)
    
    print(f"Input shapes:")
    print(f"Images: {images.shape}")
    print(f"Image mask: {image_mask.shape}")
    print(f"\nNumber of images per product in batch:")
    print(num_images_per_product)
    print(f"\nOutput shapes:")
    print(f"Product embedding: {product_embedding.shape}")
    print(f"Query embedding: {query_embedding.shape}")
