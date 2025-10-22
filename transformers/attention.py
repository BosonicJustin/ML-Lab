import torch

from torch import nn

embedding_dim = 128
query_dim = key_dim = 256
value_dim = 256


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Positional encodings tracked here
        pe = torch.zeros(max_len, d_model)

        # Construct a sequence (from 0 to maximum length) and then add a dimenson to the end
        # Shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Efficient way to compute the inverse of the angular frequency component
        denominator = 1 / (10000 ** torch.arange(0, d_model, 2).float() / d_model)

        # Apply sine to even indices, cosine to odd. here 0::2 - takes 0, 2, 4,...
        # 1:2 takes 1, 3, 5, ...
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)

        # Here just cache the positional encodings - don't do gradient updates on it
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :]


class Attention(nn.Module):
    def __init__(self, embedding_dim, query_dim, key_dim, value_dim):
        super().__init__()
        self.embedding_dim = torch.tensor(embedding_dim)
        self.query_dim = torch.tensor(query_dim)
        self.key_dim = torch.tensor(key_dim)
        self.value_dim = torch.tensor(value_dim)

        self.Q = nn.Linear(embedding_dim, query_dim)
        self.K = nn.Linear(embedding_dim, key_dim)
        self.V = nn.Linear(embedding_dim, value_dim)

    def get_qkv(self, x):
        assert x.shape[1] == self.embedding_dim, "Invalid embedding dimension"

        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        return q, k, v

    # The input here should be (sequence_length, embedding_dim)
    def forward(self, x):
        q, k, v = self.get_qkv(x)

        scores = torch.softmax(q @ k.T / torch.sqrt(self.key_dim), dim=1)
        att = scores @ v

        return att


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, query_dim, key_dim, value_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads

        self.positional_encoder = PositionalEncoding(embedding_dim)
        self.heads = nn.ModuleList([Attention(embedding_dim, query_dim, key_dim, value_dim) for _ in range(num_heads)])

        self.projection_matrix = nn.Linear(num_heads * value_dim, embedding_dim)

    # Input is expected to be (seq_len, embedding_dim)
    def forward(self, x):
        x_pos = self.positional_encoder(x)
        head_outputs = torch.cat([head(x_pos) for head in self.heads], dim=1)

        return self.projection_matrix(head_outputs)

multi_head_attention = MultiHeadAttention(embedding_dim, query_dim, key_dim, value_dim, num_heads=8)
sequence_of_representations = multi_head_attention(torch.randn(1000, embedding_dim))

print("Representation sequence where each element is a context-aware representation of the input sequence:", sequence_of_representations)