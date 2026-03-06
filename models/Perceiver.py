import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, seq_len: int, patch_size: int, num_features: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        patch_dim = patch_size * num_features
        
        self.projection = nn.Linear(patch_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch, seq_len, num_features]
        return: [batch, num_patches, hidden_dim]
        """
        batch_size, seq_len, num_features = x.shape
        
        x = x.reshape(batch_size, self.num_patches, self.patch_size, num_features)
        
        x = x.reshape(batch_size, self.num_patches, -1)
        
        x = self.projection(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = hidden_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        
        return x


class Perceiver(nn.Module):
    def __init__(
        self, 
        hidden_dim: int = 128,
        num_layers: int = 3,
        seq_size: int = 96,
        num_features: int = 4,
        num_heads: int = 8,
        num_classes: int = 3,
        dropout: float = 0.1,
        patch_size: int = 8,
        mlp_ratio: int = 4
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.patch_size = patch_size
        
        self.instance_norm = nn.InstanceNorm1d(num_features, affine=True)
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(seq_size, patch_size, num_features, hidden_dim, dropout)
        num_patches = seq_size // patch_size  # 96 // 8 = 12
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_patches, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, num_features]
        返回: [batch_size, num_classes]
        """
        # Instance Normalization
        # [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        x = self.instance_norm(x)
        # [batch, features, seq_len] -> [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        
        x = self.patch_embed(x)
        for block in self.transformer_blocks:
            x = block(x)
            
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x