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

        self.factor_names = (
            "macd",
            "rsi",
            "bb_width",
            "log_ret",
            "log_vol_ret",
            "vol_ma_ratio",
        )
        self.raw_num_features = num_features
        self.total_num_features = num_features + len(self.factor_names)
        self.seq_size = min(seq_size, 96)
        if self.seq_size < patch_size:
            raise ValueError(f"seq_size must be >= patch_size, got seq_size={self.seq_size}, patch_size={patch_size}")
        self.seq_size = (self.seq_size // patch_size) * patch_size
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_features = self.total_num_features
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.patch_size = patch_size
        
        self.instance_norm = nn.InstanceNorm1d(self.total_num_features, affine=True)
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(self.seq_size, patch_size, self.total_num_features, hidden_dim, dropout)
        num_patches = self.seq_size // patch_size
        
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

    @staticmethod
    def _ema(values: torch.Tensor, span: int = None, alpha: float = None) -> torch.Tensor:
        if alpha is None:
            if span is None:
                raise ValueError("Either span or alpha must be provided")
            alpha = 2.0 / (span + 1.0)

        ema = torch.zeros_like(values)
        ema[:, 0] = values[:, 0]
        for i in range(1, values.shape[1]):
            ema[:, i] = alpha * values[:, i] + (1.0 - alpha) * ema[:, i - 1]
        return ema

    @staticmethod
    def _rolling_mean_std(values: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, length = values.shape
        mean = torch.zeros_like(values)
        std = torch.zeros_like(values)
        if length < window:
            return mean, std

        series = values.unsqueeze(1)
        mean_valid = F.avg_pool1d(series, kernel_size=window, stride=1).squeeze(1)
        sq_mean_valid = F.avg_pool1d(series * series, kernel_size=window, stride=1).squeeze(1)
        var_valid = torch.clamp(sq_mean_valid - mean_valid * mean_valid, min=0.0)
        std_valid = torch.sqrt(var_valid)

        mean[:, window - 1:] = mean_valid
        std[:, window - 1:] = std_valid
        return mean, std

    def _generate_factors(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        close = x[:, :, 3].float()
        if x.shape[2] >= 5:
            volume = x[:, :, 4].float()
        else:
            volume = torch.ones_like(close)

        macd = self._ema(close, span=12) - self._ema(close, span=26)

        delta = torch.zeros_like(close)
        delta[:, 1:] = close[:, 1:] - close[:, :-1]
        gain = torch.clamp(delta, min=0.0)
        loss = torch.abs(torch.clamp(delta, max=0.0))
        avg_gain = self._ema(gain, alpha=1.0 / 14.0)
        avg_loss = self._ema(loss, alpha=1.0 / 14.0)
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = (100.0 - (100.0 / (1.0 + rs))) / 100.0

        bb_mean, bb_std = self._rolling_mean_std(close, window=20)
        bb_width = (4.0 * bb_std) / (bb_mean + eps)
        bb_width[:, :19] = 0.0

        log_ret = torch.zeros_like(close)
        prev_close = torch.clamp(close[:, :-1], min=eps)
        log_ret[:, 1:] = torch.log(torch.clamp(close[:, 1:], min=eps) / prev_close)

        safe_vol = torch.where(volume > 0, volume, torch.ones_like(volume))
        log_vol_ret = torch.zeros_like(close)
        log_vol_ret[:, 1:] = torch.log(safe_vol[:, 1:] / (safe_vol[:, :-1] + eps))

        vol_mean, _ = self._rolling_mean_std(safe_vol, window=20)
        vol_ma_ratio = torch.ones_like(close)
        vol_ma_ratio[:, 19:] = safe_vol[:, 19:] / (vol_mean[:, 19:] + eps)

        factors = torch.stack(
            [macd, rsi, bb_width, log_ret, log_vol_ret, vol_ma_ratio],
            dim=-1,
        )
        return torch.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)

    def _append_factors_and_crop(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] < 4:
            raise ValueError(f"Expected at least 4 features with close at index 3, got {x.shape[2]}")
        if x.shape[1] < self.seq_size:
            raise ValueError(f"Expected sequence length >= {self.seq_size}, got {x.shape[1]}")

        x = x[:, -self.seq_size:, :]
        factors = self._generate_factors(x)
        return torch.cat([x.float(), factors], dim=-1)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, num_features]
        返回: [batch_size, num_classes]
        """
        x = self._append_factors_and_crop(x)

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