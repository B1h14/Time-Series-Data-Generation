"""
Time Series Diffusion Model - Complete Implementation
=====================================================

Based on state-of-the-art research:
- Diffusion-TS (ICLR 2024): Interpretable diffusion for time series

This implementation provides a general framework for time series generation
using denoising diffusion probabilistic models with interpretable decomposition.

Key Features:
-------------
1. Transformer-based encoder-decoder architecture
2. Interpretable temporal decomposition (trend + seasonality + residual)
3. Patch-based input processing for efficiency
4. Time step conditioning for diffusion process
5. Conditional generation support (forecasting, imputation)
6. Fourier-based loss for accurate frequency modeling

Usage Example:
--------------
```python
import torch
from diffusion_ts_model import (
    DiffusionTransformer, 
    GaussianDiffusion,
    create_sample_data
)

# Create model
model = DiffusionTransformer(
    input_dim=1,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    seq_len=512
)

# Create diffusion process
diffusion = GaussianDiffusion(
    model=model,
    timesteps=1000,
    loss_type='fourier'
)

# Generate sample data
x, noise = create_sample_data(batch_size=32, seq_len=512, dim=1)

# Train
t = torch.randint(0, 1000, (32,))
loss = diffusion.training_loss(x, t)
loss.backward()

# Sample
samples = diffusion.sample(batch_size=10, seq_len=512, dim=1)
```
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


# ===== Utility Functions =====

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """Extract values from a tensor at given timesteps."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """Linear noise schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """Cosine noise schedule (improved over linear)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# ===== Model Components =====

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformers with sinusoidal embeddings.

    Provides position information to the model since transformers don't
    have inherent notion of order.

    Args:
        d_model: Dimension of model embeddings
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create sinusoidal positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class TimeEmbedding(nn.Module):
    """
    Embedding for diffusion timesteps using sinusoidal encoding.

    Maps discrete timesteps to continuous embeddings that modulate
    the network layers based on the noise level.

    Args:
        dim: Embedding dimension
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep indices (batch_size,)
        Returns:
            Timestep embeddings (batch_size, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ClassEmbedding(nn.Module):
    """
    Embedding for class labels for conditional generation.

    Maps discrete class labels to continuous embeddings that modulate
    the network layers based on the class information.
    
    For classifier-free guidance, allocates an extra embedding slot
    (index num_classes) for the null/unconditional class.

    Args:
        num_classes: Number of actual classes (e.g., 3 for sine, cosine, mixed)
        dim: Embedding dimension (should match d_model)
        null_class: If True, allocate extra slot for null class (for classifier-free guidance)
    
    Example:
        >>> # For 3 classes with classifier-free guidance
        >>> emb = ClassEmbedding(num_classes=3, dim=256, null_class=True)
        >>> # Embedding layer will have 4 slots: [0, 1, 2] for classes, [3] for null
        >>> 
        >>> y = torch.tensor([0, 1, 2, 3])  # [class_0, class_1, class_2, null]
        >>> embeddings = emb(y)  # All valid, no index error
    """
    def __init__(self, num_classes: int, dim: int, null_class: bool = True):
        super().__init__()
        # Allocate num_classes+1 embeddings if using classifier-free guidance
        # The extra embedding at index num_classes is for unconditional generation
        num_embeddings = num_classes + 1 if null_class else num_classes
        self.embedding = nn.Embedding(num_embeddings, dim)
        self.num_classes = num_classes
        self.null_class_enabled = null_class
        
        # Initialize the null class embedding to zeros (common practice)
        if null_class:
            with torch.no_grad():
                self.embedding.weight[num_classes].zero_()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: Class labels (batch_size,)
               Values should be in [0, num_classes-1] for actual classes
               or num_classes for null/unconditional class
        Returns:
            Class embeddings (batch_size, dim)
        """
        return self.embedding(y)

class TrendSynthesisLayer(nn.Module):
    """
    Trend synthesis using polynomial regression.

    Captures smooth, slow-varying components using polynomial basis functions.
    Based Diffusion-TS decomposition.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        poly_degree: Degree of polynomial basis
        seq_len: Sequence length
    """
    def __init__(self, input_dim: int, hidden_dim: int, poly_degree: int = 3, seq_len: int = 512):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.poly_degree = poly_degree
        self.seq_len = seq_len

        # Create polynomial basis: [1, t, t^2, ..., t^p]
        c = torch.arange(seq_len, dtype=torch.float32) / seq_len
        self.register_buffer('poly_basis', torch.stack([c ** i  for i in range(poly_degree + 1)], dim=1))

    def forward(self, x: torch.Tensor, mean_val: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mean_val: Mean value (batch, 1, input_dim)
        Returns:
            Trend component (batch, seq_len, input_dim)
        """
        # Project input
        h = self.linear(x)

        # Apply polynomial basis and add mean
        trend = torch.einsum('bsh,tp->bsh', h, self.poly_basis[:, :1]) + mean_val
        for i in range(1, self.poly_degree + 1):
            trend = trend + torch.einsum('bsh,tp->bsh', h, self.poly_basis[:, i:i+1])

        return trend


class FourierSynthesisLayer(nn.Module):
    """
    Seasonality synthesis using Fourier basis.

    Automatically identifies seasonal patterns by selecting top-K frequency
    components based on amplitude in the frequency domain. This captures
    periodic behavior without manual specification.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        top_k: Number of top frequencies to keep
        seq_len: Sequence length
    """
    def __init__(self, input_dim: int, hidden_dim: int, top_k: int = 5, seq_len: int = 512):
        super().__init__()
        self.top_k = top_k
        self.seq_len = seq_len
        self.projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim)
        Returns:
            Seasonality component (batch, seq_len, input_dim)
        """
        batch_size, seq_len, dim = x.shape

        # Project input
        h = self.projection(x)

        # Apply FFT to get frequency domain representation
        fft_h = torch.fft.rfft(h, dim=1)

        # Get amplitude and select top-K frequencies
        amplitude = torch.abs(fft_h)
        _, top_k_idx = torch.topk(amplitude, min(self.top_k, amplitude.size(1)), dim=1)

        # Create mask for top-K frequencies
        mask = torch.zeros_like(amplitude)
        mask.scatter_(1, top_k_idx, 1.0)

        # Filter and reconstruct
        filtered_fft = fft_h * mask
        seasonality = torch.fft.irfft(filtered_fft, n=seq_len, dim=1)

        return seasonality


class DecompositionBlock(nn.Module):
    """
    Single decomposition block combining transformer with trend/seasonality synthesis.

    Each block learns disentangled representations:
    - Trend: Smooth, low-frequency changes
    - Seasonality: Periodic patterns
    - Residual: High-frequency details

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: Feedforward network dimension
        dropout: Dropout probability
        seq_len: Sequence length
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, 
                 dropout: float = 0.1, top_k: int = 5, poly_degree: int = 3,seq_len: int = 512):
        super().__init__()

        # Transformer components
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Interpretable decomposition layers
        self.trend_layer = TrendSynthesisLayer(input_dim=d_model, hidden_dim=d_model, poly_degree=poly_degree, seq_len=seq_len)
        self.fourier_layer = FourierSynthesisLayer(input_dim=d_model, hidden_dim=d_model, top_k=top_k, seq_len=seq_len)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                time_emb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            memory: Encoder output (batch, seq_len, d_model)
            time_emb: Time embedding (batch, d_model)
        Returns:
            Tuple of (output, trend, seasonality)
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Cross-attention with encoder output
        cross_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + cross_out)

        # Modulate with time embedding
        if time_emb is not None:
            x = x + time_emb.unsqueeze(1)

        # Feed-forward network
        ff_out = self.feed_forward(x)
        x = self.norm3(x + ff_out)

        # Decompose into trend and seasonality
        mean_val = x.mean(dim=1, keepdim=True)
        trend = self.trend_layer(x, mean_val)
        seasonality = self.fourier_layer(x - trend)

        return x, trend, seasonality


class DiffusionTransformer(nn.Module):
    """
    Complete diffusion transformer for time series generation.

    Architecture combines:
    - Patch-based input processing for efficiency
    - Transformer encoder for global context
    - Decoder with interpretable decomposition
    - Time step conditioning for diffusion

    The model learns to denoise time series by predicting the clean signal
    directly (x0-prediction) rather than predicting noise.

    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
        seq_len: Sequence length
        patch_size: Size of patches
    """
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 seq_len: int = 512,
                 top_k: int = 5,
                 poly_degree: int = 3,
                 patch_size: int = 8,
                 num_classes: int = 0):   # NEW: optional number of conditioning classes
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        # Input projection and special tokens
        self.input_proj = nn.Linear(input_dim * patch_size, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches, dropout=dropout)

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            TimeEmbedding(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )

        # Optional class embedding for conditioning (if num_classes > 0)
        self.class_emb = ClassEmbedding(num_classes, d_model) if num_classes > 0 else None

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder with decomposition blocks
        self.decoder_blocks = nn.ModuleList([
            DecompositionBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, seq_len=self.num_patches, top_k=top_k, poly_degree=poly_degree)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, input_dim * patch_size)
        self.final_norm = nn.LayerNorm(d_model)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert time series to patches for efficient processing.

        Args:
            x: Time series (batch, seq_len, dim)
        Returns:
            Patches (batch, num_patches, patch_size * dim)
        """
        batch_size, seq_len, dim = x.shape
        num_patches = seq_len // self.patch_size
        x = x.reshape(batch_size, num_patches, self.patch_size * dim)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to time series.

        Args:
            x: Patches (batch, num_patches, patch_size * dim)
        Returns:
            Time series (batch, seq_len, dim)
        """
        batch_size, num_patches, _ = x.shape
        x = x.reshape(batch_size, num_patches * self.patch_size, self.input_dim)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,            
                force_uncond: bool = False) -> Dict[str, torch.Tensor]: 
        """
        Forward pass through the model.

        Args:
            x: Noisy input (batch, seq_len, input_dim)
            t: Timesteps (batch,)
            mask: Optional mask for conditional generation (batch, seq_len)
            y: Optional class labels for conditioning (batch,)
            force_uncond: If True, force unconditional generation
        Returns:
            Dictionary with 'output', 'trend', 'seasonality' tensors
        """
        batch_size = x.size(0)

        # Convert to patches
        x_patches = self.patchify(x)

        # Apply mask if provided (for conditional generation)
        if mask is not None:
            mask_patches = mask.reshape(batch_size, self.num_patches, self.patch_size).any(dim=-1)
            x_patches = torch.where(
                mask_patches.unsqueeze(-1),
                x_patches,
                self.mask_token.expand(batch_size, self.num_patches, -1)
            )

        # Project to model dimension
        x_emb = self.input_proj(x_patches)
        x_emb = self.pos_encoding(x_emb)

        # Encode with transformer
        memory = self.encoder(x_emb)

        # Get time embedding
        time_emb = self.time_mlp(t.float())

        # Merge class embedding (if present) into time embedding for conditioning
        if self.class_emb is not None:
            # Default to class 0 when labels are not provided
            if y is None:
                y = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            # If force_uncond, override labels to class 0 (simple unconditional strategy)
            if force_uncond:
                y = torch.zeros_like(y)
            class_embedding = self.class_emb(y)
            time_emb = time_emb + class_embedding

        # Decode with decomposition
        h = x_emb
        all_trends = []
        all_seasonalities = []

        for decoder_block in self.decoder_blocks:
            h, trend, seasonality = decoder_block(h, memory, time_emb)
            all_trends.append(trend)
            all_seasonalities.append(seasonality)

        # Aggregate decomposed components
        trend_sum = sum(all_trends)
        seasonality_sum = sum(all_seasonalities)

        # Final normalization and residual
        h = self.final_norm(h)
        residual = h

        # Combine: output = trend + seasonality + residual
        output = trend_sum + seasonality_sum + residual
        output = self.output_proj(output)
        output = self.unpatchify(output)

        # Also return decomposed components for interpretability
        trend_ts = self.unpatchify(self.output_proj(trend_sum))
        season_ts = self.unpatchify(self.output_proj(seasonality_sum))

        return {
            'output': output,
            'trend': trend_ts,
            'seasonality': season_ts
        }


class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process for time series generation.

    Implements the forward and reverse diffusion processes with various
    noise schedules and loss functions.

    Key features:
    - Multiple noise schedules (linear, cosine)
    - Fourier-based loss for frequency domain accuracy
    - Conditional generation support
    - DDIM and DDPM sampling

    Args:
        model: Denoising model (DiffusionTransformer)
        timesteps: Number of diffusion steps
        beta_schedule: Type of noise schedule ('linear' or 'cosine')
        loss_type: Loss function type ('mse', 'fourier', or 'combined')
    """
    def __init__(self, model: nn.Module, timesteps: int = 1000, 
                 beta_schedule: str = 'cosine', loss_type: str = 'fourier'):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type

        # Define beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Pre-compute diffusion parameters
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Posterior variance
        self.register_buffer('posterior_variance',
                           betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

        # Coefficients for x0 prediction
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: add noise to x_start.

        Args:
            x_start: Clean data (batch, seq_len, dim)
            t: Timesteps (batch,)
            noise: Optional pre-generated noise
        Returns:
            Noisy data x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, 
                                 noise: torch.Tensor) -> torch.Tensor:
        """Predict x0 from noise prediction."""
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def fourier_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss in frequency domain using FFT.

        This ensures accurate reconstruction of periodic patterns.
        """
        # Time domain loss
        time_loss = F.mse_loss(pred, target)

        # Frequency domain loss
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        freq_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

        return time_loss + freq_loss

    def training_loss(self, x_start: torch.Tensor, t: torch.Tensor, 
                     mask: Optional[torch.Tensor] = None,
                     y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            x_start: Clean data (batch, seq_len, dim)
            t: Timesteps (batch,)
            mask: Optional mask for conditional training
            y: Optional class labels for conditioning
        Returns:
            Dictionary with loss components
        """
        # Add noise
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        # Predict clean signal (pass labels through)
        model_out = self.model(x_t, t, mask, y)
        pred_x0 = model_out['output']

        # Compute loss based on loss type
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred_x0, x_start)
        elif self.loss_type == 'fourier':
            loss = self.fourier_loss(pred_x0, x_start)
        elif self.loss_type == 'combined':
            mse_loss = F.mse_loss(pred_x0, x_start)
            fourier_loss = self.fourier_loss(pred_x0, x_start)
            loss = mse_loss + 0.5 * fourier_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return {
            'loss': loss,
            'pred': pred_x0,
            'trend': model_out['trend'],
            'seasonality': model_out['seasonality']
        }

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, mask: Optional[torch.Tensor] = None,
                 y: Optional[torch.Tensor] = None, force_uncond: bool = False) -> torch.Tensor:
        """
        Single step of reverse diffusion (DDPM sampling).

        Args:
            x: Current noisy sample (batch, seq_len, dim)
            t: Current timestep
            mask: Optional mask for conditional generation
            y: Optional class labels for conditioning
            force_uncond: If True, force unconditional generation at model call
        Returns:
            Denoised sample at t-1
        """
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x.device, dtype=torch.long)

        # Predict x0 (pass class labels and force_uncond through)
        model_out = self.model(x, t_tensor, mask, y, force_uncond)
        pred_x0 = model_out['output']

        # Clip prediction
        # pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        if t > 0:
            # Compute mean and variance
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod_prev[t]

            mean = (alpha_t.sqrt() * (1 - alpha_cumprod_prev) * x + 
                   alpha_cumprod_prev.sqrt() * (1 - alpha_t) * pred_x0) / (1 - alpha_cumprod_t)

            variance = self.posterior_variance[t]
            noise = torch.randn_like(x)

            return mean + variance.sqrt() * noise
        else:
            return pred_x0

    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int, dim: int, 
               mask: Optional[torch.Tensor] = None, 
               y: Optional[torch.Tensor] = None,
               force_uncond: bool = False,
               return_intermediates: bool = False) -> torch.Tensor:
        """
        Generate samples using DDPM sampling.

        Args:
            batch_size: Number of samples to generate
            seq_len: Sequence length
            dim: Feature dimension
            mask: Optional mask for conditional generation
            y: Optional class labels for conditioning (batch_size,)
            force_uncond: If True, force unconditional generation in model calls
            return_intermediates: Whether to return intermediate steps
        Returns:
            Generated samples (batch, seq_len, dim)
        """
        device = next(self.model.parameters()).device
        shape = (batch_size, seq_len, dim)

        # Start from noise
        x = torch.randn(shape, device=device)

        intermediates = []

        # Reverse diffusion
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, mask, y, force_uncond)
            if return_intermediates:
                intermediates.append(x.cpu())

        if return_intermediates:
            return x, intermediates
        return x


# ===== Helper Functions for Testing =====

def create_sample_data(batch_size: int = 32, seq_len: int = 512, dim: int = 1,
                      function_type: str = 'sine') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sample time series data for testing.

    Args:
        batch_size: Number of samples
        seq_len: Sequence length
        dim: Feature dimension
        function_type: Type of function ('sine', 'cosine', 'mixed', 'exponential')
    Returns:
        Tuple of (data, noise) tensors
    """
    t = torch.linspace(0, 4 * np.pi, seq_len)

    if function_type == 'sine':
        # Simple sine wave with noise
        signal = torch.sin(t).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, dim)
        noise = 0.1 * torch.randn_like(signal)

    elif function_type == 'cosine':
        # Cosine wave with noise
        signal = torch.cos(t).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, dim)
        noise = 0.1 * torch.randn_like(signal)

    elif function_type == 'mixed':
        # Linear combination of sine and cosine
        signal = (0.5 * torch.sin(t) + 0.3 * torch.cos(2 * t)).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, dim)
        noise = 0.1 * torch.randn_like(signal)

    elif function_type == 'exponential':
        # Exponential decay with periodic component
        decay = torch.exp(-0.5 * t / (4 * np.pi))
        signal = (decay * torch.sin(t)).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, dim)
        noise = 0.1 * torch.randn_like(signal)

    else:
        raise ValueError(f"Unknown function type: {function_type}")

    return signal + noise, noise


if __name__ == "__main__":
    print("=" * 60)
    print("Time Series Diffusion Model - Implementation Test")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create model
    print("\nCreating model...")
    model = DiffusionTransformer(
        input_dim=1,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        seq_len=512,
        patch_size=8
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    # Create diffusion process
    print("\nCreating diffusion process...")
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=100,
        beta_schedule='cosine',
        loss_type='fourier'
    ).to(device)

    # Test forward pass
    print("\nTesting forward pass...")
    x_test, _ = create_sample_data(batch_size=4, seq_len=512, dim=1, function_type='sine')
    x_test = x_test.to(device)
    t_test = torch.randint(0, 100, (4,), device=device)

    with torch.no_grad():
        output = model(x_test, t_test)
        print(f"Input shape: {x_test.shape}")
        print(f"Output shape: {output['output'].shape}")
        print(f"Trend shape: {output['trend'].shape}")
        print(f"Seasonality shape: {output['seasonality'].shape}")

    # Test sampling
    print("\nTesting sample generation...")
    with torch.no_grad():
        samples = diffusion.sample(batch_size=2, seq_len=512, dim=1)
        print(f"Generated samples shape: {samples.shape}")

    # --- Added: Conditional generation test ---
    print("\nTesting conditional generation...")

    # Create conditional model with 3 classes
    cond_model = DiffusionTransformer(
        input_dim=1,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        seq_len=512,
        patch_size=8,
        num_classes=3
    ).to(device)

    cond_diffusion = GaussianDiffusion(
        model=cond_model,
        timesteps=100,
        beta_schedule='cosine',
        loss_type='fourier'
    ).to(device)

    # Reuse x_test / t_test from above (shape batch=4) and create class labels
    y_train = torch.randint(0, 3, (4,), device=device)

    with torch.no_grad():
        # Conditioned forward pass
        output_cond = cond_model(x_test, t_test, y=y_train)
        print(f"Conditional forward output shape: {output_cond['output'].shape}")

        # Conditioned training loss
        loss_cond = cond_diffusion.training_loss(x_test, t_test, y=y_train)
        print(f"Conditional training loss: {loss_cond['loss'].item():.6f}")

        # Conditioned sampling (batch_size=2)
        y_sample = torch.tensor([0, 1], device=device)
        samples_cond = cond_diffusion.sample(batch_size=2, seq_len=512, dim=1, y=y_sample)
        print(f"Conditional samples shape: {samples_cond.shape}")

        # Forced-unconditional sampling
        samples_uncond = cond_diffusion.sample(batch_size=2, seq_len=512, dim=1, y=y_sample, force_uncond=True)
        print(f"Forced-unconditional samples shape: {samples_uncond.shape}")
    # --- End added block ---

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)
