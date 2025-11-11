# Time Series Diffusion Model - Complete Implementation

A general-purpose diffusion model for time series generation based on state-of-the-art research from ICLR 2024, ICML 2024, and NeurIPS 2025.

## Overview

This implementation combines several cutting-edge techniques:

1. **Interpretable Decomposition** (from Diffusion-TS, ICLR 2024)
   - Explicit modeling of trend, seasonality, and residual components
   - Polynomial basis for smooth trends
   - Fourier synthesis for periodic patterns

2. **Transformer Architecture** (from MOMENT, ICML 2024)
   - Encoder-decoder with attention mechanisms
   - Patch-based processing for efficiency
   - Positional encodings (sinusoidal + RoPE)

3. **Diffusion Process** (from multiple papers)
   - Denoising score matching
   - Multiple noise schedules (linear, cosine)
   - Fourier-based loss for frequency accuracy

4. **Conditional Generation**
   - Forecasting
   - Imputation
   - Mask-based guidance

## Project Structure

```
.
├── diffusion_ts_model.py      # Core model implementation
├── train_diffusion_ts.py       # Training script
├── test_functions.py           # Test on synthetic functions
├── test_datasets.py            # Test on real datasets
├── README.md                   # This file
└── experiments/                # Output directory
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib tqdm pandas scikit-learn
```

## Quick Start

### 1. Test Basic Functionality

```python
import torch
from diffusion_ts_model import DiffusionTransformer, GaussianDiffusion

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

# Generate samples
samples = diffusion.sample(batch_size=10, seq_len=512, dim=1)
print(f"Generated samples shape: {samples.shape}")
```

### 2. Train on Synthetic Data

```bash
# Train on sine waves
python train_diffusion_ts.py

# The script will:
# - Generate synthetic data
# - Train the model
# - Save checkpoints
# - Generate visualizations
```

### 3. Test Different Functions

```bash
# Test all function types
python test_functions.py

# Tests include:
# - Simple sine/cosine waves
# - Mixed periodic functions
# - Exponential decay
# - Custom decay functions
# - All with Gaussian noise
```

## Model Architecture

### Core Components

#### 1. Positional Encoding
```python
class PositionalEncoding(nn.Module):
    """Sinusoidal positional embeddings for transformers."""
    pass
```

#### 2. Trend Synthesis
```python
class TrendSynthesisLayer(nn.Module):
    """Polynomial basis for smooth trend components."""
    pass
```

#### 3. Fourier Synthesis
```python
class FourierSynthesisLayer(nn.Module):
    """Top-K frequency selection for seasonality."""
    pass
```

#### 4. Decomposition Block
```python
class DecompositionBlock(nn.Module):
    """Transformer + interpretable decomposition."""
    pass
```

#### 5. Main Model
```python
class DiffusionTransformer(nn.Module):
    """Complete diffusion transformer."""
    pass
```

### Diffusion Process

```python
class GaussianDiffusion(nn.Module):
    """
    Forward process: q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
    Reverse process: p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²I)
    """
    pass
```

## Training

### Configuration Options

```python
config = {
    # Data
    'function_type': 'mixed',  # sine, cosine, mixed, exponential_decay, custom_decay
    'n_train': 5000,
    'n_val': 500,
    'seq_len': 512,
    'dim': 1,

    # Training
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,

    # Model
    'd_model': 256,
    'nhead': 4,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'dim_feedforward': 512,
    'patch_size': 8,

    # Diffusion
    'timesteps': 100,
    'beta_schedule': 'cosine',  # linear or cosine
    'loss_type': 'fourier',  # mse, fourier, or combined

    # Other
    'save_dir': 'experiments',
    'lambda_decay': 0.5
}
```

### Custom Training Loop

```python
from train_diffusion_ts import train_epoch, evaluate

# Training loop
for epoch in range(epochs):
    train_loss = train_epoch(model, diffusion, train_loader, optimizer, device)
    val_loss = evaluate(model, diffusion, val_loader, device)
    scheduler.step()
```

## Testing

### Test on Synthetic Functions

The implementation is tested on 5 different function types:

#### 1. Simple Sine Wave
```python
x(t) = sin(ωt + φ) + ε
```

#### 2. Simple Cosine Wave
```python
x(t) = cos(ωt + φ) + ε
```

#### 3. Mixed Periodic
```python
x(t) = a₁·sin(ω₁t + φ₁) + a₂·cos(ω₂t + φ₂) + ε
```

#### 4. Exponential Decay
```python
x(t) = exp(-λt)·sin(ωt + φ) + ε
```

#### 5. Custom Decay
```python
x(t) = f(t)·[sin(ω₁t + φ₁) + cos(ω₂t + φ₂)] + ε
where f(t) = 1/(1 + λt)
```

All functions include Gaussian noise: ε ~ N(0, σ²)

### Example Usage

```python
from test_functions import test_function

# Test exponential decay
model, diffusion, data = test_function(
    function_type='exponential_decay',
    lambda_decay=0.5,
    n_samples=1000,
    seq_len=512
)
```

## Conditional Generation

### Forecasting

```python
# Create mask for known history
mask = torch.zeros(batch_size, seq_len)
mask[:, :history_length] = 1.0

# Generate forecast
forecast = diffusion.sample(
    batch_size=batch_size,
    seq_len=seq_len,
    dim=dim,
    mask=mask
)
```

### Imputation

```python
# Create mask for observed values
observed_mask = torch.ones(batch_size, seq_len)
observed_mask[:, missing_indices] = 0.0

# Impute missing values
imputed = diffusion.sample(
    batch_size=batch_size,
    seq_len=seq_len,
    dim=dim,
    mask=observed_mask
)
```

## Interpretability

The model provides interpretable decomposition:

```python
# Get decomposition
output = model(x, t)

trend = output['trend']          # Smooth, low-frequency component
seasonality = output['seasonality']  # Periodic patterns
reconstructed = output['output']     # Full reconstruction

# Visualize
from train_diffusion_ts import visualize_decomposition
visualize_decomposition(model, sample, save_path='decomposition.png')
```

## Advanced Features

### Custom Noise Schedule

```python
def custom_beta_schedule(timesteps):
    # Define your own schedule
    return torch.linspace(0.0001, 0.02, timesteps)

diffusion = GaussianDiffusion(
    model=model,
    timesteps=1000,
    beta_schedule='custom'  # Add custom schedule support
)
```

### Custom Loss Function

```python
class CustomDiffusion(GaussianDiffusion):
    def fourier_loss(self, pred, target):
        # Implement custom frequency-based loss
        time_loss = F.mse_loss(pred, target)
        freq_loss = custom_frequency_loss(pred, target)
        return time_loss + freq_loss
```

## Performance Tips

1. **Use Patch-Based Processing**
   - Reduces computational complexity
   - Enables longer sequences
   - Patch size of 8-16 works well

2. **Choose Appropriate Model Size**
   - Small: d_model=128, 2-3 layers (fast, less accurate)
   - Medium: d_model=256, 3-4 layers (balanced)
   - Large: d_model=512, 6+ layers (accurate, slow)

3. **Noise Schedule**
   - Cosine schedule generally works better than linear
   - Fewer timesteps (50-100) for fast training
   - More timesteps (1000) for best quality

4. **Loss Function**
   - MSE: Fast, simple
   - Fourier: Better for periodic patterns
   - Combined: Best overall quality

## Troubleshooting

### Common Issues

1. **NaN Loss**
   - Reduce learning rate
   - Use gradient clipping
   - Check data normalization

2. **Poor Generation Quality**
   - Increase model size
   - Train for more epochs
   - Use Fourier loss
   - Increase diffusion timesteps

3. **Slow Training**
   - Reduce patch size
   - Decrease model size
   - Use fewer diffusion timesteps
   - Enable mixed precision training

