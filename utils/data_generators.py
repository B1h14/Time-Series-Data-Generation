""""
Dataset generators for simple time series :
    - Sine waves
    - Cosine waves
    - Mixed sine and cosine waves
    - Exponential decay with periodic components
=========================================================================
Each time series is generated with added Gaussian noise and normalized.

Usage Example:
```python
from utils.data_generators import TimeSeriesDataset, create_synthetic_dataset

# Create synthetic dataset
data = create_synthetic_dataset(n_samples=10000, seq_len=512, dim=1, function_type='sine')

# Create dataset
dataset = TimeSeriesDataset(data)   


"""
import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    """Dataset for time series generation."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_synthetic_dataset(n_samples: int = 10000, 
        seq_len: int = 512,
        dim: int = 1, 
        function_type: str = 'sine',
        lambda_decay: float = 0.5,
        custom_func: callable = None
    ) -> torch.Tensor:
    """
    Create synthetic time series dataset with various patterns.

    Args:
        n_samples: Number of samples to generate
        seq_len: Length of each time series
        dim: Dimensionality
        function_type: Type of function to generate
        lambda_decay: Decay parameter for exponential functions
    """
    t = torch.linspace(0, 4 * np.pi, seq_len)
    all_samples = []

    for i in range(n_samples):
        # Add randomness to frequencies and phases
        freq1 = np.random.uniform(2.0, 4.0)
        freq2 = np.random.uniform(2.0, 4.0)
        phase1 = np.random.uniform(0, 2 * np.pi)
        phase2 = np.random.uniform(0, 2 * np.pi)

        if function_type == 'sine':
            signal = torch.sin(freq1 * t + phase1)

        elif function_type == 'cosine':
            signal = torch.cos(freq1 * t + phase1)

        elif function_type == 'mixed':
            # Linear combination of sine and cosine
            amp1 = np.random.uniform(0.3, 0.7)
            amp2 = np.random.uniform(0.3, 0.7)
            signal = amp1 * torch.sin(freq1 * t + phase1) + amp2 * torch.cos(freq2 * t + phase2)

        elif function_type == 'exponential_decay':
            # Exponential decay with periodic component
            decay = torch.exp(-lambda_decay * t / (4 * np.pi))
            signal = decay * torch.sin(freq1 * t + phase1)
        elif function_type == 'linear_sum':
            # Linear decay with periodic component
            a_1 = np.random.uniform(0.2, 0.8)
            a_2 = np.random.uniform(0.2, 0.8)
            a_3 = np.random.uniform(0.2, 0.8)
            start = np.random.uniform(0.5, 1.0)
            end = np.random.uniform(-1.0, 0.0) 
            X = torch.linspace(start, end, seq_len)
            signal = a_1 * X + a_2 * torch.sin(2 * np.pi * t + phase1) + a_3 * torch.cos(4 * np.pi * t + phase2)
        elif function_type == 'custom_decay':
            # Custom decay function f(t) that tends to 0
            if custom_func is not None:
                decay = custom_func(t)
            else:
                decay = 1.0 / (1.0 + lambda_decay * t)
            signal = decay * (torch.sin(freq1 * t + phase1) + 0.3 * torch.cos(freq2 * t + phase2))

        # Add Gaussian noise
        noise_level = np.random.uniform(0.05, 0.15)
        noise = noise_level * torch.randn_like(signal)
        signal = signal + noise

        # Normalize
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)

        all_samples.append(signal.unsqueeze(-1))

    return torch.stack(all_samples)