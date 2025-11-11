"""
Training script for diffusion-based time series generation model.
================================================================
Contains functions for training and evaluating the model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader


def train_epoch(
        model : nn.Module,
        diffusion : nn.Module,
        dataloader : DataLoader,
        optimizer : optim.Optimizer,
        device : torch.device
        ) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = batch.to(device)
        batch_size = batch.shape[0]

        # Sample random timesteps
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)

        # Compute loss
        loss_dict = diffusion.training_loss(batch, t)
        loss = loss_dict['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model : nn.Module,
    diffusion : nn.Module,
    dataloader : DataLoader,
    device : torch.device
    ) -> float:
    """Evaluate model."""
    model.eval()
    total_loss = 0

    for batch in dataloader:
        batch = batch.to(device)
        batch_size = batch.shape[0]
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)

        loss_dict = diffusion.training_loss(batch, t)
        total_loss += loss_dict['loss'].item()

    return total_loss / len(dataloader)

