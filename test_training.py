"""
Training script for diffusion-based time series generation model.
================================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from models.diffusion_ts_model import (
    DiffusionTransformer,
    GaussianDiffusion,
)
from utils.logger_setup import setup_logging
from utils.data_generators import TimeSeriesDataset, create_synthetic_dataset
from utils.visualisation import visualize_samples, visualize_decomposition
import argparse
from models.train_diffusion_ts import train_epoch, evaluate

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train a diffusion-based time series generation model."
    )
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to the configuration JSON file.')
    parser.add_argument('--log_dir', type=str, default='logs/training.log',
                        help='Path to the log file.')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix to append to the file names.')
    
    args = parser.parse_args()

    logger = setup_logging(args.log_dir)
    config_filename = args.config

    # Read the dictionary from the file  # Options: 'sine', 'cosine', 'mixed', 'exponential_decay', 'custom_decay'

    with open(config_filename, 'r') as f:
        config = json.load(f)

    # Create save directory
    os.makedirs(os.path.join(config['save_dir'], args.suffix), exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    # Create datasets
    logger.info(f"Creating {config['function_type']} dataset...")
    train_data = create_synthetic_dataset(
        n_samples=config['n_train'],
        seq_len=config['seq_len'],
        dim=config['dim'],
        function_type=config['function_type'],
        lambda_decay=config['lambda_decay']
    )

    val_data = create_synthetic_dataset(
        n_samples=config['n_val'],
        seq_len=config['seq_len'],
        dim=config['dim'],
        function_type=config['function_type'],
        lambda_decay=config['lambda_decay']
    )

    train_dataset = TimeSeriesDataset(train_data)
    val_dataset = TimeSeriesDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create model
    logger.info("Creating model...")
    model = DiffusionTransformer(
        input_dim=config['dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        seq_len=config['seq_len'],
        patch_size=config['patch_size'],
        top_k=config['top_k'],
        poly_degree=config['poly_degree']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Create diffusion process
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=config['timesteps'],
        beta_schedule=config['beta_schedule'],
        loss_type=config['loss_type']
    ).to(device)

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(config['epochs']), desc="Training Epochs"):
        # Train
        train_loss = train_epoch(model, diffusion, train_loader, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = evaluate(model, diffusion, val_loader, device)
        val_losses.append(val_loss)

        # Update scheduler
        scheduler.step()

        # Print progress
        logger.info(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, os.path.join(config['save_dir'], args.suffix, 'best_model.pt'))

        # Generate samples periodically
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(config['save_dir'], args.suffix, f'samples_epoch_{epoch+1}.png')
            visualize_samples(model, diffusion, n_samples=5, seq_len=config['seq_len'], save_path=save_path)

    # Plot training curves
    plt.figure(figsize=(10, 5))
    
    plt.plot(np.log10(train_losses[1:]), label='Train Loss')
    plt.plot(np.log10(val_losses), label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

    # Final visualization
    print("\nGenerating final visualizations...")
    visualize_samples(model, diffusion, n_samples=10, seq_len=config['seq_len'],
                     save_path=os.path.join(config['save_dir'], args.suffix, 'final_samples.png'))
    visualize_decomposition(model, train_data[0],
                          save_path=os.path.join(config['save_dir'], args.suffix, 'decomposition.png'))

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to {config['save_dir']}/")


if __name__ == '__main__':
    main()
