import torch
import os
from models.diffusion_ts_model import (
    DiffusionTransformer,
    GaussianDiffusion,
)
from utils.logger_setup import setup_logging
from utils.visualisation import visualize_diffusion_process
import argparse

def main():
    
    parser = argparse.ArgumentParser(
        description="Test a diffusion-based time series generation model."
    )
    parser.add_argument('--model', type=str, default='experiment/model.pth',
                        help='Path to the trained model file.')
    parser.add_argument("--save_dir", type=str, default="results/",
                        help="Directory to save the results.")
    parser.add_argument('--experiment_name', type=str, default='test_experiment',
                        help='Name of the experiment for saving results.')
    args = parser.parse_args()

    logger = setup_logging('logs/test_trained_model.log')
    model_filename = args.model
    logger.info(f"Loading model from: {model_filename}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    # Load model
    saved_model = torch.load(model_filename, map_location=device)
    model_state = saved_model['model_state_dict']
    config = saved_model['config']
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
    model.load_state_dict(model_state)

    save_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    diffusion = GaussianDiffusion(
        model=model,
        timesteps=config['timesteps'],
        beta_schedule=config['beta_schedule'],
        loss_type=config['loss_type']
    ).to(device)

    visualize_diffusion_process(model=model, diffusion=diffusion, n_samples=5, seq_len=config['seq_len'], save_path=os.path.join(save_dir, 'diffusion_process.png'))
if __name__ == "__main__":
    main()