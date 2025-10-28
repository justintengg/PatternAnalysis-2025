import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import HipMRIDataset
from modules import VQVAE
from utils import ensure_dir, save_pair_grid

def visualise(checkpoint, data_root, output_dir, batch_size=8, max_slices=40):
    """
    Visualise reconstructions from a trained VQ-VAE model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    model = VQVAE(in_ch=1, hidden=128, z_channels=64, num_embeddings=512, beta=0.25).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    ds = HipMRIDataset(data_root, image_size=256, max_slices_per_volume=max_slices, recursive=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    ensure_dir(output_dir)
    with torch.no_grad():
        batch = next(iter(loader))
        x = batch.to(device)
        x_recon, _, indices = model(x)
        save_pair_grid(x.cpu(), x_recon.cpu(), os.path.join(output_dir, 'pred_sample.png'))
        print("saved reconstructions to", os.path.join(output_dir, "pred_sample.png"))
        print("indices shape:", indices.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/pred")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_slices", type=int, default=40)
    args = parser.parse_args()
    # run visualisation
    visualise(args.checkpoint, args.data_root, args.output_dir, args.batch_size, args.max_slices)