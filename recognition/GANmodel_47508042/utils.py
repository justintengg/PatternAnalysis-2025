import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_pair_grid(x_orig, x_rec, path, n=4):
    """
    Save a grid with n original / reconstructed pairs (first n in batch).
    x_*: tensors (B, C, H, W) in [0,1]
    """
    ensure_dir(os.path.dirname(path))
    B = x_orig.shape[0]
    n = min(n, B)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3*n))
    for i in range(n):
        axes[i,0].imshow(x_orig[i].cpu().squeeze(), cmap='gray')
        axes[i,0].set_title("Original")
        axes[i,0].axis('off')
        axes[i,1].imshow(x_rec[i].cpu().squeeze(), cmap='gray')
        axes[i,1].set_title("Reconstruction")
        axes[i,1].axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

def batch_ssim(x, x_recon):
    """
    Compute mean SSIM across a batch. Assumes x and x_recon are (B,1,H,W) in [0,1].
    """
    x_np = x.detach().cpu().numpy()
    xr_np = x_recon.detach().cpu().numpy()
    scores = []
    for i in range(x_np.shape[0]):
        im1 = x_np[i,0]
        im2 = xr_np[i,0]
        try:
            s = ssim(im1, im2, data_range=1.0)
        except Exception:
            s = 0.0
        scores.append(s)
    return float(np.mean(scores))
