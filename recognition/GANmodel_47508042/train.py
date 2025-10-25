import os
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm

from modules import VQVAE
from dataset import HipMRIDataset
from utils import ensure_dir, save_pair_grid, batch_ssim

def train_loop(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    ds = HipMRIDataset(args.data_root, image_size=args.image_size, max_slices_per_volume=args.max_slices, recursive=True)
    # split training and testing 90/10
    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_set, val_set = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = VQVAE(in_ch=1, hidden=args.hidden, z_channels=args.z_ch, num_embeddings=args.num_embeddings, beta=args.beta).to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    ensure_dir(args.output_dir)
    best_ssim = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [train]'):
            x = batch.to(device)
            x_recon, vq_loss, _ = model(x)
            recon_loss = F.mse_loss(x_recon,x)
            loss = recon_loss + vq_loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            train_loss += loss.item() * x.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        ssim_scores = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]")):
                x = batch.to(device)
                x_recon, vq_loss, _ = model(x)
                recon_loss = F.mse_loss(x_recon, x)
                loss = recon_loss + vq_loss
                val_loss += loss.item() * x.size(0)
                ssim_scores.append(batch_ssim(x, x_recon))

                # save first validation batch reconstructions
                if i == 0:
                    save_pair_grid(x.cpu(), x_recon.cpu(),
                                   os.path.join(args.output_dir, f"epoch_{epoch:03d}_recon.png"), n=min(4, x.size(0)))

        val_loss = val_loss / len(val_loader.dataset)
        mean_ssim = float(sum(ssim_scores) / len(ssim_scores))

        print(f"Epoch {epoch}  TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  ValSSIM={mean_ssim:.4f}")

        # checkpoint by SSIM
        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimiser.state_dict(),
                'ssim': best_ssim
            }, os.path.join(args.output_dir, "best_checkpoint.pth"))
            print(f"  Saved best model (SSIM={best_ssim:.4f})")

        # save periodic snapshot
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"vqvae_epoch_{epoch}.pt"))

    print("Training complete. Best SSIM:", best_ssim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--z_ch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--max_slices", type=int, default=40)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()
    ensure_dir(args.output_dir)
    train_loop(args)