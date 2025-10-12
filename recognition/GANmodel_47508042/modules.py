import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_ch=1, hidden=128, z_channels=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, hidden//2, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden//2, hidden, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden, z_channels, 3, 1, 1),
        )
        
    def forward(self, x):
        return self.enc(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, hidden=128, z_channels=64):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(z_channels, hidden, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden, hidden//2, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden//2, out_ch, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.dec(z)