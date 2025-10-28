import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    compresses the input image to latent representation
    """
    def __init__(self, in_ch=1, hidden=128, z_channels=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, hidden//2, 4, 2, 1),  # downsample 1
            nn.ReLU(True),
            nn.Conv2d(hidden//2, hidden, 4, 2, 1),  # downsample 2
            nn.ReLU(True),
            nn.Conv2d(hidden, z_channels, 3, 1, 1),  # project to latent space
        )

    def forward(self, x):
        """
        forward pass through the encoder
        """
        return self.enc(x)  # returns latent representation z_e

class Decoder(nn.Module):
    """
    reconstructs the image from latent representation
    """
    def __init__(self, out_ch=1, hidden=128, z_channels=64):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(z_channels, hidden, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden, hidden//2, 4, 2, 1),  # upsample 1
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden//2, out_ch, 4, 2, 1),  # upsample 2
            nn.Sigmoid()  # output in [0,1]
        )

    def forward(self, z):
        """
        forward pass through the decoder
        """
        return self.dec(z)  # returns reconstructed image

class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE
    """
    def __init__(self, num_embeddings=512, embedding_dim=64, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings  # number of discrete embeddings
        self.embedding_dim = embedding_dim  # dimension of each embedding
        self.beta = beta  # commitment loss weight

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        """
        forward pass through the VQ layer
        """
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        # compute distances
        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        # encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        # quantize and reshape
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(z_perm.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.beta * e_latent_loss

        # estimator
        quantized = z + (quantized - z).detach()

        # visualisation
        encoding_indices = encoding_indices.view(z.shape[0], z.shape[2], z.shape[3])
        return quantized, loss, encoding_indices

class VQVAE(nn.Module):
    """
    VQ-VAE model combining encoder, vector quantizer, and decoder
    """
    def __init__(self, in_ch=1, hidden=128, z_channels=64, num_embeddings=512, embedding_dim=64, beta=0.25):
        super().__init__()
        assert z_channels == embedding_dim
        self.encoder = Encoder(in_ch, hidden, z_channels)
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, beta=beta)
        self.decoder = Decoder(in_ch, hidden, z_channels)

    def forward(self, x):
        """
        forward pass through the VQ-VAE
        """
        z_e = self.encoder(x)
        quantized, vq_loss, indices = self.vq(z_e)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, indices
