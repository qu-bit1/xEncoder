import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiEncoderAutoencoder(nn.Module):
    def __init__(self, source_dim, target_dim, latent_dim, hidden_dim):
        super(MultiEncoderAutoencoder, self).__init__()

        # Source Encoder (Deeper)
        self.source_encoder = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim)  # Bottleneck
        )

        # Target Encoder (Deeper)
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim)  # Bottleneck
        )

        # Decoder (Skip Connections)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, target_dim)
        )

    def encode_source(self, x):
        return self.source_encoder(x)

    def encode_target(self, x):
        return self.target_encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_source, x_target=None):
        if x_target is not None:
            z_source = self.encode_source(x_source)
            z_target = self.encode_target(x_target)

            recon_from_source = self.decode(z_source)
            recon_from_target = self.decode(z_target)

            return recon_from_source, recon_from_target, z_source, z_target
        else:
            z_source = self.encode_source(x_source)
            recon_from_source = self.decode(z_source)
            return recon_from_source, z_source
