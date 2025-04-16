import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiEncoderAutoencoder(nn.Module):
    def __init__(self, source_dim, target_dim, latent_dim, hidden_dim):
        super(MultiEncoderAutoencoder, self).__init__()

        # Shared bottleneck layer to align encoders
        self.bottleneck = nn.Linear(hidden_dim, latent_dim)

        # Source Encoder
        self.source_encoder = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            self.bottleneck  # Shared bottleneck
        )

        # Target Encoder (uses same bottleneck)
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            self.bottleneck  # Shared bottleneck
        )

        # Decoder (predicts larger gene set)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
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