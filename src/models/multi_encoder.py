import torch
import torch.nn as nn

class MultiEncoderAutoencoder(nn.Module):
    def __init__(self, source_dim, target_dim, latent_dim, hidden_dim):
        super(MultiEncoderAutoencoder, self).__init__()

        # Shared initial transformation for both encoders
        self.shared_encoder_layer = nn.Linear(source_dim, hidden_dim)

        # Source Encoder (smaller gene set)
        self.source_encoder = nn.Sequential(
            nn.BatchNorm1d(source_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            self.shared_encoder_layer,  # Shared weight layer
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Target Encoder (larger gene set)
        self.target_encoder = nn.Sequential(
            nn.BatchNorm1d(target_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(target_dim, hidden_dim),  # Not shared since target_dim is different
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Shared Decoder (predicts the larger gene set)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 4, target_dim),
            nn.Softplus()  # Softplus ensures non-negative gene expression without truncating small values
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