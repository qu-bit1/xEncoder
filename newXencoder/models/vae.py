import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiEncoderVAE(nn.Module):
    def __init__(self, source_dim, target_dim, latent_dim, hidden_dim):
        super(MultiEncoderVAE, self).__init__()

        # Source Encoder
        self.source_encoder = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.mu_source = nn.Linear(hidden_dim, latent_dim)
        self.logvar_source = nn.Linear(hidden_dim, latent_dim)

        # Target Encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.mu_target = nn.Linear(hidden_dim, latent_dim)
        self.logvar_target = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, target_dim)
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample z from a normal distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_source(self, x):
        h = self.source_encoder(x)
        mu, logvar = self.mu_source(h), self.logvar_source(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode_target(self, x):
        h = self.target_encoder(x)
        mu, logvar = self.mu_target(h), self.logvar_target(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_source, x_target=None):
        if x_target is not None:
            z_source, mu_source, logvar_source = self.encode_source(x_source)
            z_target, mu_target, logvar_target = self.encode_target(x_target)

            recon_from_source = self.decode(z_source)
            recon_from_target = self.decode(z_target)

            return recon_from_source, recon_from_target, z_source, z_target, mu_source, logvar_source, mu_target, logvar_target
        else:
            z_source, mu_source, logvar_source = self.encode_source(x_source)
            recon_from_source = self.decode(z_source)
            return recon_from_source, z_source, mu_source, logvar_source
