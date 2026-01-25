import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvVAEEncoder(nn.Module):
    def __init__(self, img_channels, latent_size):
        super(ConvVAEEncoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2 * 2 * 256, latent_size)
        self.fc_logsigma = nn.Linear(2 * 2 * 256, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        log_sigma = self.fc_logsigma(x)

        return mu, log_sigma


class ConvVAEDecoder(nn.Module):
    def __init__(self, latent_size, img_channels):
        super(ConvVAEDecoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.sigmoid(self.deconv4(x))
        return x


class ConvVAE(nn.Module):
    def __init__(self, img_channels, latent_size):
        super(ConvVAE, self).__init__()
        self.encoder = ConvVAEEncoder(img_channels, latent_size)
        self.decoder = ConvVAEDecoder(latent_size, img_channels)

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        sigma = log_sigma.exp()
        eps = torch.randn_like(sigma)
        # Reparameterization trick
        z = eps.mul(sigma).add_(mu)
        recon_x = self.decoder(z)
        return recon_x, mu, log_sigma
