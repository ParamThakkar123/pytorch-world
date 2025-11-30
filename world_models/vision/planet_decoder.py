import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDecoder(nn.Module):
    """A Convolutional Neural Network (CNN) decoder for reconstructing image outputs."""

    def __init__(
        self, state_size, latent_size, embedding_size, activation_function="relu"
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(latent_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, latent, state):
        hidden = self.fc1(torch.cat([state, latent], dim=1))
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation
