import torch
import torch.nn as nn

_str_to_activation = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}


class ConvEncoder(nn.Module):

    def __init__(self, input_shape, embed_size, activation, depth=32):

        super().__init__()

        self.input_shape = input_shape
        self.act_fn = _str_to_activation[activation]
        self.depth = depth
        self.kernels = [4, 4, 4, 4]

        self.embed_size = embed_size

        layers = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = input_shape[0] if i == 0 else self.depth * (2 ** (i - 1))
            out_ch = self.depth * (2**i)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=2))
            layers.append(self.act_fn)

        self.conv_block = nn.Sequential(*layers)
        self.fc = (
            nn.Identity()
            if self.embed_size == 1024
            else nn.Linear(1024, self.embed_size)
        )

    def forward(self, inputs):
        reshaped = inputs.reshape(-1, *self.input_shape)
        embed = self.conv_block(reshaped)
        embed = torch.reshape(embed, (*inputs.shape[:-3], -1))
        embed = self.fc(embed)

        return embed
