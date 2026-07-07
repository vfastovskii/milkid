import torch.nn as nn


class EmbedderBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        x: shape [N, input_dim]
        => returns embedding [N, hidden_dim]
        """
        raise NotImplementedError