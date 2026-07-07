import torch.nn as nn


class AggregatorBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h):
        """
        h: shape [N, hidden_dim]
        => returns (bag_repr [hidden_dim], extras (dict))
        """
        raise NotImplementedError