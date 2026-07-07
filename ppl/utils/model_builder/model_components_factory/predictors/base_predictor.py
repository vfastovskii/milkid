import torch.nn as nn

class PredictorBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bag_repr):
        """
        bag_repr: shape [hidden_dim]
        => returns logit (scalar)
        """
        raise NotImplementedError




