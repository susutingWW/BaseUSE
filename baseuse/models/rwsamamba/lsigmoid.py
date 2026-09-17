# Vendored from https://github.com/NikolaiKyhne/RWSAMamba-UNet (models/lsigmoid.py)
# Modification (SFI multi-sampling-rate): the slope parameter is a single
# frequency-indexed table sized for the maximum sampling rate; at runtime it
# is sliced to the current number of frequency bins. Under SFI-STFT bin k maps
# to the same physical frequency at every sampling rate, so one table is
# shared across all rates.

import torch
import torch.nn as nn


class LearnableSigmoid2D(nn.Module):
    """Learnable Sigmoid Activation Function for 2D inputs, SFI-sliced."""

    def __init__(self, in_features, beta=1):
        """
        Args:
        - in_features (int): Size of the frequency table (max bins across fs).
        - beta (float, optional): Scaling factor for the sigmoid function. Defaults to 1.
        """
        super(LearnableSigmoid2D, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        """
        Args:
        - x (torch.Tensor): [B, F, T] with F <= slope table size.

        Returns:
        - torch.Tensor: [B, F, T]
        """
        n_bins = x.size(-2)
        slope = self.slope if n_bins == self.slope.size(0) else self.slope[:n_bins]
        return self.beta * torch.sigmoid(slope * x)
