import torch
import torch.nn as nn


class ConcatWithDDAWA(nn.Module):
    """
    Modified Concat module that applies DynamicDualAttentionWA
    to each input feature map before concatenation.
    """
    def __init__(self, dimension=1, channels_list=None):
        super(ConcatWithDDAWA, self).__init__()
        self.dim = dimension

        # Create a DDAWA module for each input in the list
        # channels_list is a list of channel sizes for each input feature map
        self.ddawa_modules = nn.ModuleList()
        if channels_list is not None:
            for ch in channels_list:
                self.ddawa_modules.append(DynamicDualAttentionWA(channels=ch))
        else:
            # Fallback if channels_list is not provided
            self.ddawa_modules = nn.ModuleList()

    def forward(self, x):
        # x is expected to be a tuple or list of tensors to concatenate
        if not isinstance(x, (list, tuple)):
            return x

        # Apply DDAWA to each input feature map
        weighted_inputs = []
        for i, feature in enumerate(x):
            if i < len(self.ddawa_modules):
                # Process the feature with its corresponding DDAWA module
                weighted_feature = self.ddawa_modules[i](feature)
                weighted_inputs.append(weighted_feature)
            else:
                # If no DDAWA module is defined for this input, use original
                weighted_inputs.append(feature)

        # Concatenate the weighted features along the specified dimension
        return torch.cat(weighted_inputs, dim=self.dim)# concat_ddawa.py
import torch.nn as nn
from .dda_wa import DynamicDualAttentionWA  # Import our custom module

class ConcatWithDDAWA(nn.Module):
    """
    Modified Concat module that applies DynamicDualAttentionWA
    to each input feature map before concatenation.
    """
    def __init__(self, dimension=1, channels_list=None):
        super(ConcatWithDDAWA, self).__init__()
        self.dim = dimension

        # Create a DDAWA module for each input in the list
        # channels_list is a list of channel sizes for each input feature map
        self.ddawa_modules = nn.ModuleList()
        if channels_list is not None:
            for ch in channels_list:
                self.ddawa_modules.append(DynamicDualAttentionWA(channels=ch))
        else:
            # Fallback if channels_list is not provided
            self.ddawa_modules = nn.ModuleList()

    def forward(self, x):
        # x is expected to be a tuple or list of tensors to concatenate
        if not isinstance(x, (list, tuple)):
            return x

        # Apply DDAWA to each input feature map
        weighted_inputs = []
        for i, feature in enumerate(x):
            if i < len(self.ddawa_modules):
                # Process the feature with its corresponding DDAWA module
                weighted_feature = self.ddawa_modules[i](feature)
                weighted_inputs.append(weighted_feature)
            else:
                # If no DDAWA module is defined for this input, use original
                weighted_inputs.append(feature)

        # Concatenate the weighted features along the specified dimension
        return torch.cat(weighted_inputs, dim=self.dim)