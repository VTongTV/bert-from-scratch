import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
