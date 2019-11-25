import torch
from efficientnet_pytorch import EfficientNet
from torch import nn
from torch.nn import functional as F
from ssd import ssd

from .BiFPN import BiFPN


class EfficientDet(nn.Module):
    def __init__(self, compound_coefficient=0):
        super().__init__()
        self.compound_coefficient = compound_coefficient
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0') 
        self.BiFPN = create_BiFPN(compound_coefficient)
        self.prediction_net = ssd()
    
    def create_BiFPN(self, compound_coefficient):
        return BiFPN(compound_coefficient)

    def forward(self, x):
        x = self.backbone(x)
        x = self.BiFPN(x)
        out = self.prediction_net(x)
        return out

    