# by kentaroy47

import torch
from torch import nn
from torch.nn import functional as F

class BiFPN(nn.Module):
    def __init__(self,
                num_channels):
        super(BiFPN, self).__init__()
        self.num_channels = num_channels

    def forward(self, inputs):
        num_channels = self.num_channels
        P3_in, P4_in, P5_in, P6_in, P7_in = inputs
        for input in inputs:
            print(input.size())

        P7_up = self.Conv(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels)(P7_in)
        scale = (P6_in.size(3)/P7_up.size(3))
        
        P6_up = self.Conv(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels)(P6_in+self.Resize(scale_factor=scale)(P7_up))
        scale = (P5_in.size(3)/P6_up.size(3))
        P5_up = self.Conv(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels)(P5_in+self.Resize(scale_factor=scale)(P6_up))
        scale = (P4_in.size(3)/P5_up.size(3))
        P4_up = self.Conv(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0)(P4_in+self.Resize(scale_factor=scale)(P5_up))
        scale = (P3_in.size(3)/P4_up.size(3))
        P3_out = self.Conv(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0)(P3_in+self.Resize(scale_factor=scale)(P4_up))

        # fix to downsample by interpolation
        #print("P6_up scale",scale)
        P4_out = self.Conv(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels)(P4_in + P4_up+F.interpolate(P3_out, P4_up.size()[2:]))
        P5_out = self.Conv(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels)(P5_in + P5_up+F.interpolate(P4_out, P5_up.size()[2:]))
        P6_out = self.Conv(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels)(P6_in + P6_up+F.interpolate(P5_out, P6_up.size()[2:]))
        P7_out = self.Conv(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels)(P7_in + P7_up+F.interpolate(P6_out, P7_up.size()[2:]))
        return P3_out, P4_out, P5_out, P6_out, P7_out

    @staticmethod
    def Conv(in_channels, out_channels, kernel_size, stride, padding, groups = 1):
        features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        return features 
    @staticmethod
    def Resize(scale_factor=2, mode='bilinear'):
        upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        return upsample