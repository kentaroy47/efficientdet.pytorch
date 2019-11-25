# from https://github.com/Michael-Jing/EfficientDet-pytorch/blob/master/efficientdet_pytorch/BiFPN.py

import torch
from torch import nn
from torch.nn import functional as F

class BiFPNBlock(nn.Module):
    def __init__(self, W_bifpn):
        # TODO:
        # determine the number of in_channels 
        super().__init__()
        self.W_bifpn = W_bifpn 
        self.p6_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p7_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p7_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_td_conv = nn.Conv2d(in_channels, self.W_bifpn, kernel_size, stride=1, padding=0, dilation=1, groups=in_channels, bias=True, padding_mode='zeros')
        self.p5_td_conv = nn.Conv2d(in_channels, self.W_bifpn, kernel_size, stride=1, padding=0, dilation=1, groups=in_channels, bias=True, padding_mode='zeros')
        self.p4_td_conv = nn.Conv2d(in_channels, self.W_bifpn, kernel_size, stride=1, padding=0, dilation=1, groups=in_channels, bias=True, padding_mode='zeros')
        self.p3_out_conv = nn.Conv2d(in_channels, self.W_bifpn, kernel_size, stride=1, padding=0, dilation=1, groups=in_channels, bias=True, padding_mode='zeros')
        self.p4_out_conv = nn.Conv2d(in_channels, self.W_bifpn, kernel_size, stride=1, padding=0, dilation=1, groups=in_channels, bias=True, padding_mode='zeros')
        self.p5_out_conv = nn.Conv2d(in_channels, self.W_bifpn, kernel_size, stride=1, padding=0, dilation=1, groups=in_channels, bias=True, padding_mode='zeros')
        self.p6_out_conv = nn.Conv2d(in_channels, self.W_bifpn, kernel_size, stride=1, padding=0, dilation=1, groups=in_channels, bias=True, padding_mode='zeros')
        self.p7_out_conv = nn.Conv2d(in_channels, self.W_bifpn, kernel_size, stride=1, padding=0, dilation=1, groups=in_channels, bias=True, padding_mode='zeros')

    def forward(self, input):
        epsilon = 0.0001
        p3, p4, p5, p6, p7 = input 
        size_of_p3 = p3.shape[2:]
        size_of_p4 = p4.shape[2:]
        size_of_p5 = p5.shape[2:]
        size_of_p6 = p6.shape[2:]
        size_of_p7 = p7.shape[2:]
        # I'm not sure if each of the convolution here share weights, 
        # I'll implement as each convolution has their own weights by now
        p6_td = self.p6_td_conv((self.p6_td_w1 * p6 + self.p6_td_w2 * resize(p7, size_of_p6)) /
                                 (self.p6_td_w1 + self.p6_td_w2 + epsilon))
        p5_td = self.p5_td_conv((self.p5_td_w1 * p5 + self.p5_td_w2 * resize(p6, size_of_p5)) /
                                  (self.p5_td_w1 + self.p5_td_w2 + epsilon))
        p4_td = self.p4_td_conv((self.p4_td_w1 * p4 + self.p4_td_w2 * resize(p5, size_of_p4)) /
                                   (self.p4_td_w1 + self.p4_td_w2 + epsilon))
        p3_out = self.p3_out_conv((self.p3_out_w1 * p3 + self.p3_out_w2 * resize(p4_td, size_of_p3)) /
                                    (self.p3_out_w1 + self.p3_out_w2 + epsilon))
        p4_out = self.p4_out_conv((self.p4_out_w1 * p4 + self.p4_out_w2 * p4_td + self.p4_out_w3 * resize(p3_out, size_of_p4))
                                / (self.p4_out_w1 + self.p4_out_w2 + self.p4_out_w3 + epsilon))
        p5_out = self.p5_out_conv((self.p5_out_w1 * p5 + self.p5_out_w2 * p5_td + self.p5_out_w3 * resize(p4_out, size_of_p5))
                                / (self.p5_out_w1 + self.p5_out_w2 + self.p5_out_w3 + epsilon))
        p6_out = self.p6_out_conv((self.p6_out_w1 * p6 + self.p6_out_w2 * p6_td + self.p6_out_w3 * resize(p5_out, size_of_p6)) 
                                    / (self.p6_out_w1 + self.p6_out_w2 + self.p6_out_w3 + epsilon))
        p7_out = self.p7_out_conv((self.p7_out_w1 * p7 + self.p7_out_w3 * resize(p6_out, size_of_p7)) /
                                    (self.p7_out_w1 + self.p7_out_w3 + epsilon))
        return [p3_out, p4_out, p5_out, p6_out, p7_out]

class BiFPN(nn.Module):

    def __init__(self, compound_coefficient):
        super().__init__()
        self.compound_coefficient = compound_coefficient
        self.body = self.create_body(compound_coefficient)

    def create_body(self, compound_coefficient):
        D_bifpn = compound_coefficient + 2 
        W_bifpn_dict = {0: 64,
                      1: 88,
                      2: 112,
                      3: 160,
                      4: 224,
                      5: 288,
                      6: 384}
        return [BiFPNBlock(W_bifpn_dict.get(compound_coefficient)) for _ in range(D_bifpn)]

    def forward(self, input):
        x = input
        for block in self.body:
            x = block.forward(x)
        return x