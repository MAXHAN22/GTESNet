#!/usr/bin/env python


import torch
import torch.nn.functional as F
import torch.nn as nn


BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d


class SpatialGCN1(nn.Module):

    def __init__(self, channel):
        super(SpatialGCN1, self).__init__()
        self.node_v = nn.Conv2d(channel, channel // 2, 1)
        self.node_q = nn.Conv2d(channel, channel // 2, 1)
        self.node_k = nn.Conv2d(channel, channel // 2, 1)
        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()
        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.conv_wg = nn.Conv1d(channel // 2, channel // 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(channel // 2)
        self.out = nn.Sequential(nn.Conv2d(channel // 2, channel, kernel_size=1),
                                 BatchNorm2d(channel))

    def forward(self, x):
        b, c, h, w = x.size()

        v = self.node_v(x)  # b,c//2,h,w
        v = v.reshape(b, c // 2, -1)  # bs,c//2,h*w

        q = self.node_q(x)  # b,c//2,h,w
        q1 = self.avepool(q)  # b,c//2,1,1
        q2 = self.maxpool(q)   # b,c//2,1,1
        q = q1 + q2 # b,c//2,1,1

        k = self.node_k(x)
        k = k.reshape(b, c // 2, -1) # b,c//2,h*w

        q = q.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # b,1,c//2
        q = self.softmax(q)  # 对c进行softmax b,1,c//2
        AV = torch.matmul(q, v)  # b,1,h*w
        weight = self.sigmoid(AV)  # b,1,hw
        AV = weight * k   # b,c//2,h*w
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c//2, h, w)
        out = F.relu_(self.out(AVW) + x)

        return out




if __name__ == '__main__':
    input = torch.randn(1, 512, 7, 7)
    psa = SpatialGCN1(channel=512)
    output = psa(input)
    print(output.shape)

