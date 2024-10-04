
import torch
from torch import nn
from torch.nn import functional as F
from toolbox.paper2.paper2_7.myGCN import SpatialGCN1



#使用 max+mean
class GTE(nn.Module):


    def __init__(self, in_c, num_p):
        super(GTE, self).__init__()
        self.num_cluster = num_p
        self.netup = torch.nn.Sequential(
                torch.nn.Conv2d(in_c, num_p, 3, padding=1)
                )
        self.centroids = torch.nn.Parameter(torch.rand(num_p, 1))   #生成（K, 1)可训练学习的张量

        self.upfc = torch.nn.Linear(num_p, in_c)

        self.gcn = SpatialGCN1(in_c)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c,in_c ,3 ,1, 1),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU()
        )

        self.transform = torch.nn.Sequential(
            nn.Conv2d(2*in_c, in_c, kernel_size=1),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU(),
            )


    def UP(self, scene):
        x = scene



        N, C, W, H = x.size()

        x = F.normalize(x, p=2, dim=1)          #对x做正则化，除2范数  对c  把c相当于个数，
        soft_assign = self.netup(x)                #通道数变为out_c 变为得分   n, K , h , w


        soft_assign = F.softmax(soft_assign, dim=1)  #通道注意力机制  对 c 进行softmax   n, K , h , w
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1)    # n, K , hw


        #调整图的大小
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        x_flatten = max_result + avg_result

        x_flatten = x_flatten.view(N, 1, -1)   # n, 1, h * w

        centroid = self.centroids       # K , 1

        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3) #对第一个维度进行扩展，其余不变； n, K, 1, h*w
        x2 = centroid.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)#在0处增加一个维度   1, K, 1, h*w

        residual = x1 - x2          #n, K, 1, h*w
        residual = residual * soft_assign.unsqueeze(2)  #n, K, 1, h*w  ;  n, K , 1, hw
        up = residual.sum(dim=-1)   #n, K, 1

        up = up.view(x.size(0), -1)  #n, K
        up = F.normalize(up, p=2, dim=1)    #对K

        up = self.upfc(up).unsqueeze(2).unsqueeze(3).repeat(1,1,W,H)

        return up

    def forward(self, feature):

        up = self.UP(feature)
        new_feature = torch.mul(up, feature) + up + feature

        gcn = self.gcn(new_feature)

        new1 = self.conv1(gcn + up)

        new2 = self.conv2(gcn + feature)

        # print(new1.shape)

        out = self.transform(torch.cat([new1, new2], dim = 1))


        return out, up, gcn



