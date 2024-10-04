import torch
import torch.nn as nn
from torch.nn import functional as F

from toolbox.backbone.ResNet import resnet34


from toolbox.paper2.paper2_7.fusion import CSIB2_2
from toolbox.paper2.paper2_7.prototype import prototype2
from toolbox.paper2.paper2_7.decoder0 import decoder0, decoder01


"""decoder1 use +  ; docoder0 dont transfer to decoder1
    """
###############################################################################

class student(nn.Module):
    def __init__(self,  channels=[64, 64, 128, 256, 512], p = 24):
        super(student, self).__init__()
        self.channels = channels

        resnet_raw_model1 = resnet34(pretrained=True)
        resnet_raw_model2 = resnet34(pretrained=True)
        ###############################################
        # Backbone model
        self.encoder_thermal_conv1 = resnet_raw_model1.conv1
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool

        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool

        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        ###############################################
        # funsion encoders #
        self.p1_r = prototype2(self.channels[1], p)
        self.p2_r = prototype2(self.channels[2], p)
        self.p3_r = prototype2(self.channels[3], p)
        self.p4_r = prototype2(self.channels[4], p)
        self.p1_d = prototype2(self.channels[1], p)
        self.p2_d = prototype2(self.channels[2], p)
        self.p3_d = prototype2(self.channels[3], p)
        self.p4_d = prototype2(self.channels[4], p)

        self.fu_1 = CSIB2_2(self.channels[1])

        
        self.fu_2 = CSIB2_2(self.channels[2])

        
        self.fu_3 = CSIB2_2(self.channels[3])


        self.fu_4 = CSIB2_2(self.channels[4])


        ###############################################
        # decoders #
        ###############################################
        # enhance receive field #
        self.decoder03 = decoder0(self.channels[4], self.channels[3])
        self.decoder02 = decoder0(self.channels[3], self.channels[2])
        self.decoder01 = decoder0(self.channels[2], self.channels[1])

        self.decoder13 = decoder01(self.channels[4], self.channels[3], self.channels[1])
        self.decoder12 = decoder01(self.channels[3], self.channels[2], self.channels[1])
        self.decoder11 = decoder01(self.channels[2], self.channels[1], self.channels[1])

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.channels[4], self.channels[4], 1),
            nn.BatchNorm2d(self.channels[4]),
            nn.LeakyReLU(inplace=True),
        )

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(self.channels[1], self.channels[4], 3, 1, 1),
            nn.BatchNorm2d(self.channels[4]),
            nn.LeakyReLU(True)
        )

        self.ful_conv_out = nn.Sequential(
            nn.Conv2d(64, 6, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
                                          )

        self.out2 = nn.Sequential(
            nn.Conv2d(128, 6, 1, 1, 0),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

        self.out3 = nn.Sequential(
            nn.Conv2d(256, 6, 1, 1, 0),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        )





                

    def forward(self, rgb, d):
        ###############################################
        # Backbone model
        rgb0 = self.encoder_rgb_conv1(rgb)
        rgb0 = self.encoder_rgb_bn1(rgb0)
        rgb0 = self.encoder_rgb_relu(rgb0)

        d0 = self.encoder_thermal_conv1(d)
        d0 = self.encoder_thermal_bn1(d0)
        d0 = self.encoder_thermal_relu(d0)
        ####################################################
        ## fusion
        ####################################################
        ## layer0
        rgb1 = self.encoder_rgb_maxpool(rgb0)
        rgb1 = self.encoder_rgb_layer1(rgb1)

        d1 = self.encoder_thermal_maxpool(d0)
        d1 = self.encoder_thermal_layer1(d1)

        ## layer1 融合
        rgb1, upr1, gr1= self.p1_r(rgb1)
        d1, upd1, gd1 = self.p1_d(d1)
        f1, wr1, wd1 = self.fu_1(rgb1, d1)

        ## 传输到encoder2
        rgb2 = self.encoder_rgb_layer2(rgb1)
        d2 = self.encoder_thermal_layer2(d1)

        ## layer2 融合
        rgb2, upr2, gr2 = self.p2_r(rgb2)
        d2, upd2, gd2 = self.p2_d(d2)
        f2, wr2, wd2 = self.fu_2(rgb2, d2)

        ## 传输到encoder3
        rgb3 = self.encoder_rgb_layer3(rgb2)
        d3 = self.encoder_thermal_layer3(d2)

        ## layer3 融合
        rgb3, upr3, gr3 = self.p3_r(rgb3)
        d3, upd3, gd3 = self.p3_d(d3)
        f3, wr3, wd3= self.fu_3(rgb3, d3)

        ## 传输到encoder4
        rgb4 = self.encoder_rgb_layer4(rgb3)
        d4 = self.encoder_thermal_layer4(d3)

        ## layer4 融合
        rgb4, upr4, gr4 = self.p4_r(rgb4)
        d4, upd4, gd4= self.p4_d(d4)
        f4, wr4, wd4 = self.fu_4(rgb4, d4)

        ####################################################
        ## decoder0
        ####################################################


        decoder03 = self.decoder03(f4, f3)
        decoder02 = self.decoder02(decoder03, f2)
        decoder01 = self.decoder01(decoder02, f1)

        ####################################################
        ## decoder1
        ####################################################


        decoder13 = self.decoder13(f4, decoder03, decoder01)
        decoder12 = self.decoder12(decoder13, decoder02, decoder01)
        decoder11 = self.decoder11(decoder12, decoder01, decoder01)

        out = self.ful_conv_out(decoder11)
        out2 = self.out2(decoder12)
        out3 = self.out3(decoder13)
        out0 = self.ful_conv_out(decoder01)


        # return out, out2, out3
        return out, out2, out3, decoder01, decoder02, decoder03, \
            rgb0, d0, rgb1, d1, rgb2, d2, rgb3, d3, rgb4, d4, f1, f2, f3, f4, \
            upr1, upd1, upr2, upd2, upr3, upd3, upr4, upd4,\
            gr1, gd1, gr2, gd2, gr3, gd3, gr4, gd4

    #
if __name__ == '__main__':
    rgb = torch.randn(4, 3, 256, 256)
    d = torch.randn(4, 3, 256, 256)

    model = student()

    a = model(rgb, d)

    print(a[28].shape)
    print(a[30].shape)
    print(a[32].shape)
    print(a[34].shape)
# if __name__ == '__main__':
#     # input_rgb = torch.randn(2, 3, 256, 256)
#     # input_depth = torch.randn(2, 1, 256, 256)
#     # net = teacher_model()
#     # out = net(input_rgb, input_depth)
#     # # print("out", out.shape)
#     # print("out1", out[0].shape)
#     # print("out2", out[1].shape)
#     # print("out3", out[2].shape)
#     # print("out4", out[3].shape)
#
#     #
#     # a = torch.randn(1, 3, 256, 256)
#     # b = torch.randn(1, 3, 256, 256)
#     #
#     # model = student()
#     # from FLOP import CalParams
#     #
#     # CalParams(model, a, b)
#     # print('Total params % .2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
#     loss_t = []
#     loss1 = 1
#     loss2 = 2
#     loss_t.append(loss1)
#     loss_t.append(loss2)
#     loss_t = torch.stack(loss_t, dim=0)
#     weight = (1.0 - F.softmax(loss_t, dim=0))
#     print(weight)