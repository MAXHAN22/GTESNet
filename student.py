import torch
import torch.nn as nn
from torch.nn import functional as F

from toolbox.backbone.mobilenet.MobileNetv2 import mobilenet_v2


from toolbox.paper2.paper2_7.fusion import ECF
from toolbox.paper2.paper2_7.prototype import GTE
from toolbox.paper2.paper2_7.FCD import FCDL, FCDH



###############################################################################

class student(nn.Module):
    def __init__(self,  channels=[16, 24, 32, 160, 320], p = 24):
        super(student, self).__init__()
        self.channels = channels

        mobilenet_raw_model1 = mobilenet_v2(pretrained=True)
        mobilenet_raw_model2 = mobilenet_v2(pretrained=True)
        ###############################################
        #Backbone model
        self.encoder_thermal_layer0 = mobilenet_raw_model1.features[0:2]
        self.encoder_thermal_layer1 = mobilenet_raw_model1.features[2:4]
        self.encoder_thermal_layer2 = mobilenet_raw_model1.features[4:7]
        self.encoder_thermal_layer3 = mobilenet_raw_model1.features[7:17]
        self.encoder_thermal_layer4 = mobilenet_raw_model1.features[17:18]

        self.encoder_rgb_layer0 = mobilenet_raw_model2.features[0:2]
        self.encoder_rgb_layer1 = mobilenet_raw_model2.features[2:4]
        self.encoder_rgb_layer2 = mobilenet_raw_model2.features[4:7]
        self.encoder_rgb_layer3 = mobilenet_raw_model2.features[7:17]
        self.encoder_rgb_layer4 = mobilenet_raw_model2.features[17:18]

        ###############################################
        # funsion encoders #
        self.p1_r = GTE(self.channels[1], p)
        self.p2_r = GTE(self.channels[2], p)
        self.p3_r = GTE(self.channels[3], p)
        self.p4_r = GTE(self.channels[4], p)

        self.p1_d = GTE(self.channels[1], p)
        self.p2_d = GTE(self.channels[2], p)
        self.p3_d = GTE(self.channels[3], p)
        self.p4_d = GTE(self.channels[4], p)

        self.fu_1 = ECF(self.channels[1])

        
        self.fu_2 = ECF(self.channels[2])

        
        self.fu_3 = ECF(self.channels[3])


        self.fu_4 = ECF(self.channels[4])


        ###############################################
        # decoders #
        ###############################################
        # enhance receive field #
        self.decoder03 = FCDL(self.channels[4], self.channels[3])
        self.decoder02 = FCDL(self.channels[3], self.channels[2])
        self.decoder01 = FCDL(self.channels[2], self.channels[1])

        self.decoder13 = FCDH(self.channels[4], self.channels[3], self.channels[1])
        self.decoder12 = FCDH(self.channels[3], self.channels[2], self.channels[1])
        self.decoder11 = FCDH(self.channels[2], self.channels[1], self.channels[1])


        self.ful_conv_out = nn.Sequential(
            nn.Conv2d(24, 6, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
                                          )

        self.out2 = nn.Sequential(
            nn.Conv2d(32, 6, 1, 1, 0),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

        self.out3 = nn.Sequential(
            nn.Conv2d(160, 6, 1, 1, 0),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        )





                

    def forward(self, rgb, d):
        ###############################################
        # Backbone model
        rgb0 = self.encoder_rgb_layer0(rgb)

        d0 = self.encoder_thermal_layer0(d)
        ####################################################
        ## fusion
        ####################################################
        ## layer0
        rgb1 = self.encoder_rgb_layer1(rgb0)

        d1 = self.encoder_thermal_layer1(d0)

        ## layer1 融合
        rgb1, upr1, gr1 = self.p1_r(rgb1)
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
        f3, wr3, wd3 = self.fu_3(rgb3, d3)

        ## 传输到encoder4
        rgb4 = self.encoder_rgb_layer4(rgb3)
        d4 = self.encoder_thermal_layer4(d3)

        ## layer4 融合
        rgb4, upr4, gr4 = self.p4_r(rgb4)
        d4, upd4, gd4 = self.p4_d(d4)
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

    print(a[8].shape)
    print(a[10].shape)
    print(a[12].shape)
    print(a[14].shape)
