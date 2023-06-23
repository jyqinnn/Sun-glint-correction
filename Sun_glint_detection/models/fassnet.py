from torch import nn
import torch
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return x*F.sigmoid(out)
    
class FAM(nn.Module):
    def __init__(self, d_model, d_atten, BatchNorm):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_atten, 1)
        self.activation = nn.GELU()
        self.atten_branch = SE(d_atten,ratio=4)
        self.proj_2 = nn.Conv2d(d_atten, d_model, 1)
        self.pixel_norm = BatchNorm(d_model)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.atten_branch(x)
        x = self.proj_2(x)
        x = x + shorcut

        x = self.pixel_norm(x)

        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out,BatchNorm):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out,BatchNorm):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class FASSNet(nn.Module):
    def __init__(self, img_ch=3, num_classes=3,sync_bn=False):
        super(FASSNet, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64,BatchNorm=BatchNorm)
        self.Conv2 = conv_block(ch_in=64, ch_out=128,BatchNorm=BatchNorm)
        self.Conv3 = conv_block(ch_in=128, ch_out=256,BatchNorm=BatchNorm)
        self.Conv4 = conv_block(ch_in=256, ch_out=512,BatchNorm=BatchNorm)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024,BatchNorm=BatchNorm)

        self.Up5 = up_conv(ch_in=1024, ch_out=512,BatchNorm=BatchNorm)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512,BatchNorm=BatchNorm)
        self.classifier5=nn.Sequential(nn.Conv2d(512,64,1,1),BatchNorm(64),nn.ReLU(),FAM(64,32,BatchNorm=BatchNorm),nn.Conv2d(64,2,1,1))

        self.Up4 = up_conv(ch_in=512, ch_out=256,BatchNorm=BatchNorm)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256,BatchNorm=BatchNorm)
        self.classifier4=nn.Sequential(nn.Conv2d(256,64,1,1),BatchNorm(64),nn.ReLU(),FAM(64,32,BatchNorm=BatchNorm),nn.Conv2d(64,2,1,1))

        self.Up3 = up_conv(ch_in=256, ch_out=128,BatchNorm=BatchNorm)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128,BatchNorm=BatchNorm)
        self.classifier3=nn.Sequential(nn.Conv2d(128,64,1,1),BatchNorm(64),nn.ReLU(),FAM(64,32,BatchNorm=BatchNorm),nn.Conv2d(64,2,1,1))

        self.Up2 = up_conv(ch_in=128, ch_out=64,BatchNorm=BatchNorm)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64,BatchNorm=BatchNorm)

        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.initialize_weights()

        self.upscore5 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=2, mode='bilinear')

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m,nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, SynchronizedBatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        seg5 = self.classifier5(d5)
        seg5 = self.upscore5(seg5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        seg4 = self.classifier4(d4)
        seg4 = self.upscore4(seg4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        seg3 = self.classifier3(d3)
        seg3 = self.upscore3(seg3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1,F.sigmoid(seg3),F.sigmoid(seg4),F.sigmoid(seg5)