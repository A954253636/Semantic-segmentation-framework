import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)
        return self.gamma * out + input


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rates[0],
                               padding=dilation_rates[0])
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rates[1],
                               padding=dilation_rates[1])
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rates[2],
                               padding=dilation_rates[2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 5, out_channels=out_channels * 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.pool(x)
        x5 = self.conv5(x5)
        x5 = torch.nn.functional.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)
        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x6 = self.conv1x1(x6)
        return x6


class Spatial_Attention_Module(nn.Module):
    def __init__(self, k: int):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size=(k, k), stride=(1, 1), padding=((k - 1) // 2, (k - 1) // 2),
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim=1, keepdim=True)
        max_x, _ = self.max_pooling(x, dim=1, keepdim=True)
        v = self.conv(torch.cat((max_x, avg_x), dim=1))
        v = self.sigmoid(v)
        return x * v


class Conv_change_channels(nn.Module):
    def __init__(self, inchanels: int, outchannels: int):
        super(Conv_change_channels, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchanels, out_channels=outchannels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Res_Block(nn.Module):
    def __init__(self, chanels, kernel_size=3, stride=1, padding=1):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=chanels, out_channels=chanels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(chanels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=chanels, out_channels=chanels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(chanels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        mid = x
        x = self.conv1(x)
        x = mid + x
        x = F.relu(x, inplace=True)
        return x


class Res_Attention(nn.Module):
    def __init__(self, chanels, kernel_size=3, stride=1, padding=1):
        super(Res_Attention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=chanels, out_channels=chanels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(chanels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=chanels, out_channels=chanels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(chanels),
            nn.ReLU(inplace=True),
        )
        self.SAattention = Spatial_Attention_Module(3)

    def forward(self, x):
        mid = x
        x = self.conv1(x)
        x = mid + x
        x = F.relu(x, inplace=True)
        x = self.SAattention(x)
        return x


class TranConv(nn.Module):
    def __init__(self, inchanels, outchanels, kernel_size=2, stride=2):
        super(TranConv, self).__init__()
        self.tranconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=inchanels, out_channels=outchanels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(outchanels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchanels, out_channels=outchanels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchanels),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.tranconv(x)
        return x


class MyNet(nn.Module):
    def __init__(self, in_channel=3, num_class=2):
        super(MyNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.change_chanel1 = Conv_change_channels(3, 32)
        self.resconv11 = Res_Block(chanels=32)
        self.resconv12 = Res_Block(chanels=32)
        self.resconv13 = Res_Block(chanels=32)
        self.resconv14 = Res_Block(chanels=32)
        self.aspp = ASPP(in_channels=32, out_channels=16, dilation_rates=[6, 12, 18])
        self.Res_Attention1 = Res_Attention(chanels=32)
        self.pool1 = nn.MaxPool2d(2)
        self.change_chanel2 = Conv_change_channels(32, 64)
        self.resconv21 = Res_Block(chanels=64)
        self.resconv22 = Res_Block(chanels=64)
        self.Res_Attention2 = Res_Attention(chanels=64)
        self.pool2 = nn.MaxPool2d(2)
        self.change_chanel3 = Conv_change_channels(64, 128)
        self.resconv31 = Res_Block(chanels=128)
        self.resconv32 = Res_Block(chanels=128)
        self.Res_Attention3 = Res_Attention(chanels=128)
        self.pool3 = nn.MaxPool2d(2)
        self.change_chanel4 = Conv_change_channels(128, 256)
        self.resconv41 = Res_Block(chanels=256)
        self.resconv42 = Res_Block(chanels=256)
        self.Res_Attention4 = Res_Attention(chanels=256)
        self.pool4 = nn.MaxPool2d(2)
        self.resconv51 = Res_Block(chanels=256)
        self.self_attention = SelfAttention(256)
        self.resconv52 = Res_Block(chanels=256)
        self.tran16 = TranConv(256, 256)
        self.tran8 = TranConv(512, 128)
        self.tran4 = TranConv(256, 64)
        self.tran2 = TranConv(128, 32)
        self.out_conv1 = Conv_change_channels(96, 32)
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=num_class, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x1 = self.resconv12(self.resconv11(self.change_chanel1(x)))
        x_aspp = self.aspp(self.resconv14(self.resconv13(x1)))
        xA1 = self.Res_Attention1(x1)  # 经过注意力
        x2 = self.pool1(xA1)  # 下采样
        x2 = self.Res_Attention2(self.resconv22(self.resconv21(self.change_chanel2(x2))))
        x3 = self.pool2(x2)
        x3 = self.Res_Attention3(self.resconv32(self.resconv31(self.change_chanel3(x3))))
        x4 = self.pool3(x3)
        x4 = self.Res_Attention4(self.resconv42(self.resconv41(self.change_chanel4(x4))))
        x5 = self.pool4(x4)
        x5 = self.resconv52(self.self_attention(self.resconv52(x5)))
        t1 = self.tran16(x5)
        t2 = self.tran8(torch.cat([t1, x4], dim=1))
        t3 = self.tran4(torch.cat([t2, x3], dim=1))
        t4 = self.tran2(torch.cat([t3, x2], dim=1))
        out = torch.cat([t4, x1, x_aspp], dim=1)
        out = self.out_conv2(self.out_conv1(out))
        return (out)