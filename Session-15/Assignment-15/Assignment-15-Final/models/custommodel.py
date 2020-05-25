import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, xchannels, ychannels, kernel_size=(3, 3)):
        super(Encoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(xchannels, ychannels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(ychannels, ychannels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small


class Decoder(nn.Module):
    def __init__(self, xbigchannels, xchannels, ychannels, kernel_size=3):
        super(Decoder, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu2d(xbigchannels + xchannels, ychannels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(ychannels, ychannels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(ychannels, ychannels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.upsample(x, size=(height, width), mode='bilinear')
        x = torch.cat([x, down_tensor], 1)
        x = self.decode(x)
        return x


class CustomNet(nn.Module):
    def __init__(self, in_shape):
        super(CustomNet, self).__init__()
        channels, height, width = in_shape

        self.down1 = Encoder(channels, 32, kernel_size=3)
        self.down2 = Encoder(32, 64, kernel_size=3)
        self.down3 = Encoder(64, 128, kernel_size=3)
        self.down4 = Encoder(128, 256, kernel_size=3)

        self.center = nn.Sequential(
            ConvBnRelu2d(256, 256, kernel_size=3, padding=1),
        )

        self.up4 = Decoder(256, 256, 128, kernel_size=3)
        self.up3 = Decoder(128, 128, 64, kernel_size=3)
        self.up2 = Decoder(64, 64, 32, kernel_size=3)
        self.up1 = Decoder(32, 32, 32, kernel_size=3)
        self.classify_m = nn.Conv2d(32, 1, kernel_size=1, bias=True)
        self.classify_d = nn.Conv2d(32, 3, kernel_size=1, bias=True)

    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        
        xd = self.center(out)
        out = self.center(out)

        out = self.up4(out, down4)
        out = self.up3(out, down3)
        out = self.up2(out, down2)
        out = self.up1(out, down1)
        
        # Mask
        out = self.classify_m(out)

        outd = self.up4(xd, down4)
        outd = self.up3(outd, down3)
        outd = self.up2(outd, down2)
        outd = self.up1(outd, down1)

        # Depth map
        outd = self.classify_d(outd)

        return out, outd