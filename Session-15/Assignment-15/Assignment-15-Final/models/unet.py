# Reference : https://raw.githubusercontent.com/usuyama/pytorch-unet/master/pytorch_unet.py
import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_channels_mask, n_channels_depth):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.sigmoid = nn.Sigmoid()
        self.conv_last_mask = nn.Conv2d(64, n_channels_mask, 1)
        self.conv_last_depth = nn.Conv2d(64, n_channels_depth, 1)
        
        
    def forward(self, sample):
        bgfgs = sample['bg_fg']

        conv1 = self.dconv_down1(bgfgs)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        conv4 = self.dconv_down4(x)
        conv4d = self.dconv_down4(x)
       
        x = self.upsample(conv4)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        x = self.conv_last_mask(x)
        outMask = self.sigmoid(x) # Mask
        
        # Second output
        x1 = self.upsample(conv4d)        
        x1 = torch.cat([x1, conv3], dim=1)
        
        x1 = self.dconv_up3(x1)
        x1 = self.upsample(x1)        
        x1 = torch.cat([x1, conv2], dim=1)       

        x1 = self.dconv_up2(x1)
        x1 = self.upsample(x1)        
        x1 = torch.cat([x1, conv1], dim=1)   
        
        x1 = self.dconv_up1(x1)

        outDepth = self.conv_last_depth(x1) # Depth
        
        return outMask, outDepth