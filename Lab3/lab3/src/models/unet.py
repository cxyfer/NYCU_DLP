# Implement your UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

"""
    References:
    - [Tutorial 117 - Building your own U-Net using encoder and decoder blocks](https://www.youtube.com/watch?v=T6h-mVVpafI&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=121)
    - https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial117_building_unet_using_encoder_decoder_blocks.ipynb
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(in_channels=out_channels + skip_channels, out_channels=out_channels)
        
    def forward(self, x, skip_features):
        x = self.upconv(x)
        x = torch.cat((x, skip_features), dim=1)
        x = self.conv(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(UNet, self).__init__()

        self.enc1 = EncoderBlock(input_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        
        self.bridge = ConvBlock(512, 1024) # bridge

        self.dec1 = DecoderBlock(1024, 512, 512)
        self.dec2 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec4 = DecoderBlock(128, 64, 64)
        
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid() # for binary classification

    def forward(self, x):
        # Encoder
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        
        # Bridge
        b1 = self.bridge(p4)

        # Decoder
        d1 = self.dec1(b1, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)
        
        # Final Conv
        outputs = self.final_conv(d4)

        # Activation
        outputs = self.sigmoid(outputs)
        
        return outputs
    
if __name__ == '__main__':
    model = UNet(input_channels=3, n_classes=1)
    input = torch.randn(8, 3, 256, 256) # Batch size 8, 3 channels, 256 x 256 image
    output = model(input)
    print(f"Output shape: {output.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=(3, 256, 256))