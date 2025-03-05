# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

"""
    Reference:
    - ResNet34
        - https://meetonfriday.com/posts/fb19d450/
        - https://github.com/chenyuntc/pytorch-book/blob/master/Chapter4/Chapter4.md
    - ResNet34-UNet
        - https://github.com/GohVh/resnet34-unet/
        - https://www.researchgate.net/figure/UNet-architecture-with-a-ResNet-34-encoder-The-output-of-the-additional-1x1-convolution_fig3_350858002
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = shortcut
        # if stride != 1 or in_channels != out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_channels)
        #     )
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        # out = self.relu(out)
        return F.relu(out)
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks):
        super(EncoderBlock, self).__init__()
        blocks = [ResidualBlock(in_channels, out_channels, 2)]
        for _ in range(1, n_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)
        return out, x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, skip):
        x = torch.cat([skip, x], dim=1)
        x = self.block(self.up(x))
        return x
    
class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet34_UNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.enc2 = EncoderBlock(64, 64, 3)
        self.enc3 = EncoderBlock(64, 128, 4)
        self.enc4 = EncoderBlock(128, 256, 6)
        self.enc5 = EncoderBlock(256, 512, 3)
        self.enc2 = self._make_layer(64, 64, 3, 1, is_shortcut=False)

        
        self.center = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.decoder4 = DecoderBlock(256+512, 32)
        self.decoder3 = DecoderBlock(32+256, 32)
        self.decoder2 = DecoderBlock(32+128, 32)
        self.decoder1 = DecoderBlock(32+64, 32)
        
        self.output = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def _make_layer(self, in_channels, out_channels, n_blocks, stride, is_shortcut=True):
        if is_shortcut:
            shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:   
            shortcut = None

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, shortcut))
        
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2, _ = self.enc2(enc1)
        enc3, skip1 = self.enc3(enc2)
        enc4, skip2 = self.enc4(enc3)
        enc5, skip3 = self.enc5(enc4)
        
        skip4 = enc5
        center = self.center(enc5)
        
        dec4 = self.decoder4(center, skip4)
        dec3 = self.decoder3(dec4, skip3)
        dec2 = self.decoder2(dec3, skip2)
        dec1 = self.decoder1(dec2, skip1)
        
        return self.output(dec1)
    
if __name__ == '__main__':
    model = ResNet34_UNet(3, 1)
    input = torch.randn(8, 3, 256, 256) # Batch size 8, 3 channels, 256 x 256 image
    output = model(input)
    print(f"Output shape: {output.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=(3, 256, 256))