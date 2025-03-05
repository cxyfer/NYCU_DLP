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

    def forward(self, x):
        out = self.block(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        # out = self.relu(out)
        return F.relu(out)
    
class EncoderBlock(nn.Module):
    """
        Actually, this is a layer of ResNet34
    """
    def __init__(self, in_channels, out_channels, n_blocks, stride, is_shortcut=True):
        super(EncoderBlock, self).__init__()

        if is_shortcut:
            shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_channels))
        else:   
            shortcut = None

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, shortcut))

        for _ in range(1, n_blocks):
             layers.append(ResidualBlock(out_channels, out_channels))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        out = self.blocks(x)
        return out
    
class DecoderBlock(nn.Module):
    # def __init__(self, in_channels, out_channels):
    def __init__(self, in_channels, out_channels, up_in_channels=None, up_out_channels=None):
        super(DecoderBlock, self).__init__()
        if up_in_channels is None:
            up_in_channels = in_channels
        if up_out_channels is None:
            up_out_channels = out_channels
        self.up = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    # x1-upconv , x2-downconv
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.block(x)
    
class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ResNet34_UNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.enc2 = EncoderBlock(64, 64, 3, 1, is_shortcut=False)
        self.enc3 = EncoderBlock(64, 128, 4, 2)
        self.enc4 = EncoderBlock(128, 256, 6, 2)
        self.enc5 = EncoderBlock(256, 512, 3, 2)

        self.bridge = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)
        self.dec5 = DecoderBlock(128, 64, 64, 64)

        self.lastlayer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        maxpool = self.maxpool(enc1)
        enc2 = self.enc2(maxpool)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        center = self.bridge(enc5)
        
        dec1 = self.dec1(center, enc5)
        dec2 = self.dec2(dec1, enc4)
        dec3 = self.dec3(dec2, enc3)
        dec4 = self.dec4(dec3, enc2)
        dec5 = self.dec5(dec4, enc1)

        out = self.lastlayer(dec5)
        return out
    
if __name__ == '__main__':
    model = ResNet34_UNet(3, 1)
    input = torch.randn(8, 3, 256, 256) # Batch size 8, 3 channels, 256 x 256 image
    output = model(input)
    print(f"Output shape: {output.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=(3, 256, 256))