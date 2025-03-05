import math
import torch
import torch.nn as nn

class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x ** 2

class Permute2d(nn.Module):
    def __init__(self, shape):
        super(Permute2d, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.permute(x, self.shape)
    
# reference paper: https://ieeexplore.ieee.org/document/8716937

class SCCNet_bak(nn.Module):
    """
        C: number of EEG input channels
        timeSample: number of EEG input time samples.
    """
    def __init__(self, numClasses=4, timeSample=438, Nu=None, C=22, Nc=20, Nt=1, dropoutRate=0.5):
        super(SCCNet_bak, self).__init__()

        Nu = C if Nu is None else Nu
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(C, Nt), stride=1, padding=0),
            nn.BatchNorm2d(Nu),
            SquareLayer(),
            nn.Dropout(dropoutRate),
        )
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=Nu, out_channels=Nc, kernel_size=(1, 12), stride=1, padding=(0, 11)),
            nn.BatchNorm2d(Nc),
            SquareLayer(),
            nn.Dropout(dropoutRate),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 62))
        self.fc = nn.Linear(in_features= Nc * ((timeSample - 62) // 62 + 1), out_features=numClasses)

    def forward(self, x):
        print(f'Input shape: {x.shape}')
        x = x.unsqueeze(1)
        print(f'After unsqueeze: {x.shape}')
        print(x.dtype)
        x = self.conv1(x)
        print(f'After conv1: {x.shape}')
        x = self.conv2(x)
        print(f'After conv2: {x.shape}')
        x = self.avgpool(x)
        print(f'After avgpool: {x.shape}')
        x = torch.log(x)
        print(f'After log: {x.shape}')
        x = x.view(x.size()[0], -1)
        print(f'After view: {x.shape}')
        x = self.fc(x)
        print(f'After fc: {x.shape}')
        return x

    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        pass

if __name__ == '__main__':
    model = SCCNet_bak(numClasses=4, timeSample=438, Nu=22, Nc=20, dropoutRate=0.5, Nt=1)
    print(model)
    input = torch.randn(32, 22, 438)  # Batch size of 32, 22 EEG channels, 438 time samples
    output = model(input)
    print(f"Output shape: {output.shape}")