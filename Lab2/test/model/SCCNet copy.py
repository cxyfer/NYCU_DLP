import math
import torch
import torch.nn as nn

# reference paper: https://ieeexplore.ieee.org/document/8716937

class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x ** 2
    
class SCCNet_bak(nn.Module):
    """
        C: number of EEG input channels
        timeSample: number of EEG input time samples.
    """
    def __init__(self, numClasses=4, timeSample=438, Nu=None, C=22, Nc=20, Nt=1, dropoutRate=0.5):
        super(SCCNet, self).__init__()

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

class Permute2d(nn.Module):
    def __init__(self, shape):
        super(Permute2d, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.permute(x, self.shape)

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=438, Nu=None, C=22, Nt=1, Nc=20,dropoutRate=0.5):
        super(SCCNet, self).__init__()
        Nu = C if Nu is None else Nu

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, Nu, (C, Nt)),
            Permute2d((0, 2, 1, 3)),
            nn.BatchNorm2d(1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                1, Nc, (Nu, 12), padding=(0, 6)
            ),
            nn.BatchNorm2d(Nc)
        )
        self.dropout = nn.Dropout(dropoutRate)
        self.avgpool = nn.AvgPool2d((1, 62), stride=(1, 12))
        fc_inSize = self.get_size(C, timeSample)[1]
        self.fc = nn.Linear(fc_inSize, numClasses, bias=True)

    def forward(self, x):
        # print(f'Input shape: {x.shape}')
        x = x.unsqueeze(1)
        # print(f'After unsqueeze: {x.shape}')
        x = self.conv1(x)
        # print(f'After conv1: {x.shape}')
        x = self.conv2(x)
        # print(f'After conv2: {x.shape}')
        x = x ** 2
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def get_size(self, n_chan, n_sample):
        data = torch.ones((1, 1, n_chan, n_sample))
        x = self.conv1(data)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        return x.size()

class Permute2d(nn.Module):
    def __init__(self, shape):
        super(Permute2d, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.permute(x, self.shape)

class SCCNet_V2(nn.Module):
    # def __init__(self, numClasses=4, timeSample=438, Nu=None, C=22, Nc=20, Nt=1, dropoutRate=0.5):
    def __init__(self, numClasses=4, timeSample=438, Nu=44, C=22, Nc=20, Nt=2, dropoutRate=0.5):
        super(SCCNet, self).__init__()

        Nu = C if Nu is None else Nu

        # First convolutional block
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(C, Nt), stride=1, padding=0),
            # nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(Nc, 2), stride=2, padding=0),
            nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(C, Nt), stride=Nt, padding=0),
            nn.BatchNorm2d(Nu),
        )

        # Second convolutional block
        # 12: 0.1s, 6: 0.05s
        self.conv2 = nn.Sequential(
            # nn.Conv2d(1, Nc, (Nu, 12), padding=(0, 6)),
            # nn.Conv2d(in_channels=Nu, out_channels=Nc, kernel_size=(1, 12), stride=1, padding=(0, 6)),
            nn.Conv2d(in_channels=Nu, out_channels=Nc, kernel_size=(1, 12), stride=1, padding=(0, 115)),
            nn.BatchNorm2d(Nc),
        )

        # Square layer
        self.square = SquareLayer()

        # Dropout layer
        self.dropout = nn.Dropout(dropoutRate)

        # Pooling layer: Temporal smoothing
        # 62: 0.5s, 12: 0.1s
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))

        # Fully connected layer
        # fc_inSize = self.get_size(C, timeSample)[1]
        fc_inSize = Nc * ((timeSample - 62) // 12 + 1)
        # self.fc = nn.Linear(fc_inSize, numClasses, bias=True)
        self.fc = nn.Linear(in_features=fc_inSize, out_features=numClasses)

        # Softmax layer
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(f'Input shape: {x.shape}')

        # Add channel dimension
        x = x.unsqueeze(1)
        # print(f'After unsqueeze: {x.shape}')

        # First convolutional block
        x = self.conv1(x)
        # print(f'After conv1: {x.shape}')

        # Second convolutional block
        x = self.conv2(x)
        # print(f'After conv2: {x.shape}')

        # Square layer
        x = self.square(x)

        # Dropout layer
        x = self.dropout(x)

        # Pooling layer: Temporal smoothing
        x = self.avgpool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # print(f'After flatten: {x.shape}')

        # Fully connected layer
        x = self.fc(x)
        # print(f'After fc: {x.shape}')

        # Softmax layer
        # x = self.softmax(x)
        return x
    
    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, n_chan, n_sample):
        pass
    
if __name__ == '__main__':
    # model = SCCNet(numClasses=4, timeSample=438, Nu=22, Nc=20, dropoutRate=0.5, Nt=2)
    model = SCCNet(numClasses=4, timeSample=438, Nu=44, C=22, Nc=20, Nt=2, dropoutRate=0.5)
    print(model)
    sample_input = torch.randn(32, 22, 438)  # Batch size of 32, 22 EEG channels, 438 time samples
    output = model(sample_input)
    print(f"Output shape: {output.shape}")