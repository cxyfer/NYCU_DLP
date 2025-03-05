import torch
import torch.nn as nn
from torchsummary import summary # pip install torchsummary

# reference paper: https://ieeexplore.ieee.org/document/8716937

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

class SCCNet(nn.Module):
    # def __init__(self, numClasses=4, timeSample=438, Nu=22, C=22, Nt=1, Nc=20, dropoutRate=0.5):
    def __init__(self, numClasses=4, timeSample=438, Nu=44, C=22, Nt=1, Nc=20, dropoutRate=0.5):    
        super(SCCNet, self).__init__()

        # First convolutional: Spatial component analysis
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(C, Nt), stride=1, padding=0),
            Permute2d((0, 2, 1, 3)), 
            nn.BatchNorm2d(1)
        )

        # Second convolutional: Spactial-Temporal convolution
        # 12: 0.1s, 6: 0.05s
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=Nc, kernel_size=(Nu, 12), stride=1, padding=(0, 6)),
            nn.BatchNorm2d(Nc),
        )

        # Square layer
        self.square = SquareLayer()

        # Dropout layer: prevent overfitting
        self.dropout = nn.Dropout(dropoutRate)

        # Pooling layer: perform smoothing in the temporal domain
        # 62: 0.5s, 12: 0.1s
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))

        # Fully connected layer
        # fc_inSize = self.get_size(C, timeSample)[1]
        fc_inSize = Nc * ((timeSample - 62) // 12 + 1)
        self.fc = nn.Linear(in_features=fc_inSize, out_features=numClasses, bias=True)

        # Softmax layer
        self.softmax = nn.Softmax(dim=1)

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
        # print(f'After avgpool: {x.shape}')

        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # print(f'After flatten: {x.shape}')

        # Fully connected layer
        x = self.fc(x)
        # print(f'After fc: {x.shape}')

        # Softmax layer
        x = self.softmax(x)

        # print(f'Output shape: {x.shape}')
        return x

    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, n_chan, n_sample):
        data = torch.ones((1, 1, n_chan, n_sample))
        x = self.conv1(data)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        return x.size()

if __name__ == '__main__':
    # model = SCCNet(numClasses=4, timeSample=438, Nu=22, Nc=20, dropoutRate=0.5, Nt=2)
    model = SCCNet(numClasses=4, timeSample=438, Nu=44, C=22, Nc=20, Nt=2, dropoutRate=0.5)
    print(model)
    input = torch.randn(32, 22, 438) # Batch size 32, 22 EEG channels, 438 time samples
    output = model(input)
    print(f"Output shape: {output.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=(22, 438))