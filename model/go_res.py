import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


# 1x1 convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, padding=0, bias=False)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# 3x3 convolution
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, 
                     stride=stride, padding=2, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class GO_Res(nn.Module):
    def __init__(self):
        super(GO_Res, self).__init__()
        self.in_channels = 256
        self.conv5x5 = conv5x5(13, 256)
        self.conv1x1 = conv1x1(13, 256)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer4 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer5 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer6 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer7 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer8 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer9 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer10 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer11 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer12 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer13 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer14 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer15 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer16 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer17 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer18 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer19 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.layer20 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.output = conv3x3(256, 1)
        nn.LogSoftmax()

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv5x5(x) + self.conv1x1(x)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        out = self.layer19(out)
        out = self.layer20(out)
        out = self.output(out)
        out = out.view(out.size(0), -1)
        return out

def save_model_stru(path, model, data_input):
    dummy_input = torch.randn(data_input).cuda()       
    torch.onnx.export(model,dummy_input, path,verbose=True)