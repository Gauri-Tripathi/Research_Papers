import os 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from torch.utils.tensorboard import SummaryWriter


def same_padding(kernel_size, stride):
    return max(kernel_size - stride, 0) // 2

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(ConvBnRelu, self).__init__()
        
        if padding == 'same':
            if isinstance(kernel_size, tuple):
                padding_val = tuple(same_padding(k, s) for k, s in zip(kernel_size, (stride, stride)))
            else:
                padding_val = same_padding(kernel_size, stride)
        else:
            padding_val = padding
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding_val, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv1 = ConvBnRelu(3, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvBnRelu(32, 32, kernel_size=3, padding=0)
        self.conv3 = ConvBnRelu(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv4 = ConvBnRelu(64, 80, kernel_size=1, padding=0)
        self.conv5 = ConvBnRelu(80, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool2(x)
        return x


class InceptionBlockA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlockA, self).__init__()
        
        self.branch1_1 = ConvBnRelu(in_channels, 64, kernel_size=1)
        self.branch1_2 = ConvBnRelu(64, 96, kernel_size=3)
        self.branch1_3 = ConvBnRelu(96, 96, kernel_size=3)
        
        self.branch2_1 = ConvBnRelu(in_channels, 48, kernel_size=1)
        self.branch2_2 = ConvBnRelu(48, 64, kernel_size=3)
        
        self.branch3_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch3_2 = ConvBnRelu(in_channels, 64, kernel_size=1)
        
        self.branch4 = ConvBnRelu(in_channels, 64, kernel_size=1)
        
    def forward(self, x):
        branch1 = self.branch1_1(x)
        branch1 = self.branch1_2(branch1)
        branch1 = self.branch1_3(branch1)
        
        branch2 = self.branch2_1(x)
        branch2 = self.branch2_2(branch2)
        
        branch3 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3)
        
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class InceptionBlockB(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(InceptionBlockB, self).__init__()
        
        self.branch1_1 = ConvBnRelu(in_channels, mid_channels, kernel_size=(1, 1))
        self.branch1_2 = ConvBnRelu(mid_channels, mid_channels, kernel_size=(7, 1))
        self.branch1_3 = ConvBnRelu(mid_channels, mid_channels, kernel_size=(1, 7))
        self.branch1_4 = ConvBnRelu(mid_channels, mid_channels, kernel_size=(7, 1))
        self.branch1_5 = ConvBnRelu(mid_channels, 192, kernel_size=(1, 7))
        
        self.branch2_1 = ConvBnRelu(in_channels, mid_channels, kernel_size=(1, 1))
        self.branch2_2 = ConvBnRelu(mid_channels, mid_channels, kernel_size=(1, 7))
        self.branch2_3 = ConvBnRelu(mid_channels, 192, kernel_size=(7, 1))
        
        self.branch3_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch3_2 = ConvBnRelu(in_channels, 192, kernel_size=(1, 1))
        
        self.branch4 = ConvBnRelu(in_channels, 192, kernel_size=(1, 1))
        
    def forward(self, x):
        branch1 = self.branch1_1(x)
        branch1 = self.branch1_2(branch1)
        branch1 = self.branch1_3(branch1)
        branch1 = self.branch1_4(branch1)
        branch1 = self.branch1_5(branch1)
        
        branch2 = self.branch2_1(x)
        branch2 = self.branch2_2(branch2)
        branch2 = self.branch2_3(branch2)
        
        branch3 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3)
        
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class InceptionBlockC(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlockC, self).__init__()
        
        self.branch1_1 = ConvBnRelu(in_channels, 448, kernel_size=1)
        self.branch1_2 = ConvBnRelu(448, 384, kernel_size=3)
        self.branch1_3a = ConvBnRelu(384, 384, kernel_size=(1, 3))
        self.branch1_3b = ConvBnRelu(384, 384, kernel_size=(3, 1))
        
        self.branch2_1 = ConvBnRelu(in_channels, 384, kernel_size=1)
        self.branch2_2a = ConvBnRelu(384, 384, kernel_size=(1, 3))
        self.branch2_2b = ConvBnRelu(384, 384, kernel_size=(3, 1))
        
        self.branch3_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch3_2 = ConvBnRelu(in_channels, 192, kernel_size=1)
        
        self.branch4 = ConvBnRelu(in_channels, 320, kernel_size=1)
        
    def forward(self, x):
        branch1 = self.branch1_1(x)
        branch1 = self.branch1_2(branch1)
        branch1_a = self.branch1_3a(branch1)
        branch1_b = self.branch1_3b(branch1)
        branch1 = torch.cat([branch1_a, branch1_b], dim=1)
        
        branch2 = self.branch2_1(x)
        branch2_a = self.branch2_2a(branch2)
        branch2_b = self.branch2_2b(branch2)
        branch2 = torch.cat([branch2_a, branch2_b], dim=1)
        
        branch3 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3)
        
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class ReductionBlockA(nn.Module):
    def __init__(self, in_channels):
        super(ReductionBlockA, self).__init__()
        
        self.branch1_1 = ConvBnRelu(in_channels, 64, kernel_size=1)
        self.branch1_2 = ConvBnRelu(64, 96, kernel_size=3)
        self.branch1_3 = ConvBnRelu(96, 96, kernel_size=3, stride=2, padding=0)
        
        self.branch2 = ConvBnRelu(in_channels, 384, kernel_size=3, stride=2, padding=0)
        
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def forward(self, x):
        branch1 = self.branch1_1(x)
        branch1 = self.branch1_2(branch1)
        branch1 = self.branch1_3(branch1)
        
        branch2 = self.branch2(x)
        
        branch3 = self.branch3(x)
        
        return torch.cat([branch1, branch2, branch3], dim=1)


class ReductionBlockB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionBlockB, self).__init__()
        
        self.branch1_1 = ConvBnRelu(in_channels, 192, kernel_size=1)
        self.branch1_2 = ConvBnRelu(192, 192, kernel_size=(1, 7))
        self.branch1_3 = ConvBnRelu(192, 192, kernel_size=(7, 1))
        self.branch1_4 = ConvBnRelu(192, 192, kernel_size=3, stride=2, padding=0)
        
        self.branch2_1 = ConvBnRelu(in_channels, 192, kernel_size=1)
        self.branch2_2 = ConvBnRelu(192, 320, kernel_size=3, stride=2, padding=0)
        
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def forward(self, x):
        branch1 = self.branch1_1(x)
        branch1 = self.branch1_2(branch1)
        branch1 = self.branch1_3(branch1)
        branch1 = self.branch1_4(branch1)
        
        branch2 = self.branch2_1(x)
        branch2 = self.branch2_2(branch2)
        
        branch3 = self.branch3(x)
        
        return torch.cat([branch1, branch2, branch3], dim=1)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBnRelu(in_channels, 128, kernel_size=1)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.adaptive_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super(InceptionV3, self).__init__()
        
        self.aux_logits = aux_logits
        
        self.stem = StemBlock()
        
        self.inception_a1 = InceptionBlockA(192)
        self.inception_a2 = InceptionBlockA(288)
        self.inception_a3 = InceptionBlockA(288)
        
        self.reduction_a = ReductionBlockA(288)
        
        self.inception_b1 = InceptionBlockB(768, 128)
        self.inception_b2 = InceptionBlockB(768, 160)
        self.inception_b3 = InceptionBlockB(768, 160)
        self.inception_b4 = InceptionBlockB(768, 192)
        
        if aux_logits:
            self.aux_classifier = AuxiliaryClassifier(768, num_classes)
        
        self.reduction_b = ReductionBlockB(768)
        
        self.inception_c1 = InceptionBlockC(1280)
        self.inception_c2 = InceptionBlockC(2048)
        
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        
        x = self.inception_a1(x)
        x = self.inception_a2(x)
        x = self.inception_a3(x)
        
        x = self.reduction_a(x)
        
        x = self.inception_b1(x)
        x = self.inception_b2(x)
        x = self.inception_b3(x)
        x = self.inception_b4(x)
        
        aux_out = None
        if self.aux_logits and self.training:
            aux_out = self.aux_classifier(x)
        
        x = self.reduction_b(x)
        
        x = self.inception_c1(x)
        x = self.inception_c2(x)
        
        x = self.adaptive_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.aux_logits and self.training:
            return x, aux_out
        else:
            return x