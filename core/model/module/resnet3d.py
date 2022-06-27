import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, 
                 block,
                 layers,
                 block_inplanes,
                 args):
        super().__init__()
        
        self.args = args
        self.device = self.args.device
        n_input_channels = self.args.n_input_channels
        conv1_t_size = self.args.conv1_t_size
        conv1_t_stride = self.args.conv1_t_stride
        self.no_max_pool = self.args.no_max_pool
        shortcut_type = self.args.shortcut_type
        widen_factor = self.args.widen_factor
        self.seq_size = self.args.clip_size

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1],
                                       layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2],
                                       layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3],
                                       layers[3], shortcut_type, stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear_dim = 2048 # HG modify 

        self.linear = torch.nn.Linear(self.linear_dim, self.linear_dim)

        if self.args.use_normsoftmax:
            self.to_seq = torch.nn.Linear(self.linear_dim, self.linear_dim * self.seq_size)

        self.classifiers = []
        self.Softmax = torch.nn.Softmax(dim=-1)

    def set_classifiers(self, n_class_list):
        for n_class in n_class_list:
            self.classifiers.append(torch.nn.Linear(self.linear_dim, n_class).to(self.device))
            # self.classifiers.append(torch.nn.Linear(self.linear_dim, n_class).cuda())
            # self.classifiers.append(torch.nn.Linear(self.linear_dim, n_class * self.seq_size).to(self.device))
        
        self.classifiers = nn.ModuleList(self.classifiers) # @HG.modifty: for ddp
            
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def get_feature(self, x, key):
        x = x[key]

        x = self.conv1(x)
        x = self.bn1(x)     
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        
        x = x.view(x.size(0), -1) # resnet 50, 101 = [2, 2048]
        # x = self.linear(x)

        return x

    def forward(self, x, key='video'):
        feat = self.get_feature(x, key)
                
        outputs = []
        
        for ci in range(len(self.classifiers)): # 512 -> [3,13,7,7]
            x = self.classifiers[ci](feat)
            # x = x.view(feat.size(0), self.seq_size, -1)

            out = self.Softmax(x)
            outputs.append(out)
        
        if self.args.use_normsoftmax:
            emb = self.to_seq(feat).view(x.size(0), self.seq_size, -1)
            outputs.append(emb)
        
        return outputs


def generate_resnet(args):
    model_depth = args.model_depth
    
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), args)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), args)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), args)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), args)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), args)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), args)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), args)

    return model
