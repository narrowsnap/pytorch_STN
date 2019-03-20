import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicStn(nn.Module):

    def __init__(self, parallel, in_feature, **kwargs):
        super(BasicStn, self).__init__()
        self.conv = conv1x1(in_feature, 128)
        self.fc_loc = nn.Sequential(
            nn.Linear(128*14*14, 64),
            nn.Tanh(),
            nn.Linear(64, 2*len(parallel)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128*14*14)
        x = self.fc_loc(x)
        return x


class BasicFc(nn.Module):

    def __init__(self, in_feature, out_feature, p=0, **kwargs):
        super(BasicFc, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, p=0):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # [64, 112, 112]
        self.layer1 = self._make_layer(block, 64, layers[0])
        # [64, 112, 112]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # [128, 56, 56]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # [256, 28, 28]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # [512, 14, 14]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, crop):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class ResNetStn(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, p=0, is_train=False):
        super(ResNetStn, self).__init__()
        self.is_train = is_train
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # [64, 112, 112]
        self.layer1 = self._make_layer(block, 64, layers[0])
        # [64, 112, 112]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # [128, 56, 56]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # [256, 28, 28]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # [512, 14, 14]
        self.conv = conv1x1(512 * block.expansion, 128)
        self.fc_loc = nn.Sequential(
            nn.Linear(128*14*14, 64),
            nn.Tanh(),
            nn.Linear(64, 6),
            nn.Tanh()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout2 = nn.Dropout(p)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)

        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)
        x1 = self.conv(feature)
        theta = self.fc_loc(x1.view(-1, 128*14*14))

        mask = torch.tensor([[1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]], dtype=torch.float).cuda()
        theta = torch.mm(theta, mask)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, feature.size())
        x1 = F.grid_sample(feature, grid)

        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.dropout(x1)
        x1 = self.fc(x1)


        x2 = self.avgpool2(feature)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.dropout2(x2)
        x2 = self.fc2(x2)

        return x1, x2


class ResNetMultiStn(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, p=0, parallel=[0.9, 0.7, 0.5]):
        super(ResNetMultiStn, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # [64, 112, 112]
        self.layer1 = self._make_layer(block, 64, layers[0])
        # [64, 112, 112]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # [128, 56, 56]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # [256, 28, 28]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # [512, 14, 14]
        self.stn_fc = StnFc975(parallel, 512 * block.expansion, num_classes)
        #         self.stn_fc = StnFc8642(parallel, 512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.stn_fc(x)

        return x


class StnFc975(nn.Module):

    def __init__(self, parallel, in_feature, out_feature):
        super(StnFc975, self).__init__()
        self.parallel = parallel
        self.out_feature = out_feature
        self.stn = BasicStn(parallel, in_feature)
        self.fc1 = BasicFc(in_feature, out_feature)
        self.fc2 = BasicFc(in_feature, out_feature)
        self.fc3 = BasicFc(in_feature, out_feature)
        self.fc4 = BasicFc(in_feature, out_feature)

    def forward(self, feature):
        x = self.fc1(feature)
        thetas = self.stn(feature)
        i = 0
        theta = thetas[:, (i)*2:(i+1)*2]
        theta = theta.view(-1, 2, 1)
        crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        theta = torch.cat((crop_matrix, theta), dim=2)
        grid = F.affine_grid(theta, feature.size())
        xs = F.grid_sample(feature, grid)
        x += self.fc2(xs)
        i += 1

        theta = thetas[:, (i)*2:(i+1)*2]
        theta = theta.view(-1, 2, 1)
        crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        theta = torch.cat((crop_matrix, theta), dim=2)
        grid = F.affine_grid(theta, feature.size())
        xs = F.grid_sample(feature, grid)
        x += self.fc3(xs)
        i += 1

        theta = thetas[:, (i)*2:(i+1)*2]
        theta = theta.view(-1, 2, 1)
        crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        theta = torch.cat((crop_matrix, theta), dim=2)
        grid = F.affine_grid(theta, feature.size())
        xs = F.grid_sample(feature, grid)
        x += self.fc4(xs)

        return x


class StnFc8642(nn.Module):

    def __init__(self, parallel, in_feature, out_feature):
        super(StnFc8642, self).__init__()
        self.parallel = parallel
        self.out_feature = out_feature
        self.stn = BasicStn(parallel, in_feature)
        self.fc1 = BasicFc(in_feature, out_feature)
        self.fc2 = BasicFc(in_feature, out_feature)
        self.fc3 = BasicFc(in_feature, out_feature)
        self.fc4 = BasicFc(in_feature, out_feature)
        self.fc5 = BasicFc(in_feature, out_feature)

    def forward(self, feature):
        x = self.fc1(feature)
        thetas = self.stn(feature)
        i = 0
        theta = thetas[:, (i)*2:(i+1)*2]
        theta = theta.view(-1, 2, 1)
        crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        theta = torch.cat((crop_matrix, theta), dim=2)
        grid = F.affine_grid(theta, feature.size())
        xs = F.grid_sample(feature, grid)
        x += self.fc2(xs)
        i += 1

        theta = thetas[:, (i)*2:(i+1)*2]
        theta = theta.view(-1, 2, 1)
        crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        theta = torch.cat((crop_matrix, theta), dim=2)
        grid = F.affine_grid(theta, feature.size())
        xs = F.grid_sample(feature, grid)
        x += self.fc3(xs)
        i += 1

        theta = thetas[:, (i)*2:(i+1)*2]
        theta = theta.view(-1, 2, 1)
        crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        theta = torch.cat((crop_matrix, theta), dim=2)
        grid = F.affine_grid(theta, feature.size())
        xs = F.grid_sample(feature, grid)
        x += self.fc4(xs)
        i += 1

        theta = thetas[:, (i)*2:(i+1)*2]
        theta = theta.view(-1, 2, 1)
        crop_matrix = torch.tensor([[self.parallel[i], 0], [0, self.parallel[i]]], dtype=torch.float).cuda()
        crop_matrix = crop_matrix.repeat(theta.size(0), 1).reshape(theta.size(0), 2, 2)
        theta = torch.cat((crop_matrix, theta), dim=2)
        grid = F.affine_grid(theta, feature.size())
        xs = F.grid_sample(feature, grid)
        x += self.fc5(xs)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict.update(
            {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()})
        model.load_state_dict(model_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict.update(
            {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()})
        model.load_state_dict(model_dict)
    return model


def resnet_stn18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetStn(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict.update(
            {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()})
        model.load_state_dict(model_dict)
    return model


def resnet_stn50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetStn(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict.update(
            {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()})
        model.load_state_dict(model_dict)
    return model


def resnet_multi_stn50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMultiStn(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict.update(
            {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()})
        model.load_state_dict(model_dict)
    return model


def resnet_multi_stn101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetMultiStn(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict.update(
            {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()})
        model.load_state_dict(model_dict)
    return model