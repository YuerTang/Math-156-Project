import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

KERNAL_SIZE = 7

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [C,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x
    


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size=KERNAL_SIZE
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)         # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)         # [B,2,H,W]
        return self.sigmoid(self.conv(x_cat)) * x
    


# Combined ChannelAttention and SpatialAttention
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# CBAM block
class BasicBlockWithCBAM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.cbam = CBAM(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# PoseRegressor Module
class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)

# Pose Loss Function
class CameraPoseLoss(nn.Module):
    """
    A class to represent camera pose loss with optional learnable weights.
    """

    def __init__(self, s_x, s_q, trainable=False, norm=2):
        """
        :param config: (dict) Configuration dictionary with keys:
            - "learnable": (bool) Whether s_x and s_q are learnable
            - "s_x": (float) Initial weight for position loss
            - "s_q": (float) Initial weight for orientation loss
            - "norm": (int or float) Norm type for distance (1 for L1, 2 for L2, etc.)
        """
        super(CameraPoseLoss, self).__init__()
        self.learnable = trainable
        self.norm = norm

        s_x = torch.tensor([s_x], dtype=torch.float32)
        s_q = torch.tensor([s_q], dtype=torch.float32)

        self.s_x = nn.Parameter(s_x, requires_grad=self.learnable)
        self.s_q = nn.Parameter(s_q, requires_grad=self.learnable)

    def forward(self, est_pose, gt_pose):
        """
        Compute the camera pose loss.

        :param est_pose: (torch.Tensor) Nx7 estimated pose tensor
        :param gt_pose: (torch.Tensor) Nx7 ground truth pose tensor
        :return: l_x, l_q, total_loss
        """
        # Position loss
        l_x = torch.norm(gt_pose[:, :3] - est_pose[:, :3], p=self.norm, dim=1).mean()

        # Orientation loss using normalized quaternions
        est_q = F.normalize(est_pose[:, 3:], p=2, dim=1)
        gt_q = F.normalize(gt_pose[:, 3:], p=2, dim=1)
        l_q = torch.norm(gt_q - est_q, p=self.norm, dim=1).mean()

        # Total loss
        if self.learnable:
            total_loss = l_x * torch.exp(-self.s_x) + self.s_x + l_q * torch.exp(-self.s_q) + self.s_q
        else:
            total_loss = self.s_x * l_x + self.s_q * l_q

        return total_loss


# ResNet with CBAM
class ResNetCBAM(nn.Module):
    def __init__(self, block_name, layers, pos_dim=3, orien_dim=4):
        super().__init__()

        block_path, block_name = block_name.rsplit('.', 1)
        block = importlib.import_module(block_path)
        block_class = getattr(block, block_name)

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_class, 64,  layers[0])
        self.layer2 = self._make_layer(block_class, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_class, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_class, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pos_fc = PoseRegressor(512 * block_class.expansion, pos_dim)
        self.orien_fc = PoseRegressor(512 * block_class.expansion, orien_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)  # -> [B, 64, H/4, W/4]
        x = self.layer2(x)  # -> [B, 128, H/8, W/8]
        x = self.layer3(x)  # -> [B, 256, H/16, W/16]
        x = self.layer4(x)  # -> [B, 512, H/32, W/32]

        x = self.avgpool(x)  # -> [B, 512, 1, 1]
        x = torch.flatten(x, 1)

        # Split position and orientation head
        position = self.pos_fc(x)
        orientation = self.orien_fc(x)

        return torch.cat([position, orientation], dim=1)