import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

class StaticLoss(nn.Module):
    def __init__(self, type="torch.nn.MSELoss", beta=3.0):
        super(StaticLoss, self).__init__()
        self.learnable = False
        loss_path, loss_name = type.rsplit('.', 1)
        loss = importlib.import_module(loss_path)
        loss_class = getattr(loss, loss_name)
        self.loss_fnc = loss_class()
        self.beta = beta
        pass
    
    def forward(self, outputs, labels):
        return self.loss_fnc(labels[:,:3], outputs[:,:3]) + self.beta * self.loss_fnc(labels[:,3:], F.normalize(outputs[:,3:]))

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