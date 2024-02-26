import torch
import torch.nn as nn
import torch.nn.functional as F


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).repeat(1, 3, 1, 1).cuda()
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).repeat(1, 3, 1, 1).cuda()

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        grad_output = torch.abs(grad_x) + torch.abs(grad_y)

        grad_x = F.conv2d(y, self.weight_x, padding=1)
        grad_y = F.conv2d(y, self.weight_y, padding=1)
        grad_label = torch.abs(grad_x) + torch.abs(grad_y)

        # delta = 1

        # grad_output[grad_output < delta] = 0
        # grad_output[grad_output > delta] = 1

        # grad_label[grad_label < delta] = 0
        # grad_label[grad_label > delta] = 1

        # grad_output = x * grad_output
        # grad_label = y * grad_label
        
        loss = self.criterion(grad_output, grad_label)
        return loss