import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifyNet(nn.Module):
    def __init__(self):
        super(ClassifyNet, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(16*8*8, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax()
        )
        # self.sig = nn.Sigmoid()
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x