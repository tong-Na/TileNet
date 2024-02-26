import torch
import torch.nn as nn
from .unet2 import UnetGenerator


class MaskNet(nn.Module):
    def __init__(self, image_size, num_points=11):
        super(MaskNet, self).__init__()

        self.unet = UnetGenerator(1, 1, 8, use_dropout=True)

    def forward(self, x):
        x = self.unet(x)

        # x = torch.sigmoid(x)

        return x


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from PIL import Image

    A = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.float32,
    )

    A = A.unsqueeze(0)

    model = MaskNet()
    print(A.shape)
    out1 = model(A)
    print(out1.shape)
