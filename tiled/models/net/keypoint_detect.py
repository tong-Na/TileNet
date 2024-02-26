import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class KeypointDetectNet(nn.Module):
    def __init__(self):
        super(KeypointDetectNet, self).__init__()

        self.resnet = models.resnet34(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 128)
        self.resnet.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 22),
            # nn.ReLU(),
            # nn.Linear(64, 22),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from PIL import Image

    transform_list = []
    method = transforms.InterpolationMode.BICUBIC
    transform_list += [transforms.Resize((224, 224))]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(0.5, 0.5)]
    tf = transforms.Compose(transform_list)

    A = Image.open("~/try-on/data/zalando-hd-resized/train/cloth/00000_00.jpg").convert(
        "L"
    )

    A = tf(A)
    A = A.unsqueeze(0)

    model = KeypointDetectNet()
    print(A.shape)
    out1 = model(A)
    print(out1.shape)
