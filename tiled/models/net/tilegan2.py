import torch
import torch.nn as nn
import torch.nn.functional as F
from .tilegan import Up


class ChannelAttention(nn.Module):
    def __init__(self, input_nc, kernel):
        super(ChannelAttention, self).__init__()
        self.avg = nn.AvgPool2d(kernel, 1)
        self.net = nn.Sequential(
            nn.Conv2d(input_nc, input_nc // 4, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(input_nc // 4, input_nc, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg = self.avg(x)
        weight = self.net(avg)
        return x * weight


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ca_kernel,
        norm_layer=nn.BatchNorm2d,
        innermost=False,
    ):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.downconv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.norm = norm_layer(out_channels)
        self.innermost = innermost
        if not self.innermost:
            self.ca = ChannelAttention(out_channels, ca_kernel)

    def forward(self, x):
        x = self.relu(x)
        x = self.downconv(x)
        x = self.norm(x)
        x_ca = None
        if not self.innermost:
            x_ca = self.ca(x)
        return x, x_ca


class TileGAN(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(TileGAN, self).__init__()
        self.down0 = Down(input_nc, ngf, ngf * 2)
        self.down1 = Down(ngf, ngf * 2, ngf)
        self.down2 = Down(ngf * 2, ngf * 4, ngf // 2)
        self.down3 = Down(ngf * 4, ngf * 8, ngf // 4)
        self.down4 = Down(ngf * 8, ngf * 8, ngf // 8)
        self.down5 = Down(ngf * 8, ngf * 8, ngf // 16)
        self.down6 = Down(ngf * 8, ngf * 8, ngf // 32)
        self.down7 = Down(ngf * 8, ngf * 8, ngf // 64, innermost=True)

        self.up7 = Up(ngf * 8, ngf * 8)
        self.up6 = Up(ngf * 16, ngf * 8)
        self.up5 = Up(ngf * 16, ngf * 8)
        self.up4 = Up(ngf * 16, ngf * 8)
        self.up3 = Up(ngf * 16, ngf * 4)
        self.up2 = Up(ngf * 8, ngf * 2)
        self.up1 = Up(ngf * 4, ngf)
        self.up0 = Up(ngf * 2, output_nc, outermost=True)

    def forward(self, raw, coarse):
        x = torch.cat([coarse, raw], dim=1)
        d0, d0_ca = self.down0(x)
        d1, d1_ca = self.down1(d0)
        d2, d2_ca = self.down2(d1)
        d3, d3_ca = self.down3(d2)
        d4, d4_ca = self.down4(d3)
        d5, d5_ca = self.down5(d4)
        d6, d6_ca = self.down6(d5)
        d7, _ = self.down7(d6)

        u7 = self.up7(d7)
        u6 = self.up6(torch.cat([u7, d6_ca], dim=1))
        u5 = self.up5(torch.cat([u6, d5_ca], dim=1))
        u4 = self.up4(torch.cat([u5, d4_ca], dim=1))
        u3 = self.up3(torch.cat([u4, d3_ca], dim=1))
        u2 = self.up2(torch.cat([u3, d2_ca], dim=1))
        u1 = self.up1(torch.cat([u2, d1_ca], dim=1))
        u0 = self.up0(torch.cat([u1, d0_ca], dim=1))
        return u0


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from PIL import Image

    transform_list = []
    method = transforms.InterpolationMode.BICUBIC
    transform_list.append(transforms.Resize([286, 286], method))
    transform_list.append(transforms.RandomCrop(256))
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    tf = transforms.Compose(transform_list)

    A = Image.open("~/try-on/data/zalando-hd-resized/train/cloth/00000_00.jpg").convert(
        "RGB"
    )
    A_1 = tf(A)
    A_1 = A_1.unsqueeze(0)
    A_2 = tf(A)
    A_2 = A_2.unsqueeze(0)
    A = torch.cat([A_1, A_2], dim=0)
    model = TileGAN(6, 3)
    output = model(A, A)
    print(output.shape)
