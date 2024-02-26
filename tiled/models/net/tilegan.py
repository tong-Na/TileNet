import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self, input_nc):
        super(STN, self).__init__()
        self.loc_net_conv = nn.Sequential(
            nn.Conv2d(input_nc, 20, kernel_size=5, stride=2, padding=0),
            # nn.Conv2d(input_nc, 8, kernel_size=7),  # 256->250
            nn.MaxPool2d(kernel_size=2, stride=2),  # 125
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=5, stride=2, padding=0),
            # nn.Conv2d(8, 10, kernel_size=5),  # 121
            nn.MaxPool2d(kernel_size=2, stride=2),  # 60
            nn.ReLU(),
        )
        self.loc_net_fc = nn.Sequential(
            nn.Linear(20 * 15 * 15, 50),
            # nn.Linear(10 * 60 * 60, 32),
            nn.ReLU(),
            nn.Linear(50, 6),
            # nn.Linear(32, 6),
        )

        self.loc_net_fc[2].weight.data.zero_()
        self.loc_net_fc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        theta = self.loc_net_conv(x)
        theta = theta.view(-1, 20 * 15 * 15)
        # theta = theta.view(-1, 10 * 60 * 60)
        theta = self.loc_net_fc(theta)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.shape)
        output = F.grid_sample(x, grid)
        return output


class InterpolationConvolutionModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InterpolationConvolutionModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2, True)
        self.downconv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.relu(x)
        x = self.downconv(x)
        x = self.norm(x)
        return x


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer=nn.BatchNorm2d,
        outermost=False,
        use_dropout=False,
    ):
        super().__init__()
        self.use_dropout = use_dropout
        self.relu = nn.ReLU(True)
        # self.upconv = InterpolationConvolutionModel(in_channels, out_channels)
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.norm = norm_layer(out_channels)
        self.outermost = outermost
        self.use_dropout = use_dropout
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(x)
        x = self.upconv(x)
        if self.outermost:
            x = self.tanh(x)
        else:
            x = self.norm(x)
            if self.use_dropout:
                x = self.dropout(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(UNet, self).__init__()
        self.down0 = nn.Conv2d(input_nc, ngf, kernel_size=5, stride=2, padding=2)
        self.down1 = Down(ngf, ngf * 2)
        self.down2 = Down(ngf * 2, ngf * 4)
        self.down3 = Down(ngf * 4, ngf * 8)
        self.down4 = Down(ngf * 8, ngf * 8)
        self.down5 = Down(ngf * 8, ngf * 8)
        self.down6 = Down(ngf * 8, ngf * 8)
        self.down7 = Down(ngf * 8, ngf * 8)

        self.up7 = Up(ngf * 8, ngf * 8, use_dropout=True)
        self.up6 = Up(ngf * 16, ngf * 8, use_dropout=True)
        self.up5 = Up(ngf * 16, ngf * 8, use_dropout=True)
        self.up4 = Up(ngf * 16, ngf * 8, use_dropout=False)
        self.up3 = Up(ngf * 16, ngf * 4, use_dropout=False)
        self.up2 = Up(ngf * 8, ngf * 2, use_dropout=False)
        self.up1 = Up(ngf * 4, ngf, use_dropout=False)
        self.up0 = Up(ngf * 2, output_nc, outermost=True, use_dropout=False)

    def forward(self, x):
        d0 = self.down0(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u7 = self.up7(d7)
        u6 = self.up6(torch.cat([d6, u7], dim=1))
        u5 = self.up5(torch.cat([d5, u6], dim=1))
        u4 = self.up4(torch.cat([d4, u5], dim=1))
        u3 = self.up3(torch.cat([d3, u4], dim=1))
        u2 = self.up2(torch.cat([d2, u3], dim=1))
        u1 = self.up1(torch.cat([d1, u2], dim=1))
        u0 = self.up0(torch.cat([d0, u1], dim=1))
        return u0


class TileGAN(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(TileGAN, self).__init__()
        self.stn1 = STN(input_nc)
        self.stn2 = STN(output_nc)
        self.unet = UNet(input_nc, output_nc)

    def forward(self, x):
        x = self.stn1(x)
        x = self.unet(x)
        x = self.stn2(x)
        return x


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
    model = TileGAN(3, 3)
    output = model(A)
    print(output.shape)
