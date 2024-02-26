import torch
import torch.nn as nn

"""
重新实现unet
"""


def map2image(feature, layer):
    dir_path = "../map_image"
    for i in range(feature.shape[1]):
        temp = feature[0, i, :, :]
        # print(temp.shape)
        temp = temp.data.numpy()
        temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp) + 1e-5)
        temp = np.round(temp * 255)
        img = Image.fromarray(temp)
        img_dir = os.path.join(dir_path, layer)
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        img.convert("RGB").save(os.path.join(img_dir, str(i) + ".jpg"))


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.relu = nn.LeakyReLU(0.2, True)
        self.downconv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        if not self.outermost:
            x = self.relu(x)
        x = self.downconv(x)
        if not self.outermost and not self.innermost:
            x = self.norm(x)
        return x


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.use_dropout = use_dropout
        self.relu = nn.ReLU(True)
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.norm = norm_layer(out_channels)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.relu(x)
        x = self.upconv(x)
        if self.outermost:
            x = self.tanh(x)
        else:
            x = self.norm(x)
            if not self.innermost and self.use_dropout:
                x = self.dropout(x)
        return x


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(
        self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        self.down1 = Down(input_nc, ngf, outermost=True, norm_layer=norm_layer)
        self.down2 = Down(ngf, ngf * 2, norm_layer=norm_layer)
        self.down3 = Down(ngf * 2, ngf * 4, norm_layer=norm_layer)
        self.down4 = Down(ngf * 4, ngf * 8, norm_layer=norm_layer)
        self.down5 = Down(ngf * 8, ngf * 8, norm_layer=norm_layer)
        self.down6 = Down(ngf * 8, ngf * 8, norm_layer=norm_layer)
        self.down7 = Down(ngf * 8, ngf * 8, norm_layer=norm_layer)
        self.down8 = Down(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)

        self.up8 = Up(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        self.up7 = Up(ngf * 16, ngf * 8, use_dropout=use_dropout, norm_layer=norm_layer)
        self.up6 = Up(ngf * 16, ngf * 8, use_dropout=use_dropout, norm_layer=norm_layer)
        self.up5 = Up(ngf * 16, ngf * 8, use_dropout=use_dropout, norm_layer=norm_layer)
        self.up4 = Up(ngf * 16, ngf * 4, use_dropout=use_dropout, norm_layer=norm_layer)
        self.up3 = Up(ngf * 8, ngf * 2, use_dropout=use_dropout, norm_layer=norm_layer)
        self.up2 = Up(ngf * 4, ngf, norm_layer=norm_layer)
        self.up1 = Up(ngf * 2, output_nc, outermost=True, norm_layer=norm_layer)

        # self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.cbam1 = attention.CBAM(channel=64)
        # self.cbam2 = attention.CBAM(channel=128)
        # self.cbam3 = attention.CBAM(channel=256)
        # self.cbam4 = attention.CBAM(channel=512)
        # self.cbam5 = attention.CBAM(channel=512)
        # self.cbam6 = attention.CBAM(channel=512)
        # self.cbam7 = attention.CBAM(channel=512)

    def forward(self, x):
        """Standard forward"""
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        # d6 = self.down6(d5)
        # d7 = self.down7(d6)
        # d8 = self.down8(d7)
        # d8 = self.down8(d4)
        d8 = self.down8(d5)

        u8 = self.up8(d8)
        # u7 = self.up7(torch.cat([d7, u8], dim=1))
        # u6 = self.up6(torch.cat([d6, u7], dim=1))
        # u6 = self.up6(torch.cat([d6, u8], dim=1))
        # u5 = self.up5(torch.cat([d5, u6], dim=1))
        u5 = self.up5(torch.cat([d5, u8], dim=1))
        u4 = self.up4(torch.cat([d4, u5], dim=1))
        # u4 = self.up4(torch.cat([d4, u8], dim=1))
        u3 = self.up3(torch.cat([d3, u4], dim=1))
        u2 = self.up2(torch.cat([d2, u3], dim=1))
        u1 = self.up1(torch.cat([d1, u2], dim=1))

        # u1 = (u1 + 1) / 2
        # u1 = u1 * 2 - 1

        return u1


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    import os

    transform_list = []
    method = transforms.InterpolationMode.BICUBIC
    transform_list.append(transforms.Resize([286, 286], method))
    transform_list.append(transforms.RandomCrop(256))
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5), (0.5))]
    tf = transforms.Compose(transform_list)

    A = Image.open("~/try-on/data/zalando-hd-resized/train/cloth/00000_00.jpg").convert(
        "RGB"
    )
    A = tf(A)
    A = A.unsqueeze(0)

    print(A.shape)
    A = torch.ones((2, 3, 2, 2), dtype=torch.float32)
    A[:, 0, :, :] -= 10
    print(A)
    # print(A.shape)
    # model = UnetGenerator(1, 3)
    # output = model(A)
    # print(output.shape)
    # for key in output:
    #     feature = output[key]
    #     dir_path = '../vgg_map'
    #     temp = torch.sum(feature, dim=1, keepdim=True)
    #     print(temp.shape)
    # temp = temp.data.numpy()
    # temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp) + 1e-5)
    # temp = np.round(temp*255)
    # img = Image.fromarray(temp)
    # img_dir = os.path.join(dir_path, key)
    # if not os.path.exists(img_dir):
    #     os.mkdir(img_dir)
    # img.convert('RGB').save(os.path.join(img_dir, str(key) + '.jpg'))
    # for i in range(feature.shape[1]):
    #     temp = feature[0, i, :, :]
    #     # print(temp.shape)
    #     temp = temp.data.numpy()
    #     temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp) + 1e-5)
    #     temp = np.round(temp*255)
    #     img = Image.fromarray(temp)
    #     img_dir = os.path.join(dir_path, key)
    #     if not os.path.exists(img_dir):
    #         os.mkdir(img_dir)
    #     img.convert('RGB').save(os.path.join(img_dir, str(i) + '.jpg'))
