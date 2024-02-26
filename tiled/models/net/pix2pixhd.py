import torch
import torch.nn as nn


class LocalEnhancer(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=32,
        n_downsample_global=3,
        n_blocks_global=9,
        n_local_enhancers=1,
        n_blocks_local=3,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
    ):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2**n_local_enhancers)
        # 共享
        model_global = GlobalGenerator(
            input_nc,
            output_nc,
            ngf_global,
            n_downsample_global,
            n_blocks_global,
            norm_layer,
        ).model
        model_global = [model_global[i] for i in range(len(model_global) - 3)]
        self.model = nn.Sequential(*model_global)

        # 非共享
        # model_global1 = GlobalGenerator(
        #     1, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        # model_global1 = [model_global1[i] for i in range(len(model_global1)-3)]

        # model_global2 = GlobalGenerator(
        #     input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        # model_global2 = [model_global2[i] for i in range(len(model_global2)-3)]

        # setattr(self, 'model'+'1_0global', nn.Sequential(*model_global1))
        # setattr(self, 'model'+'1_1global', nn.Sequential(*model_global2))

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            # downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            # model_downsample0 = [nn.ReflectionPad2d(3), nn.Conv2d(1, ngf_global, kernel_size=7, padding=0),
            #                     norm_layer(ngf_global), nn.ReLU(True),
            #                     nn.Conv2d(ngf_global, ngf_global * 2,
            #                               kernel_size=3, stride=2, padding=1),
            #                     norm_layer(ngf_global * 2), nn.ReLU(True)]
            model_downsample = [
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                norm_layer(ngf_global),
                nn.ReLU(True),
                nn.Conv2d(
                    ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1
                ),
                norm_layer(ngf_global * 2),
                nn.ReLU(True),
            ]
            # residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [
                    ResnetBlock(
                        ngf_global * 2,
                        padding_type=padding_type,
                        norm_layer=norm_layer,
                        use_dropout=True,
                    )
                ]

            # upsample
            model_upsample += [
                nn.ConvTranspose2d(
                    ngf_global * 2,
                    ngf_global,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                norm_layer(ngf_global),
                nn.ReLU(True),
            ]

            # final convolution
            if n == n_local_enhancers:
                model_upsample += [
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                    nn.Tanh(),
                ]

            # setattr(self, 'model'+str(n)+'_0',
            #         nn.Sequential(*model_downsample0))
            setattr(self, "model" + str(n) + "_1", nn.Sequential(*model_downsample))
            setattr(self, "model" + str(n) + "_2", nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def forward(self, input):
        # create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        # output at coarest level
        output_prev = self.model(input_downsampled[-1])
        # build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, "model" + str(n_local_enhancers) + "_1")
            model_upsample = getattr(self, "model" + str(n_local_enhancers) + "_2")
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            # todo
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

    # def branch(self, input, model_name):
    #     input_downsampled = [input]
    #     input_downsampled.append(self.downsample(input_downsampled[-1]))

    #     # output at coarest level
    #     # output_prev = self.model(input_downsampled[-1])
    #     model = getattr(self, model_name + 'global')
    #     output_prev = model(input_downsampled[-1])
    #     # build up one layer at a time

    #     model_downsample = getattr(self, model_name)
    #     input_i = input_downsampled[self.n_local_enhancers - 1]
    #     return model_downsample(input_i) + output_prev

    # def forward(self, input1, input2):
    #     # create input pyramid
    #     model_upsample = getattr(self, 'model1_2')
    #     # todo
    #     output_prev = model_upsample(self.branch(input1, 'model1_0') +
    #      self.branch(input2, 'model1_1'))
    #     return output_prev


class GlobalGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        n_downsampling=3,
        n_blocks=9,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
    ):
        assert n_blocks >= 0
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        ]
        # b * ngf * h * w
        # downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1
                ),
                norm_layer(ngf * mult * 2),
                activation,
            ]

        # resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=True,
                )
            ]
            #  norm_layer=norm_layer)]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                norm_layer(int(ngf * mult / 2)),
                activation,
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(
        self, dim, padding_type, norm_layer, use_dropout=False, use_bias=False
    ):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from PIL import Image

    transform_list = []
    method = transforms.InterpolationMode.BICUBIC
    transform_list += [transforms.Resize((256, 256))]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(0.5, 0.5)]
    tf = transforms.Compose(transform_list)

    A = Image.open("~/try-on/data/zalando-hd-resized/train/cloth/00000_00.jpg").convert(
        "RGB"
    )
    # B = Image.open('../img_for_loss/pic_large.png').convert('RGB')
    # C = Image.open('../img_for_loss/pic_dif.png').convert('RGB')

    A = tf(A)
    A = A.unsqueeze(0)
    # B = tf(B)
    # B = B.unsqueeze(0)
    # C = tf(C)
    # C = C.unsqueeze(0)

    model1 = LocalEnhancer(3, 3, 64, n_local_enhancers=3)
    model2 = GlobalGenerator(3, 3, 64)
    print(A.shape)
    out1 = model1(A)
    print(out1.shape)
