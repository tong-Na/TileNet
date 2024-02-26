import torch
from .base_model import BaseModel
from .net import ffc
from . import networks
from .losses import masked


class InpaintModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", dataset_mode="inpaint")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument(
                "--lambda_L1", type=float, default=100.0, help="weight for L1 loss"
            )

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "G_L1", "G_FM", "G_Resnetpl"]

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>

        if self.isTrain:
            self.visual_names = ["real_A", "fake_B", "real_B"]
            self.model_names = ["G", "D"]
        else:  # during test time, only load G
            self.visual_names = ["fake_B"]
            self.model_names = ["G"]

        # define networks (both generator and discriminator)

        # todo 模型结构

        self.netG = ffc.FFCResNetGenerator(**ffc.config)
        self.netG.to(self.gpu_ids[0])
        self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(
                opt.input_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionResnetpl = masked.ResNetPL(weight=30, weights_path="").to(
                self.device
            )

            params_to_update = []
            for _, param in self.netG.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)

            self.optimizer_G = torch.optim.Adam(
                params_to_update, lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr * 10, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if self.isTrain and not opt.continue_train:
                self.load_checkpoint(
                    "~/try-on/tiled/util/lama-main/big-lama/models/best.ckpt"
                )

    def set_input(self, input):
        self.real_A = input["A"].to(self.device)
        self.real_B = input["B"].to(self.device)
        self.mask = input["mask"].to(self.device)

        self.masked_A = self.real_A * (1 - self.mask)
        self.masked_A = torch.cat([self.masked_A, self.mask], dim=1)

        self.image_paths = input["A_paths"]

    def forward(self):
        self.fake_B = self.netG(self.masked_A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # pred_fake = self.netD(self.fake_B.detach())
        pred_fake, _ = self.netD(self.fake_B.detach())
        # self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        # pred_real = self.netD(self.real_B)
        self.real_B.requires_grad = True
        pred_real, _ = self.netD(self.real_B)
        # self.loss_D_real = self.criterionGAN(pred_real, True)

        # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D = masked.discriminator_loss(
            self.real_B, pred_real, pred_fake, self.mask
        )
        self.loss_D.backward()

    def backward_G(self):
        self.loss_G_L1 = masked.masked_l1_loss(
            self.fake_B, self.real_B, self.mask, 10, 0
        )
        pred_fake, features_fake = self.netD(self.fake_B)
        _, features_real = self.netD(self.real_B)
        # self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1 * 1e3
        self.loss_G_GAN = masked.generator_loss(pred_fake) * 10

        self.loss_G_FM = (
            masked.feature_matching_loss(features_fake, features_real) * 100
        )

        self.loss_G_Resnetpl = self.criterionResnetpl(self.fake_B, self.real_B)

        self.loss_G = (
            self.loss_G_GAN + self.loss_G_L1 + self.loss_G_FM + self.loss_G_Resnetpl
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.set_requires_grad(self.netG, False)
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(
            self.netD, False
        )  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def load_checkpoint(self, path):
        lama_dict = torch.load(path)
        sd = lama_dict["state_dict"]
        sd_gen = {}
        for k in sd.keys():
            prefix = k.split("_")[0]
            if prefix != "val" and prefix != "test":  # 生成器部分的权重参数
                k_new = "module." + k[10:]

                sd_gen[k_new] = sd[k]
        self.netG.load_state_dict(sd_gen, strict=True)
