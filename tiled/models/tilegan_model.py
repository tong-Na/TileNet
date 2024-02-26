import torch
from .base_model import BaseModel
from .net import tilegan
from .net import tilegan2
from . import networks


class TileGanModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", netG="tilegan", dataset_mode="end2end")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument(
                "--lambda_L1", type=float, default=100.0, help="weight for L1 loss"
            )

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G_GAN", "D_real", "D_fake", "G_L1", "G_Per"]

        if self.isTrain:
            self.visual_names = ["real_A", "fake_B", "real_B"]
            self.model_names = ["G", "D"]
        else:  # during test time, only load G
            self.visual_names = ["fake_B"]
            self.model_names = ["G"]
        # define networks (both generator and discriminator)

        self.tilegan2 = opt.tilegan2
        if opt.tilegan2:
            self.netG = tilegan2.TileGAN(opt.input_nc * 2, opt.output_nc)
        else:
            self.netG = tilegan.TileGAN(opt.input_nc, opt.output_nc)

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

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.lambda_L1 = opt.lambda_L1
            self.criterionPer = networks.PerceptualLoss(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr / 2, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input["A"].to(self.device)
        self.real_B = input["B"].to(self.device)
        if self.tilegan2:
            self.coarse = input["C"].to(self.device)
        self.image_paths = input["A_paths"]

    def forward(self):
        if self.tilegan2:
            self.fake_B = self.netG(self.real_A, self.coarse)  # G(A)
        else:
            self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        pred_fake = self.netD(self.fake_B.detach())
        pred_real = self.netD(self.real_B)

        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        pred_fake = self.netD(self.fake_B)
        self.loss_G = 0
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G += self.loss_G_GAN

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
        self.loss_G += self.loss_G_L1

        self.loss_G_Per = self.criterionPer(self.fake_B, self.real_B)
        self.loss_G += self.loss_G_Per

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
