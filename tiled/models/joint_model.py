import torch
from .base_model import BaseModel
from . import networks
from .losses import ssim


class JointModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        parser.set_defaults(norm="batch", netG="joint", dataset_mode="joint")
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
        self.loss_names = ["G_GAN", "D_real", "D_fake"]
        if self.isTrain:
            if opt.l1_loss:
                self.loss_names.append("G_L1A")
                self.loss_names.append("G_L1B")
                # self.loss_names.append('G_fakeinput')
                # self.loss_names.append('G_fakelabel')
                # self.loss_names.append('G_mask')
            if opt.per_loss:
                self.loss_names.append("G_Per")
            if opt.feature_matching_loss:
                self.loss_names.append("G_FMatch")
            if opt.zoom_loss:
                self.loss_names.append("G_ZOOM")
            if opt.grad_loss:
                self.loss_names.append("G_Grad")
            if opt.ssim_loss:
                self.loss_names.append("G_SSIM")
            if opt.symm_loss:
                self.loss_names.append("G_Symmetry")

        self.visual_names = ["real_A", "fake_B", "real_B", "coarse"]

        if self.isTrain:
            self.model_names = ["G", "D"]
        else:  # during test time, only load G
            self.model_names = ["G"]
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

        if (
            self.isTrain
        ):  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = None
            self.criterionPer = None
            self.criterionFMatch = None
            self.criterionZoom = None
            self.criterionGrad = None
            self.criterionSSIM = None
            self.criterionSymmetry = None

            if opt.l1_loss:
                self.criterionL1 = torch.nn.L1Loss()
            if opt.per_loss:
                self.criterionPer = networks.PerceptualLoss(self.device)
            if opt.feature_matching_loss:
                self.criterionFMatch = networks.FeatureMatchingLoss().to(self.device)
            if opt.zoom_loss:
                self.criterionZoom = networks.ZoomLoss(self.device)
            if opt.grad_loss:
                self.criterionGrad = networks.GradLoss().to(self.device)
            if opt.ssim_loss:
                self.criterionSSIM = ssim.SSIM()
            if opt.symm_loss:
                self.criterionSymmetry = networks.SymmetryLoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr / 2, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr / 2, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.coarse, self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.fake_B.detach())[-1]
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = self.netD(self.real_B)[-1]
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_B)
        pred_real = self.netD(self.real_B)
        self.loss_G = 0
        self.loss_G_GAN = self.criterionGAN(pred_fake[-1], True)
        self.loss_G = self.loss_G_GAN

        if self.criterionL1 is not None:
            self.loss_G_L1B = (
                (self.criterionL1(self.fake_B, self.real_B)) * self.opt.lambda_L1 * 2
            )
            self.loss_G_L1A = (
                self.criterionL1(self.fake_B, self.real_A) * self.opt.lambda_L1 * 0.8
            )
            self.loss_G += self.loss_G_L1A + self.loss_G_L1B
            # self.loss_G_fakeinput = self.criterionL1(self.fake_B, self.real_A) * self.opt.lambda_L1 * 0.8
            # self.loss_G_fakelabel = (self.criterionL1(self.fake_B*(1-self.coarse), self.real_B*(1-self.coarse))) * self.opt.lambda_L1 * 2
            # self.loss_G_mask = self.coarse.sum() / (self.coarse.shape[0]*self.coarse.shape[-2]*self.coarse.shape[-1]) * 5
            # self.loss_G += self.loss_G_fakelabel + self.loss_G_mask
        if self.criterionPer is not None:
            self.loss_G_Per = self.criterionPer(self.fake_B, self.real_B) * 1000
            self.loss_G += self.loss_G_Per
        if self.criterionFMatch is not None:
            self.loss_G_FMatch = self.criterionFMatch(pred_fake, pred_real)
            self.loss_G += self.loss_G_FMatch
        if self.criterionZoom is not None:
            self.loss_G_ZOOM = self.criterionZoom(self.fake_B, self.real_B) * 1000
            self.loss_G += self.loss_G_ZOOM
        if self.criterionGrad is not None:
            self.loss_G_Grad = (
                self.criterionGrad(self.fake_B, self.coarse) * 10
            )  # default 10
            self.loss_G += self.loss_G_Grad
        if self.criterionSSIM is not None:
            self.loss_G_SSIM = self.criterionSSIM(self.fake_B, self.real_B) * 100
            self.loss_G += self.loss_G_SSIM
        if self.criterionSymmetry is not None:
            self.loss_G_Symmetry = self.criterionSymmetry(self.fake_B) * 10
            self.loss_G += self.loss_G_Symmetry

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(
            self.netD, False
        )  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
