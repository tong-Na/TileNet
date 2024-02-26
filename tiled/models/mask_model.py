import torch
from .base_model import BaseModel
from . import networks
from .losses import symm
from torch.nn import functional as F


class MaskModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', dataset_mode='mask')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1', 'G_Symm']
            
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['out_mask']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)

        self.netG = networks.MaskGenerator()
        self.netG.to(self.gpu_ids[0])
        self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionSymm = symm.SymmetryLoss().to(self.device)

        if self.isTrain:
            # define loss functions
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.points = input['points'].to(self.device)
        self.real_mask = input['real_mask'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.out_mask = self.netG(self.points)
        self.loss_G_L1 = self.criterionL1(self.real_mask, self.out_mask) * 1e2
        self.loss_G_Symm = self.criterionSymm(self.out_mask) * 1e1
        self.loss_G = self.loss_G_L1 + self.loss_G_Symm

    def backward_G(self):
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

