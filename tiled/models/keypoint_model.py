import torch
from .base_model import BaseModel
from . import networks
from .net import hrnet


class KeypointModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', dataset_mode='keypoint')
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
        self.loss_names = ['G', 'G']
            
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)

        # self.netG = networks.KeypointGenerator()
        self.netG = hrnet.HighResolutionNet()
        self.netG.to(self.gpu_ids[0])
        self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)

        self.criterionLoss = torch.nn.MSELoss()

        if self.isTrain:
            # define loss functions
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_heatmap = input['heatmap'].to(self.device)
        
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.out_heatmap = self.netG(self.real_A)
        self.loss_G = self.criterionLoss(self.real_heatmap, self.out_heatmap) * 1e3

    def backward_G(self):
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

