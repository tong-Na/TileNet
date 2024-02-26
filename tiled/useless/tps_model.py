import torch
from .base_model import BaseModel
from . import networks


class TPSModel(BaseModel):
    """ This class implements the tps model.

    The model training requires '--dataset_mode aligned' dataset.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='tps')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the tps class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_Per', 'G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['GMM']
        # define networks (both generator and discriminator)
        self.netGMM = networks.define_GMM(gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionPer = networks.PerceptualLoss(self.device)
            self.criterionSymmetry = networks.SymmetryLoss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_GMM = torch.optim.Adam(self.netGMM.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_GMM)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.pose_source = input['C'].to(self.device)
        self.pose_center = input['D'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        self.real_A = self.pose_source
        self.real_B = self.pose_center

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        self.fake_B = self.netGMM(self.pose_source, self.pose_center)  # G(A)

    def backward_GMM(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # self.loss_G_Symmetry = self.criterionSymmetry(self.fake_B) * 5
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L1

        self.loss_G_Per = self.criterionPer(self.fake_B, self.real_B) * 100
        self.loss_G += self.loss_G_Per
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update GMM
        self.optimizer_GMM.zero_grad()        # set G's gradients to zero
        self.backward_GMM()                   # calculate graidents for G
        self.optimizer_GMM.step()             # udpate G's weights

