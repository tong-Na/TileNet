import torch
from .base_model import BaseModel
from . import networks
from .net import trans_unet


class ContentModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", dataset_mode="content")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument(
                "--lambda_L1", type=float, default=100.0, help="weight for L1 loss"
            )

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G_L2", "reg"]
        if self.isTrain:
            self.visual_names = ["input", "out_mask", "masked_img", "label"]
        else:
            self.visual_names = ["out_mask"]

        self.model_names = ["G"]

        self.netG = trans_unet.VisionTransformer(num_classes=1)
        self.netG.to(self.gpu_ids[0])
        self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)

        self.criterionL2 = torch.nn.MSELoss()
        # self.criterionL2 = torch.nn.L1Loss()

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.input = input["input"].to(self.device)
        self.label = input["label"].to(self.device)
        self.image_paths = input["A_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.out_mask = self.netG(self.input)

        if not self.isTrain:
            delta = -0.7
            self.out_mask[self.out_mask > delta] = 1
            self.out_mask[self.out_mask <= delta] = -1
        #     self.masked_img = self.input * (1 - self.out_mask)

    def backward_G(self):
        self.loss_G_L2 = self.criterionL2(self.out_mask, self.label) * 1e2
        lambda_reg = 5e-3
        self.loss_reg = 0.0
        for param in self.netG.parameters():
            self.loss_reg += torch.norm(param, p=2)  # 以L2范数作为正则化项

        self.loss_G = self.loss_G_L2 + lambda_reg * self.loss_reg

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

        delta = -0.7
        self.out_mask[self.out_mask > delta] = 1
        self.out_mask[self.out_mask <= delta] = 0
        self.masked_img = self.input * (1 - self.out_mask)
