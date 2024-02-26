import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class TPSDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # data/zalando-hd-resized/train(test)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.dir_A = os.path.join(self.dir_AB, 'cloth')
        self.dir_B = os.path.join(self.dir_AB, 'image')
        self.dir_C = os.path.join(self.dir_AB, 'pose_joint')
        self.path_D = os.path.join(opt.dataroot, 'center_pose.jpg')
        self.D = Image.open(self.path_D).convert('RGB')

        self.A_paths = sorted(make_dataset(self.dir_A))  # get image paths
        self.B_paths = sorted(make_dataset(self.dir_B))  # get image paths
        self.C_paths = sorted(make_dataset(self.dir_C))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        C = Image.open(C_path).convert('RGB')
        D_cp = self.D.copy()

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        A_transform = get_transform(self.opt, {'crop_pos': (15, 15), 'flip': transform_params['flip']}, grayscale=(self.input_nc == 1))
        # A_transform = get_transform(self.opt, {'crop_pos': (15, 15), 'flip': transform_params['flip']}, grayscale=1)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        C_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        D_transform = get_transform(self.opt, {'crop_pos': (15, 15), 'flip': transform_params['flip']}, grayscale=(self.input_nc == 1))

        # C = C_transform(B)
        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)
        D_cp = D_transform(D_cp)

        return {'A': A, 'B': B, 'C': C, 'D': D_cp, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
