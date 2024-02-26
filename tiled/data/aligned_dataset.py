import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import pandas as pd


class AlignedDataset(BaseDataset):
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
        phase = opt.phase
        self.dir_AB = os.path.join(opt.dataroot, phase)  # get the image directory
        self.dir_A = os.path.join(self.dir_AB, "cloth")
        # self.dir_A = os.path.join(self.dir_AB, 'cloth-mask')
        # self.dir_B = os.path.join(self.dir_AB, 'image')
        # self.dir_B = os.path.join(self.dir_AB, 'segment')
        self.dir_B = os.path.join(self.dir_AB, "image-tps/image_to_gen_cloth")
        # self.dir_B = "./results/tryon_pix2pix_norm/train"

        csv_dir = "~/try-on/tiled/landmarks/csv_data/result_without_sleeveless/"
        # csv_path = os.path.join(csv_dir, f'{phase}_cloth_gen_result.csv')
        csv_path = os.path.join(csv_dir, f"{phase}_cloth_rm_sless.csv")
        self.imgid, self.landmarks_cloth = get_all_landmarks(csv_path)
        # self.A_paths = sorted(make_dataset(self.dir_A))  # get image paths
        self.A_paths = [os.path.join(self.dir_A, img) for img in self.imgid]
        # self.B_paths = sorted(make_dataset(self.dir_B))  # get image paths
        self.B_paths = [os.path.join(self.dir_B, img) for img in self.imgid]
        # self.B_paths = [os.path.join(self.dir_B, img[0:8]+'_fake_B.png') for img in self.imgid]
        self.dir_mask = f"~/try-on/tiled/results/pix2pix_unet8_mask/{phase}/images"
        self.mask_paths = [
            os.path.join(self.dir_mask, img[0:8] + "_fake_B.png") for img in self.imgid
        ]

        assert (
            self.opt.load_size >= self.opt.crop_size
        )  # crop_size should be smaller than the size of loaded image
        self.input_nc = (
            self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        )
        self.output_nc = (
            self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc
        )

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

        A = Image.open(A_path).convert("RGB")
        B = Image.open(B_path).convert("RGB")

        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path).convert("L")

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        A_transform = get_transform(
            self.opt, {"flip": transform_params["flip"]}, grayscale=(self.input_nc == 1)
        )
        # A_transform = get_transform(self.opt, {'crop_pos': (15, 15), 'flip': transform_params['flip']}, grayscale=1)
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        B_transform = get_transform(
            self.opt, {"flip": transform_params["flip"]}, grayscale=(self.input_nc == 1)
        )
        C_transform = get_transform(
            self.opt, {"flip": transform_params["flip"]}, grayscale=1
        )

        A = A_transform(A)
        B = B_transform(B)
        mask = C_transform(mask)

        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        B = B * mask
        mask = (1 - mask) * (246 / 255)
        B = B + mask

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)


def get_all_landmarks(path):
    df = pd.read_csv(path)
    value = []
    for j in df["landmarks"]:
        value.append(eval(j))
    return [list(df["image_id"]), value]
