import os
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import torch


class ClassifyDataset(BaseDataset):
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
        # phase = 'train'
        self.mask_dir = f"~/try-on/tiled/results/pix2pix_unet8_mask/{phase}/images"

        label_path = f"~/try-on/data/zalando-hd-resized/sleeveless_{phase}.txt"
        self.imgid, self.labels = get_all_labels(label_path)

        self.mask_paths = [
            os.path.join(self.mask_dir, img[0:8] + "_fake_B.png") for img in self.imgid
        ]

        assert (
            self.opt.load_size >= self.opt.crop_size
        )  # crop_size should be smaller than the size of loaded image
        self.input_nc = 1

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
        mask_path = self.mask_paths[index]

        mask = Image.open(mask_path).convert("L")

        label = self.labels[index]
        label = [label, 1 - label]

        mask_transform = get_transform(self.opt, None, grayscale=1, unchanged=False)

        mask = mask_transform(mask)

        # mask[mask <= 0] = 0
        # mask[mask > 0] = 1
        label = torch.tensor(label, dtype=torch.float32)

        return {"mask": mask, "label": label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.mask_paths)


def get_all_labels(path):
    imgids = []
    labels = []
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            imgid, label = line.split(" ")
            imgids.append(imgid)
            labels.append(int(label))

    return [imgids, labels]
