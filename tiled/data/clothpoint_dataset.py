import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import pandas as pd
import numpy
import torch


class ClothpointDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # data/zalando-hd-resized/train(test)
        # get the image directory
        self.dir_A = os.path.join(opt.dataroot, f"{opt.phase}/image")
        # self.dir_A = os.path.join(opt.dataroot, 'test/image')

        point_dir = (
            f"~/try-on/data/zalando-hd-resized/{opt.phase}/{opt.phase}_cloth.csv"
        )

        # read all control points in csv files
        self.imgid, self.keypoint = get_all_landmarks(point_dir)

        self.keypoint = torch.tensor(self.keypoint, dtype=torch.float32)
        self.keypoint = self.keypoint.reshape(-1, 22)

        self.A_paths = [os.path.join(self.dir_A, img[6:]) for img in self.imgid]

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert("RGB")

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(
            self.opt, {"flip": transform_params["flip"]}, grayscale=True
        )
        A = A_transform(A)

        keypoint = self.keypoint[index]

        return {"A": A, "heatmap": keypoint, "A_paths": A_path}

    def __len__(self):
        """Return the total number of control point sets in the file."""
        return len(self.keypoint)


def get_all_landmarks(path):
    df = pd.read_csv(path)
    value = []
    for j in df["landmarks"]:
        value.append(eval(j))
    return [list(df["image_id"]), value]
