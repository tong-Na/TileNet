"""用于端到端模型的数据读取"""

import os
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import pandas as pd


class End2endDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # opt.phase = "train"
        self.dir_A = os.path.join(opt.dataroot, f"{opt.phase}/image")
        self.dir_B = os.path.join(opt.dataroot, f"{opt.phase}/cloth")

        csv_dir = os.path.join(opt.dataroot, f"{opt.phase}/{opt.phase}_cloth_rm.csv")

        self.imgid, _ = get_all_landmarks(csv_dir)

        self.A_paths = [
            os.path.join(self.dir_A, img.split("/")[-1]) for img in self.imgid
        ]
        self.B_paths = [
            os.path.join(self.dir_B, img.split("/")[-1]) for img in self.imgid
        ]

        self.tilegan2 = opt.tilegan2
        if opt.tilegan2:
            self.dir_C = os.path.join(opt.dataroot, f"{opt.phase}/tilegan_coarse")
            self.C_paths = [
                os.path.join(self.dir_C, img.split("/")[-1][:-3] + "png")
                for img in self.imgid
            ]

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert("RGB")

        B_path = self.B_paths[index]
        B = Image.open(B_path).convert("RGB")

        transform_params = get_params(self.opt, A.size)
        img_transform = get_transform(self.opt, {"flip": transform_params["flip"]})
        A = img_transform(A)
        B = img_transform(B)

        if self.tilegan2:
            C_path = self.C_paths[index]
            C = Image.open(C_path).convert("RGB")
            C = img_transform(C)
            return {"A": A, "B": B, "C": C, "A_paths": A_path}

        return {"A": A, "B": B, "A_paths": A_path}

    def __len__(self):
        """Return the total number of control point sets in the file."""
        return len(self.imgid)


def get_all_landmarks(path):
    df = pd.read_csv(path)
    value = []
    for j in df["landmarks"]:
        value.append(eval(j))
    return [list(df["image_id"]), value]
