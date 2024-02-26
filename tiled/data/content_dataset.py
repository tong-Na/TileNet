import os
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import pandas as pd


class ContentDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # opt.phase = "train"
        # self.dir_A = os.path.join(opt.dataroot, f"{opt.phase}/shape_cloth")
        self.dir_A = os.path.join(opt.dataroot, f"{opt.phase}/{opt.phase}_tps")
        # self.dir_A = os.path.join(opt.dataroot, f"{opt.phase}/raw_shape_cloth")
        ###
        # self.dir_B = os.path.join(opt.dataroot, f"{opt.phase}/content_cloth")
        ###
        # self.dir_M = os.path.join(opt.dataroot, f"{opt.phase}/content_mask_coarse")
        self.dir_M = os.path.join(
            opt.dataroot, f"{opt.phase}/0shaped_content_mask_coarse"
        )

        point_dir = (
            f"~/try-on/data/zalando-hd-resized/{opt.phase}/{opt.phase}_cloth_rm.csv"
        )

        self.imgid, _ = get_all_landmarks(point_dir)

        ###
        # dirs = [self.dir_A, self.dir_B]
        # self.A_paths = [
        #     os.path.join(d, img[-12:-3] + "png") for img in self.imgid for d in dirs
        # ]
        # self.M_paths = [
        #     os.path.join(self.dir_M, img[-12:-3] + "png")
        #     for img in self.imgid
        #     for d in dirs
        # ]
        ###
        self.A_paths = [os.path.join(self.dir_A, img[-12:]) for img in self.imgid]
        # self.A_paths = [
        #     os.path.join(self.dir_A, img[-12:-3] + "png") for img in self.imgid
        # ]
        self.M_paths = [
            os.path.join(self.dir_M, img[-12:-3] + "png") for img in self.imgid
        ]

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert("RGB")

        M_path = self.M_paths[index]
        M = Image.open(M_path).convert("L")

        transform_params = get_params(self.opt, A.size)
        transform_img = get_transform(
            self.opt, {"flip": transform_params["flip"]}, grayscale=False
        )
        A = transform_img(A)

        mask_transform = get_transform(
            self.opt, {"flip": transform_params["flip"]}, grayscale=1
        )
        mask = mask_transform(M)

        return {"input": A, "label": mask, "A_paths": A_path}

    def __len__(self):
        """Return the total number of control point sets in the file."""
        return len(self.imgid)


def get_all_landmarks(path):
    df = pd.read_csv(path)
    value = []
    for j in df["landmarks"]:
        value.append(eval(j))
    return [list(df["image_id"]), value]
