import os
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import pandas as pd


class InpaintDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # ###
        # self.dir_A = os.path.join(opt.dataroot, f"{opt.phase}/shape_cloth")
        self.dir_A = os.path.join(opt.dataroot, f"{opt.phase}/{opt.phase}_tps")
        # self.dir_C = (
        #     "~/try-on/tiled/results/content_l2_tps/test_latest/content_mask_gen"
        # )
        self.dir_C = f"~/try-on/tiled/results/content_0shaped/{opt.phase}_latest"
        # self.dir_M = os.path.join(opt.dataroot, f'{opt.phase}/mask')
        # ###
        # self.dir_A = os.path.join(opt.dataroot, f"{opt.phase}/content_cloth")
        # self.dir_A = os.path.join(opt.dataroot, f"{opt.phase}/0shaped_content_cloth")
        self.dir_B = os.path.join(opt.dataroot, f"{opt.phase}/cloth")
        # self.dir_C = os.path.join(opt.dataroot, f"{opt.phase}/content_mask_coarse")
        # self.dir_C = os.path.join(
        #     opt.dataroot, f"{opt.phase}/0shaped_content_mask_coarse"
        # )

        csv_dir = os.path.join(opt.dataroot, f"{opt.phase}/{opt.phase}_cloth_rm.csv")

        self.imgid, _ = get_all_landmarks(csv_dir)

        # self.A_paths = [
        #     os.path.join(self.dir_A, img.split("/")[-1][:-3] + "png")
        #     for img in self.imgid
        # ]
        self.A_paths = [
            os.path.join(self.dir_A, img.split("/")[-1]) for img in self.imgid
        ]
        self.B_paths = [
            os.path.join(self.dir_B, img.split("/")[-1]) for img in self.imgid
        ]
        self.C_paths = [
            os.path.join(self.dir_C, img.split("/")[-1][:-3] + "png")
            for img in self.imgid
        ]

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert("RGB")

        B_path = self.B_paths[index]
        B = Image.open(B_path).convert("RGB")

        C_path = self.C_paths[index]
        C = Image.open(C_path).convert("L")

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        img_transform = get_transform(self.opt, {"flip": transform_params["flip"]})
        A = img_transform(A)
        B = img_transform(B)

        mask_transform = get_transform(
            self.opt, {"flip": transform_params["flip"]}, grayscale=1
        )
        mask = mask_transform(C)

        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        return {"A": A, "B": B, "mask": mask, "A_paths": A_path}

    def __len__(self):
        """Return the total number of control point sets in the file."""
        return len(self.imgid)


def get_all_landmarks(path):
    df = pd.read_csv(path)
    value = []
    for j in df["landmarks"]:
        value.append(eval(j))
    return [list(df["image_id"]), value]
