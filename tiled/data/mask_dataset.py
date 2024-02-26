import os
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import pandas as pd
import torch


class MaskDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # opt.phase = 'train'
        self.dir_A = os.path.join(opt.dataroot, f"{opt.phase}/cloth-mask")

        point_dir = (
            f"~/try-on/data/zalando-hd-resized/{opt.phase}/{opt.phase}_cloth_rm.csv"
        )

        self.imgid, self.keypoint = get_all_landmarks(point_dir)
        self.heatmap_size = opt.crop_size
        self.keypoint = torch.tensor(self.keypoint, dtype=torch.float32)
        self.keypoint *= self.heatmap_size

        self.A_paths = [os.path.join(self.dir_A, img[-12:]) for img in self.imgid]

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert("RGB")

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(
            self.opt, {"flip": transform_params["flip"]}, grayscale=1
        )
        A = A_transform(A)

        keypoint = self.keypoint[index]
        keypoint = keypoint.long()
        heatmap = get_heatmap(keypoint, (self.heatmap_size, self.heatmap_size))

        return {"points": heatmap, "real_mask": A, "A_paths": A_path}

    def __len__(self):
        """Return the total number of control point sets in the file."""
        return len(self.imgid)


def get_heatmap(keypoints, heatmap_size, sigma=3):
    num_keypoints, _ = keypoints.size()
    h, w = heatmap_size

    # 生成热图坐标的网格
    X = torch.linspace(0, h - 1, h)
    Y = torch.linspace(0, w - 1, w)
    xx, yy = torch.meshgrid(X, Y, indexing="ij")
    xx = xx.unsqueeze(0).repeat(num_keypoints, 1, 1)  # n, h, w
    yy = yy.unsqueeze(0).repeat(num_keypoints, 1, 1)  # n, h, w

    x0 = keypoints[..., 1].unsqueeze(-1).unsqueeze(-1)
    y0 = keypoints[..., 0].unsqueeze(-1).unsqueeze(-1)
    # 计算欧氏距离
    distances = torch.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    weights = torch.exp(-0.5 * (distances / sigma) ** 2)

    # heatmap = torch.zeros((num_keypoints, h, w))
    # heatmap = torch.where(distances <= 3 * sigma, weights, heatmap)
    heatmap = weights
    heatmap = heatmap.sum(dim=0).unsqueeze(0)

    return heatmap  # 1, h, w


def get_all_landmarks(path):
    df = pd.read_csv(path)
    value = []
    for j in df["landmarks"]:
        value.append(eval(j))
    return [list(df["image_id"]), value]
