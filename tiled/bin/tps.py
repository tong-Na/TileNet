"""
根据源关键点和目标关键点进行tps插值变换
"""

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import functional as F


class TpsGridGen(nn.Module):
    def __init__(self, size=(768, 1024), dtype=torch.float):
        super(TpsGridGen, self).__init__()

        # Create a grid in numpy.
        self.size = size
        grid_x, grid_y = np.meshgrid(
            np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1])
        )
        self.grid_x = torch.tensor(grid_x, dtype=dtype)
        self.grid_y = torch.tensor(grid_y, dtype=dtype)

    # TODO: refactor
    def apply_transformation(self, points_source, points_target):
        """
        inputs:
            (points_source, points_target): 控制点对
            points_source: 控制点对中来自原图的点 N*2 ，值归一化到[-1,1]之间
            points_target: 控制点对中来自目标图（扭曲图）的点 N*2 ，值归一化到[-1,1]之间
        return:
            grid: m*m*2的向量，其中grid[i][j]表示原图中i,j位置的点与扭曲图中哪个点对应，
            最后一维'2'表示对应的点的坐标(x,y)

        """
        points_source = points_source.transpose(1, 0)
        points_target = points_target.transpose(1, 0)

        self.N = points_source.shape[0]
        # 计算控制点之间的距离的平方，进一步计算r^2log(r^2)
        source_x = points_source[:, :1]
        source_y = points_source[:, 1:]
        distance_source = (source_x - source_x.transpose(1, 0)) ** 2 + (
            source_y - source_y.transpose(1, 0)
        ) ** 2
        # distance_source = torch.sqrt(distance_source)
        distance_source[distance_source == 0] = 1
        distance_source = torch.mul(distance_source, torch.log(distance_source))
        # N*N

        # 计算待插值点与控制点之间的距离
        m, n = self.size
        points_x = self.grid_x.reshape(-1, 1)  # m*m * 1
        points_y = self.grid_y.reshape(-1, 1)  # m*m * 1
        distance_all = (points_x - source_x.transpose(1, 0)) ** 2 + (
            points_y - source_y.transpose(1, 0)
        ) ** 2
        distance_all[distance_all == 0] = 1
        distance_all = torch.mul(distance_all, torch.log(distance_all))
        # m*m * N

        One = torch.ones((self.N, 1), dtype=torch.float)
        Zero = torch.zeros((3, 3), dtype=torch.float)
        P = torch.cat((One, points_source), dim=1)
        R = torch.cat(
            (
                torch.cat((distance_source, P), dim=1),
                torch.cat((P.transpose(1, 0), Zero), dim=1),
            ),
            dim=0,
        )
        # R:N+3 * N+3
        # R A = Z    A:N+3 * 2
        Z = torch.cat((points_target, torch.zeros((3, 2), dtype=torch.float)), dim=0)
        # 解方程 RA=Z，计算A
        R_inv = torch.linalg.pinv(R)
        A = torch.mm(R_inv, Z)
        # A = torch.linalg.solve(R, Z)

        One2 = torch.ones((m * n, 1), dtype=torch.float)

        P2 = torch.cat((One2, torch.cat((points_x, points_y), dim=1)), dim=1)
        R2 = torch.cat((distance_all, P2), dim=1)
        # R2:m*m * N+3
        R2 = R2.unsqueeze(dim=0)
        A = A.unsqueeze(dim=0)
        grid = torch.bmm(R2, A)

        # grid:b * m*m * 2
        grid = grid.reshape(-1, n, m, 2)
        return grid

    def warp(self, img_point, cloth_point, img):
        points_source = img_point
        points_target = cloth_point

        points_source = torch.tensor(points_source, dtype=torch.float)
        points_target = torch.tensor(points_target, dtype=torch.float)
        points_source = points_source.transpose(1, 0)
        points_target = points_target.transpose(1, 0)
        # 到此为止，points_source格式为：n * 2，每一行表示某个控制点的坐标
        # 每个坐标的数值在0-1之间，即将x坐标除以宽度，y坐标除以高度，左上角(0,0)，右下角(1,1)

        # 将控制点坐标从0,1转换到-1,1区间
        points_source = points_source * 2 - 1
        points_target = points_target * 2 - 1

        grid = self.apply_transformation(points_target, points_source)

        warped = F.grid_sample(img, grid, padding_mode="border", align_corners=True)
        # warped = tensor2im(warped)
        # save_image(warped, '~/try-on/data/zalando-hd-resized/train/image-tps/image_to_gen_cloth/'+name)
        return warped


def tensor2im(input_image, imtype=np.uint8):
    """ "Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (
            (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        )  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def csv_to_dict(path):
    df = pd.read_csv(path)
    key = []
    value = []
    for i in df["image_id"]:  # “score”用作值
        key.append(i[6:])
    for j in df["landmarks"]:  # “score”用作值
        # value.append(eval(j))
        value.append(np.array(eval(j)).reshape(11, 2))
    r = zip(key, value)
    return r


if __name__ == "__main__":
    transform_list = []
    method = transforms.InterpolationMode.BICUBIC
    # transform_list.append(transforms.Resize([256, 256], method))
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(0.5, 0.5)]
    tf = transforms.Compose(transform_list)

    tps = TpsGridGen()

    phase = "train"

    img_points = csv_to_dict(
        f"~/try-on/data/zalando-hd-resized/{phase}/{phase}_image.csv"
    )
    cloth_points = dict(
        csv_to_dict(f"~/try-on/data/zalando-hd-resized/{phase}/{phase}_cloth.csv")
    )

    k = 4  # 将变换过程分成k份，每次移动1/k，结果通常比一次到位更好些
    for img_point in img_points:
        name = img_point[0]
        img_path = f"~/try-on/data/zalando-hd-resized/{phase}/image/" + name
        img = Image.open(img_path).convert("RGB")
        img = tf(img)
        img = img.unsqueeze(dim=0)

        start = img_point[1]
        end = cloth_points[img_point[0]]
        start[3, :] = end[3, :]
        start[5, :] = end[5, :]
        if end[-1][0] <= 0 or end[-1][1] <= 0 or end[-2][0] <= 0 or end[-2][1] <= 0:
            start = start[:-2]
            end = end[:-2]
        delta = [
            [(end[i][0] - start[i][0]) / k, (end[i][1] - start[i][1]) / k]
            for i in range(len(start))
        ]
        mid = [[start[i][0], start[i][1]] for i in range(len(start))]
        warped = img
        for i in range(k):
            start = [pos for pos in mid]
            mid = [
                [mid[i][0] + delta[i][0], mid[i][1] + delta[i][1]]
                for i in range(len(mid))
            ]
            warped = tps.warp(start, mid, warped)

        warped = tensor2im(warped)
        save_image(warped, f"~/try-on/tiled/landmarks/tps_test/{phase}_tps/" + name)
