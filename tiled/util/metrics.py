#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:2.py
# author:74049
# datetime:2023/6/12 18:39
# software: PyCharm
"""
this is functiondescription
"""
# import module your need
from tqdm import tqdm
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2

# import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import argparse

# import matplotlib.pyplot as plt

import pytorch_fid

'''
---------------ssim, psnr, lpips------------------------------------------------
1. 在conda虚拟环境中下载模块lpips, 修改lpips包中_init_.py中的load_image函数如下
def load_image(path):
    if(path[-3:] == 'dng'):
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
    elif(path[-3:]=='bmp' or path[-3:]=='jpg' or path[-3:]=='png' or path[-4:]=='jpeg'):
        import cv2
        img = cv2.imread(path)[:,:,::-1]
        img = cv2.resize(img, (192,256), interpolation=cv2.INTER_AREA)
        return img
    else:
        img = (255*plt.imread(path)[:,:,:3]).astype('uint8')
        img = cv2.resize(img, (192,256), interpolation=cv2.INTER_AREA)

    return img


2. 运行参数
    input_images_path:真实图像所在位置
    image2smiles2image_save_path:生成图像所在位置
    ****运行:补充上两个参数运行这个py文件就可以得到ssim, psnr, lpips


3. 注意
    line108:可以保存查看resize过后的图像(filepath包含了前缀地址, 切片是为了得到每张图像的名字)
    line191:使得图片两两对应(关键是生成图像的后缀)
----------------------------------------------------------------

---------------FID------------------------------------------------
1. 在conda虚拟环境中下载模块pytorch_fid, 首先修改pytorch_fid包中fid_score.py中的imread函数如下
def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    img = Image.open(filename)

    img = np.asarray(img, dtype=np.uint8)[..., :3]

    img = cv2.resize(img, (192, 256), interpolation=cv2.INTER_AREA)  

    return img


2. 运行参数
    GT_path:真实图像所在位置
    generated_path:生成图像所在位置
    ****终端运行:python -m pytorch_fid GT_path generated_path --dims 768

----------------------------------------------------------------
'''

parser = argparse.ArgumentParser(description="PSNR SSIM script", add_help=False)
parser.add_argument(
    "--input_images_path",
    default="~/try-on/tiled/results/refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1-350_rmsless_dp_t100_noshare/test_latest/images",
)
parser.add_argument(
    "--image2smiles2image_save_path",
    default="~/try-on/tiled/results/refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1-350_rmsless_dp_t100_noshare/test_latest/images",
)
parser.add_argument("-v", "--version", type=str, default="0.1")
args = parser.parse_args()


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg"])


# def addBounding(image, bound=128):

#     h, w, c = image.shape

#     if h == w:
#         return image

#     image_bound = np.ones((h, w+bound*2, c))*255
#     image_bound = image_bound.astype(np.uint8)
#     image_bound[:, bound:bound+w] = image

#     return image_bound

# def load_img(filepath):
#     img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
#     img = addBounding(img)

#     img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)


#     img = img.astype(np.float32)
#     img = img / 255.
#     return img


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (192, 256), interpolation=cv2.INTER_AREA)

    # cv2.imwrite("D:/000/"+filepath[102:], img)

    img = img.astype(np.float32)
    img = img / 255.0

    return img


class DataLoaderVal(Dataset):
    def __init__(self, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = args.input_images_path
        input_dir = args.image2smiles2image_save_path

        clean_files = sorted(os.listdir(os.path.join(gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(input_dir)))

        self.clean_filenames = [
            os.path.join(gt_dir, x)
            for x in clean_files
            if is_png_file(x) and x[-10:-6] == "fake"
        ]
        self.noisy_filenames = [
            os.path.join(input_dir, x)
            for x in noisy_files
            if is_png_file(x) and x[-10:-6] == "real"
        ]

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        return clean, noisy, clean_filename, noisy_filename


def get_validation_data():
    return DataLoaderVal(None)


test_dataset = get_validation_data()
test_loader = DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
)

## Initializing the model
# loss_fn = lpips.LPIPS(net='alex', version=args.version)


if __name__ == "__main__":

    # ---------------------- PSNR + SSIM ----------------------
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_groundtruth = data_test[0].numpy().squeeze().transpose((1, 2, 0))
        rgb_restored = data_test[1].cuda()

        rgb_restored = (
            torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))
        )
        psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_groundtruth))
        ssim_val_rgb.append(ssim_loss(rgb_restored, rgb_groundtruth, channel_axis=2))

    psnr_val_rgb = sum(psnr_val_rgb) / len(test_dataset)
    ssim_val_rgb = sum(ssim_val_rgb) / len(test_dataset)

    # ---------------------- LPIPS ----------------------
    # files = os.listdir(args.input_images_path)
    # i = 0
    # total_lpips_distance = 0
    average_lpips_distance = 0
    # for file in files:

    #     try:
    #         # Load images

    #         img0 = lpips.im2tensor(lpips.load_image(os.path.join(args.input_images_path, file)))

    #         file2 = file[-12: -4] + "_fake_B.png"

    #         img1 = lpips.im2tensor(lpips.load_image(os.path.join(args.image2smiles2image_save_path, file2)))

    #         if (os.path.exists(os.path.join(args.input_images_path, file)), os.path.exists(os.path.join(args.image2smiles2image_save_path, file2))):
    #             i = i + 1

    #         # Compute distance
    #         current_lpips_distance = loss_fn.forward(img0, img1)

    #         total_lpips_distance = total_lpips_distance + current_lpips_distance

    #     except Exception as e:
    #         print(e)

    # average_lpips_distance = float(total_lpips_distance) / i
    #
    # print("The processed iamges is ", i)
    print(
        "PSNR: %f, SSIM: %f, LPIPS: %f "
        % (psnr_val_rgb, ssim_val_rgb, average_lpips_distance)
    )
