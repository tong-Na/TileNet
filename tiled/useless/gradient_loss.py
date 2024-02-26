import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

import numpy as np

# np.set_printoptions(threshold=np.inf)
import time
import cv2
import util
import os
import pandas as pd

import torch.nn.functional as F

transform_list = []
method = transforms.InterpolationMode.BICUBIC
transform_list.append(transforms.Resize([512, 512], method))
# transform_list.append(transforms.RandomCrop(256))
transform_list += [transforms.ToTensor()]
transform_list += [transforms.Normalize(0.5, 0.5)]
tf = transforms.Compose(transform_list)


def get_all_landmarks(path):
    df = pd.read_csv(path)
    value = []
    for j in df["landmarks"]:
        value.append(eval(j))
    return [list(df["image_id"]), value]


def calDiff(mask):
    mask_flip = torch.flip(mask, [-2])
    temp = mask - mask_flip
    return temp[temp != 0].shape


def rmAsym(phase):
    csv_dir = "~/try-on/tiled/landmarks/csv_data/result_without_sleeveless/"
    csv_path = os.path.join(csv_dir, f"{phase}_cloth_gen_result.csv")
    img_dir = f"../results/pix2pix_unet8_mask/{phase}/images/"

    imgs, landmarks = get_all_landmarks(csv_path)
    imgs = [img for img in imgs]
    landmarks = [ld for ld in landmarks]

    data = {"00": "image_id", "11": "landmarks"}
    frame = pd.DataFrame(data, index=[0])
    csv_path = os.path.join(csv_dir, f"{phase}_cloth_rm_asym.csv")
    frame.to_csv(csv_path, mode="w", index=False, header=False)

    sum = 0
    for idx in range(len(imgs)):
        im = imgs[idx]
        img_path = os.path.join(img_dir, f"{im[0:8]}_fake_B.png")
        A = Image.open(img_path).convert("L")
        A = tf(A)
        A = A.unsqueeze(0)

        A[A <= 0] = 0
        A[A > 0] = 1

        # 筛选不对称衣物
        score = calDiff(A)
        if score[0] >= 35000:
            print(im)
            sum += 1
        else:
            data = {"name": im, "value": [landmarks[idx]]}
            frame = pd.DataFrame(data)
            frame.to_csv(csv_path, mode="a", index=False, header=False)

    print(f"remove {sum} images")


def rmSleeveless(phase):
    csv_dir = "~/try-on/tiled/landmarks/csv_data/result_without_sleeveless/"
    csv_path = os.path.join(csv_dir, f"{phase}_cloth_rm_asym.csv")
    mask_dir = f"../results/pix2pix_unet8_mask/{phase}/images/"
    image_dir = f"../../data/zalando-hd-resized/{phase}/image-tps/image_to_gen_cloth"

    imgs, landmarks = get_all_landmarks(csv_path)
    imgs = [img for img in imgs]
    # landmarks = [ld for ld in landmarks]

    imgs = imgs[1400:1600]
    with open(f"../../data/zalando-hd-resized/sleeveless_{phase}.txt", "a") as f:
        for im in imgs:
            f.write(f"{im} 1\n")

    return

    imgs = [
        "00000_00",
        "00014_00",
        "00023_00",
        "00076_00",
        "00210_00",
        "00213_00",
        "00822_00",
        "00081_00",
        "00119_00",
        "00238_00",
        "00335_00",
        "00356_00",
        "00451_00",
        "00472_00",
        "00479_00",
    ]

    # data = {"00":"image_id", "11":"landmarks"}
    # frame = pd.DataFrame(data, index=[0])
    # csv_path = os.path.join(csv_dir, f'{phase}_cloth_rm_sless.csv')
    # frame.to_csv(csv_path, mode="w", index=False, header=False)

    sum = 0
    for idx in range(len(imgs)):
        im = imgs[idx]
        mask_path = os.path.join(mask_dir, f"{im[:8]}_fake_B.png")
        mask = Image.open(mask_path).convert("L")
        mask = tf(mask)
        mask = mask.unsqueeze(0)

        # image_path = os.path.join(image_dir, f'{im}_00.jpg')
        # image = Image.open(image_path).convert('RGB')
        # image = tf(image)
        # image = image.unsqueeze(0)

        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        # image = image * mask
        # mask = 1 - mask
        # image = image + mask

        # 筛选无袖衣物
        score = calDiff(mask)
        print(score)
        # if score[0] <= 40000:
        #     print(im)
        #     sum += 1
        # else:
        #     data = {"name": im, "value": [landmarks[idx]]}
        #     frame = pd.DataFrame(data)
        #     frame.to_csv(csv_path, mode="a", index=False, header=False)

    print(f"remove {sum} images")


def getLabels():
    imgids = []
    labels = []
    with open("~/try-on/data/zalando-hd-resized/sleeveless_train.txt", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            imgid, label = line.split(" ")
            imgids.append(imgid)
            labels.append(label)

    return [imgids, labels]


if __name__ == "__main__":

    # from networks import PerceptualLoss
    phase = "train"
    a, b = getLabels()
    print(len(a), len(b))
    # rmAsym(phase)
    # rmSleeveless(phase)

    # imgs = ['00000', '00014', '00023', '00076', '00210', '00213', '00752', '00791', '00822', '01045']

    # A = Image.open('../checkpoints/pix2pix_unet_num6/web/images/epoch083_real_B.png').convert('RGB')
    # B = Image.open('../checkpoints/pix2pix_unet_num6/web/images/epoch192_fake_B.png').convert('RGB')
    # A = Image.open('../results/pix2pix_unet8_mask/train/images/00000_00_fake_B.png').convert('L')
    # B = Image.open('../../data/zalando-hd-resized/train/image-tps/image_to_gen_cloth/00007_00.jpg').convert('RGB')
    # A = np.array(A)
    # B = np.array(B)

    # A = torch.tensor(A)
    # B = torch.tensor(B)

    # B = tf(B)
    # B = B.unsqueeze(0)

    # B = util.tensor2im(mask_flip)
    # util.save_image(B, 'temp.jpg')
