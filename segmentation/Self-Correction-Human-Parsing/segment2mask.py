"""
根据outouts目录中的20种颜色的分割图计算衣物掩膜，原始人物图像叠加衣物掩膜，得到只有衣物的掩码,
将该掩码取反后与轮廓掩码取交集，得到无关内容掩码
"""

import cv2
import os
import pandas as pd
import numpy as np

"""
0, 0, 0         背景
128, 0, 0       帽子
0, 128, 0       头发
128, 128, 0     手套
0, 0, 128       太阳镜
128, 0, 128     上衣    34
0, 128, 128     连衣裙  40
128, 128, 128   外套    42
64, 0, 0        袜子
192, 0, 0       裤子
64, 128, 0      连衣裤
192, 128, 0     围巾
64, 0, 128      短裙    33
192, 0, 128     脸
64, 128, 128    左臂
192, 128, 128   右臂
0, 64, 0        左腿
128, 64, 0      右腿
0, 192, 0       左鞋
128, 192, 0     右鞋
"""

seg_dict = {
    "0_0_0": 0,
    "128_0_0": 0,
    "0_128_0": 0,
    "128_128_0": 0,
    "0_0_128": 0,
    "128_0_128": 1,
    "0_128_128": 1,
    "128_128_128": 1,
    "64_0_0": 0,
    "192_0_0": 0,
    "64_128_0": 0,
    "192_128_0": 0,
    "64_0_128": 0,
    "192_0_128": 0,
    "64_128_128": 0,
    "192_128_128": 0,
    "0_64_0": 0,
    "128_64_0": 0,
    "0_192_0": 0,
    "128_192_0": 0,
}

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(img, img_path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_mask(img):
    # 获取图像img对应的衣物掩膜，其中衣物部分为False，其余部分为True
    img //= 64  # 将诸如rgb128, 0, 192转换为2, 0, 3便于计算对应的唯一值
    pix_id = (
        img[:, :, 0] + img[:, :, 1] * 4 + img[:, :, 2] * 16
    )  # 按照4进制计算/64后的rgb对应的数值
    # seg_dict中衣物部分rgb对应的4进制数字
    cloth_mask = (
        (pix_id == 34) | (pix_id == 40) | (pix_id == 42)
    )  # | (pix_id == 3) | (pix_id == 9) | (pix_id == 33)
    # 由于要将clothmask中True的位置置为0，因此需要取反，保留衣物部分，去除其余部分
    return ~cloth_mask


phase = "test"
seg_dir = f"~/try-on/segmentation/Self-Correction-Human-Parsing/{phase}_outputs"
# raw_dir = f"~/try-on/data/zalando-hd-resized/{phase}/{phase}_tps"
raw_dir = f"~/try-on/data/zalando-hd-resized/{phase}/image"
mask_dir = f"~/try-on/data/zalando-hd-resized/{phase}/mask"
# mask_dir = '~/try-on/segmentation/Self-Correction-Human-Parsing/mask'
# raw_dir = '~/try-on/segmentation/Self-Correction-Human-Parsing/inputs'
target_dir = f"~/try-on/data/zalando-hd-resized/{phase}/0shaped_content_mask_coarse"
# target_dir = '~/try-on/segmentation/Self-Correction-Human-Parsing/output2'
# target_dir = './outputs'
mkdir(target_dir)

img_names = pd.read_csv(
    f"~/try-on/data/zalando-hd-resized/{phase}/{phase}_cloth_rm.csv"
)
img_names = list(img_names["image_id"])
img_names = sorted(img_names)
sum = len(img_names)
counts = 0
for fname in img_names:
    fname = fname.split("/")[-1]
    prefix = fname.split(".")[0]
    seg_path = os.path.join(seg_dir, prefix + ".png")
    seg_img = load_image(seg_path)
    cloth_mask = get_mask(seg_img)

    # cloth_mask = ~cloth_mask
    cloth_mask_pic = np.zeros((seg_img.shape[0], seg_img.shape[1]), dtype=np.uint8)
    cloth_mask_pic[cloth_mask] = 255
    cv2.imwrite(os.path.join(target_dir, prefix + ".png"), cloth_mask_pic)

    # raw_path = os.path.join(raw_dir, fname)
    # raw_img = load_image(raw_path)

    # mask_path = os.path.join(mask_dir, prefix + ".png")
    # mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # mask_img = cv2.resize(mask_img, (raw_img.shape[1], raw_img.shape[0]))
    # shape_mask = mask_img >= 128

    # content_mask = shape_mask & cloth_mask
    # content_mask_pic = np.zeros_like(mask_img)
    # content_mask_pic[content_mask] = 255
    # cv2.imwrite(os.path.join(target_dir, prefix + ".png"), content_mask_pic)

    counts += 1
    print(prefix + ".png", f"{(counts/sum * 100):.2f}%\t{counts}/{sum}")

"""
斑点
7175,7383,8312,14442,00200,01459,03468,03687
错误
8109,9318
"""
