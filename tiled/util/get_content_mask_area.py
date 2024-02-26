import cv2
import os
import pandas as pd
import numpy as np


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


phase = "train"
img_names = pd.read_csv(
    f"~/try-on/data/zalando-hd-resized/{phase}/{phase}_cloth_rm.csv"
)
img_names = list(img_names["image_id"])
img_names = sorted(img_names)
# img_names = [
#     'c/00006_00.jpg', 'c/00064_00.jpg', 'c/00035_00.jpg', 'c/00069_00.jpg',
#     'c/00373_00.jpg',
# ]
sum = len(img_names)
counts = 0
large_counts = 0

content_mask_dir = f"~/try-on/data/zalando-hd-resized/{phase}/content_mask_coarse"
shape_mask_dir = f"~/try-on/data/zalando-hd-resized/{phase}/mask"


delta = 0.6

for fname in img_names:
    fname = fname.split("/")[-1]
    prefix = fname.split(".")[0]

    content_mask_path = os.path.join(content_mask_dir, prefix + ".png")
    content_mask_img = cv2.imread(content_mask_path, cv2.IMREAD_GRAYSCALE)
    content_mask = content_mask_img >= 128

    h, w = content_mask_img.shape

    shape_mask_path = os.path.join(shape_mask_dir, prefix + ".png")
    shape_mask_img = cv2.imread(shape_mask_path, cv2.IMREAD_GRAYSCALE)
    shape_mask_img = cv2.resize(shape_mask_img, (w, h), interpolation=cv2.INTER_CUBIC)
    shape_mask = shape_mask_img >= 128

    content_area = np.sum(content_mask)
    shape_area = np.sum(shape_mask)

    ratio = content_area / shape_area
    # print(ratio)
    if ratio > delta:
        large_counts += 1

    counts += 1
    print(prefix + ".png", f"{(counts/sum * 100):.2f}%\t{counts}/{sum}")

print(f"Total number of masks that are too large: {large_counts}")
