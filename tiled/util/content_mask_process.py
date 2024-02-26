"""
对生成的内容掩码进行后处理，包括resize、轮廓掩码规范形状等
"""

import cv2
import os
import pandas as pd
import numpy as np


def save_image(img, img_path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img)


phase = "test"
img_names = ["./00006_00.jpg"]

shape_mask_dir = f"~/try-on/data/zalando-hd-resized/{phase}/mask"
# content_mask_dir = f'~/try-on/data/zalando-hd-resized/{phase}/content_mask_coarse'
content_mask_dir = (
    f"~/try-on/tiled/results/content_l2_tps/{phase}_latest/content_mask_gen"
)
target_dir = f"."

h, w = 1024, 768

for fname in img_names:
    fname = fname.split("/")[-1]
    prefix = fname.split(".")[0]

    content_mask_path = os.path.join(content_mask_dir, prefix + ".png")
    content_mask_img = cv2.imread(content_mask_path, cv2.IMREAD_GRAYSCALE)
    # content_mask_img = cv2.resize(content_mask_img, (w, h), interpolation = cv2.INTER_CUBIC)
    content_mask = content_mask_img >= 128

    shape_mask_path = os.path.join(shape_mask_dir, prefix + ".png")
    shape_mask_img = cv2.imread(shape_mask_path, cv2.IMREAD_GRAYSCALE)
    shape_mask_img = cv2.resize(
        shape_mask_img, (224, 224), interpolation=cv2.INTER_CUBIC
    )
    shape_mask = shape_mask_img >= 128

    content_mask_img[~(content_mask & shape_mask)] = 0

    save_image(content_mask_img, os.path.join(target_dir, prefix + ".png"))
