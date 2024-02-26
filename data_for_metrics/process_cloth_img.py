"""
用于处理图像，以便计算指标
对数据集中的label数据和生成的fake数据重命名，保持命名一致，后缀都为png格式，一一匹配
"""

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


def get_all_landmarks(path):
    df = pd.read_csv(path)
    value = []
    for j in df["landmarks"]:
        value.append(eval(j))
    return [list(df["image_id"]), value]


if __name__ == "__main__":
    phase = "test"
    dataroot = r"~/try-on/data/zalando-hd-resized"
    csv_dir = os.path.join(dataroot, f"{phase}/{phase}_cloth_rm.csv")
    imgid, _ = get_all_landmarks(csv_dir)

    data_real_dir = os.path.join(dataroot, f"{phase}")
    ###
    # data_fake_dir = r"~/try-on/tiled/results/pix2pix_cgan"
    # data_fake_dir = os.path.join(data_fake_dir, f"{phase}_latest/images")
    data_fake_dir = r"~/try-on/data/zalando-hd-resized/test/shape_cloth"
    ###

    target_real_dir = r"~/try-on/data_for_metrics/real_data"
    ###
    target_fake_dir = r"~/try-on/data_for_metrics/fake_data_tps_shaped"
    ###

    if not os.path.exists(target_real_dir):
        os.makedirs(target_real_dir)

    if not os.path.exists(target_fake_dir):
        os.makedirs(target_fake_dir)

    for i, img in enumerate(tqdm(imgid)):
        # 保存数据集中原图
        # real_path = os.path.join(data_real_dir, img)
        # target_real_path = os.path.join(
        #     target_real_dir, img.split("/")[-1][:-4] + "_real_B.png"
        # )
        # real_img = Image.open(real_path).convert("RGB")
        # real_img = real_img.resize((256, 256), Image.Resampling.BICUBIC)
        # real_img.save(target_real_path, format="PNG")

        # 保存生成图
        ###
        # fake_path = os.path.join(data_fake_dir, img.split("/")[-1][:-4] + "_fake_B.png")
        fake_path = os.path.join(data_fake_dir, img.split("/")[-1][:-4] + ".png")
        ###
        target_fake_path = os.path.join(
            target_fake_dir, img.split("/")[-1][:-4] + "_fake_B.png"
        )
        fake_img = Image.open(fake_path).convert("RGB")
        ###
        fake_img = fake_img.resize((256, 256), Image.Resampling.BICUBIC)
        ###
        fake_img.save(target_fake_path, format="PNG")
