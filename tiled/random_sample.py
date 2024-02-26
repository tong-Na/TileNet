import os
import pandas as pd
import random
from PIL import Image


def get_all_landmarks(path):
    df = pd.read_csv(path)
    return list(df["image_id"])


def combine(imgs, mode="h"):

    w, h = imgs[0].size
    n = len(imgs)

    joint = Image.new("RGB", (w * n, h))
    x, y = 0, 0
    if mode == "h":
        # joint = Image.new('RGB', (img1.size[0] + img2.size[0], img1.size[1]))
        for i in range(n):
            joint.paste(imgs[i], (x, y))
            x += w
    else:
        joint = Image.new("RGB", (w, h * n))
        for i in range(n):
            joint.paste(imgs[i], (x, y))
            y += h

    return joint


def getTotalImage(sample_list, img_dirs, out_img="temp.png"):
    joint_list = []
    for sample in sample_list:
        img_list = []
        for i in range(len(img_dirs)):
            img = Image.open(os.path.join(img_dirs[i], f"{sample[:-4]}_fake_B.png"))
            img_list.append(img)

        imgB = Image.open(os.path.join(img_dirs[0], f"{sample[:-4]}_real_B.png"))
        imgA = Image.open(os.path.join(img_dirs[0], f"{sample[:-4]}_real_A.png"))
        img_list += [imgB, imgA]
        joint = combine(img_list, "h")
        joint_list.append(joint)

    joint = combine(joint_list, "v")
    joint.save(out_img)


if __name__ == "__main__":
    csv_dir = "~/try-on/tiled/landmarks/csv_data/result_without_sleeveless/"
    csv_path = os.path.join(csv_dir, "test_cloth_rm_sless.csv")
    img_id = get_all_landmarks(csv_path)
    # print(img_id)
    counts = len(img_id)
    print(counts)
    random.seed(2022)
    random.shuffle(img_id)
    sample_list = random.sample(img_id, 20)
    # sample_list.append("00460_00.jpg")
    # sample_list.append("00705_00.jpg")
    # sample_list.append("00470_00.jpg")
    sample_list.sort()
    print(sample_list)
    # ['00814_00.jpg', '01035_00.jpg', '02388_00.jpg', '02760_00.jpg', '03721_00.jpg',
    #  '04517_00.jpg', '04672_00.jpg', '06208_00.jpg', '06663_00.jpg', '07966_00.jpg',
    #  '08114_00.jpg', '09555_00.jpg', '09691_00.jpg', '09912_00.jpg', '11635_00.jpg',
    #  '11675_00.jpg', '11709_00.jpg', '11906_00.jpg', '12066_00.jpg', '13140_00.jpg']

    img_dir6 = "~/try-on/tiled/results/transunet0-200_wmask_gradB_symm_L1A1.3_L1B1-605_rmsless/test_latest/images"
    img_dir0 = "~/try-on/tiled/results/transunet0-200_wmask_gradB_symm_L1A1.3_L1B1-615_rmsless/test_latest/images"
    img_dir1 = "~/try-on/tiled/results/transunet0-200_wmask_gradB_symm_L1A1.3_L1B1-625_rmsless/test_latest/images"
    img_dir2 = "~/try-on/tiled/results/joint_no-condition_L1A0.8B2-lambda100-full_dp_noshare_seg/test_latest/images"
    img_dir3 = "~/try-on/tiled/results/joint_no-condition_L1A0.8B2-lambda100-full_dp_noshare_unseg/test_latest/images"
    img_dir4 = "~/try-on/tiled/results/joint_no-condition_L1A0.8B2-lambda300-rmsless_dp_noshare/test_latest/images"
    img_dir5 = "~/try-on/tiled/results/refine_pixhd_wmask_gradB_symm_L1A0.85_L1B1-250_rmsless_dp_t0-200_noshare/test_latest/images"
    getTotalImage(sample_list, [img_dir3])
