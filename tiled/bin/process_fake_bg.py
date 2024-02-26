import cv2
import os
import pandas as pd


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(img, img_path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


phase = "test"
img_names = pd.read_csv(
    f"~/try-on/data/zalando-hd-resized/{phase}/{phase}_cloth_rm.csv"
)
img_names = list(img_names["image_id"])
img_names = sorted(img_names)
sum = len(img_names)
counts = 0

raw_dir = r"~/try-on/tiled/results/inpaint_lama_raw/test_latest/images"
shape_mask_dir = f"~/try-on/data/zalando-hd-resized/{phase}/mask"
target_dir = r"~/try-on/data_for_metrics/fake_data_0tps"
mkdir(target_dir)

for fname in img_names:
    fname = fname.split("/")[-1]
    prefix = fname.split(".")[0]

    raw_path = os.path.join(raw_dir, prefix + "_fake_B.png")
    raw_img = load_image(raw_path)
    h, w, _ = raw_img.shape

    shape_mask_path = os.path.join(shape_mask_dir, prefix + ".png")
    shape_mask_img = cv2.imread(shape_mask_path, cv2.IMREAD_GRAYSCALE)
    shape_mask_img = cv2.resize(shape_mask_img, (w, h), interpolation=cv2.INTER_CUBIC)
    shape_mask = shape_mask_img < 10

    raw_img[shape_mask] = 246
    save_image(raw_img, os.path.join(target_dir, prefix + "_fake_B.png"))

    counts += 1
    print(prefix + ".png", f"{(counts/sum * 100):.2f}%\t{counts}/{sum}")
