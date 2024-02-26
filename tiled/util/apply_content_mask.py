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

raw_dir = f"~/try-on/data/zalando-hd-resized/{phase}/{phase}_tps"
# raw_dir = f"~/try-on/data/zalando-hd-resized/{phase}/image"
# content_mask_dir = (
#     f"~/try-on/data/zalando-hd-resized/{phase}/content_mask_coarse"
# )
content_mask_dir = (
    f"~/try-on/data/zalando-hd-resized/{phase}/0shaped_content_mask_coarse"
)
cloth_dir = f"~/try-on/data/zalando-hd-resized/{phase}/cloth"
# target_dir = f"~/try-on/data/zalando-hd-resized/{phase}/content_cloth"
target_dir = f"~/try-on/data/zalando-hd-resized/{phase}/0shaped_content_cloth"
mkdir(target_dir)

for fname in img_names:
    fname = fname.split("/")[-1]
    prefix = fname.split(".")[0]

    raw_path = os.path.join(raw_dir, fname)
    raw_img = load_image(raw_path)

    content_mask_path = os.path.join(content_mask_dir, prefix + ".png")
    content_mask_img = cv2.imread(content_mask_path, cv2.IMREAD_GRAYSCALE)
    content_mask = content_mask_img >= 128

    cloth_path = os.path.join(cloth_dir, fname)
    cloth_img = load_image(cloth_path)

    cloth_img[content_mask] = raw_img[content_mask]
    save_image(cloth_img, os.path.join(target_dir, prefix + ".png"))

    counts += 1
    print(prefix + ".png", f"{(counts/sum * 100):.2f}%\t{counts}/{sum}")
