import cv2
import os


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


img_name = r"~/try-on/data_for_metrics/real_data/00006_00_real_B.png"
img = load_image(img_name)
print(img[0:3, 0:3, :])  # 246
