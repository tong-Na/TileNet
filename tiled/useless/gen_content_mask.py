import torch
import os

# import util
import torchvision.transforms as transforms
from PIL import Image

# 读取图像A和图像B
cloth_dir = "~/try-on/data/zalando-hd-resized/train/cloth"
content_cloth_dir = "~/try-on/data/zalando-hd-resized/train/content_cloth"
fname = "00001_00.jpg"
image_A = Image.open(os.path.join(cloth_dir, fname)).convert("RGB")
image_B = Image.open(os.path.join(content_cloth_dir, fname)).convert("RGB")

# 转换为灰度图像方便处理
transform = transforms.Grayscale()
image_A_gray = transform(image_A)
image_B_gray = transform(image_B)

# 转换为Tensor
tensor_A = transforms.ToTensor()(image_A_gray).unsqueeze(0)
tensor_B = transforms.ToTensor()(image_B_gray).unsqueeze(0)

# 生成掩码
threshold = 0.02  # 设置阈值范围
diff = torch.abs(tensor_A - tensor_B)
mask = torch.where(diff <= threshold, torch.tensor(0), torch.tensor(255)).byte()

# 将Tensor转换为PIL图像并显示或保存掩码图像
mask_image = transforms.ToPILImage()(mask.squeeze(0))
# mask_image.show()
mask_image.save("mask_image.png")
