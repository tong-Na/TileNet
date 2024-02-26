"""计算fid、ssim、lpips指标"""

import os
import cv2
import torch
from tqdm import tqdm
from lpips import LPIPS
from pytorch_fid import fid_score
from skimage.metrics import structural_similarity as ssim_loss

real_dir = r"~/try-on/data_for_metrics/real_data"
###
fake_dir = r"~/try-on/data_for_metrics/fake_data_0shaped"
###

# fid
fid_value = fid_score.calculate_fid_given_paths(
    [real_dir, fake_dir], batch_size=1, cuda=True, dims=2048
)

print("fid: ", fid_value)  # 14.239389871283123

img_real_names = sorted(os.listdir(os.path.join(real_dir)))
img_fake_names = sorted(os.listdir(os.path.join(fake_dir)))

# ssim
ssim_value = 0

# lpips
lpips_model = LPIPS(net="alex")
lpips_value = 0

for i in tqdm(range(len(img_real_names))):
    img_real_path = os.path.join(real_dir, img_real_names[i])
    img_fake_path = os.path.join(fake_dir, img_fake_names[i])
    img_real = cv2.imread(img_real_path)
    img_fake = cv2.imread(img_fake_path)
    img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
    img_fake = cv2.cvtColor(img_fake, cv2.COLOR_BGR2RGB)

    ssim_value += ssim_loss(img_real, img_fake, channel_axis=2)

    # img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)
    # img_fake = cv2.cvtColor(img_fake, cv2.COLOR_BGR2GRAY)

    # ssim_value += ssim_loss(img_real, img_fake)

    img_real_tensor = torch.from_numpy(img_real).permute(2, 0, 1).float() / 255.0
    img_fake_tensor = torch.from_numpy(img_fake).permute(2, 0, 1).float() / 255.0

    lpips_value += lpips_model(
        img_real_tensor.unsqueeze(0), img_fake_tensor.unsqueeze(0)
    )

print("ssim ", ssim_value / len(img_real_names))  # 0.7427256777323276
print("lpips: ", lpips_value.item() / len(img_real_names))  # 0.18324577945402298
