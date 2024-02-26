import torch
from PIL import Image
import torchvision.transforms as transforms


# mask_path = r'~/try-on/data/zalando-hd-resized/train/content_mask_coarse/00024_00.jpg'
# m = Image.open(mask_path).convert('L')

# transform_list = []
# method = transforms.InterpolationMode.BICUBIC
# transform_list.append(transforms.Grayscale(1))
# transform_list += [transforms.Resize((224, 224), method)]
# transform_list += [transforms.ToTensor()]
# transform_list += [transforms.Normalize(0.5, 0.5)]
# tf = transforms.Compose(transform_list)
# # print(m.shape)
# mask = tf(m)
# # m = m.unsqueeze(0)

# mask[mask <= 0] = 0
# mask[mask > 0] = 1
# print(mask[0][50:100, 50:100])

a = torch.tensor([[1, 2], [1, 2]], dtype=torch.float32)

b = torch.tensor([[1, 3], [0, 2]], dtype=torch.float32)

m = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)

a = a.unsqueeze(0).unsqueeze(0)
a = a.repeat(2, 3, 1, 1)

b = b.unsqueeze(0).unsqueeze(0)
b = b.repeat(2, 3, 1, 1)

m = m.unsqueeze(0).unsqueeze(0)
m = m.repeat(2, 1, 1, 1)


l1 = torch.abs(a - b).sum(dim=-1).sum(dim=-1).sum(dim=-1, keepdim=True)
S = m.sum(dim=-1).sum(dim=-1)
l1 /= S + 1e-7
l1 = l1.sum() / l1.shape[0]
print(l1)
