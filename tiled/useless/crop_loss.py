import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

import numpy as np
# np.set_printoptions(threshold=np.inf)
import time
import cv2


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        # h_relu4 = self.slice4(h_relu3)
        # h_relu5 = self.slice5(h_relu4)
        # out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        out = [h_relu1, h_relu2, h_relu3]
        return out


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        # self.criterion = nn.MSELoss()
        # self.weights = [1.0, 1.0/4, 1.0/8, 1.0/16, 1.0/32]

    def forward(self, x, y):
        # begin = time.time() 

        # tensor转为numpy, 直接用tensor.where的话返回的像素坐标也是tensor类型，在查找像素坐标最大值和最小值时耗时比numpy类型更长
        x = x.numpy()
        y = y.numpy()
   
        # 使用numpy.where查找到非背景像素，背景的RGB值在一些图片中不同，但大多都是[245 245 245]和[246 246 246] 所以使用<
        crop_x = np.where(x<[240,240,240])
        a0, b0, a, b = min(crop_x[0]), max(crop_x[0]), min(crop_x[1]), max(crop_x[1])
        img_x = x[int(a0)-10:int(b0)+10, int(a)-10:int(b)+10]
        # 需要计算时间时不可查看裁切结果
        # cv2.imshow("0", img_x)
        # cv2.waitKey()  # 按下任何键盘按键后
        # cv2.destroyAllWindows()  # 释放所有窗体

        crop_y = np.where(y<[240,240,240])
        a0, b0, a, b = min(crop_y[0]), max(crop_y[0]), min(crop_y[1]), max(crop_y[1])
        img_y = y[int(a0)-10:int(b0)+10, int(a)-10:int(b)+10]
        # cv2.imshow("1", img_y)
        # cv2.waitKey()  # 按下任何键盘按键后
        # cv2.destroyAllWindows()  # 释放所有窗体
       
        # 这个文件下定义的transform函数是由PIL转到tensor，所以需要先将上一步numpy转为PIL再通过tf函数转为相同尺寸大小的tensor
        # 应该可以改写ts函数使numpy直接到tensor，可省略Image.fromarray函数
        x = Image.fromarray(np.uint8(img_x))
        y = Image.fromarray(np.uint8(img_y))
        x = tf(x)
        x = x.unsqueeze(0)
        y = tf(y)
        y = y.unsqueeze(0)

        # end = time.time() 
        # print(end - begin)
      
        # 以下内容不变
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # gram_x = gram_matrix(x_vgg[i])
            # gram_y = gram_matrix(y_vgg[i].detach())
            # loss += self.weights[i] * self.criterion(gram_x, gram_y)
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


if __name__ == '__main__':
    transform_list = []
    method = transforms.InterpolationMode.BICUBIC
    transform_list.append(transforms.Resize([286, 286], method))
    # transform_list.append(transforms.RandomCrop(256))
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(0.5, 0.5)]
    tf = transforms.Compose(transform_list)

    # from networks import PerceptualLoss
    model = Vgg19()
    weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    A = Image.open('ROOT/img_for_loss/pic_dif.png').convert('RGB')
    B = Image.open('ROOT/img_for_loss/pic_large.png').convert('RGB')   
    C = Image.open('ROOT/img_for_loss/pic_small.png').convert('RGB')
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    A = torch.tensor(A)
    B = torch.tensor(B)
    C = torch.tensor(C)
  
    # A是pic_dif, B是pic_large, C是pic_small
    criterion = PerceptualLoss()
   
    print(criterion(A, B)) # tensor(0.14847)
    print(criterion(A, A)) # tensor(0.)
    print(criterion(B, C)) # tensor(0.10065)
    

    
 


 

