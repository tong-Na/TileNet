import os
# import re
import sys
sys.path.append('.')
import cv2
# import math
# import time
# import scipy
import argparse
# import matplotlib
import numpy as np
# import pylab as plt
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from collections import OrderedDict
# from scipy.ndimage.morphology import generate_binary_structure
# from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
# from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)



model = get_model('vgg19')     
model.load_state_dict(torch.load(args.weight))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

dataset_path = '../../data/zalando-hd-resized'
phase = 'train'
input_dir = os.path.join(dataset_path, phase, 'image')
out_dir = os.path.join(dataset_path, phase, 'pose_joint_json')
# out_dir = './outputs'
mkdir(out_dir)


img_names = []
for _, _, fnames in os.walk(input_dir):
    for fname in fnames:
        if is_image_file(fname):
            img_names.append(fname)
img_names = sorted(img_names)
sum = len(img_names)
counts = 0
for fname in img_names:
    key_points = {'posX': {}, 'posY': {}}
    img_path = os.path.join(input_dir, fname)
    oriImg = cv2.imread(img_path) # B,G,R order
    # oriImg = cv2.imread('../../data/zalando-hd-resized/test/image/01011_00.jpg')
    shape_dst = np.min(oriImg.shape[0:2])
    h, w, _ = oriImg.shape
    # Get results of original image

    with torch.no_grad():
        paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')

    prefix = fname.split('.')[0]
    humans = paf_to_pose_cpp(heatmap, paf, cfg)
    for k, v in humans[0].body_parts.items():
        # x = int(v.x * w + 0.5)
        # y = int(v.y * h + 0.5)
        key_points['posX'][k] = v.x
        key_points['posY'][k] = v.y

    # dark = np.zeros(oriImg.shape, dtype='uint8')
    # out = draw_humans(dark, humans)
    # out_image = os.path.join(out_dir, prefix + '.jpg')
    # cv2.imwrite(out_image, out)

    out_json = os.path.join(out_dir, prefix + '.json')
    import json
    with open(out_json, 'w') as f:
        f.write(json.dumps(key_points, ensure_ascii=False, indent=4))

    counts += 1
    print(fname, f'{(counts/sum * 100):.2f}%\t{counts}/{sum}')

##### 生成居中姿势 #########
# COORDINATE = {
#     0: [0.5, 0.08], 1: [0.5, 0.2], 2: [0.3, 0.2], 3: [0.26, 0.4], 4: [0.25, 0.6],
#     5: [0.7, 0.2], 6: [0.74, 0.4], 7: [0.75, 0.6], 8: [0.4, 0.55], 9: [0.4, 0.75],
#     10: [0.4, 0.95], 11: [0.6, 0.55], 12: [0.6, 0.75], 13: [0.6, 0.95], 14: [0.46, 0.05],
#     15: [0.54, 0.05], 16: [0.4, 0.07], 17: [0.6, 0.07]
# }
# human = Human([])
# for part_idx in range(cfg.MODEL.NUM_KEYPOINTS):
#     human.body_parts[part_idx] = BodyPart(
#         '%d-%d' % (0, part_idx), part_idx,
#         COORDINATE[part_idx][0],
#         COORDINATE[part_idx][1],
#         1
#     )

# oriImg = cv2.imread('../../data/zalando-hd-resized/test/image/01011_00.jpg')
# human.score = 1
# dark = np.zeros(oriImg.shape, dtype='uint8')
# out = draw_humans(dark, [human])
# out_image = os.path.join(out_dir, 'center_pose' + '.jpg')
# cv2.imwrite(out_image, out)


