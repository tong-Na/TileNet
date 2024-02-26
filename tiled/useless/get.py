import torch
import torchvision.transforms as transforms
import util
import numpy as np


def compute_distances(keypoints, heatmap_size, sigma=1):
    num_keypoints, _ = keypoints.size()
    h, w = heatmap_size

    # 生成热图坐标的网格
    X = torch.linspace(0, h-1, h)
    Y = torch.linspace(0, w-1, w)
    xx, yy = torch.meshgrid(X, Y, indexing='ij')
    xx = xx.unsqueeze(0).repeat(num_keypoints, 1, 1) # n, h, w
    yy = yy.unsqueeze(0).repeat(num_keypoints, 1, 1) # n, h, w

    x0 = keypoints[..., 1].unsqueeze(-1).unsqueeze(-1)
    y0 = keypoints[..., 0].unsqueeze(-1).unsqueeze(-1)
    # 计算欧氏距离
    distances = torch.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    weights = torch.exp(-0.5 * (distances / sigma) ** 2)

    heatmap = torch.zeros((num_keypoints, h, w))
    heatmap = torch.where(distances <= 3 * sigma, weights, heatmap)

    return heatmap


def find_keypoints(heatmap, threshold):
    keypoints = []
    n, h, w = heatmap.shape

    for i in range(n):
        heatmap_i = heatmap[i]
        flat_heatmap = heatmap_i.flatten()  # 将热图展平为一维张量
        max_index = torch.argmax(flat_heatmap)  # 找到最大值的索引
        max_value = flat_heatmap[max_index]  # 最大值
        if max_value > threshold:
            row = max_index // w  # 行索引
            col = max_index % w  # 列索引
            keypoints.append([round(col.item() / w, 4), round(row.item() / h, 4)])  # 将关键点坐标添加到列表中

    return keypoints

if __name__ == '__main__':
    N = 112
    keypoints = torch.tensor([[0.2031, 0.8555], [0.6823, 0.9004], [0.2109, 0.6133], [0.7839, 0.6582], [0.1302, 0.5352], [0.9219, 0.5605], [0.1953, 0.4434], [0.8438, 0.4219], [0.3385, 0.3887], [0.7109, 0.373], [0.5286, 0.4785]], dtype=torch.float32)

    # keypoints = keypoints.unsqueeze(0)
    keypoints *= 112
    keypoints = keypoints.long()
    hm = compute_distances(keypoints, (N, N))
    ret = hm[0]
    # for i in range(1, hm.shape[0]):
    #     ret += hm[i]
        # temp = hm[i].unsqueeze(0).unsqueeze(0)
        # temp = util.tensor2im(temp)
        # util.save_image(temp, f'{i}.jpg')
    ret = ret.unsqueeze(0).unsqueeze(0)
    ret = util.tensor2im(ret)
    util.save_image(ret, 'ret.jpg')

    points = find_keypoints(hm, 0)
    print(points)
    # print(hm)