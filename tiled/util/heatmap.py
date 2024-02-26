"""给定热力图，找出其中的关键点"""
import torch


def find_keypoints(heatmap, threshold=0):
    keypoints = []
    heatmap = heatmap.squeeze(0)
    n, h, w = heatmap.shape

    for i in range(n):
        heatmap_i = heatmap[i]
        flat_heatmap = heatmap_i.flatten()  # 将热图展平为一维张量
        max_index = torch.argmax(flat_heatmap)  # 找到最大值的索引
        max_value = flat_heatmap[max_index]  # 最大值
        if max_value > threshold:
            row = max_index // w  # 行索引
            col = max_index % w  # 列索引
            keypoints.append(
                [round(col.item() / w, 4), round(row.item() / h, 4)]
            )  # 将关键点坐标添加到列表中

    return keypoints
