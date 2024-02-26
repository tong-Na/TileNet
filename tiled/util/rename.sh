#!/bin/bash

# 进入目标目录
cd ~/try-on/tiled/results/content_0shaped/train_latest

# 循环处理文件名
for file in *_00_out_mask.png; do
    # 提取文件名中的前缀部分
    prefix="${file%_00_out_mask.png}"
    # 重命名文件
    mv "$file" "${prefix}_00.png"
done