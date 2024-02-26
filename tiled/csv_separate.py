import pandas as pd
import csv
import os

with open("~/try-on/data/zalando-hd-resized/keypoints_dataset.csv", "r") as f:
    csvdata = []
    reader = csv.reader(f)

    for row in reader:

        if row[0][0] == "c":
            csvdata.append(row)


with open("~/try-on/data/zalando-hd-resized/0.csv", "w", newline="") as f:
    writer = csv.writer(f)
    # 注意传入数据的格式为列表元组格式
    writer.writerows(csvdata)

# imgid = os.listdir("~/try-on/data/zalando-hd-resized/train/cloth")
# print(len(imgid))
