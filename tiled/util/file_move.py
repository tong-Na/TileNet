import os
import shutil

loss_file_name = []
path_main = r"~/try-on/tiled/results/pix2pix_unet8_wmask_gradA_L1-14_-ssim_rmsless/test_latest/images"  # 待处理文件夹路径
filelist_main = os.listdir(path_main)  # 将“待处理文件夹“下的文件名以列表的形式列出来
path_receive_fake = r"~/try-on/tiled/results/pix2pix_unet8_wmask_gradA_L1-14_-ssim_rmsless/test_latest/fake"  # 移动到这个位置
path_receive_real = r"~/try-on/tiled/results/pix2pix_unet8_wmask_gradA_L1-14_-ssim_rmsless/test_latest/real"
LENGHT = len(
    filelist_main
)  # 定义一个变量存放filelist_main的长度,用len()方法计算列表长度返回的是列表中元素的个数
name = []
for files in filelist_main:

    filename0 = os.path.splitext(files)[0]  # 读取文件名
    name = filename0
    name0 = name[0:8]  # 按照文件名前14位提取新建文件夹的文件名
    name1 = name[9:15]

    if name1 == "real_B":

        full_path = os.path.join(path_main, files)

        shutil.move(full_path, path_receive_real)

    elif name1 == "fake_B":
        full_path = os.path.join(path_main, files)

        shutil.move(full_path, path_receive_fake)

    else:
        continue

    name = []
print("完成！")
