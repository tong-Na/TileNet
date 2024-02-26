import cv2
import numpy as np
from PIL import Image
import os

def process_images(mask_path, image_path, save_mask_path, save_image_path):

    image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    kernel_size = 6
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_image = cv2.dilate(image, kernel, iterations=1)

    # cv2.imwrite('/Users/irisalpful/PycharmProjects/lama-main/imagenet/img2/dilated_image.jpg', dilated_image)

    # cv2.imshow('Dilated Image', dilated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    dilated_image = cv2.cvtColor(dilated_image, cv2.COLOR_BGR2RGB)

    mask = Image.fromarray(dilated_image)
    image = Image.open(image_path).convert('RGB')

    mask_array = np.array(mask)
    image_array = np.array(image)

    width = mask_array.shape[1]

    quarter_width = width // 4

    start = 410
    end = 820
    for x in range(80, 280, 5):

        for y in range(0, 1024, 10):
            # Get the color of the current pixel and its symmetric counterpart
            left_color = mask_array[y, x]
            right_color = mask_array[y, width - x - 1]

            # Check if one is black (0, 0, 0) and the other is white (255, 255, 255)
            if (left_color[0] <= 100 and right_color[0] >= 200 ):
                # Update the mask's black pixel to white

                mask_array[y, width - x - 1] = [0, 0, 0]

                # Update the image's corresponding pixel
                image_array[y, width - x - 1] = image_array[y, x]

            if (left_color[0] >= 200 and right_color[0] <= 100):
                # Update the mask's black pixel to white

                mask_array[y, x] = [0, 0, 0]

                # Update the image's corresponding pixel
                image_array[y, x] = image_array[y, width - x - 1]

        start = start - 10
        end = end + 10

    # Convert the arrays back to images
    new_mask = Image.fromarray(mask_array)
    new_image = Image.fromarray(image_array)

    # Save the updated images
    print(save_mask_path)
    new_mask.save(save_mask_path)
    new_image.save(save_image_path)


def process_folder(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for file in os.listdir(source_folder):
        if file.endswith(".png") and not file.endswith("_mask.png"):
            image_path = os.path.join(source_folder, file)
            mask_path = os.path.join(source_folder, file.replace(".png", "_mask.png"))

            # 检查掩码文件是否存在
            if os.path.exists(mask_path):
                # 生成保存路径
                save_image_path = os.path.join(target_folder, file)
                save_mask_path = os.path.join(target_folder, file.replace(".png", "_mask.png"))

                # 调用处理函数
                process_images(mask_path, image_path, save_mask_path, save_image_path)

# 使用示例：source_folder里面是配对的图像和掩码
source_folder = '/Users/irisalpful/PycharmProjects/lama-main/imagenet/img'
target_folder = '/Users/irisalpful/PycharmProjects/lama-main/imagenet/img2'
process_folder(source_folder, target_folder)


