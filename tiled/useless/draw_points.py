"""用于在图像上绘制关键点，论文中使用"""

from PIL import Image, ImageDraw
import numpy as np

# 读入图像
# image_path = r"~/try-on/data/zalando-hd-resized/test/image/00126_00.jpg"
# image = Image.open(image_path)

# 给定11个点的坐标

# 创建绘图对象
# draw = ImageDraw.Draw(image)

# for point in points:
#     x, y = point * np.array(image.size)  # 将归一化的坐标映射到图像大小
#     radius = 12  # 圆的半径
#     color_fill = (255, 0, 0)  # 红色实心填充
#     color_outline = (0, 0, 0)  # 黑色边框

#     draw.ellipse(
#         [(x - radius, y - radius), (x + radius, y + radius)],
#         fill=color_fill,
#         outline=color_outline,
#         width=3,
#     )

# # 保存结果图像
# output_path = "./output_image.png"  # 保存路径
# image.save(output_path)

s_points = np.array(
    [
        [0.2143, 0.7143],
        [0.6429, 0.75],
        [0.2679, 0.5],
        [0.6607, 0.5],
        [0.1964, 0.4821],
        [0.8214, 0.4643],
        [0.2857, 0.3214],
        [0.7321, 0.2857],
        [0.4107, 0.2679],
        [0.5893, 0.25],
        [0.4821, 0.2857],
    ]
)

t_points = np.array(
    [
        [0.2857, 0.75],
        [0.7143, 0.75],
        [0.25, 0.5357],
        [0.7143, 0.5179],
        [0.0893, 0.4821],
        [0.8929, 0.4643],
        [0.2321, 0.2857],
        [0.7679, 0.2857],
        [0.3929, 0.2321],
        [0.5893, 0.2321],
        [0.5, 0.25],
    ]
)

# image_size = (768, 1024)
# image = Image.new("RGB", image_size, color=(255, 255, 255))
image_path = r"~/try-on/data/zalando-hd-resized/test/test_tps/00126_00.jpg"
image = Image.open(image_path)
image_size = image.size
draw = ImageDraw.Draw(image)

for i in range(s_points.shape[0]):
    radius = 12
    # color_fill = (255, 0, 0)
    color_outline = (0, 0, 0)
    # x, y = s_points[i] * np.array(image_size)
    # draw.ellipse(
    #     [(x - radius, y - radius), (x + radius, y + radius)],
    #     fill=color_fill,
    #     outline=color_outline,
    #     width=3,
    # )
    color_fill = (0, 0, 255)
    x, y = t_points[i] * np.array(image_size)
    draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color_fill,
        outline=color_outline,
        width=3,
    )

output_path = "./output_image_target.png"  # 保存路径
image.save(output_path)
