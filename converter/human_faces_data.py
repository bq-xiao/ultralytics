import cv2
from matplotlib import pyplot as plt

x0, y0, x1, y1 = 0.585625, 0.472509, 0.59625, 0.694158

width = 800
height = 582

img_path = '../datasets/human-faces-object-detection/data/images/train/00000963.jpg'
img = cv2.imread(img_path)
plt.imshow(img)
ax = plt.gca()
# 默认框的颜色是黑色，第一个参数是左上角的点坐标
# 第二个参数是宽，第三个参数是长
x = float(x0) * width - float(x1) * width / 2
y = float(y0) * height - float(y1) * height / 2
ax.add_patch(
    plt.Rectangle(
        (x, y),
        float(x1) * width,
        float(y1) * height, color="red", fill=False, linewidth=1))

plt.show()
