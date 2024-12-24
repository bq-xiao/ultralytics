import cv2
from matplotlib import pyplot as plt

x0, y0, x1, y1 = 106, 332, 264, 454

img_path = '../../datasets/human-faces-object-detection/images/00002798.jpg'
img = cv2.imread(img_path)
plt.imshow(img)
ax = plt.gca()
# 默认框的颜色是黑色，第一个参数是左上角的点坐标
# 第二个参数是宽，第三个参数是长
ax.add_patch(
    plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="red", fill=False, linewidth=1))

plt.show()
