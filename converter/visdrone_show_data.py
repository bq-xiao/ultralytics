import cv2
from PIL import Image
from matplotlib import pyplot as plt


def convert_box(size, box):
    # Convert VisDrone box to YOLO xywh box
    dw = size[0]
    dh = size[1]
    x, y = box[0] * dw, box[1] * dh
    return x - box[2] * dw / 2, y - box[3] * dh / 2, box[2] * dw, box[3] * dh


img_path = '../../datasets/VisDrone/VisDrone2019-DET-train/images/9999956_00000_d_0000028.jpg'
img_size = Image.open(img_path).size
with open('../../datasets/VisDrone/VisDrone2019-DET-train/labels/9999956_00000_d_0000028.txt', 'r') as f:
    for row in [x.split(' ') for x in f.read().strip().splitlines()]:
        box = convert_box(img_size, tuple(map(float, row[1:])))
        img = cv2.imread(img_path)
        plt.imshow(img)
        ax = plt.gca()
        # 默认框的颜色是黑色，第一个参数是左上角的点坐标
        # 第二个参数是宽，第三个参数是长
        ax.add_patch(
            plt.Rectangle((int(box[0]), int(box[1])), int(box[2]), int(box[3]), color="red", fill=False,
                          linewidth=1))

plt.show()
