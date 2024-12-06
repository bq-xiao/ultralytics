import cv2
from matplotlib import pyplot as plt

with open('../datasets/data/WIDER_train/labels/54--Rescue/54_Rescue_rescuepeople_54_29.txt', 'r') as f:
    lines = f.readlines()
    full_path = "../datasets/data/WIDER_train/images/54--Rescue/54_Rescue_rescuepeople_54_29.jpg"
    img = cv2.imread(full_path)
    height, width, channels = img.shape
    plt.imshow(img)
    ax = plt.gca()

    for line in lines:
        list = line.split(" ")
        # 默认框的颜色是黑色，第一个参数是左上角的点坐标
        # 第二个参数是宽，第三个参数是长
        x = float(list[1]) * width - float(list[3]) * width / 2
        y = float(list[2]) * height - float(list[4]) * height / 2
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                float(list[3]) * width,
                float(list[4]) * height, color="red", fill=False, linewidth=1))
        print(line)

    plt.show()
