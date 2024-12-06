import cv2
from matplotlib import pyplot as plt

with open('WIDER_train/label_view.txt', 'r') as f:
    lines = f.readlines()
    file_name = lines[0].strip()
    face_size = int(lines[1])
    full_path = "./WIDER_train/images/" + file_name
    img = cv2.imread(full_path)
    plt.imshow(img)
    ax = plt.gca()

    for line in lines[2:face_size + 3]:
        list = line.split(" ")
        # 默认框的颜色是黑色，第一个参数是左上角的点坐标
        # 第二个参数是宽，第三个参数是长
        ax.add_patch(
            plt.Rectangle((float(list[0]), float(list[1])), float(list[2]), float(list[3]), color="red", fill=False, linewidth=1))
        print(line)

    plt.show()
