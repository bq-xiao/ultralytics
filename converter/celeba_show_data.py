import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取CSV文件
data = np.loadtxt('../datasets/celebfaces/list_landmarks_align_celeba.csv', delimiter=',', dtype=str, encoding='utf-8')
row = data[1144]
print(row)
file_name = row[0]
full_path = "../datasets/celebfaces/images/train/" + file_name
img = cv2.imread(full_path)
plt.imshow(img)
ax = plt.gca()
lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y \
    = float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), \
      float(row[8]), float(row[9]), float(row[10])
x = np.array([lefteye_x, righteye_x, nose_x, leftmouth_x, rightmouth_x])
y = np.array([lefteye_y, righteye_y, nose_y, leftmouth_y, rightmouth_y])

plt.scatter(x, y)
ax.add_patch(
    plt.Rectangle((float(lefteye_x - 7.5), float(lefteye_y - 5)), 15, 10, color="red", fill=False,
                  linewidth=1)
)
ax.add_patch(
    plt.Rectangle((float(leftmouth_x), float(leftmouth_y)), rightmouth_x - leftmouth_x, rightmouth_y - leftmouth_y,
                  color="green", fill=False,
                  linewidth=1)
)
plt.show()
