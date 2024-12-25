import numpy as np
from PIL import Image, ImageDraw

with open(r'D:\pyworkspace\datasets\human-faces-object-detection\seg-labels\00001722.txt', 'r') as f:
    lines = f.readlines()
    line = lines[0]
    list = line.split(" ")
    arr = list[1:]
    np_arr = np.asarray(arr)
    full_path = r"D:\pyworkspace\datasets\human-faces-object-detection\data\images\train\00001722.jpg"
    image = Image.open(full_path)
    height, width = image.height, image.width
    img_draw = ImageDraw.Draw(image)
    xxyy = np_arr.reshape(-1, 2)
    for xy in xxyy:
        a = float(xy[0]) * width
        b = float(xy[1]) * height
        img_draw.point((a, b), (255, 0, 0))

    image.show()
