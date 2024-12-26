import os

import pandas as pd

csv = pd.read_csv('../../datasets/human-faces-object-detection/faces.csv')


def convert_box(df):
    width, height, x0, y0, x1, y1 = float(df['width']), float(df['height']), float(df['x0']), float(df['y0']), float(
        df['x1']), float(df['y1'])

    b_width = x1 - x0
    b_heght = y1 - y0
    x_center = x0 + b_width / 2
    y_center = y0 + b_heght / 2
    return round(x_center / width, 6), round(y_center / height, 6), \
           round(b_width / width, 6), round(b_heght / height, 6)


for index, row in csv.iterrows():
    img_name = row['image_name']
    file_array = img_name.split(".")
    new_file = '../../datasets/human-faces-object-detection/labels/' + file_array[0] + ".txt"
    x_center, y_center, w, h = convert_box(row)
    line = "0 " + str(x_center) + " " + str(y_center) + " " + str(w) + " " + str(h) + "\n"
    if os.path.exists(new_file):
        with open(new_file, "a") as file:
            file.write(line)
            file.close()
    else:
        yolo_file = open(new_file, "w", encoding='utf-8')
        yolo_file.write(line)
        yolo_file.close()
