import shutil

import pandas as pd

csv = pd.read_csv('../../datasets/human-faces-object-detection/faces.csv', usecols=['image_name'])
image = csv.drop_duplicates(subset=['image_name'])
train = image.iloc[0:2000]

for index, row in train.iterrows():
    img_name = row['image_name']
    # 源文件路径
    source_img = '../../datasets/human-faces-object-detection/images/' + img_name
    # 目标文件路径
    destination_img = '../../datasets/human-faces-object-detection/data/images/train/' + img_name
    # 移动文件
    shutil.move(source_img, destination_img)

    file_array = img_name.split(".")
    # 源文件路径
    source_img = '../../datasets/human-faces-object-detection/seg-labels/' + file_array[0] + ".txt"
    # 目标文件路径
    destination_img = '../../datasets/human-faces-object-detection/data/labels/train/' + file_array[0] + ".txt"
    # 移动文件
    shutil.move(source_img, destination_img)
