import os
# 获取图像长 宽
# 'WIDER_train/labelv2.txt'
def get_images_info(meta_file, key_word = '.jpg'):
    # 图像大小
    image_map = dict()
    with open(meta_file, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            if line.lower().find(key_word) > 0:
                line_array = line.split(" ")
                img_width = float(line_array[2])
                img_height = float(line_array[3].strip())
                image_map[line_array[1]] = {'width': img_width, 'height': img_height}

    return image_map

# 坐标
# annotation_file = 'WIDER_train/label_test.txt'
# label_base_dir = 'WIDER_train/labels/'
def generate_yolo_labes(annotation_file, label_base_dir, image_info=None, end_with ='.jpg\n'):
    if image_info is None:
        image_info = dict()

    if not os.path.exists(label_base_dir):
        os.mkdir(label_base_dir)

    with open(annotation_file, 'r') as f:
        all_lines = f.readlines()
        index_array = []
        for i in range(len(all_lines)):
            if all_lines[i].lower().endswith(end_with):
                index_array.append(i)

        for index in index_array:
            file_name = all_lines[index]
            line_num = int(all_lines[index + 1])
            split_lines = all_lines[index:index + line_num + 2]
            img_info = image_info[file_name.strip()]
            file_array = file_name.split(".")
            # 打开文件
            sub_dir = file_array[0].split("/")[0]
            dir = label_base_dir + sub_dir
            if not os.path.exists(dir):
                os.mkdir(dir)

            new_file = label_base_dir + file_array[0] + ".txt"
            yolo_file = open(new_file, "w", encoding='utf-8')
            yolo_lines = []
            for l in split_lines[2:len(split_lines)]:
                line_array = l.split(" ")
                # 标记坐标起始点
                x = float(line_array[0])
                y = float(line_array[1])
                box_width = float(line_array[2])
                box_height = float(line_array[3])
                # 标记中心坐标点
                x_center = x + box_width / 2
                y_center = y + box_height / 2
                #print(f"--{x_center}---{y_center}--")
                scale_x_center = x_center / img_info['width']
                scale_box_width = box_width / img_info['width']
                scale_y_center = y_center / img_info['height']
                scale_box_height = box_height / img_info['height']
                #print(f"-scale-{scale_x_center}---{scale_y_center}--")
                # class x_center y_center width height
                yolo_line = "0 " + str(scale_x_center) + " " + str(scale_y_center) \
                            + " " + str(scale_box_width) + " " + str(scale_box_height) + "\n"
                yolo_lines.append(yolo_line)

            # 写文件
            yolo_file.writelines(yolo_lines)
            yolo_file.close()
            print(f"{new_file}:{yolo_lines}")
            print(f"{new_file} created")

if __name__ == '__main__':
   image_info = get_images_info('WIDER_val/labelv2.txt', '.jpg')
   generate_yolo_labes('WIDER_val/wider_face_val_bbx_gt.txt', 'WIDER_val/labels/', image_info, '.jpg\n')

   print("OK, Generate Yolo Labels success!")