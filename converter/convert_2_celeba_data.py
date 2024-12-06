import shutil

import pandas as pd

df_partition = pd.read_csv('../datasets/celebfaces/list_eval_partition.csv')
df_partition.set_index('partition')
train_data = df_partition.loc[df_partition['partition'] == 0]
print(f"train {train_data.shape}")
val_data = df_partition.loc[df_partition['partition'] == 1]
print(f"val {val_data.shape}")
test_data = df_partition.loc[df_partition['partition'] == 2]
print(f"test {test_data.shape}")


def move_data(df, data_set='train'):
    for index, row in df.iterrows():
        img_name = row['image_id']
        src = '../datasets/celebfaces/img_align_celeba/img_align_celeba/' + img_name
        dst = '../datasets/celebfaces/images/' + data_set + '/' + img_name
        shutil.move(src, dst)


df_label = pd.read_csv('../datasets/celebfaces/list_landmarks_align_celeba.csv')
df_label.set_index('image_id')


# 0=lefteye;1=righteye;2=nose;3=mouth
def generate_label(img_df, label_df, data_set='train'):
    img_df.reset_index()
    img_df.set_index('image_id')
    w = 178
    h = 218
    for index, row in img_df.iterrows():
        img_name = row['image_id']
        label_row = label_df.loc[label_df['image_id'] == img_name]
        label_row.set_index('image_id')
        file_array = img_name.split(".")
        new_file = '../datasets/celebfaces/labels/' + data_set + '/' + file_array[0] + ".txt"
        yolo_file = open(new_file, "w", encoding='utf-8')
        lines = []
        for l, r in label_row.iterrows():
            lines.append("0 " + str(float(r['lefteye_x'] / w)) + " " + str(float(r['lefteye_y'] / h)) + " 0 0\n")
            lines.append("1 " + str(float(r['righteye_x'] / w)) + " " + str(float(r['righteye_y'] / h)) + " 0 0\n")
            lines.append("2 " + str(float(r['nose_x'] / w)) + " " + str(float(r['nose_y'] / h)) + " 0 0\n")
            if r['rightmouth_x'] >= r['leftmouth_x']:
                w1 = r['rightmouth_x'] - r['leftmouth_x']
            else:
                w1 = r['leftmouth_x'] - r['rightmouth_x']

            if r['rightmouth_y'] >= r['leftmouth_y']:
                h1 = r['rightmouth_y'] - r['leftmouth_y']
            else:
                h1 = r['leftmouth_y'] - r['rightmouth_y']

            lines.append("3 " + str(float(r['leftmouth_x'] / w)) + " " + str(float(r['leftmouth_y'] / h))
                         + " " + str(float(w1 / w)) + " " + str(float(h1 / h)))
        yolo_file.writelines(lines)
        yolo_file.close()


generate_label(train_data, df_label, 'train')
generate_label(val_data, df_label, 'val')
generate_label(test_data, df_label, 'test')
