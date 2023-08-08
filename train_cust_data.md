## 准备环境

在开始之前，我们需要准备基本的环境python，cuda，模型底座等。当然我们也可以在cpu的环境下进行训练，这个时候整个训练速度会比较慢。

#### Python版本：

```shell
python -V

Python 3.8.10
```

#### GPU显卡驱动版本如下：

```shell
nvcc -V

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Mar_21_19:24:09_Pacific_Daylight_Time_2021
Cuda compilation tools, release 11.3, V11.3.58
Build cuda_11.3.r11.3/compiler.29745058_0
```

我的电脑上的CUDA版本为11.3

安装对应cuda版本的pytorch，参考[pytorch官网](https://pytorch.org/get-started/locally/)安装教程

![image-20230807100559103](D:\pyworkspace\ultralytics\imgs\image-20230807100559103.png)

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
```

```shell
pip list
Package                 Version
----------------------- ------------
..........
scipy                   1.10.1
seaborn                 0.12.2
setuptools              68.0.0
six                     1.16.0
tensorboard             2.13.0
tensorboard-data-server 0.7.1
torch                   1.12.1+cu113
torchvision             0.13.1+cu113
tqdm                    4.65.0
typing_extensions       4.4.0
........
```

我的电脑上pytorch版本是1.12.1+cu113。



## 准备数据集

#### 下载数据集

我们从网络上下载一个公开的带人脸标注的数据集，[WIDER FACE: A Face Detection Benchmark](http://shuoyang1213.me/WIDERFACE/)

数据集图像文件和人脸标注文件分别可以通过如下链接下载：

![image-20230807101530386](D:\pyworkspace\ultralytics\imgs\image-20230807101530386.png)

可以分开下载训练数据集，验证数据集和测试数据集。



#### 数据集预处理

下载的数据集标注格式如下：

![image-20230807103927932](D:\pyworkspace\ultralytics\imgs\image-20230807103927932.png)

标注框的点坐标格式如下：

x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose

其中，人脸标注的坐标如下图所示：


![image-20230807103340048](D:\pyworkspace\ultralytics\imgs\image-20230807103340048.png)

XY坐标轴从左上角开始，W为图像的宽度（width），H为图像的高度（height）；人脸标注框坐标也是从左上角开始，w1是标注框的宽度，h1是标注框的高度。

然而，以上原始的标注框坐标不能满足YOLO训练需要的标注点坐标，YOLO需要的标注宽格式如下：

![yolo_format](D:\pyworkspace\ultralytics\imgs\yolo_format.jpg)

标注框的x, y为标注框的中心坐标点，不是左上角的起始坐标点。因此，我们需要将WIDERFACE标注框的左上角x，y坐标转换为YOLO的中心坐标点。

> - Box coordinates must be in **normalized xywh** format (from 0 - 1). If your boxes are in pixels, divide `x_center` and `width` by image width, and `y_center` and `height` by image height.

YOLO坐标点和标记的宽高需要进行归一化处理。



#### 标注坐标转换

我们通过程序代码的方式进行转换

```python
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
```

 通过WIDERFACE自带的标注获取图像的宽度和高度。

```python
# 坐标转换
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
```

将标注框的（x1，y1）转换为中心坐标（x_center, y_center）；并且对标记的中心坐标（x_center, y_center）和标记的宽高进行归一化处理，计算中心坐标的简单公式如下：

+ x_center = x1 + w1 / 2；x1为标注框的左上坐标，w1为标注框的宽度；

+ y_center = y1 +  h1 / 2；y1为标注框的左上坐标，h1为标注框的高度；



归一化标注数据的公式如下：

+ scale_x_center = x_center / W，scale_box_width = w1  / W；w1，W分别为标注框和图像的宽度；
+ scale_y_center = y_center / H，scale_box_height = h1 / H；h1，H分别为标注框和图像的高度；



我们可以通过如下代码片段对处理后的数据集进行验证。

```python
import cv2
from matplotlib import pyplot as plt

with open('data/WIDER_train/labels/54--Rescue/54_Rescue_rescuepeople_54_29.txt', 'r') as f:
    lines = f.readlines()
    full_path = "data/WIDER_train/images/54--Rescue/54_Rescue_rescuepeople_54_29.jpg"
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

```

![image-20230807110102108](D:\pyworkspace\ultralytics\imgs\image-20230807110102108.png)

验证效果如上所示，可以看到转换后的标注，完全能够恢复到转换前的效果。

## 训练

下载模型基本框架源代码：

```shell
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics

# 安装基本的依赖
pip install -r requirements.txt
```



经过以上数据集准备，我们将数据集的结构组织成如下结构：

```shell
├───data
│   ├───WIDER_train
│   │   ├───images
│   │   │   ├───0--Parade
│   │   │   ├───1--Handshaking
│   │   │   ├───10--People_Marching
│   │   │   ├───11--Meeting
............................
│   │   └───labels
│   │       ├───0--Parade
│   │       ├───1--Handshaking
│   │       ├───10--People_Marching
│   │       ├───11--Meeting
..............................
│   └───WIDER_val
│   |   ├───images
│   |   │   ├───0--Parade
│   |   │   ├───1--Handshaking
│   |   │   ├───10--People_Marching
│   |   │   ├───11--Meeting
│   │   └───labels
│   │       ├───0--Parade
│   │       ├───1--Handshaking
│   │       ├───10--People_Marching
│   │       ├───11--Meeting
...........................
...........................
```

最终我们整个训练任务的目录结构如下图所示：

![image-20230808102159256](D:\pyworkspace\ultralytics\imgs\image-20230808102159256.png)



#### 准备训练配置文件

训练前需要准备一个yaml文件，定义数据集和训练分类，格式如下：

```yaml
path: ./data  # dataset root dir
train: WIDER_train/images  # train images (relative to 'path') 128 images
val: WIDER_val/images  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes (80 COCO classes)
names:
  0: face
```

因为我们是单分类训练，所以我们只有一个分类（face）人脸。该文件的格式和结构基本与coco数据集保持一致。



#### 准备训练脚本

我们可以通过python脚本进行训练，也可以通过yolo命令行进行训练。为了更直观的理解整个训练过程，我们采用脚本的方式进行训练，并且修改了部分训练参数。

```python
from ultralytics import YOLO
from ultralytics import settings

def train():
    # Update a setting
    settings.update({'clearml': False,
                     'comet':False,
                     'mlflow':False,
                     'neptune':False,
                     'raytune':False,
                     'wandb': False})

    # Update multiple settings
    settings.update({'tensorboard': True})
    settings.update({'datasets_dir': r'D:\pyworkspace\ultralytics\datasets'})

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('./yolov8n.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(
        data='./datasets/cust_data.yaml',
        project = 'face',
        name = 'detect',
        batch = 24,
        save_period = 3,
        cache = False,
        device = 0,
        single_cls = True,
        resume = True,

    )
    print(f"train result:{results}")

if __name__ == '__main__':
    train()
```

通过以上训练脚本我们关闭了一些不需要的组件，并且修改了部分参数。我们采用yolov8n.pt一个比较小的模型底座开始在我们自定义数据集上开始训练。

经过大概40到50个epoch训练，我们训练出一个初始版本的模型best.pt和last.pt。

我们可以分别使用这两个模型做一些推理验证。



## 验证

我们可以使用训练好的模型做一些简单的验证。

```shell
yolo predict model=best.pt source=ultralytics\assets
Ultralytics YOLOv8.0.147  Python-3.8.10 torch-1.12.1+cu113 CUDA:0 (NVIDIA GeForce MX250, 2048MiB)
Model summary (fused): 168 layers, 3005843 parameters, 0 gradients

image 1/2 D:\pyworkspace\ultralytics\ultralytics\assets\bus.jpg: 640x480 3 faces, 20.8ms
image 2/2 D:\pyworkspace\ultralytics\ultralytics\assets\zidane.jpg: 384x640 2 faces, 28.0ms
Speed: 2.6ms preprocess, 24.4ms inference, 4.4ms postprocess per image at shape (1, 3, 384, 640)
Results saved to D:\pyworkspace\ultralytics\runs\detect\predict3
```

![bus](D:\pyworkspace\ultralytics\imgs\bus.jpg)

![zidane](D:\pyworkspace\ultralytics\imgs\zidane.jpg)

使用yolov8自带的图像，我们可以使用训练后的模型进行验证，验证结果如上所示。



## 总结

#### 数据集收集

大多数情况下，我们需要根据自己的数据集训练出我们需要的模型，这个时候我们的数据一般来自我们的业务数据，也可能是我们根据实际的业务场景，收集到的公开数据，例如：通过网络爬虫定向爬取的数据。也可能我们通过商业途径购买的符合我们业务场景的数据。



#### 数据集理解

当我们获取到需要的数据后，我们需要对数据进行理解和分析。分析这些数据是否能够满足我们训练模型的需要，以及我们需要清楚的知道我们是什么类型的训练任务。从大的方向来说，机器学习分为有监督和无监督两大类型。从具体的领域来说，机器学习分为：图像，视频，音频，自然语言（NLP）等。

因此，针对不同的训练任务，我们需要不同结构的数据集。比如：我们是目标检查任务，那么我们的数据集中就必须要存在标注信息，标注出检测目标的坐标，宽度和高度等信息。



#### 数据集预处理

针对目标检测训练任务，大多数情况下我们可能只有原始的数据，没有标注信息，这个时候我们需要借助其他工具和软件进行自定义标注。并且将标注的数据转换成模型能够理解的结构，才能进行训练。这个时候我们要充分阅读模型的帮助文档，按照文档中的要求将数据和标注信息进行转换。比如：模型要求输入的数据和标注信息要做归一化，我们就要按照要求和计算公式对数据做归一化处理；需要坐标点做转换的我们要进行坐标点转换等。

如果我们是从网络上下载的公开数据集，一般情况下会包含原始数据和一些标签数据（标注框，标注点等）。这个时候我们可以用下载的数据集直接进行训练，也可能需要稍作预处理。



#### 训练

当我们准备好数据集后，我们就可以开始进行训练了。一般情况下，我们可以从0到1开始训练一个全新的模型，也可以使用一个模型底座和参数，只训练我们自己的数据集这种训练也被称为迁移学习。在迁移学习的基础上我们也可以对模型进行微调，以达到我们需要的训练效果。

一般情况下，我们可以通过模型自带的命令行工具进行训练，也可以通过python脚本进行训练两者效果是一样的。

在训练过程中，我们可以使用GPU和CPU两类设备来进行训练。GPU训练速度比较快，CPU训练速度比较慢。

在GPU上训练的时候，当我们的显存不够的时候，比较容易发生OOM。这个时候我们一般需要做两种调整：

+ 增加GPU显存
+ 调小batch size大小

增加GPU显存的难度比较大，一般情况下，我们是直接调小batch size来降低显存的占用率。

在训练的过程中，我们还需要收集训练过程的日志，损失值，梯度变化情况等数据，借助一些工具（例如：tensorboard等）来可视化训练的曲线变化，以便了解我们的模型是否过拟合还是欠拟合，也可以观察出模型损失值的变化过程等信息。



