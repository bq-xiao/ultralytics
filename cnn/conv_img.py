from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from torch import nn
from torchvision.utils import save_image

# 定义ToTensor的transform
transform = transforms.Compose([transforms.ToTensor()])

# 读入指定路径下的图片
image = Image.open('../test_image.jpeg')
# 这里对图片应用transform，就转换为了张量
x = transform(image)
#x = x.reshape((1, 1, 500, 668))

conv2d = nn.Conv2d(3, 3, kernel_size=(3, 3), bias=False)
conv2d_1 = nn.Conv2d(3, 1, kernel_size=(1, 1), bias=False)

max_pool = nn.MaxPool2d(2, stride=2)

#x = conv2d(x)
#x = conv2d_1(x)
#conv2d.zero_grad()

x = max_pool(x)

save_image(x, './y.jpg')