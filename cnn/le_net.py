import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from IPython import display

# 4、同一行画出图片和标签，可视化
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def show_fashion_mnist(images, labels):
    display.set_matplotlib_formats('svg')#用矢量图进行展示
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


class LeNetV5(nn.Module):
    def __init__(self):
        super(LeNetV5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride = 1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.f1 = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 1：池化
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        # 2：池化
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # 3：展平
        x = self.f1(x)
        # 4：线性
        x = self.fc1(x)
        x = F.relu(x)
        # 5：线性
        x = self.fc2(x)
        x = F.relu(x)
        # 6：线性
        x = self.fc3(x)

        return x

def test(model, criterion, images, labels):
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
        return 100 * correct / total, loss.item()

def validation(model, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to("cuda")
            labels = labels.to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Val Accuracy: {:.2f}%'.format(100 * correct / total))
        return 100 * correct / total

def train(train_epochs, train_loader, test_loader):
    # 定义模型、损失函数和优化器
    model = LeNetV5()
    model = model.to("cuda")
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to("cuda")
    optimizer = optim.Adam(model.parameters())

    images, labels = next(iter(train_loader))
    writer = SummaryWriter("runs/logs_LeNet")
    writer.add_graph(model, images.to("cuda"))
    # 训练模型
    for epoch in range(train_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()
            model.train(True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 梯度下降
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, train_epochs, i + 1, len(train_loader), loss.item()))
                # 损失值
                writer.add_scalar('Loss/train', loss.item(), epoch)
                # 测试准确率
                test_accuracy, test_loss = test(model, criterion, images, labels)
                writer.add_scalar('Loss/test', test_loss, epoch)
                writer.add_scalar('Accuracy/test', test_accuracy, epoch)

        # 验证准确率
        model.eval()
        val_accuracy = validation(model, test_loader)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

if __name__ == '__main__':
    # 加载MNIST数据集
    train_dataset = datasets.MNIST(root='../datasets/mnist', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='../datasets/mnist', train=False, transform=transforms.ToTensor())

    # 定义数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)
    # 5、查看部分样本并可视化
    X, y = [], []
    for i in range(10):
        X.append(train_dataset[i][0])
        y.append(train_dataset[i][1])
    #show_fashion_mnist(X, y)
    train(50, train_loader, test_loader)