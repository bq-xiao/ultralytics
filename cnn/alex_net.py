import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.models import AlexNet
from torchvision.transforms import transforms


# 4、同一行画出图片和标签，可视化
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def show_dataset(images, labels):
    display.set_matplotlib_formats('svg')  # 用矢量图进行展示
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(50, 50))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.permute(1, 2, 0))
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


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


def validation(model, test_loader, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Val Accuracy: {:.2f}%'.format(100 * correct / total))
        return 100 * correct / total


def train(train_epochs, train_loader, test_loader, device):
    torch.cuda.empty_cache()
    # 定义模型、损失函数和优化器
    model = AlexNet(num_classes=101)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters())

    images, labels = next(iter(train_loader))
    writer = SummaryWriter("runs/logs_LeNet")
    images = images.to(device)
    writer.add_graph(model, images)
    # 训练模型
    for epoch in range(train_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            model.train(True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 梯度下降
            optimizer.step()

            if (i + 1) % 50 == 0:
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
        val_accuracy = validation(model, test_loader, device)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    torch.save(model, 'Cust_AlexNet.pt')
    print("Train model done!!")


def show_model(net):
    X = torch.rand(size=(1, 3, 224, 224), dtype=torch.float32)
    print("================Model Size================")
    for layer in net.children():
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)

    print("==========================================")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, data_loader, categories, num_images=6, device='cpu'):
    model.to(device)
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(50, 50))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {categories[preds[j]]} \n actual:{categories[labels[j]]}')
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    return
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    # 加载MNIST数据集
    # train_dataset = datasets.MNIST(root='../datasets/mnist', train=True, transform=transforms.ToTensor(), download=True)
    # test_dataset = datasets.MNIST(root='../datasets/mnist', train=False, transform=transforms.ToTensor())
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = datasets.Caltech101(root='../datasets/caltech101', transform=transform, download=False)
    # print(f"Train data shape: \t{full_dataset[0][0].shape}")
    # print(f"Train label shape: \t{full_dataset[0][1].shape}")
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    # 定义数据加载器

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, pin_memory=True,
                              pin_memory_device='cuda')
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, pin_memory=True,
                             pin_memory_device='cuda')
    # 5、查看部分样本并可视化
    X, y = [], []
    for i in range(10):
        X.append(train_dataset[i][0])
        # print(full_dataset.categories[train_dataset[i][1]])
        y.append(full_dataset.categories[train_dataset[i][1]])
    # show_dataset(X, y)
    # 训练
    # train(20, train_loader, test_loader, "cpu")
    # 可视化
    # model = AlexNet()
    # show_model(model)
    # 可视化结果
    model = torch.load("Cust_AlexNet.pt")
    visualize_model(model, test_loader, full_dataset.categories, device='cuda')
