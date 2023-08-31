import lightning.pytorch as pl
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms, datasets
from torchvision.models.vgg import _vgg
from torchviz import make_dot


class VGGModule(pl.LightningModule):
    def __init__(self, model, loss, learning_rate=2e-5):
        super().__init__()
        self.model = model
        self.loss = loss
        self.train_accuracy = Accuracy(task="multiclass", num_classes=101)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=101)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=101)
        self.lr = learning_rate

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y_hat, y):
        return self.loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, labels = batch
        x_hat = self(images)
        loss = self.compute_loss(x_hat, labels)
        self.train_accuracy.update(x_hat, labels)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        images, labels = batch
        x_hat = self(images)
        test_loss = self.compute_loss(x_hat, labels)
        # 测试准确率
        self.test_accuracy.update(x_hat, labels)
        self.log_dict(
            {'test_loss': test_loss, 'test_accuracy': self.test_acc}, on_epoch=True, on_step=False, prog_bar=True)
        return test_loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        images, labels = batch
        x_hat = self(images)
        val_loss = self.loss(x_hat, labels)
        # 验证准确率
        self.val_accuracy.update(x_hat, labels)
        self.log_dict(
            {'val_loss': val_loss, 'val_accuracy': self.val_accuracy}, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3,
                                               eps=1e-9, verbose=True),
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1
            },
        }


# CIFAR100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# CIFAR100_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
def get_dataloaders(dataset_path, batch_size=64, num_workers=4):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    # train_transform = transforms.Compose([
    #     # transforms.ToPILImage(),
    #     transforms.Grayscale(3),
    #     transforms.RandomHorizontalFlip(),
    #     # transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     # transforms.Normalize((0.5,), (0.5,))
    # ])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # full_dataset = datasets.Caltech101(root=dataset_path, transform=transform, download=False)
    train_dataset = datasets.CIFAR100(root=dataset_path, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(root=dataset_path, train=False, transform=test_transform)
    # print(f"Train data shape: \t{full_dataset[0][0].shape}")
    # print(f"Train label shape: \t{full_dataset[0][1].shape}")
    # train_size = int(0.8 * len(full_dataset))
    # test_size = len(full_dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    # 定义数据加载器

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, test_loader


# 可视化模型
def show_model(model, X):
    y = model(X)
    dot = make_dot(y.mean(), params=dict(model.named_parameters()))
    dot.format = "png"
    dot.filename = "vgg_model"
    # 指定文件生成的文件夹
    dot.directory = "torchviz"
    # 生成文件
    dot.view()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(100)
    # 获取数据加载
    train_loader, test_loader = get_dataloaders('../datasets/cifar100', batch_size=32, num_workers=1)
    # 定义网络模型
    model = _vgg("A", False, None, True, num_classes=101)
    # 损失函数
    loss = nn.CrossEntropyLoss()
    lr = 1e-3
    # lighting模型
    module = VGGModule(model, loss, learning_rate=lr)
    # 回调函数
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        patience=10,
                                        mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        filename="vgg-net-{epoch:02d}-{val_accuracy:.2f}"
    )
    # 实例化trainer
    trainer = Trainer(max_epochs=100,
                      accelerator="cpu",
                      devices=1,
                      profiler="simple",
                      callbacks=[early_stop_callback, ckpt_callback, lr_monitor],
                      enable_progress_bar=True,
                      enable_model_summary=True)
    # 训练
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=test_loader)
    # images, labels = next(iter(test_loader))
    # show_model(model, images)
    # 保存模型
    # trainer.save_checkpoint('vgg_model.ckpt')
    # result = trainer.test(module, test_loader, ckpt_path='last')
    # print(f"result:\n{result}")
