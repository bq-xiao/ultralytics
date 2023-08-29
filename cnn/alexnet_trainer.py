import lightning.pytorch as pl
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms, datasets
from torchvision.models import AlexNet


class AlexNetModule(pl.LightningModule):
    def __init__(self, model, loss, learning_rate=2e-5):
        super().__init__()
        self.model = model
        self.loss = loss
        self.acc = Accuracy(task="multiclass", num_classes=101)
        self.lr = learning_rate

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, labels = batch
        x_hat = self.model(images)
        loss = self.loss(x_hat, labels)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        images, labels = batch
        x_hat = self.model(images)
        test_loss = self.loss(x_hat, labels)
        # 测试准确率
        test_acc = self.acc(x_hat, labels)
        self.log_dict({'test_loss': test_loss, 'test_accuracy': test_acc}, on_epoch=True, on_step=False, prog_bar=True)
        return test_loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        images, labels = batch
        x_hat = self.model(images)
        val_loss = self.loss(x_hat, labels)
        # 验证准确率
        val_acc = self.acc(x_hat, labels)
        self.log_dict({'val_loss': val_loss, 'val_accuracy': val_acc}, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3,
                                               eps=1e-9, verbose=True),
                "interval": "epoch",
                "monitor": "val_accuracy",
                "frequency": 1
            },
        }


def get_dataloaders():
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

    train_loader = DataLoader(dataset=train_dataset, batch_size=4, num_workers=1, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, num_workers=1, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    # 获取数据加载
    train_loader, test_loader = get_dataloaders()
    # 定义网络模型
    model = AlexNet(num_classes=101)
    # 损失函数
    loss = nn.CrossEntropyLoss()
    lr = 1e-3
    # lighting模型
    module = AlexNetModule(model, loss, learning_rate=lr)
    # 回调函数
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        patience=3,
                                        mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min'
    )
    # 实例化trainer
    trainer = Trainer(max_epochs=100,
                      accelerator="gpu",
                      profiler="simple",
                      callbacks=[early_stop_callback, ckpt_callback, lr_monitor],
                      enable_progress_bar=True,
                      enable_model_summary=True)
    # 训练
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=test_loader)
    # 保存模型
    trainer.save_checkpoint('alexnet_model.ckpt')
    # result = trainer.test(module, test_loader, ckpt_path='last')
    # print(f"result:\n{result}")
