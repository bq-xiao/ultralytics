## train

### xxx

```python
model = YOLO("yolo11n.pt")
results = model.train(data="coco8.yaml", epochs=3)
```

1. 检查是否为pytorch模型
2. 检查ultralytics是否为最新版本
3. 处理默认参数

```python
self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
```

根据task_map加载trainer，并且初始化trainer

task_map如下所示：

```json
"classify": {
	"model": ClassificationModel,
	"trainer": yolo.classify.ClassificationTrainer,
	"validator": yolo.classify.ClassificationValidator,
	"predictor": yolo.classify.ClassificationPredictor,
},
"detect": {
	"model": DetectionModel,
	"trainer": yolo.detect.DetectionTrainer,
	"validator": yolo.detect.DetectionValidator,
	"predictor": yolo.detect.DetectionPredictor,
},
"segment": {
	"model": SegmentationModel,
	"trainer": yolo.segment.SegmentationTrainer,
	"validator": yolo.segment.SegmentationValidator,
	"predictor": yolo.segment.SegmentationPredictor,
},
"pose": {
	"model": PoseModel,
	"trainer": yolo.pose.PoseTrainer,
	"validator": yolo.pose.PoseValidator,
	"predictor": yolo.pose.PosePredictor,
},
"obb": {
	"model": OBBModel,
	"trainer": yolo.obb.OBBTrainer,
	"validator": yolo.obb.OBBValidator,
	"predictor": yolo.obb.OBBPredictor,
}
```

```python
engine.trainer.BaseTrainer.__init__
```

初始化trainer

1. 初始化训练默认参数
2. 解析yaml文件，获取train/val/test dataset
3. 初始化回调函数

```python
self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
```

1. 初始化DetectionModel
2. 加载权重weights

```python
self.trainer.train()-->engine.trainer.BaseTrainer.train(self)
```

根据具体的任务获取对应的trainer，并且调用train方法

```python
engine.trainer.BaseTrainer._do_train(self, world_size=1)
```

1. 初始化trainer
2. 优化器梯度设置为0
3. torch模型每层设置为training mode

```python
engine.trainer.BaseTrainer._setup_train(self, world_size)
```

1. 设置模型
2. 检查图片大小是否为步幅的倍数，如果不是，则将其更新为大于或等于给定步幅的最接近倍数
3. 计算batch大小
4. 初始化Dataloader
5. 初始化优化器
6. 初始化训练学习率调度器
7. 初始化EarlyStopping



```python
self.loss, self.loss_items = self.model(batch)
```

1. 开始正在训练，前向传播，计算损失值
2. 缩放损失并进行反向传播
3. 更新优化器并调整缩放因子



```python
nn.tasks._predict_once

def _predict_once(self, x, profile=False, visualize=False, embed=None):
	"""
	Perform a forward pass through the network.

	Args:
		x (torch.Tensor): The input tensor to the model.
		profile (bool):  Print the computation time of each layer if True, defaults to False.
		visualize (bool): Save the feature maps of the model if True, defaults to False.
		embed (list, optional): A list of feature vectors/embeddings to return.

	Returns:
		(torch.Tensor): The last output of the model.
	"""
	y, dt, embeddings = [], [], []  # outputs
	for m in self.model:
		if m.f != -1:  # if not from previous layer
			x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
		if profile:
			self._profile_one_layer(m, x, dt)
		x = m(x)  # run
		y.append(x if m.i in self.save else None)  # save output
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
		if embed and m.i in embed:
			embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
			if m.i == max(embed):
				return torch.unbind(torch.cat(embeddings, 1), dim=0)
	return x
```



循环遍历网络执行前向传播

























