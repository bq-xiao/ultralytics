## 代码入口

yolo v8的基本训练代码如下所示：

```python
from ultralytics import YOLO

# Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='coco128.yaml', epochs=3)
```

通过上面关键的三个步骤，我们就可以开始训练一个YOLO模型了:

+ 引入基本类
+ 初始化模型
+ 调用train方法进行训练
+ 调用val方法进行验证

## 初始化

### 预训练模型加载方式

```python
model = YOLO('yolov8n.pt')
```

通过YOLO类加载一个预训练模型，YOLO类继承ultralytics.engine.model.Model类，进入`__init__`方法后会做一些基本变量的初始化和回调函数的初始化。

![ultralytics.models.yolo.model](D:\pyworkspace\ultralytics\imgs\ultralytics.models.yolo.model.png)

```python
# Check if Ultralytics HUB model from https://hub.ultralytics.com
if self.is_hub_model(model):
    from ultralytics.hub.session import HUBTrainingSession

    self.session = HUBTrainingSession(model)
    model = self.session.model_file
```

判断是否为hub模型，如果是将从hub上下载模型到本地。

```python
if suffix in ('.yaml', '.yml'):
    self._new(model, task)  # 初始化新模型
else:
    self._load(model, task)  # 加载新模型
```

通过加载模型的后缀名来判断是初始化一个新模型还是加载一个预训练的模型。

> self._load(model, task) --> attempt_load_one_weight(weights) --> torch_safe_load(weight) --> torch.load(file, map_location='cpu')

整个加载预训练模型的调用链如上所示，最终通过torch的load方法来加载模型权重数据。

![image-20230808180430478](D:\pyworkspace\ultralytics\imgs\image-20230808180430478.png)

整个预训练模型加载完成后，会初始化以上变量和参数信息。

### Yaml加载方式

```python
model = YOLO('yolov8n.yaml')
```

1.yaml文件加载的方式与上面介绍的加载预训练模型的流程基本保持一致，也初始化了一些关键数据和回调函数等，只是在下面的条件判断上做了区分：

```python
if suffix in ('.yaml', '.yml'):
    self._new(model, task)  # yaml加载方式
else:
    self._load(model, task)  # 预训练模型加载方式
```

> self._new(model, task) --> cfg_dict = yaml_model_load(cfg)

2.`yaml_model_load`函数会加载`ultralytics/cfg/models/v8/yolov8.yaml`模型参数数据。

```python
model = model or self.smart_load('model')
self.model = model(cfg_dict, verbose=verbose and RANK == -1)  # build model
```

3.model会根据不同的任务（task）返回不同的的模型，task与模型的映射关系如下所示：

```json
'classify': {
'model': ClassificationModel,
'trainer': yolo.classify.ClassificationTrainer,
'validator': yolo.classify.ClassificationValidator,
'predictor': yolo.classify.ClassificationPredictor,
},
'detect': {
'model': DetectionModel,
'trainer': yolo.detect.DetectionTrainer,
'validator': yolo.detect.DetectionValidator,
'predictor': yolo.detect.DetectionPredictor,
},
'segment': {
'model': SegmentationModel,
'trainer': yolo.segment.SegmentationTrainer,
'validator': yolo.segment.SegmentationValidator,
'predictor': yolo.segment.SegmentationPredictor,
},
'pose': {
'model': PoseModel,
'trainer': yolo.pose.PoseTrainer,
'validator': yolo.pose.PoseValidator,
'predictor': yolo.pose.PosePredictor,
}
```

因为我们使用的yolov8n.yaml是detect的任务，所以model为DetectionModel。

4.初始化DetectionModel模型

`self.model = model(cfg_dict, verbose=verbose and RANK == -1) --> DetectionModel.__init__`

该方法会根据yaml的模型配置参数初始化一个新模型。

![image-20230808185454621](D:\pyworkspace\ultralytics\imgs\image-20230808185454621.png)

parse_model是将yaml配置文件中的参数转化为torch模型，该函数会初始化模型layout。

同时该init方法会初始化一些模型参数，为后面的train，val，predict做准备。

## train方法

根据不同的任务类型，调用不同的trainer，例如：detect类型的任务，则调用DetectionTrainer类中的train函数。 DetectionTrainer继承自BaseTrainer，因此会调用BaseTrainer类的train方法。

> self.trainer.train() -> yolo.detect.DetectionTrainer.train()->ultralytics.engine.trainer.BaseTrainer.train()

```python
def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
```

分类任务的train函数如下所示，yolo.classify.ClassificationTrainer

```python
def train(cfg=DEFAULT_CFG, use_python=False):
    """Train the YOLO classification model."""
    model = cfg.model or 'yolov8n-cls.pt'  # or "resnet18"
    data = cfg.data or 'mnist160'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = ClassificationTrainer(overrides=args)
        trainer.train()
```

经过上面的yaml初始化后，分别实例化了Trainer，Validator和Predictor，因此可以直接根据不同的task，调用对应类中的方法。

## val方法

val的设计与train的设计一样，都是调用初始化后对应类的函数来完成操作。例如detect类型的任务则调用yolo.detect.DetectionValidator类的val函数

```python
def val(cfg=DEFAULT_CFG, use_python=False):
    """Validate trained YOLO model on validation dataset."""
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'

    args = dict(model=model, data=data)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).val(**args)
    else:
        validator = DetectionValidator(args=args)
        validator(model=args['model'])
```

classify类型的任务则调用yolo.classify.ClassificationValidator里的val函数

```python
def val(cfg=DEFAULT_CFG, use_python=False):
    """Validate YOLO model using custom data."""
    model = cfg.model or 'yolov8n-cls.pt'  # or "resnet18"
    data = cfg.data or 'mnist160'

    args = dict(model=model, data=data)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).val(**args)
    else:
        validator = ClassificationValidator(args=args)
        validator(model=args['model'])
```

