## 初始化

### 初始化预训练模型

```python
model = YOLO('yolo11n.pt')
models.yolo.model.YOLO.__init__
```

初始化YoLo对象，会调用父类的初始化方法，```super().__init__(model=model, task=task, verbose=verbose)```

```python
engine.model.Model.__init__
```

1. 初始化默认callbacks
2. 判断是否为Hub模型
3. 判断是否为Triton Server模型
4. 调用self._load加载模型

默认callbacks

```python
default_callbacks = {
    # Run in trainer
    "on_pretrain_routine_start": [on_pretrain_routine_start],
    "on_pretrain_routine_end": [on_pretrain_routine_end],
    "on_train_start": [on_train_start],
    "on_train_epoch_start": [on_train_epoch_start],
    "on_train_batch_start": [on_train_batch_start],
    "optimizer_step": [optimizer_step],
    "on_before_zero_grad": [on_before_zero_grad],
    "on_train_batch_end": [on_train_batch_end],
    "on_train_epoch_end": [on_train_epoch_end],
    "on_fit_epoch_end": [on_fit_epoch_end],  # fit = train + val
    "on_model_save": [on_model_save],
    "on_train_end": [on_train_end],
    "on_params_update": [on_params_update],
    "teardown": [teardown],
    # Run in validator
    "on_val_start": [on_val_start],
    "on_val_batch_start": [on_val_batch_start],
    "on_val_batch_end": [on_val_batch_end],
    "on_val_end": [on_val_end],
    # Run in predictor
    "on_predict_start": [on_predict_start],
    "on_predict_batch_start": [on_predict_batch_start],
    "on_predict_postprocess_end": [on_predict_postprocess_end],
    "on_predict_batch_end": [on_predict_batch_end],
    "on_predict_end": [on_predict_end],
    # Run in exporter
    "on_export_start": [on_export_start],
    "on_export_end": [on_export_end],
}
```



```python
engine.model.Model._load
```

1. 加载模型权重文件，调用attempt_load_one_weight
2. 从加载后的模型中获取任务类型
3. 重置模型参数，{"imgsz", "data", "task", "single_cls"}

```python
nn.tasks.attempt_load_one_weight
```

1. 加载模型checkpoint，调用torch_safe_load
2. 加载模型默认参数
3. 从加载后的模型中，获取任务类型guess_model_task
4. 更新模型（层），兼容低版本torch

默认默认参数如下：

```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Default training settings and hyperparameters for medium-augmentation COCO training

task: detect # (str) YOLO task, i.e. detect, segment, classify, pose, obb
mode: train # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark

# Train settings -------------------------------------------------------------------------------------------------------
model: # (str, optional) path to model file, i.e. yolov8n.pt, yolov8n.yaml
data: # (str, optional) path to data file, i.e. coco8.yaml
epochs: 100 # (int) number of epochs to train for
time: # (float, optional) number of hours to train for, overrides epochs if supplied
patience: 100 # (int) epochs to wait for no observable improvement for early stopping of training
batch: 16 # (int) number of images per batch (-1 for AutoBatch)
imgsz: 640 # (int | list) input images size as int for train and val modes, or list[h,w] for predict and export modes
save: True # (bool) save train checkpoints and predict results
save_period: -1 # (int) Save checkpoint every x epochs (disabled if < 1)
cache: False # (bool) True/ram, disk or False. Use cache for data loading
device: # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
workers: 8 # (int) number of worker threads for data loading (per RANK if DDP)
project: # (str, optional) project name
name: # (str, optional) experiment name, results saved to 'project/name' directory
exist_ok: False # (bool) whether to overwrite existing experiment
pretrained: True # (bool | str) whether to use a pretrained model (bool) or a model to load weights from (str)
optimizer: auto # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
verbose: True # (bool) whether to print verbose output
seed: 0 # (int) random seed for reproducibility
deterministic: True # (bool) whether to enable deterministic mode
single_cls: False # (bool) train multi-class data as single-class
rect: False # (bool) rectangular training if mode='train' or rectangular validation if mode='val'
cos_lr: False # (bool) use cosine learning rate scheduler
close_mosaic: 10 # (int) disable mosaic augmentation for final epochs (0 to disable)
resume: False # (bool) resume training from last checkpoint
amp: True # (bool) Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check
fraction: 1.0 # (float) dataset fraction to train on (default is 1.0, all images in train set)
profile: False # (bool) profile ONNX and TensorRT speeds during training for loggers
freeze: None # (int | list, optional) freeze first n layers, or freeze list of layer indices during training
multi_scale: False # (bool) Whether to use multiscale during training
# Segmentation
overlap_mask: True # (bool) merge object masks into a single image mask during training (segment train only)
mask_ratio: 4 # (int) mask downsample ratio (segment train only)
# Classification
dropout: 0.0 # (float) use dropout regularization (classify train only)

# Val/Test settings ----------------------------------------------------------------------------------------------------
val: True # (bool) validate/test during training
split: val # (str) dataset split to use for validation, i.e. 'val', 'test' or 'train'
save_json: False # (bool) save results to JSON file
save_hybrid: False # (bool) save hybrid version of labels (labels + additional predictions)
conf: # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
iou: 0.7 # (float) intersection over union (IoU) threshold for NMS
max_det: 300 # (int) maximum number of detections per image
half: False # (bool) use half precision (FP16)
dnn: False # (bool) use OpenCV DNN for ONNX inference
plots: True # (bool) save plots and images during train/val

# Predict settings -----------------------------------------------------------------------------------------------------
source: # (str, optional) source directory for images or videos
vid_stride: 1 # (int) video frame-rate stride
stream_buffer: False # (bool) buffer all streaming frames (True) or return the most recent frame (False)
visualize: False # (bool) visualize model features
augment: False # (bool) apply image augmentation to prediction sources
agnostic_nms: False # (bool) class-agnostic NMS
classes: # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
retina_masks: False # (bool) use high-resolution segmentation masks
embed: # (list[int], optional) return feature vectors/embeddings from given layers

# Visualize settings ---------------------------------------------------------------------------------------------------
show: False # (bool) show predicted images and videos if environment allows
save_frames: False # (bool) save predicted individual video frames
save_txt: False # (bool) save results as .txt file
save_conf: False # (bool) save results with confidence scores
save_crop: False # (bool) save cropped images with results
show_labels: True # (bool) show prediction labels, i.e. 'person'
show_conf: True # (bool) show prediction confidence, i.e. '0.99'
show_boxes: True # (bool) show prediction boxes
line_width: # (int, optional) line width of the bounding boxes. Scaled to image size if None.

# Export settings ------------------------------------------------------------------------------------------------------
format: torchscript # (str) format to export to, choices at https://docs.ultralytics.com/modes/export/#export-formats
keras: False # (bool) use Kera=s
optimize: False # (bool) TorchScript: optimize for mobile
int8: False # (bool) CoreML/TF INT8 quantization
dynamic: False # (bool) ONNX/TF/TensorRT: dynamic axes
simplify: True # (bool) ONNX: simplify model using `onnxslim`
opset: # (int, optional) ONNX: opset version
workspace: None # (float, optional) TensorRT: workspace size (GiB), `None` will let TensorRT auto-allocate memory
nms: False # (bool) CoreML: add NMS

# Hyperparameters ------------------------------------------------------------------------------------------------------
lr0: 0.01 # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
lrf: 0.01 # (float) final learning rate (lr0 * lrf)
momentum: 0.937 # (float) SGD momentum/Adam beta1
weight_decay: 0.0005 # (float) optimizer weight decay 5e-4
warmup_epochs: 3.0 # (float) warmup epochs (fractions ok)
warmup_momentum: 0.8 # (float) warmup initial momentum
warmup_bias_lr: 0.1 # (float) warmup initial bias lr
box: 7.5 # (float) box loss gain
cls: 0.5 # (float) cls loss gain (scale with pixels)
dfl: 1.5 # (float) dfl loss gain
pose: 12.0 # (float) pose loss gain
kobj: 1.0 # (float) keypoint obj loss gain
nbs: 64 # (int) nominal batch size
hsv_h: 0.015 # (float) image HSV-Hue augmentation (fraction)
hsv_s: 0.7 # (float) image HSV-Saturation augmentation (fraction)
hsv_v: 0.4 # (float) image HSV-Value augmentation (fraction)
degrees: 0.0 # (float) image rotation (+/- deg)
translate: 0.1 # (float) image translation (+/- fraction)
scale: 0.5 # (float) image scale (+/- gain)
shear: 0.0 # (float) image shear (+/- deg)
perspective: 0.0 # (float) image perspective (+/- fraction), range 0-0.001
flipud: 0.0 # (float) image flip up-down (probability)
fliplr: 0.5 # (float) image flip left-right (probability)
bgr: 0.0 # (float) image channel BGR (probability)
mosaic: 1.0 # (float) image mosaic (probability)
mixup: 0.0 # (float) image mixup (probability)
copy_paste: 0.0 # (float) segment copy-paste (probability)
copy_paste_mode: "flip" # (str) the method to do copy_paste augmentation (flip, mixup)
auto_augment: randaugment # (str) auto augmentation policy for classification (randaugment, autoaugment, augmix)
erasing: 0.4 # (float) probability of random erasing during classification training (0-0.9), 0 means no erasing, must be less than 1.0.
crop_fraction: 1.0 # (float) image crop fraction for classification (0.1-1), 1.0 means no crop, must be greater than 0.

# Custom config.yaml ---------------------------------------------------------------------------------------------------
cfg: # (str, optional) for overriding defaults.yaml

# Tracker settings ------------------------------------------------------------------------------------------------------
tracker: botsort.yaml # (str) tracker type, choices=[botsort.yaml, bytetrack.yaml]

```

```python
nn.tasks.torch_safe_load
```

1. 判断如果是在线模型，则下载模型权重
2. 调用torch.load(file, map_location="cpu")加载模型checkpoint
3. 如果加载报错，调用check_requirements检查并安装requirements.txt文件中的依赖，重新torch.load模型checkpoint



### 初始化空模型

```python
model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
```

初始化YoLo对象，会调用父类的初始化方法，```super().__init__(model=model, task=task, verbose=verbose)```

```python
engine.model.Model.__init__
```

1. 初始化默认callbacks
2. 判断是否为Hub模型
3. 判断是否为Triton Server模型
4. 调用self._new加载模型

```python
nn.tasks.yaml_model_load
```

加载模型定义yaml文件

yolo11.yaml定义格式如下：

```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLO11 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)

```

```python
self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)
```

根据yaml定义的模型，初始化模型

```python
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
    
nn.tasks.DetectionModel.__init__
```

根据模型映射关系，初始化对应的模型类

```python
nn.tasks.parse_model
```

根据解析的yaml文件内容，初始化PyTorch model

```python
# Build strides
m = self.model[-1]  # Detect()
if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
	s = 256  # 2x min stride
	m.inplace = self.inplace

	def _forward(x):
		"""Performs a forward pass through the model, handling different Detect subclass types accordingly."""
		if self.end2end:
			return self.forward(x)["one2many"]
		return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

	m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
	self.stride = m.stride
	m.bias_init()  # only run once
else:
	self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

# Init weights, biases
initialize_weights(self)
```

1. 对初始化后的模型进行计算步幅
2. 对初始化后的模型初始化权重















