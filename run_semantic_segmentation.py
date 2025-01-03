import logging
import os
import sys
from functools import partial
from glob import glob

import albumentations as A
import evaluate
import numpy as np
import torch
import transformers
from albumentations.pytorch import ToTensorV2
from torch import nn
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    Trainer,
    default_data_collator, TrainingArguments, EarlyStoppingCallback,
)

from datasets import DatasetDict, Dataset, Image

logger = logging.getLogger('transformers-faces-segment')


def reduce_labels_transform(labels: np.ndarray, **kwargs) -> np.ndarray:
    """Set `0` label as with value 255 and then reduce all other labels by 1.

    Example:
        Initial class labels:         0 - background; 1 - road; 2 - car;
        Transformed class labels:   255 - background; 0 - road; 1 - car;

    **kwargs are required to use this function with albumentations.
    """
    labels[labels == 0] = 255
    labels = labels - 1
    labels[labels == 254] = 255
    return labels


def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                 "label": sorted(label_paths)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())
    return dataset


def load_images(path):
    train_x = sorted(glob(os.path.join(path, 'train', 'images', '*.jpg')))
    train_y = sorted(glob(os.path.join(path, 'train', 'labels', '*.png')))

    val_x = sorted(glob(os.path.join(path, 'val', 'images', '*.jpg')))
    val_y = sorted(glob(os.path.join(path, 'val', 'labels', '*.png')))

    return (train_x, train_y), (val_x, val_y)


# INFO = 20,DEBUG = 10
log_level = 20
cache_dir = "/kaggle/working/transformers-faces-segment-cache"
model_name_or_path = "nvidia/mit-b0"
do_reduce_labels = True
main_path = "/kaggle/input/human-face-segmentation/LaPa"

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# The default of training_args.log_level is passive, so we set log level at info here to have that default.
transformers.utils.logging.set_verbosity_info()
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Load dataset
(train_x, train_y), (val_x, val_y) = load_images(main_path)
print("train_x shape : ", len(train_x))
print("train_y shape : ", len(train_y))
print("val_x shape : ", len(val_x))
print("val_y shape : ", len(val_y))

image_paths_train = train_x
label_paths_train = train_y

image_paths_validation = val_x
label_paths_validation = val_y

# step 1: create Dataset objects
train_dataset = create_dataset(image_paths_train, label_paths_train)
validation_dataset = create_dataset(image_paths_validation, label_paths_validation)

# step 2: create DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
}
)
id2label = {0: 'face'}
label2id = {v: k for k, v in id2label.items()}

# Load the mean IoU metric from the evaluate package
metric = evaluate.load("mean_iou", cache_dir=cache_dir)


# Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
@torch.no_grad()
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=len(id2label),
        ignore_index=0,
        reduce_labels=image_processor.do_reduce_labels,
    )
    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    return metrics


config = AutoConfig.from_pretrained(
    model_name_or_path,
    label2id=label2id,
    id2label=id2label,
    cache_dir=cache_dir,
)
model = AutoModelForSemanticSegmentation.from_pretrained(
    model_name_or_path,
    from_tf=False,
    config=config,
    cache_dir=cache_dir,
)
image_processor = AutoImageProcessor.from_pretrained(
    model_name_or_path,
    do_reduce_labels=True,
    cache_dir=cache_dir,
)

# Define transforms to be applied to each image and target.
if "shortest_edge" in image_processor.size:
    # We instead set the target size as (shortest_edge, shortest_edge) to here to ensure all images are batchable.
    height, width = image_processor.size["shortest_edge"], image_processor.size["shortest_edge"]
else:
    height, width = image_processor.size["height"], image_processor.size["width"]

train_transforms = A.Compose(
    [
        A.Lambda(
            name="reduce_labels",
            mask=reduce_labels_transform if do_reduce_labels else None,
            p=1.0,
        ),
        # pad image with 255, because it is ignored by loss
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=255, p=1.0),
        A.RandomCrop(height=height, width=width, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ]
)
val_transforms = A.Compose(
    [
        A.Lambda(
            name="reduce_labels",
            mask=reduce_labels_transform if do_reduce_labels else None,
            p=1.0,
        ),
        A.Resize(height=height, width=width, p=1.0),
        A.Normalize(mean=image_processor.image_mean, std=image_processor.image_std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ]
)


def preprocess_batch(example_batch, transforms: A.Compose):
    pixel_values = []
    labels = []
    for image, target in zip(example_batch["image"], example_batch["label"]):
        transformed = transforms(image=np.array(image.convert("RGB")), mask=np.array(target))
        pixel_values.append(transformed["image"])
        labels.append(transformed["mask"])

    encoding = {}
    encoding["pixel_values"] = torch.stack(pixel_values).to(torch.float)
    encoding["labels"] = torch.stack(labels).to(torch.long)

    return encoding


# Preprocess function for dataset should have only one argument,
# so we use partial to pass the transforms
preprocess_train_batch_fn = partial(preprocess_batch, transforms=train_transforms)
preprocess_val_batch_fn = partial(preprocess_batch, transforms=val_transforms)

dataset["train"].set_transform(preprocess_train_batch_fn)
dataset["validation"].set_transform(preprocess_val_batch_fn)

output_dir = "/kaggle/working/transformers-faces-segment-output"
batch_size = 32

training_args = TrainingArguments(
    output_dir=output_dir,
    do_train=True,
    learning_rate=6e-5,
    num_train_epochs=2,
    lr_scheduler_type='polynomial',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_strategy='steps',
    logging_steps=10,
    eval_strategy='steps',
    save_strategy='steps',
    seed=1337,
    save_total_limit=3,
    save_steps=5,
    eval_steps=5,
    eval_accumulation_steps=5,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
    report_to=['clearml', 'tensorboard']
)

callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
# Initialize our trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    tokenizer=image_processor,
    data_collator=default_data_collator,
    callbacks=callbacks
)

# Training
train_result = trainer.train(resume_from_checkpoint=None)
print(train_result)

trainer.save_model()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
