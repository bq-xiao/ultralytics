import numpy as np
import pandas as pd
import torch
from PIL import Image

from ultralytics.utils import ops

csv = pd.read_csv('../../datasets/human-faces-object-detection/faces.csv')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
sam2_checkpoint = r"D:\pyworkspace\sam2\sam2.1_hiera_base_plus.pt"
model_cfg = r"D:\pyworkspace\sam2\sam2\configs\sam2.1\sam2.1_hiera_b+.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)


def xyn(data, orig_shape):
    return [
        ops.scale_coords(data.shape[1:], x, orig_shape, normalize=True)
        for x in ops.masks2segments(data)
    ]


for index, row in csv.iterrows():
    img_name = row['image_name']
    x0, y0, x1, y1 = float(row['x0']), float(row['y0']), float(row['x1']), float(row['y1'])
    orig_img = '../../datasets/human-faces-object-detection/data/images/train/' + img_name
    image = Image.open(orig_img)
    image = np.array(image.convert("RGB"))
    predictor.set_image(image)
    input_box = np.array([x0, y0, x1, y1])
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    orig_shape = np.array([int(row['height']), int(row['width'])])
    segments = xyn(torch.tensor(masks), orig_shape)

    file_array = img_name.split(".")
    new_file = '../../datasets/human-faces-object-detection/seg-labels/' + file_array[0] + ".txt"
    with open(new_file, "a") as f:
        for i in range(len(segments)):
            s = segments[i]
            if len(s) == 0:
                continue
            segment = map(str, segments[i].reshape(-1).tolist())
            f.write(f"0 " + " ".join(segment) + "\n")
