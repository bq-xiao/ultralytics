import pandas as pd

from ultralytics import SAM

csv = pd.read_csv('../datasets/human-faces-object-detection/faces.csv')
sam_model = SAM('sam2.1_b.pt')
device = "cuda"

for index, row in csv.iterrows():
    img_name = row['image_name']
    x0, y0, x1, y1 = float(row['x0']), float(row['y0']), float(row['x1']), float(row['y1'])
    boxes = [x0, y0, x1, y1]
    orig_img = '../datasets/human-faces-object-detection/images/' + img_name
    sam_results = sam_model(orig_img, bboxes=boxes, verbose=False, save=False, device=device)
    segments = sam_results[0].masks.xyn  # noqa

    file_array = img_name.split(".")
    new_file = '../datasets/human-faces-object-detection/seg-labels/' + file_array[0] + ".txt"
    with open(new_file, "a") as f:
        for i in range(len(segments)):
            s = segments[i]
            if len(s) == 0:
                continue
            segment = map(str, segments[i].reshape(-1).tolist())
            f.write(f"0 " + " ".join(segment) + "\n")
