# Load a model
from ultralytics import YOLO, SAM

det_model = YOLO("human-faces-seg.pt")
sam_model = SAM("sam2.1_b.pt")
conf = 0.25,
iou = 0.45,
imgsz = 640,
max_det = 300
device = "cuda"
data = "test_data/face_test.jpeg"

det_results = det_model(data)

for result in det_results:
    class_ids = result.boxes.cls.int().tolist()  # noqa
    if len(class_ids):
        boxes = result.boxes.xyxy  # Boxes object for bbox outputs
        sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
        for seg in sam_results:
            seg.save(filename="face_test.jpg")
