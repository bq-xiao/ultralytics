from ultralytics import solutions

inf = solutions.Inference(
    model="human-faces-seg.pt",  # You can use any model that Ultralytics support, i.e. YOLO11, YOLOv10
)
inf.inference()

### Make sure to run the file using command `streamlit run <file-name.py>`
