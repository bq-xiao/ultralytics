from ultralytics import SAM

# Load a model
sam = SAM("sam2.1_l.pt")
# Run inference with bboxes prompt
segs = sam.predict(source="ultralytics/assets/zidane.jpg")
for seg in segs:
    seg.save(filename="111.jpg")
