from ultralytics import SAM

# Load a model
sam = SAM("sam2.1_b.pt")
# Run inference with bboxes prompt
segs = sam.predict(source="../datasets/human-faces-object-detection/images/00002798.jpg",
                   bboxes=[106, 332, 264, 454], stream=False)
for seg in segs:
    seg.save(filename="00002798.jpg")
