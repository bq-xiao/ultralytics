from ultralytics import YOLO

# Load a model
model = YOLO("human-faces-seg.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model.predict(source="test_data/face_test.jpeg")

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    print(boxes)
    masks = result.masks  # Masks object for segmentation masks outputs
    print(masks)
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
