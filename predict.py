import cv2
import numpy as np
from PIL import Image, ImageDraw

from ultralytics import YOLO, SAM

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model
sam = SAM("sam2.1_l.pt")

# Run batched inference on a list of images
results = model.predict(source=["ultralytics/assets/zidane.jpg"],
                        device='cpu', conf = 0.5)  # return a list of Results objects
image = Image.open("ultralytics/assets/zidane.jpg")
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    result.save(filename="detect.jpg")  # save to disk
    box = boxes[0]
    #print(box)
    print(box.xyxy.squeeze(0).tolist())
    xy = box.xyxy.squeeze(0).tolist()
    cropped_image = image.crop(xy)
    cropped_image.save('cropped_image.jpg')
    segs = sam.predict(source=cropped_image, points=box.xyxy.reshape(2, 2))
    img_draw = ImageDraw.Draw(cropped_image)
    for seg in segs:
        seg.save(filename="seg.jpg", boxes=False, labels = False)
        # masks
        masks = seg.masks
        #print(seg.masks)
        mask = masks[0]
        #print(mask)
        xy = mask.xy[0]
        #img = np.zeros((512,512,3))
        img = np.array(cropped_image)
        cv2.fillConvexPoly(img, np.int32([xy]), color=(0,0,255))
        cv2.imwrite('mask.jpg', img)
        #print(xy)
        for p in xy:
            img_draw.point((p[0], p[1]), (255,0,0))
        cropped_image.save('img_draw_point.jpg')
        #cropped_image.show()
