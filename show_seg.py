import cv2

from ultralytics.utils import ops


def xyn(data, orig_shape):
    return [
        ops.scale_coords(data.shape, x, orig_shape, normalize=True)
        for x in data
    ]


with open(r'10013349094_1.txt', 'r') as f:
    lines = f.readlines()
    xyxy = []
    for line in lines:
        xy = line.split(" ")
        xyxy.append((float(xy[0]), float(xy[1])))

    from scipy.spatial import ConvexHull
    import numpy as np

    points = [(p[0], p[1]) for p in xyxy]
    hull = ConvexHull(points)
    bb = [points[i] for i in hull.vertices]
    bb.append(bb[0])
    print(bb)
    full_path = r"10013349094_1.jpg"
    image = cv2.imread(full_path)
    height, width, channels = image.shape

    line = "0"
    for b in bb:
        x = round(b[0] / width, 6)
        y = round(b[1] / height, 6)
        line = line + " " + str(x) + " " + str(y)

    segments = []
    new_file = r'D:\pyworkspace\datasets\human-faces-object-detection\test\test.txt'
    with open(new_file, "a") as f:
        f.write(line)

    # full_path = r"10013349094_1.jpg"
    # image = cv2.imread(full_path)
    # height, width = image.height, image.width
    # img_draw = ImageDraw.Draw(image)
    contours = []
    for l in bb:
        a = float(l[0])
        b = float(l[1])
        contours.append((a, b))
        # img_draw.point((a, b), (255, 0, 0))
        # img_draw.rectangle([(490, 320), (687, 664)], fill=None, outline='red', width=2)
    aa = np.array([contours])
    cv2.drawContours(image, aa.astype(np.int32), -1, (0, 255, 0), 3)
    cv2.imshow('image', image)
    cv2.waitKey(0)
