from ultralytics.data.annotator import auto_annotate

auto_annotate(data="ultralytics/assets",
              det_model="yolo11x.pt",
              sam_model="sam2.1_l.pt")