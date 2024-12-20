from torchview import draw_graph

from ultralytics import YOLO

yolo = YOLO('yolo11n.pt')
print(yolo.model)
input_shape = (2, 3, 640, 640)
model_graph = draw_graph(yolo.model, input_size=input_shape, expand_nested=True, save_graph=True, filename="torchview",
                         directory=".")

model_graph.visual_graph
