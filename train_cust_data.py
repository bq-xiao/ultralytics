from ultralytics import YOLO
from ultralytics import settings

def train():
    # Update a setting
    settings.update({'clearml': False,
                     'comet':False,
                     'mlflow':False,
                     'neptune':False,
                     'raytune':False,
                     'wandb': False})

    # Update multiple settings
    settings.update({'tensorboard': True})
    settings.update({'datasets_dir': r'D:\pyworkspace\ultralytics\datasets'})

    # Load a pretrained YOLO model (recommended for training)
    # model = YOLO('./best.pt')
    model = YOLO('yolov8n.yaml')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(
        data='./datasets/cust_data.yaml',
        project = 'face',
        name = 'detect',
        batch = 24,
        save_period = 3,
        cache = False,
        device = 'cpu',
        single_cls = True,
        resume = True,

    )
    print(f"train result:{results}")

if __name__ == '__main__':
    train()