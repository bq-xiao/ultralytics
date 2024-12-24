from ultralytics import YOLO
from ultralytics import settings


def train():
    # Update a setting
    settings.update({'clearml': False,
                     'comet': False,
                     'mlflow': False,
                     'neptune': False,
                     'raytune': False,
                     'wandb': False})

    # Update multiple settings
    settings.update({'tensorboard': True})
    settings.update({'datasets_dir': r'D:\pyworkspace\datasets'})

    # Load a pretrained YOLO model (recommended for training)
    # model = YOLO('./best.pt')
    model = YOLO('yolo11m.pt')
    # model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(
        data='human-faces-det.yaml',
        project='human-faces-det',
        name='detect',
        batch=-1,
        save_period=100,
        epochs=200,
        imgsz=640,
        patience=60,
        cache=False,
        device='cpu',
        verbose=True,
        single_cls=True,
        resume=False
    )
    print(f"train result:{results}")


if __name__ == '__main__':
    train()
