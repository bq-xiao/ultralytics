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
    model = YOLO('yolo11n.pt')
    # model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(
        data='vehicle-detection.yaml',
        project='vehicle-detection',
        name='detect',
        batch=-1,
        save_period=10,
        cache=False,
        device='cpu',
        verbose=True,
        single_cls=False,
        resume=False
    )
    print(f"train result:{results}")


if __name__ == '__main__':
    train()
