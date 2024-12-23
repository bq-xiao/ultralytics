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
    settings.update({'datasets_dir': r'/Users/xiaobaoqiang/pyworkspace/datasets'})

    # Load a pretrained YOLO model (recommended for training)
    # model = YOLO('./best.pt')
    model = YOLO('yolo11n-seg.pt')
    # model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(
        data='package-seg.yaml',
        project='package-seg',
        name='segment',
        batch=-1,
        save_period=20,
        cache=False,
        device='mps',
        verbose=True,
        single_cls=False,
        resume=False
    )
    print(f"train result:{results}")


if __name__ == '__main__':
    train()
