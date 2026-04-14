import yaml, time
from pathlib import Path
from ultralytics import RTDETR, YOLO, settings
from src.config.paths import setup_mlflow

def main():
    cfg = yaml.safe_load(open('config.yaml'))
    settings.update({'mlflow': False})

    ModelClass = YOLO if 'yolo' in cfg['model']['architecture'] else RTDETR
    model = ModelClass(f"{cfg['model']['architecture']}.pt")

    mlflow = setup_mlflow()
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])

    with mlflow.start_run(run_name=cfg['experiment']['name']):
        mlflow.log_params(cfg['training'])
        model.train(
            data=cfg['data']['dataset_yaml'],
            epochs=cfg['training']['epochs'],
            batch=cfg['data']['batch_size'],
            imgsz=cfg['data']['imgsz'],
            project=cfg['paths']['project'],
            name='train',
            exist_ok=True
        )

        # Test-Time Augmentation (TTA) Validation
        print("Running TTA Validation...")
        val_res = model.val(data=cfg['data']['dataset_yaml'], augment=True)
        if hasattr(val_res, 'results_dict'):
            metrics = {f"tta_{k}": v for k, v in val_res.results_dict.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(metrics)

if __name__ == '__main__':
    main()
