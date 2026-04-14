import yaml
from ultralytics import RTDETR, YOLO, settings
from src.config.paths import setup_mlflow

def train_with_tta(cfg_path='config.yaml'):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
        
    settings.update({'mlflow': False, 'wandb': False})
    
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
            patience=cfg['training'].get('patience', 15),
            lr0=cfg['training'].get('learning_rate', 0.001),
            exist_ok=True
        )
        
        print("Running TTA Validation...")
        best_weights = f"{cfg['paths']['project']}/train/weights/best.pt"
        best_model = ModelClass(best_weights)
        val_res = best_model.val(data=cfg['data']['dataset_yaml'], augment=True)
        if hasattr(val_res, 'results_dict'):
            metrics = {f"tta_{k}": v for k, v in val_res.results_dict.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(metrics)
