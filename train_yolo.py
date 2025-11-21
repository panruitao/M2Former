import os
os.environ['YOLO_VERBOSE'] = str(False)

from ultralytics import YOLO
import multiprocessing


def train_model(cfg):
    name_parts = [
        os.path.splitext(os.path.basename(cfg["model"]))[0],
        os.path.splitext(os.path.basename(cfg["data"]))[0],
    ]
    if cfg["data"] == "ispark.yaml":
        name_parts.append(str(cfg["imgsz"]))
    if "representation" in cfg:
        name_parts.append(cfg["representation"])
    name = "_".join(name_parts)

    train_args = {
        "data": cfg["data"],
        "imgsz": cfg["imgsz"],
        "epochs": cfg["epochs"],
        "patience": cfg["patience"],
        "batch": cfg["batch"],
        "augmentation": cfg["augmentation"],
        "cos_lr": cfg["cos_lr"],
        "device": cfg["device"],
        "name": "test",
    }
    if "representation" in cfg:
        train_args["representation"] = cfg["representation"]

    model = YOLO(model=cfg["model"])
    model.train(**train_args)
    print(f"Start training with name: {name} on device {cfg['device']}")

if __name__ == '__main__':
    cfgs = [
        {"model": "yolo11s.pt", "data": "espark.yaml", "imgsz": 640, "representation": "histogram",
         "epochs": 200, "patience": 40, "batch": 64, "augmentation": False, "device": 0},
        {"model": "yolo11m.pt", "data": "espark.yaml", "imgsz": 640, "representation": "histogram",
         "epochs": 200, "patience": 40, "batch": 64, "augmentation": False, "device": 1},
        {"model": "yolov8s.pt", "data": "espark.yaml", "imgsz": 640, "representation": "histogram",
         "epochs": 200, "patience": 40, "batch": 64, "augmentation": False, "device": 2},
        {"model": "yolov8m.pt", "data": "espark.yaml", "imgsz": 640, "representation": "histogram",
         "epochs": 200, "patience": 40, "batch": 64, "augmentation": False, "device": 3},
        {"model": "yolov5su.pt", "data": "espark.yaml", "imgsz": 640, "representation": "histogram",
         "epochs": 200, "patience": 40, "batch": 64, "augmentation": False, "device": 4},
        {"model": "yolov5mu.pt", "data": "espark.yaml", "imgsz": 640, "representation": "histogram",
         "epochs": 200, "patience": 40, "batch": 64, "augmentation": False, "device": 5},
    ]

    processes = []
    for cfg in cfgs:
        p = multiprocessing.Process(target=train_model, args=(cfg,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
