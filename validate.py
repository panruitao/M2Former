from ultralytics import YOLO, RTDETR

model = RTDETR("./runs/detect/m2former_espark640_histogram_baseline/weights/best.pt")
metrics = model.val(data="espark.yaml",
                    split="test",
                    representation="histogram",
                    imgsz=640,
                    batch=64,
                    device=0,
                    save_json=True,
                    project='./runs/val',
                    name='m2former_espark640_histogram_baseline')