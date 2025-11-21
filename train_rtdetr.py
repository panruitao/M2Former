from ultralytics import RTDETR

import warnings
warnings.filterwarnings("ignore", message=".*grid_sampler_2d_backward_cuda.*")


model = RTDETR("m2former.yaml")
# model = RTDETR("rtdetr-resnet18.yaml")
results = model.train(name="m2former_espark640_histogram_baseline",
                      data="espark.yaml",
                      imgsz=640,  # input size
                      representation="histogram",  # event representation
                      device=0,
                      epochs=50,
                      patience=10,  # early stop
                      batch=16,
                      nbs=16,
                      use_aal=False,  # enable AAL loss
                      optimizer="AdamW",
                      lr0=0.0001,
                      lrf=0.01,
                      cos_lr=False,
                      momentum=0.900,
                      weight_decay=0.0001,
                      warmup_epochs=3.0,
                      warmup_momentum=0.8,
                      warmup_bias_lr=0.1,
                      amp=False,  # mixed precision
                      max_norm=0.1,  # gradient clip
                      multi_scale=False,  # multi-scale inputs
                      augmentation=False,  # enable augmentation
                      degrees=0.0,  # affine augmentation
                      translate=0.1,
                      scale=0.1,
                      shear=0.0,
                      perspective=0.0,
                      flipud=0.0,  # flip augmentation
                      fliplr=0.5,
                      mosaic=0.5,  # mosaic augmentation
                      mixup=0.5,  # mixup augmentation
                      )