# M²Former for Event-based Space Object Detection

## 1. Acknowledgement

This project is built upon [Ultralytics](https://github.com/ultralytics/ultralytics) (version **8.3.104**). 
We gratefully acknowledge the Ultralytics team for their open-source contribution, which enabled this work. 
Further details of the paper can be found on the [project page](https://iamie-vision.github.io/M2Former/).


## 2. Environment Setup
```bash
# Create and activate a conda environment
conda create -n m2former python=3.9 -y
conda activate m2former

# Clone and install Ultralytics
git clone https://github.com/panruitao/M2Former.git
pip install -e .

# Install other dependencies
pip install -r requirements.txt
```


## 3. Dataset

The experiments in this project are conducted on the **E-SPARK** dataset. Please [download](https://zenodo.org/) it before running inference or training.


## 4. Pretrained Models (Inference)

We provide several pretrained models that can be directly used for inference with **`predict.py`**:
<table>
  <tr>
    <td><a href="https://drive.google.com/file/d/1i-ENNkdQppkNj898WfxqBavr45fCkerM/view?usp=drive_link">
        m2former_espark640_histogram_baseline</a></td>
    <td><a href="https://drive.google.com/file/d/1DiFkDuhitIqgueLSqDuvXDKK7OV7Xs7m/view?usp=drive_link">
        m2former_espark320_histogram_baseline</a></td>
  </tr>
  <tr>
    <td><a href="https://drive.google.com/file/d/1OUw2fNyQj-imXCznUgu8Yt2TYRi3k8A0/view?usp=drive_link">
        m2former_espark640_histogram_aal</a></td>
    <td><a href="https://drive.google.com/file/d/12wOeAB7QRQHsnXeE7SPhteLkN7ghiydg/view?usp=drive_link">
        m2former_espark320_histogram_aal</a></td>
  </tr>
  <tr>
    <td><a href="https://drive.google.com/file/d/1vJW7i2DJ2c9PfntmRycf6V5GEwu83Vw2/view?usp=drive_link">
        m2former_espark640_histogram_aal_aug</a></td>
    <td><a href="https://drive.google.com/file/d/1V937GuSTW0j5fpK_7WWOZXUSiWBME6Uf/view?usp=drive_link">
        m2former_espark320_histogram_aal_aug</a></td>
  </tr>
</table>

Download and put these models on `./pretrained` folder.

### Example Usage

For preprocessed .h5 files:
```bash
python predict.py \
    --weights ./pretrained/m2former_espark640_histogram_baseline/weights/best.pt \
    --source /path/to/histogram.h5 \
    --imgsz 640 \
    --representation histogram
```
For raw event .txt files:
```bash
python predict.py \
    --weights ./pretrained/m2former_espark640_histogram_baseline/weights/best.pt \
    --source /path/to/txt_folder \
    --imgsz 640 \
    --representation histogram
```


## 5. Model Training

This project supports training with both **RT-DETR** and **YOLO** frameworks.

Use `train_rtdetr.py` to train M²Former or RT-DETR-R18 on E-SPARK:
```bash
# default config inside train_rtdetr.py
python train_rtdetr.py
```
The model architectures are defined in [m2former.yaml](ultralytics/cfg/models/rt-detr/m2former.yaml) and [rtdetr-resnet18.yaml](ultralytics/cfg/models/rt-detr/rtdetr-resnet18.yaml). 
Source codes are located in [M2FormerNet.py](ultralytics/nn/AddModules/M2FormerNet.py) and [ResNet.py](ultralytics/nn/AddModules/ResNet.py).

Use `train_yolo.py` to train YOLOv5, YOLOv8 and YOLO11 (multiprocessing) on E-SPARK:
```bash
# default config inside train_rtdetr.py
python train_yolo.py
```
**Note:** Use the pretrained models from Ultralytics.


## 6. Model Validation

After training, the performance of the models can be evaluated in two ways to obtain **AP@50** and **AP@50:95**.

(1) Use `map_metric.py` to parse the training log (`results.csv`) and report the epoch with the highest AP@50:95.
```bash
python map_metric.py --model_dir /path/to/model_training_dir
```
**Note:** Experimental results reported in the paper are obtained by this way. 
In particular, **APs** (small object detection performance) is calculated by [COCOAPI](https://github.com/cocodataset/cocoapi).

(2) Run `val.py` with the trained weights to evaluate the model on the validation or test set.
```bash
# default config inside validate.py
python validate.py
```
**Note:** Metrics computed by `val.py` are slightly lower than those in the training log, since `val.py` runs a stricter evaluation with different mode, 
while the log records the best metrics during training.


## Citations

If you utilize this code in your research, please cite our paper:
```bibtex
Coming soon.
```
