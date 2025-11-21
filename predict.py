import os
import argparse
import numpy as np
import h5py
from natsort import natsorted
from ultralytics import YOLO, RTDETR


def parse_event_txt(file_path):
    events = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            t, x, y, p = float(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            events.append((t, x, y, p))
    return np.array(events, dtype=[('t', 'f4'), ('x', 'u2'), ('y', 'u2'), ('p', 'u1')])


def generate_event_histogram(events, height, width):
    histogram = np.zeros(shape=(height, width), dtype=np.float32)
    for event in events:
        x, y, p = event['x'], event['y'], event['p']
        if 0 <= x < width and 0 <= y < height:
            histogram[y, x] += (2 * int(p) - 1)
    # Normalize
    pos_mask = histogram > 0
    neg_mask = histogram < 0
    if pos_mask.any():
        histogram[pos_mask] /= histogram[pos_mask].max()
    if neg_mask.any():
        histogram[neg_mask] /= abs(histogram[neg_mask].min())
    return histogram[..., np.newaxis]


def generate_event_surface(events, height, width, tau=50e-3):
    surface = np.zeros(shape=(height, width), dtype=np.float32)
    t_ref = events['t'][-1]
    for event in events:
        x, y, t, p = event['x'], event['y'], event['t'], event['p']
        if 0 <= x < width and 0 <= y < height:
            decay_value = np.exp(-(t_ref - t) / tau)
            surface[y, x] = decay_value if p > 0 else -decay_value
    return surface[..., np.newaxis]


def generate_event_volume(events, height, width, n_bins=1):
    volume = np.zeros(shape=(height, width, n_bins, 2), dtype=np.float32)
    min_time, max_time = events['t'].min(), events['t'].max()
    time_bins = np.linspace(min_time, max_time, n_bins + 1)
    for event in events:
        t, x, y, p = event['t'], event['x'], event['y'], event['p']
        if 0 <= x < width and 0 <= y < height:
            bin_index = np.searchsorted(time_bins, t, side='right') - 1
            bin_index = np.clip(bin_index, 0, n_bins - 1)
            volume[y, x, bin_index, p] += 1
    for polarity in range(2):
        max_val = volume[:, :, :, polarity].max()
        if max_val > 0:
            volume[:, :, :, polarity] /= max_val
    return volume.reshape(height, width, n_bins * 2)


def run_inference(weights, source, representation="histogram", imgsz=640, model_type="RTDETR",
                  project="./runs/predict", name="exp", show=False, save=True, save_txt=True, save_conf=True):
    """Unified inference interface"""
    dvs_res = {"dvs640": (480, 640), "dvs346": (260, 346)}
    event_rep = {"histogram": generate_event_histogram,
                 "surface": generate_event_surface,
                 "volume": generate_event_volume}

    # Load model
    model = RTDETR(weights) if model_type.upper() == "RTDETR" else YOLO(weights)

    # Prepare input data
    inputs = []
    if os.path.isdir(source):  # TXT directory
        height, width = dvs_res["dvs640"]  # default resolution
        filenames = natsorted(os.listdir(source))
        for filename in filenames[:100]:   # only read first 100 files
            if filename.endswith(".txt"):
                events = parse_event_txt(os.path.join(source, filename))
                inputs.append(event_rep[representation](events, height, width))
    elif source.endswith(".h5"):  # HDF5 file
        with h5py.File(source, 'r') as f:
            keys = list(f.keys())[:100]    # only read first 100 datasets
            for key in keys:
                data = f[key][()]
                inputs.append(data)
    else:
        raise ValueError("Unsupported source type, must be a directory of TXT files or an HDF5 file.")

    # Run inference
    return model.predict(inputs,
                         imgsz=imgsz,
                         representation=representation,
                         rect=False,
                         project=project,
                         name=name,
                         show=show,
                         save=save,
                         save_txt=save_txt,
                         save_conf=save_conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--source", type=str, required=True, help="Input data source (TXT dir or HDF5 file)")
    parser.add_argument("--representation", type=str, default="histogram", choices=["histogram", "surface", "volume"])
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    parser.add_argument("--model_type", type=str, default="RTDETR", choices=["RTDETR", "YOLO"])
    parser.add_argument("--project", type=str, default="./runs/predict")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-txt", action="store_true")
    parser.add_argument("--no-conf", action="store_true")

    args = parser.parse_args()
    run_inference(
        weights=args.weights,
        source=args.source,
        representation=args.representation,
        imgsz=args.imgsz,
        model_type=args.model_type,
        project=args.project,
        name=args.name,
        show=args.show,
        save=not args.no_save,
        save_txt=not args.no_txt,
        save_conf=not args.no_conf,
    )
