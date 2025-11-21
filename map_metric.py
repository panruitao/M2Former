import os
import argparse
import pandas as pd


def get_max_map_and_epoch(csv_path: str):
    """Read results.csv and return the epoch with max mAP@50-95"""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    max_row = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
    max_epoch = int(max_row['epoch'])
    max_map_50 = float(max_row['metrics/mAP50(B)'])
    max_map_50_95 = float(max_row['metrics/mAP50-95(B)'])
    return max_epoch, max_map_50, max_map_50_95


def main(run_path: str):
    csv_path = os.path.join(run_path, "results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"results.csv not found in {run_path}")
    epoch, max_map_50, max_map_50_95 = get_max_map_and_epoch(csv_path)
    print(f"Run: {run_path}")
    print(f"Epoch {epoch}: mAP@50: {max_map_50:.5f}, mAP@50-95: {max_map_50_95:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get best mAP from Ultralytics results.csv")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to training run folder (should contain results.csv)"
    )
    args = parser.parse_args()
    main(args.model_dir)
