import os
import numpy as np
from extract_logmel import build_dataset

BASE_DIR = "data/train/base"
AMR_DIR = "data/train/amr"
OUT_DIR = "features_out"

os.makedirs(OUT_DIR, exist_ok=True)


def build_and_save(data_root, tag):
    print(f"\nBuilding features for: {tag}")

    X_bona, y_bona = build_dataset(os.path.join(data_root, "bonafide"), label=0)
    X_spoof, y_spoof = build_dataset(os.path.join(data_root, "spoof"), label=1)

    X = np.vstack([X_bona, X_spoof])
    y = np.concatenate([y_bona, y_spoof])

    np.save(os.path.join(OUT_DIR, f"X_{tag}.npy"), X)
    np.save(os.path.join(OUT_DIR, f"y_{tag}.npy"), y)

    print(f"Saved {tag}: X shape {X.shape}, y shape {y.shape}")


if __name__ == "__main__":
    build_and_save(BASE_DIR, "base")
    build_and_save(AMR_DIR, "amr")
