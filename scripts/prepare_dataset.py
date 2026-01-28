import os
import sys
import numpy as np
from tqdm import tqdm

from src.utils.config import load_config
# -----------------------
# Load config
# -----------------------

cfg = load_config("configs/prepare_dataset.yaml")

RAW_PATH  = cfg.paths.raw_path


# -----------------------
# Helper Functions
# -----------------------

def train_val_split(samples, val_frac=0.2, seed=0):
    np.random.seed(seed)
    N = samples.shape[0]
    idx = np.random.permutation(N)
    n_val = int(val_frac * N)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return samples[train_idx], samples[val_idx]



def create_timeseries_samples(
    month,
    feature_list,
    train_save_dir,
    val_save_dir, 
    val_frac,
    seed,
    horizon,
    stride,
):
    train_data = {}
    val_data = {}

    print(f"\n==============================")
    print(f"Month: {month}")
    print(f"Train Save path: {train_save_dir}")
    print(f"Val Save path: {val_save_dir}") 
    print(f"Horizon={horizon}, Stride={stride}")
    print(f"==============================\n")

    for feat in tqdm(feature_list):
        month_name=month.split('_')[0]

        file_path = os.path.join( RAW_PATH, month, f"{month_name}_{feat}.npy")    #change month here if needed
        arr = np.load(file_path)

        print(f"\nFeature: {feat}")
        print("Original shape:", arr.shape)

        T = arr.shape[0]

        idx = range(0, T - horizon + 1, stride)
        samples = np.stack([arr[i:i+horizon] for i in idx], axis=0)

        print("Total samples:", samples.shape[0])

        train_samples, val_samples = train_val_split(
            samples, val_frac=val_frac, seed=seed
        )
        
        train_data[feat] = train_samples
        val_data[feat] = val_samples

        del arr, samples

    return train_data, val_data


# -----------------------
# Run
# -----------------------
# -----------------------
# Run (feature-wise, no buffer)
# -----------------------

os.makedirs(cfg.paths.train_savepath, exist_ok=True)
os.makedirs(cfg.paths.val_savepath, exist_ok=True)

all_features = cfg.features.met_variables_raw + cfg.features.emission_variables_raw

for feat in all_features:

    print("\n===================================")
    print("Processing feature:", feat)
    print("===================================\n")

    train_chunks = []
    val_chunks   = []

    for month in cfg.data.months:

        train_m, val_m = create_timeseries_samples(
            month=month,
            feature_list=[feat],   # <-- single feature
            train_save_dir=cfg.paths.train_savepath,
            val_save_dir=cfg.paths.val_savepath,
            val_frac=cfg.data.val_frac,
            seed=cfg.data.seed,
            horizon=cfg.data.horizon,
            stride=cfg.data.stride,
        )

        train_chunks.append(train_m[feat])
        val_chunks.append(val_m[feat])

        del train_m, val_m

    # merge only this feature across months
    train_merged = np.concatenate(train_chunks, axis=0)
    val_merged   = np.concatenate(val_chunks, axis=0)

    print(
        f"\nFinal {feat} -> Train:", train_merged.shape,
        "Val:", val_merged.shape
    )

    np.save(
        os.path.join(cfg.paths.train_savepath, f"train_{feat}.npy"),
        train_merged.astype(np.float32)
    )
    np.save(
        os.path.join(cfg.paths.val_savepath, f"val_{feat}.npy"),
        val_merged.astype(np.float32)
    )

    del train_chunks, val_chunks, train_merged, val_merged
