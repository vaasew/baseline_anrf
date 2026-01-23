import os
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
    os.makedirs(train_save_dir, exist_ok=True)
    os.makedirs(val_save_dir, exist_ok=True)

    print(f"\n==============================")
    print(f"Month: {month}")
    print(f"Train Save path: {train_save_dir}")
    print(f"Val Save path: {val_save_dir}") 
    print(f"Horizon={horizon}, Stride={stride}")
    print(f"==============================\n")

    for feat in tqdm(feature_list):

        file_path = os.path.join( RAW_PATH, f"{month}_{feat}.npy")
        arr = np.load(file_path)

        print(f"\nFeature: {feat}")
        print("Original shape:", arr.shape)

        # arr = arr[spin_up:]
        T = arr.shape[0]
        print("After spin-up:", arr.shape)

        idx = range(0, T - horizon + 1, stride)
        samples = np.stack([arr[i:i+horizon] for i in idx], axis=0)

        print("Total samples:", samples.shape[0])

        train_samples, val_samples = train_val_split(
            samples, val_frac=val_frac, seed=seed
        )

        print("Train:", train_samples.shape[0], "Val:", val_samples.shape[0])

        np.save(os.path.join(train_save_dir, f"train_{feat}.npy"), train_samples)
        np.save(os.path.join(val_save_dir, f"val_{feat}.npy"), val_samples)

        del arr, samples, train_samples, val_samples

    print("\nAll features processed.")


# -----------------------
# Run
# -----------------------

for month in cfg.data.months:

    create_timeseries_samples(
        month=month,
        feature_list=cfg.features.met_variables_raw,
        train_save_dir=cfg.paths.train_savepath,
        val_save_dir=cfg.paths.val_savepath,
        val_frac=cfg.data.val_frac,
        seed=cfg.data.seed,
        horizon=cfg.data.horizon,
        stride=cfg.data.stride,
    )

    create_timeseries_samples(
        month=month,
        feature_list=cfg.features.emission_variables_raw,
        train_save_dir=cfg.paths.train_savepath,
        val_save_dir=cfg.paths.val_savepath,
        val_frac=cfg.data.val_frac,
        seed=cfg.data.seed,
        horizon=cfg.data.horizon,
        stride=cfg.data.stride,
    )
