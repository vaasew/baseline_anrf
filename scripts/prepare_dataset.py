import os
import numpy as np
from tqdm import tqdm
from scipy import io
from src.utils.config import load_config

# -----------------------
# Load config
# -----------------------

cfg = load_config("configs/prepare_dataset.yaml")
RAW_PATH = cfg.paths.raw_path

OUT_TRAIN = cfg.paths.train_savepath + "_blocks"
OUT_VAL   = cfg.paths.val_savepath   + "_blocks"

os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_VAL, exist_ok=True)

# -----------------------
# Load min–max stats
# -----------------------

min_max = io.loadmat(cfg.paths.min_max_file)
all_features = cfg.features.met_variables_raw + cfg.features.emission_variables_raw
V = len(all_features)

min_vals = {f: min_max[f"{f}_min"].item() for f in all_features}
max_vals = {f: min_max[f"{f}_max"].item() for f in all_features}

# -----------------------
# Settings
# -----------------------

horizon = cfg.data.horizon
stride  = cfg.data.stride
val_frac = cfg.data.val_frac
seed = cfg.data.seed
BLOCK_SIZE = 200   # 🔴 adjust so one block is ~3–6 GB

np.random.seed(seed)

# -----------------------
# Helper
# -----------------------

def normalize(arr, feat):
    arr = arr.astype(np.float32)
    arr = (arr - min_vals[feat]) / (max_vals[feat] - min_vals[feat])
    if feat in ["u10", "v10"]:
        arr = 2.0 * arr - 1.0
    if feat in cfg.features.emission_variables_raw:
        arr = np.clip(arr, 0, 1)
    return arr

# -----------------------
# Build blocks
# -----------------------

def build_blocks(split="train"):

    print(f"\n==============================")
    print(f"Building {split.upper()} blocks")
    print(f"==============================\n")

    all_samples = []

    # ---- First pass: count samples ----
    for month in cfg.data.months:
        any_feat = all_features[0]
        T = np.load(os.path.join(RAW_PATH, month, f"{any_feat}.npy"), mmap_mode="r").shape[0]
        n = len(range(0, T - horizon + 1, stride))
        all_samples.extend([(month, i) for i in range(n)])

    all_samples = np.array(all_samples)
    np.random.shuffle(all_samples)

    split_idx = int((1 - val_frac) * len(all_samples))
    if split == "train":
        sample_list = all_samples[:split_idx]
    else:
        sample_list = all_samples[split_idx:]

    print("Total samples:", len(sample_list))

    block_id = 0

    for i in range(0, len(sample_list), BLOCK_SIZE):

        block_samples = sample_list[i:i+BLOCK_SIZE]
        B = len(block_samples)

        print(f"\nBlock {block_id} | Samples: {B}")

        # ---- load all features month-wise ----
        month_cache = {}

        X_block = None

        for b, (month, idx) in enumerate(tqdm(block_samples)):

            if month not in month_cache:
                month_cache[month] = {
                    f: normalize(np.load(os.path.join(RAW_PATH, month, f"{f}.npy")), f)
                    for f in all_features
                }

            if X_block is None:
                T0, H, W = month_cache[month][all_features[0]].shape
                X_block = np.empty((B, horizon, H, W, V), dtype=np.float32)

            for v, f in enumerate(all_features):
                start = idx * stride
                X_block[b, :, :, :, v] = month_cache[month][f][start:start+horizon]

        out_dir = OUT_TRAIN if split == "train" else OUT_VAL
        out_path = os.path.join(out_dir, f"{split}_block_{block_id:03d}.npy")

        np.save(out_path, X_block)
        print("Saved:", out_path, X_block.shape)

        del X_block, month_cache
        block_id += 1


# -----------------------
# Run
# -----------------------

build_blocks("train")
build_blocks("val")

print("\n✅ All blocks built successfully.")
