from src.utils.utilities3 import *
from src.utils.adam import Adam
from models.baseline_model import FNO2D
from src.utils.config import load_config
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

import torch
torch.set_num_threads(1)

import numpy as np
import json
from tqdm import tqdm
import os
import time

# -----------------------
# Load config
# -----------------------

cfg = load_config("configs/train.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.cuda.empty_cache()

torch.manual_seed(0)
np.random.seed(0)

# -----------------------
# Settings from YAML
# -----------------------

time_input = cfg.data.time_input
time_out   = cfg.data.time_out
T = time_input + time_out

S1 = cfg.data.S1
S2 = cfg.data.S2

met_variables      = cfg.features.met_variables
emission_variables = cfg.features.emission_variables
all_features = met_variables + emission_variables
V = len(all_features)

batch_size = cfg.training.batch_size
epochs     = cfg.training.epochs

# 🔴 block directories
TRAIN_BLOCK_DIR = cfg.paths.train_savepath + "_blocks"
VAL_BLOCK_DIR   = cfg.paths.val_savepath   + "_blocks"

# =========================================================
# Block Dataset
# =========================================================

class BlockDataset(torch.utils.data.Dataset):
    def __init__(self, block_dir, time_input):

        self.block_files = sorted([
            os.path.join(block_dir, f)
            for f in os.listdir(block_dir)
            if f.endswith(".npy")
        ])

        self.blocks = []
        self.block_sizes = []
        self.cum_sizes = []

        total = 0
        for bf in self.block_files:
            arr = np.load(bf, mmap_mode="r")  # (B, T, S1, S2, V)
            self.blocks.append(arr)
            n = arr.shape[0]
            self.block_sizes.append(n)
            total += n
            self.cum_sizes.append(total)

        self.N = total
        self.time_input = time_input

    def __len__(self):
        return self.N

    def _locate(self, idx):
        for i, cs in enumerate(self.cum_sizes):
            if idx < cs:
                prev = 0 if i == 0 else self.cum_sizes[i-1]
                return i, idx - prev
        raise IndexError

    def __getitem__(self, idx):

        block_id, local_idx = self._locate(idx)
        arr = self.blocks[block_id]

        X = arr[local_idx]                 # (T, S1, S2, V)
        x = torch.from_numpy(X[:self.time_input])
        y = torch.from_numpy(X[self.time_input:, ..., 0]).permute(1, 2, 0)

        return x, y


train_dataset = BlockDataset(TRAIN_BLOCK_DIR, time_input)
test_dataset  = BlockDataset(VAL_BLOCK_DIR,   time_input)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

# =========================================================
# Model
# =========================================================

model = FNO2D(
    time_in=time_input,
    features=V,
    time_out=time_out,
    width=cfg.model.width,
    modes=cfg.model.modes,
).to(device)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total parameters:", count_params(model))

optimizer = Adam(
    model.parameters(),
    lr=float(cfg.training.lr),
    weight_decay=float(cfg.training.weight_decay)
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg.training.scheduler_step,
    gamma=cfg.training.scheduler_gamma
)

myloss = LpLoss(size_average=False)

path1 = cfg.paths.model_save_path
log_save = cfg.paths.save_dir

os.makedirs(os.path.dirname(log_save), exist_ok=True)
os.makedirs(os.path.dirname(path1), exist_ok=True)

log = []

# =========================================================
# Training
# =========================================================

for ep in tqdm(range(epochs)):
    model.train()
    t_epoch_start = time.time()

    train_l2 = 0.0
    data_time = 0.0
    transfer_time = 0.0
    compute_time = 0.0

    end = time.time()

    for x, y in train_loader:

        data_time += time.time() - end

        t0 = time.time()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        transfer_time += time.time() - t0

        t1 = time.time()
        optimizer.zero_grad(set_to_none=True)
        out = model(x).view(x.size(0), S1, S2, time_out)
        l2 = myloss(out, y)
        l2.backward()
        optimizer.step()
        compute_time += time.time() - t1

        train_l2 += l2.item()
        end = time.time()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x).view(x.size(0), S1, S2, time_out)
            test_l2 += myloss(out, y).item()

    train_l2 /= len(train_dataset)
    test_l2  /= len(test_dataset)

    t_epoch_end = time.time()
    n_batches = len(train_loader)

    print("\n========== Epoch profile ==========")
    print(f"Epoch total time     : {t_epoch_end - t_epoch_start:.2f} s")
    print(f"Avg data loading    : {data_time / n_batches:.4f} s/batch")
    print(f"Avg H2D transfer    : {transfer_time / n_batches:.4f} s/batch")
    print(f"Avg compute (GPU)   : {compute_time / n_batches:.4f} s/batch")
    print("===================================\n")

    log.append({
        "epoch": ep,
        "duration": t_epoch_end - t_epoch_start,
        "train_data_loss": train_l2,
        "val_data_loss": test_l2
    })

    print(ep, t_epoch_end - t_epoch_start, train_l2, test_l2)

    if (ep + 1) % cfg.training.checkpoint_every == 0:
        ckpt_path = path1.replace(".pt", f"_ep{ep}.pt")
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, ckpt_path)

        with open(log_save, "w") as f:
            json.dump(log, f)
