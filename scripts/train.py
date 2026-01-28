from src.utils.utilities3 import *
from src.utils.adam import Adam
from models.baseline_model import FNO2D
from src.utils.config import load_config

import torch
import numpy as np
from scipy import io
import json
from tqdm import tqdm
import os
from timeit import default_timer

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
V  = cfg.features.V

met_variables      = cfg.features.met_variables
emission_variables = cfg.features.emission_variables
all_features = met_variables + emission_variables

batch_size = cfg.training.batch_size
epochs     = cfg.training.epochs

savepath_train = cfg.paths.savepath_train
savepath_val   = cfg.paths.savepath_val

# -------------------------------
# Load min-max statistics
# -------------------------------

min_max = io.loadmat(cfg.paths.min_max_file)

# =========================================================
# Dataloader
# =========================================================

class DataLoaders(torch.utils.data.Dataset):

    def __init__(self, split, cfg, min_max, savepath_train, savepath_val):

        self.time_input = cfg.data.time_input
        self.time_out   = cfg.data.time_out
        self.T = self.time_input + self.time_out

        self.S1 = cfg.data.S1
        self.S2 = cfg.data.S2

        self.met_variables = cfg.features.met_variables
        self.emi_variables = cfg.features.emission_variables
        self.all_features  = self.met_variables + self.emi_variables

        self.min_max = min_max

        if split == "train":
            base_path = savepath_train
        elif split == "val":
            base_path = savepath_val
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.arrs = {}
        for feat in self.all_features:
            path = os.path.join(base_path, f"{split}_{feat}.npy")
            self.arrs[feat] = np.load(path, mmap_mode="r")

        self.N = self.arrs[self.all_features[0]].shape[0]

    def __len__(self):
        return self.N

    def _normalize(self, x, key):

        maxx = float(self.min_max[f"{key}_max"])
        minn = float(self.min_max[f"{key}_min"])
        den = maxx - minn

        if key in ["u10", "v10"]:
            x = (2 * (x - minn) / den) - 1
        else:
            x = (x - minn) / den

        if key in self.emi_variables:
            x = np.clip(x, 0, 1)

        return x.astype(np.float32)

    def __getitem__(self, idx):

        x = np.empty((self.time_input, S1, S2, V), dtype=np.float32)

        for c, feat in enumerate(self.all_features):
            arr = self.arrs[feat][idx, :self.T]
            arr = self._normalize(arr, feat)
            x[..., c] = arr[:self.time_input]

        y = self.arrs[self.all_features[0]][idx, self.time_input:self.T]
        y = self._normalize(y, self.all_features[0])
        y = torch.from_numpy(y).permute(1, 2, 0)

        x = torch.from_numpy(x)

        return x, y


train_dataset = DataLoaders("train", cfg, min_max, savepath_train, savepath_val)
test_dataset  = DataLoaders("val",   cfg, min_max, savepath_train, savepath_val)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
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
    lr=cfg.training.lr,
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
    t1 = default_timer()
    train_l2 = 0.0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(x).view(x.size(0), S1, S2, time_out)
        l2 = myloss(out, y)
        l2.backward()
        optimizer.step()

        train_l2 += l2.item()

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

    t2 = default_timer()

    log.append({
        "epoch": ep,
        "duration": t2 - t1,
        "train_data_loss": train_l2,
        "val_data_loss": test_l2
    })

    print(ep, t2 - t1, train_l2, test_l2)

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
