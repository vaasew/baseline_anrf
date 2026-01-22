from src.utils.utilities3 import *
from src.utils.adam import Adam
from models.baseline_model import FNO2D

# -----------------------
# Load config
# -----------------------

from src.utils.config import load_config
cfg = load_config("configs/train.yaml")


import torch
import numpy as np
from scipy import io

import json
from tqdm import tqdm
import math
import os
from timeit import default_timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.cuda.empty_cache()



torch.manual_seed(0)
np.random.seed(0)


# -----------------------
# Settings from YAML
# -----------------------

ntrain = cfg.data.ntrain
ntest = cfg.data.ntest
total_time = cfg.data.total_time
time_input = cfg.data.time_input
time_out = cfg.data.time_out
S1 = cfg.data.S1
S2 = cfg.data.S2

met_variables = cfg.features.met_variables
emission_variables = cfg.features.emission_variables
single_variables = cfg.features.single_variables
mean_names = cfg.features.mean_names

savepath_emissions = cfg.paths.savepath_emissions
savepath_met = cfg.paths.savepath_met

V = cfg.features.V
modes = cfg.model.modes
width = cfg.model.width
batch_size = cfg.training.batch_size
epochs = cfg.training.epochs
learning_rate = cfg.training.lr
scheduler_step = cfg.training.scheduler_step
scheduler_gamma = cfg.training.scheduler_gamma


def denormalisation(concentration, max_pm, min_pm):
    return (concentration * (max_pm.unsqueeze(-1) - min_pm.unsqueeze(-1))) + min_pm.unsqueeze(-1)

# -------------------------------
# Normalization helper
# -------------------------------

def normalize_data(data, min_max, key, *, wind=False, clip=False):
    maxx = float(min_max[f"{key}_max"])
    minn = float(min_max[f"{key}_min"])
    den = maxx - minn

    if den == 0:
        raise ValueError(f"{key}_max == {key}_min")

    if wind:
        data = (2 * (data - minn) / den) - 1
    else:
        data = (data - minn) / den

    if clip:
        data = np.clip(data, 0, 1)

    return data.astype(np.float32)


# -------------------------------
# Load min-max statistics
# -------------------------------

min_max = io.loadmat(cfg.paths.min_max_file)


# -------------------------------
# Allocate arrays
# -------------------------------

total = np.zeros((ntrain, time_input + time_out, S1, S2, V), dtype=np.float32)
test  = np.zeros((ntest,  time_input + time_out, S1, S2, V), dtype=np.float32)

counter = 0


# -------------------------------
# Meteorology variables
# -------------------------------

for met_variable in met_variables:
 

    train_path = os.path.join(savepath_met, f"train_{met_variable}.npy")
    val_path   = os.path.join(savepath_met, f"val_{met_variable}.npy")

    train_data = np.load(train_path, mmap_mode="r")[:, :time_input + time_out]
    val_data   = np.load(val_path,   mmap_mode="r")[:, :time_input + time_out]

    train_data = normalize_data(
        train_data, min_max, met_variable,
        wind=met_variable in ["u10", "v10"]
    )
    val_data = normalize_data(
        val_data, min_max, met_variable,
        wind=met_variable in ["u10", "v10"]
    )

    total[..., counter] = train_data
    test[...,  counter] = val_data
    counter += 1

    del train_data, val_data



# -------------------------------
# Emission variables
# -------------------------------

for variable in emission_variables:

    train_data = np.load(os.path.join(savepath_emissions, f"train_{variable}.npy"), mmap_mode="r")[:, :time_input + time_out]
    val_data = np.load(os.path.join(savepath_emissions, f"val_{variable}.npy"), mmap_mode="r")[:, :time_input + time_out]

    train_data = normalize_data(train_data, min_max, variable, clip=True)
    val_data   = normalize_data(val_data,   min_max, variable, clip=True)

    total[..., counter] = train_data
    test[...,  counter] = val_data
    counter += 1

    del train_data, val_data




# -------------------------------
# Safety check
# -------------------------------

if counter != V:
    raise ValueError(f"Feature mismatch: counter={counter}, V={V}")


train_a = torch.tensor(total[:ntrain, :time_input, :, :, :], dtype = torch.float32)
train_u = torch.tensor(total[:ntrain, time_input:, :, :, 0], dtype = torch.float32)
train_u = train_u.permute(0, 2, 3, 1)

print(train_a.shape, train_u.shape)
del total
print(train_a.shape, train_u.shape)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True,drop_last=True)
del train_a, train_u
test_a = torch.tensor(test[:ntest, :time_input, :, :, :], dtype = torch.float32)
test_u = torch.tensor(test[:ntest, time_input:, :, :, 0], dtype = torch.float32)
test_u = test_u.permute(0, 2, 3, 1)
print(test_a.shape, test_u.shape)
del test
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False,drop_last=True)
del test_u, test_a




model = FNO2D(
    time_in=cfg.data.time_input,
    features=cfg.features.V,
    time_out=cfg.data.time_out,
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



path1 = cfg.paths.model_save_path
log_save = cfg.paths.save_dir

myloss = LpLoss(size_average=False)
log = []

os.makedirs(os.path.dirname(log_save), exist_ok=True)
os.makedirs(os.path.dirname(path1), exist_ok=True)


for ep in tqdm(range(epochs)):
    model.train()
    t1 = default_timer()
    train_l2 = 0.0

    for x,y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x).view(batch_size, S1, S2, time_out)
        l2 = myloss(out, y)
        total_loss = l2
        total_loss.backward()
        optimizer.step()
        train_l2 += l2.item()
    scheduler.step()
    
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).view(batch_size, S1, S2, time_out)
            test_l2 += myloss(out, y).item()
            

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
        
    log.append({
        "epoch": ep,
        "duration": t2 - t1,
        "train_data_loss": train_l2,
        "val_data_loss": test_l2, 
        })
    print(ep, t2-t1, train_l2, test_l2)
    if (ep + 1) % cfg.training.checkpoint_every == 0:
        ckpt_path = path1.replace(".pt", f"_ep{ep}.pt")
        torch.save({'epoch': ep, 'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, ckpt_path)

        with open(log_save, "w") as final_log:
        	json.dump(log, final_log)
            


      

