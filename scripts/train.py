from src.utils.utilities3 import *
from src.utils.adam import Adam
from src.models.baseline import FNO2D

from src.utils.config import load_config
cfg = load_config("configs/train.yaml")


import torch
import numpy as np
from scipy import io

import json
from tqdm import tqdm
import operator
import math
import os
from math import pi


from functools import reduce
from functools import partial
from timeit import default_timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.cuda.empty_cache()



torch.manual_seed(0)
np.random.seed(0)

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

min_max = io.loadmat(cfg.paths.min_max_file)
min_max_pm = np.zeros((1, 1, 2), dtype = np.float32)
min_max_pm[:, :, 0] = min_max['pm25_max']
min_max_pm[:, :, 1] = min_max['pm25_min']
min_max_tensor = torch.tensor(min_max_pm, dtype = torch.float32)
del min_max_pm




total = np.zeros((ntrain, time_input + time_out, S1, S2, V), dtype = np.float32)
test = np.zeros((ntest, time_input + time_out, S1, S2, V), dtype = np.float32)
##open a file normalise add to total
counter = 0
#First all met variables then combined rain, then combined emissions, then single emissions
for met_variable in met_variables:
    train_1_path = os.path.join(savepath_met, f"train_{met_variable}.npy")
    maxx = min_max[f'{met_variable}_max']
    minn = min_max[f'{met_variable}_min']
    den = (maxx-minn)
    data = np.load(train_1_path, mmap_mode="r")[:, :time_input + time_out, :, :]
    if met_variable in ['u10', 'v10']:
        data = (2*(data - minn)/ den) - 1
    else:
        data = (data - minn)/ den
    total[:, :, :, :, counter] = data

    validation_path = os.path.join(savepath_met, f"val_{met_variable}.npy")
    data = np.load(validation_path, mmap_mode = "r")[:, :time_input + time_out, :, :]
    if met_variable in ['u10', 'v10']:
        data = (2*(data - minn)/ den) - 1
    else:
        data = (data - minn)/ den
    test[:, :, :, :, counter] = data   
    counter = counter + 1
    del data
    
train_1_path_rainc = os.path.join(savepath_met, f"train_rainc.npy")
train_1_path_rainnc = os.path.join(savepath_met, f"train_rainnc.npy")


maxx = min_max[f'rain_combined_max']
minn = min_max[f'rain_combined_min']
den = maxx - minn

data1 = np.load(train_1_path_rainc, mmap_mode="r")[:, :time_input + time_out, :, :]
data2 = np.load(train_1_path_rainnc, mmap_mode="r")[:, :time_input + time_out, :, :]
data = data1 + data2
del data1, data2
data = (data - minn)/ den
total[:, :, :, :, counter] = data


validation_path_rainc = os.path.join(savepath_met, f"val_rainc.npy")
validation_path_rainnc = os.path.join(savepath_met, f"val_rainnc.npy")
data1 = np.load(validation_path_rainc, mmap_mode = "r")[:, :time_input + time_out, :, :]
data2 = np.load(validation_path_rainnc, mmap_mode = "r")[:, :time_input + time_out, :, :]
data = data1 + data2
del data1, data2
data = (data - minn)/ den
test[:, :, :, :, counter] = data
counter = counter + 1
del validation_path, validation_path_rainc, validation_path_rainnc, train_1_path
del train_1_path_rainc, train_1_path_rainnc

for variables in emission_variables:
    pollutant = variables[0].split('_')[0]
    train_1_path_v1 = os.path.join(savepath_emissions, f"train_{variables[0]}.npy")
    train_1_path_v2 = os.path.join(savepath_emissions, f"train_{variables[1]}.npy")


    maxx = min_max[f'{pollutant}_max']
    minn = min_max[f'{pollutant}_min']
    den = maxx - minn
    
    data1 = np.load(train_1_path_v1, mmap_mode="r")[:, :time_input + time_out, :, :]
    data2 = np.load(train_1_path_v2, mmap_mode="r")[:, :time_input + time_out, :, :]
    data = data1 + data2
    del data1, data2
    data = (data - minn)/ den
    total[:, :, :, :, counter] = data


    validation_path_v1 = os.path.join(savepath_emissions, f"val_{variables[0]}.npy")
    validation_path_v2 = os.path.join(savepath_emissions, f"val_{variables[1]}.npy")
    data1 = np.load(validation_path_v1, mmap_mode = "r")[:, :time_input + time_out, :, :]
    data2 = np.load(validation_path_v2, mmap_mode = "r")[:, :time_input + time_out, :, :]
    data = data1 + data2
    del data1, data2
    data = (data - minn)/ den
    test[:, :, :, :, counter] = data
    counter = counter + 1
    del data
    
for k, met_variable in enumerate(single_variables):
    train_1_path = os.path.join(savepath_emissions, f"train_{met_variable}.npy")
    maxx = min_max[f'{mean_names[k]}_max']
    minn = min_max[f'{mean_names[k]}_min']
    den = maxx - minn   
    data = np.load(train_1_path, mmap_mode="r")[:, :time_input + time_out, :, :]
    data = (data - minn)/ den
    total[:, :, :, :, counter] = data


    validation_path = os.path.join(savepath_emissions, f"val_{met_variable}.npy")
    data = np.load(validation_path, mmap_mode = "r")[:, :time_input + time_out, :, :]
    data = (data - minn)/ den
    test[:, :, :, :, counter] = data
    counter = counter + 1
    del data


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


T = cfg.data.time_out
weights = np.array([math.exp((i+1)/T) for i in range(T)])
total = np.sum(weights)
weights = weights/total
print(weights)
w = torch.tensor(weights, dtype=torch.float32, device=device)
w = w.view(1, 1, 1, T)


model = FNO2D(
    time_in=cfg.data.time_input,
    features=cfg.features.V,
    time_out=cfg.data.time_out,
    width=cfg.model.width,
    modes=cfg.model.modes,
).to(device)


print(count_params(model))

optimizer = Adam(
    model.parameters(),
    lr=cfg.training.lr,
    weight_decay=cfg.training.weight_decay
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg.training.scheduler_step,
    gamma=cfg.training.scheduler_gamma
)



min_max_tensor = min_max_tensor.to(device)
del min_max

path1 = cfg.paths.model_save_path
log_save = cfg.paths.save_dir

myloss = LpLoss_weighted(weights = weights, size_average=False)
log = []

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
            


      

