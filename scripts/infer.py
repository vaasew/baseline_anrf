from models.baseline_model import FNO2D

import torch
import numpy as np
from scipy import io
import os
from utils.utilities3 import *  

# -----------------------
# Load config
# -----------------------

from src.utils.config import load_config
cfg = load_config("configs/infer.yaml")


torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def testing(model, forward_steps, test_loader, model_output_steps, name_file, d, V, time_input,lat=52,long=68):
    prediction = torch.zeros((d, lat, long, forward_steps * model_output_steps))
    '''
    Works for cases where output steps are greater than the number of input steps or equal for example 10 input step output 16 pm steps
    
    '''
    index = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            xx = batch[0].to(device)
            if index%100 == 0:
                print(index)
            for k in range(forward_steps):
                if k == 0:
                    fp = xx[:, :time_input, :, :, :].reshape(1, time_input, lat, long, V)
                    im1 = model(fp)
                    im1 = torch.clamp(im1, min = 1e-6)
                    pred = im1
                else:
                    next_pass_pm = im1.permute(0, 3, 1, 2).unsqueeze(-1)[:, -time_input:, :, :, :]
                    next_pass_met = xx[:, model_output_steps*k : model_output_steps*k + time_input, :, :, 1:].reshape(1, time_input, lat, long, V-1)
                    im1 = model(torch.concat((next_pass_pm, next_pass_met), dim = -1))
                    im1 = torch.clamp(im1, min = 1e-6)
                    pred = torch.concat((pred, im1), dim = -1)
                
            prediction[index, :, :, :] = pred
            index = index + 1
            
    np.save(name_file,np.array(prediction.cpu().detach().tolist(),dtype=np.float32))

    return None



# -----------------------
# Settings from YAML
# -----------------------

dataset = cfg.data.dataset
ntest = cfg.data.ntest
total_time = cfg.data.total_time
time_input = cfg.data.time_input
time_out = cfg.data.time_out
S1 = cfg.data.S1
S2 = cfg.data.S2
lat = cfg.data.S1
long = cfg.data.S2

met_variables = cfg.features.met_variables
emission_variables = cfg.features.emission_variables
single_variables = cfg.features.single_variables
mean_names = cfg.features.mean_names

savepath_emissions = cfg.paths.savepath_emissions
savepath_met = cfg.paths.savepath_met

V = cfg.features.V
modes = cfg.model.modes
width = cfg.model.width

checkpoint = torch.load(cfg.paths.checkpoint, map_location=device)


print(f"[INFO] Running inference on dataset: {dataset}")



def normalize_data(data, min_max, key, *, wind=False, clip=False):
    """
    data: np.ndarray [N, T, H, W]
    key:  base name like 'pm25', 'NOx', 'rain_combined', 'NMVOCE'
    """

    maxx = float(min_max[f"{key}_max"])
    minn = float(min_max[f"{key}_min"])
    den = maxx - minn

    if den == 0:
        raise ValueError(f"{key}_max == {key}_min in min_max file")


    if wind:
        data = (2 * (data - minn) / den) - 1
    else:
        data = (data - minn) / den

    if clip:
        data = np.clip(data, 0, 1)

    return data.astype(np.float32)



def denormalisation(concentration, max_pm, min_pm):
    return (concentration * (max_pm.unsqueeze(-1) - min_pm.unsqueeze(-1))) + min_pm.unsqueeze(-1)

min_max = io.loadmat(cfg.paths.min_max_file)





test = np.zeros((ntest, time_input + time_out, S1, S2, V), dtype = np.float32)
counter = 0

for met_variable in met_variables:
    if met_variable=='rain':
        continue

    test_path = os.path.join(savepath_met, f"{dataset}_{met_variable}.npy")
    data = np.load(test_path, mmap_mode = "r")[:, :time_input + time_out, :, :].astype(np.float32)

    data = normalize_data(
            data,
            min_max,
            key=met_variable,
            wind=met_variable in ["u10", "v10"]
            )

    test[:, :, :, :, counter] = data   
    counter = counter + 1
    del data
    del test_path


if 'rain' in met_variables:
 
    test_path_rainc = os.path.join(savepath_met, f"{dataset}_rainc.npy")
    test_path_rainnc = os.path.join(savepath_met, f"{dataset}_rainnc.npy")
    data1 = np.load(test_path_rainc, mmap_mode = "r")[:, :time_input + time_out, :, :].astype(np.float32)
    data2 = np.load(test_path_rainnc, mmap_mode = "r")[:, :time_input + time_out, :, :].astype(np.float32)
    data = data1 + data2
    data = normalize_data(
        data,
        min_max,
        key="rain_combined"
    )

    del data1, data2
    test[:, :, :, :, counter] = data
    counter = counter + 1
    del test_path_rainc, test_path_rainnc, data


for variables in emission_variables:
    pollutant = variables[0].split('_')[0]


    test_path_v1 = os.path.join(savepath_emissions, f"{dataset}_{variables[0]}.npy")
    test_path_v2 = os.path.join(savepath_emissions, f"{dataset}_{variables[1]}.npy")
    data1 = np.load(test_path_v1, mmap_mode = "r")[:, :time_input + time_out, :, :].astype(np.float32)
    data2 = np.load(test_path_v2, mmap_mode = "r")[:, :time_input + time_out, :, :].astype(np.float32)
    data = data1 + data2
    del data1, data2

    data = normalize_data(
        data,
        min_max,
        key=pollutant,
        clip=True
    )

    test[:, :, :, :, counter] = data
    counter = counter + 1
    del data

for k, met_variable in enumerate(single_variables):


    test_path = os.path.join(savepath_emissions, f"{dataset}_{met_variable}.npy")
    data = np.load(test_path, mmap_mode = "r")[:, :time_input + time_out, :, :].astype(np.float32)

    data = normalize_data(
    data,
    min_max,
    key=mean_names[k]
    )

    test[:, :, :, :, counter] = data
    counter = counter + 1
    del data

if counter != V:
    print("Wrong assignment")



test_a = torch.tensor(test, dtype = torch.float32) 
print(test_a.shape)
del test
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a), batch_size=1, shuffle=False)
d = test_a.shape[0]
del test_a

model = FNO2D(
    time_in=time_input,
    features=V,
    time_out=time_out,
    width=width,
    modes=modes,
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

os.makedirs(os.path.dirname(cfg.paths.output_loc), exist_ok=True)
testing(
    model,
    cfg.inference.forward_steps,
    test_loader,
    cfg.data.time_out,
    os.path.join(cfg.paths.output_loc,f'{dataset}.npy'),
    d, V, time_input, lat, long
)


del test_loader
