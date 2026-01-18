from src.utils.config import load_config
cfg = load_config("configs/infer.yaml")
from src.models.baseline import FNO2D

import torch
import numpy as np
from scipy import io
import os
from utils.utilities3 import *  


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



def denormalisation(concentration, max_pm, min_pm):
    return (concentration * (max_pm.unsqueeze(-1) - min_pm.unsqueeze(-1))) + min_pm.unsqueeze(-1)


checkpoint = torch.load(cfg.paths.checkpoint, map_location=device)

model = FNO2D(
    time_in=time_input,
    features=V,
    time_out=time_out,
    width=width,
    modes=modes,
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

min_max = io.loadmat(cfg.paths.min_max_file)





test = np.zeros((ntest, time_input + time_out, S1, S2, V), dtype = np.float32)
counter = 0

for met_variable in met_variables:
    maxx = min_max[f'{met_variable}_max'].item()
    minn = min_max[f'{met_variable}_min'].item()
    den = (maxx-minn)

    validation_path = os.path.join(savepath_met, f"anrf_val_{met_variable}.npy")
    data = np.load(validation_path, mmap_mode = "r")[:, :time_input + time_out, :, :].astype(np.float32)
    if met_variable in ['u10', 'v10']:
        data = (2*(data - minn)/den) - 1
    else:
        data = (data - minn)/ den
    test[:, :, :, :, counter] = data   
    counter = counter + 1
    del data


maxx = min_max[f'rain_combined_max'].item()
minn = min_max[f'rain_combined_min'].item()
den = maxx - minn
validation_path_rainc = os.path.join(savepath_met, f"anrf_val_rainc.npy")
validation_path_rainnc = os.path.join(savepath_met, f"anrf_val_rainnc.npy")
data1 = np.load(validation_path_rainc, mmap_mode = "r")[:, :time_input + time_out, :, :].astype(np.float32)
data2 = np.load(validation_path_rainnc, mmap_mode = "r")[:, :time_input + time_out, :, :].astype(np.float32)
data = data1 + data2
del data1, data2
data = (data - minn)/den
test[:, :, :, :, counter] = data
counter = counter + 1
del validation_path, validation_path_rainc, validation_path_rainnc, data


for variables in emission_variables:
    pollutant = variables[0].split('_')[0]

    maxx = min_max[f'{pollutant}_max'].item()
    minn = min_max[f'{pollutant}_min'].item()
    den = maxx - minn  

    validation_path_v1 = os.path.join(savepath_emissions, f"anrf_val_{variables[0]}.npy")
    validation_path_v2 = os.path.join(savepath_emissions, f"anrf_val_{variables[1]}.npy")
    data1 = np.load(validation_path_v1, mmap_mode = "r")[:, :time_input + time_out, :, :].astype(np.float32)
    data2 = np.load(validation_path_v2, mmap_mode = "r")[:, :time_input + time_out, :, :].astype(np.float32)
    data = data1 + data2
    del data1, data2
    data = (data - minn)/den
    data = np.clip(data, 0, 1)
    test[:, :, :, :, counter] = data
    counter = counter + 1
    del data

for k, met_variable in enumerate(single_variables):

    maxx = min_max[f'{mean_names[k]}_max'].item()
    minn = min_max[f'{mean_names[k]}_min'].item()
    den = maxx - minn 

    validation_path = os.path.join(savepath_emissions, f"anrf_val_{met_variable}.npy")
    data = np.load(validation_path, mmap_mode = "r")[:, :time_input + time_out, :, :].astype(np.float32)
    data = (data - minn)/den
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

os.makedirs(os.path.dirname(cfg.paths.output_file), exist_ok=True)
testing(
    model,
    cfg.inference.forward_steps,
    test_loader,
    cfg.data.time_out,
    cfg.paths.output_file,
    d, V, time_input, lat, long
)


del test_loader
