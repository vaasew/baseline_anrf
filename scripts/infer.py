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


min_max = io.loadmat(cfg.paths.min_max_file)

max_pm = float(min_max["cpm25_max"])
min_pm = float(min_max["cpm25_min"])

def denorm(x):
    # to denormalize the prediction
    return x * (max_pm - min_pm) + min_pm



def testing(model, forward_steps, test_loader, model_output_steps, name_file, d, V, time_input,lat=52,long=68):
    prediction = torch.zeros((d, lat, long, forward_steps * model_output_steps))
 
    index = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            xx = batch[0].to(device)
            if index%100 == 0:
                print(index)
            fp = xx[:, :time_input, :, :, :].reshape(1, time_input, lat, long, V)
            im1 = model(fp)
            im1 = torch.clamp(im1, min = 1e-6)
            pred = im1

                
            prediction[index, :, :, :] = pred
            index = index + 1
            
    np.save(name_file,denorm(np.array(prediction.cpu().detach().tolist(),dtype=np.float32)))


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

savepath = cfg.paths.input_loc

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

    test_path = os.path.join(savepath, f"{met_variable}.npy")
    data = np.load(test_path, mmap_mode = "r")[:, :time_input , :, :].astype(np.float32)

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


for variable in emission_variables:

    data=np.load(os.path.join(savepath, f"{variable}.npy"), mmap_mode="r")[:, :time_input]

    data = normalize_data(
        data,
        min_max,
        key=variable,
        clip=True
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
    test_loader,
    cfg.data.time_out,
    os.path.join(cfg.paths.output_loc,f'preds.npy'),
    d, V, time_input, lat, long
)


del test_loader
