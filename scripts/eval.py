import os
import numpy as np
import pandas as pd
from scipy import io
from src.utils.config import load_config
from src.utils.metrics import rmse, smape, mfb

# -----------------------
# Load config
# -----------------------

cfg = load_config("configs/eval.yaml")

# -----------------------
# Settings from YAML
# -----------------------

pred_path   = cfg.paths.pred_path
actual_path = cfg.paths.actual_path
save_dir    = cfg.paths.save_dir
f_type   = cfg.data.dataset
time_in  = cfg.data.time_input

cities = cfg.cities

metrics = {
    "rmse": rmse,
    # "smape": smape,
    # "mfb": mfb
}

os.makedirs(save_dir, exist_ok=True)



# -----------------------
# Load prediction
# -----------------------

pred = np.load(pred_path)                 
pred = pred.astype(np.float32)


# -----------------------
# Load actual
# -----------------------

act = np.load(actual_path)                
act = np.transpose(act, (0, 2, 3, 1))     
act = act[..., time_in:]                  


# -----------------------
# Full domain evaluation
# -----------------------

domain_results = {}

for name, fn in metrics.items():
    res = fn(act, pred)                   
    domain_results[name] = np.nanmean(res, axis=(0))  

df_domain = pd.DataFrame(domain_results)
df_domain.to_csv(os.path.join(save_dir, f"{f_type}_domain.csv"), index=False)

print("Saved domain evaluation.")


# -----------------------
# City-wise evaluation
# # -----------------------

# save_dir = os.path.join(cfg.paths.save_dir, "cities")
# os.makedirs(save_dir, exist_ok=True)

# T = pred.shape[-1]

# for city, pts in vars(cities).items():

#     act_list, pred_list = [], []

#     for (lat, lon) in pts:
#         act_list.append(act[:, lat:lat+1, lon:lon+1, :])
#         pred_list.append(pred[:, lat:lat+1, lon:lon+1, :])

#     city_act = np.mean(np.stack(act_list, axis=0), axis=0)
#     city_pred = np.mean(np.stack(pred_list, axis=0), axis=0)

#     rows = []

#     for t in range(T):

#         row = {"hour": t+1}

#         for name, fn in metrics.items():
#             res = fn(city_act, city_pred)      # (N, T)
#             row[name] = np.nanmean(res[:, t])

#         rows.append(row)

#     df = pd.DataFrame(rows)
#     df.to_csv(os.path.join(save_dir, f"{cfg.data.dataset}_{city}.csv"), index=False)

#     print(f"saved {city} results")