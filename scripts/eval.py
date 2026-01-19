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
ntest    = cfg.data.ntest
time_in  = cfg.data.time_input

cities = cfg.cities

metrics = {
    "rmse": rmse,
    "smape": smape,
    "mfb": mfb
}

os.makedirs(save_dir, exist_ok=True)


# -----------------------
# Load prediction
# -----------------------

pred = np.load(pred_path)                  # (N, H, W, T)
pred = pred.astype(np.float32)


# -----------------------
# Load actual
# -----------------------

act = np.load(actual_path)                 # (N, T_total, H, W)
act = np.transpose(act, (0, 2, 3, 1))      # (N, H, W, T_total)
act = act[..., time_in:]                   # keep only forecast horizon


# -----------------------
# Full domain evaluation
# -----------------------

domain_results = {}

for name, fn in metrics.items():
    res = fn(act, pred)                    # (N, H, W, T)
    domain_results[name] = np.nanmean(res, axis=(0,1,2))   # mean over N,H,W

df_domain = pd.DataFrame(domain_results)
df_domain.to_csv(os.path.join(save_dir, f"{f_type}_domain.csv"), index=False)

print("Saved domain evaluation.")


# -----------------------
# City-wise evaluation
# -----------------------

rows = []

for city, pts in cities.items():

    act_list, pred_list = [], []

    for (lat, lon) in pts:
        act_list.append(act[:, lat:lat+1, lon:lon+1, :])
        pred_list.append(pred[:, lat:lat+1, lon:lon+1, :])

    city_act = np.mean(np.stack(act_list, axis=0), axis=0)
    city_pred = np.mean(np.stack(pred_list, axis=0), axis=0)

    city_row = {"city": city}

    for name, fn in metrics.items():
        res = fn(city_act, city_pred)                   # (N,1,1,T)
        city_row[name] = np.nanmean(res, axis=(0,1,2))  # (T,)

    rows.append(city_row)

# explode into per-hour columns
final_rows = []

T = rows[0]["rmse"].shape[0]

for r in rows:
    for t in range(T):
        final_rows.append({
            "city": r["city"],
            "hour": t+1,
            "rmse": r["rmse"][t],
            "smape": r["smape"][t],
            "mfb": r["mfb"][t]
        })

df_city = pd.DataFrame(final_rows)
df_city.to_csv(os.path.join(save_dir, f"{f_type}_cities.csv"), index=False)
