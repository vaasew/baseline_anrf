# Demo Run Instructions

This guide explains **exactly how to set up and run a demo pipeline** from -
dataset preparation → training → inference → evaluation.

If a **GPU is available**, the **entire pipeline finishes in under ~30 minutes**.

---

#  Important Notes

*  The demo uses **only certain features from April 2016 data**
*  Training epochs is set to **50 epochs**
*  The experiment name is fixed as: **`demo`**
*  All paths and configs are already set in the repo

---

# 1️⃣ Clone the Repository (run_demo branch)

```bash
git clone -b run_demo https://github.com/vaasew/baseline_anrf.git
cd baseline_anrf
```

Make sure you are on the correct branch:

```bash
git branch
# should show: * run_demo
```

---

# 2️⃣ Download Demo Dataset

Download the data from:

👉 [https://drive.google.com/file/d/1kajlQ_FVpaZjoUSdFV_dyd0TVHtcQdpu/view?usp=sharing](https://drive.google.com/file/d/1kajlQ_FVpaZjoUSdFV_dyd0TVHtcQdpu/view?usp=sharing)

The data contains:

* Features (April 2016 only):

  * `pm25`
  * `t2`
  * `rainc`
  * `rainnc`
  * `PM25_e`
  * `PM25_f`
* Normalization file(for pre-processing):

  * `min_max.mat`

---

# 3️⃣ Place Files in Correct Directories (Very Important)

### 🔹 Feature files

All feature files must be named in this format:

```
APRIL_2016_HOURLY_<feature_name>.npy
```

Place them inside:

```
data/raw/
```

Example:

```
data/raw/APRIL_2016_HOURLY_pm25.npy
data/raw/APRIL_2016_HOURLY_t2.npy
...
```

---

### 🔹 Normalization file

Place `min_max.mat` inside:

```
data/stats/
```

Final structure should look like:

```
data/
 ├── raw/
 │    ├── APRIL_2016_HOURLY_pm25.npy
 │    ├── APRIL_2016_HOURLY_t2.npy
 │    ├── ...
 └── stats/
      └── min_max.mat
```

---

# 4️⃣ Run the Demo Pipeline

You can run the full pipeline in **one command**.

---

## ✅ Option A: Local machine


```bash
bash run.sh
```

---

## ✅ Option B: HPC cluster

```bash
qsub run_job.pbs
```

(Modify `run_job.pbs` if your environment paths differ.)

---

# 5️⃣ What the Pipeline Does

Running `run.sh` or `run_job.pbs` will automatically execute the following stages:

---

## 🔹 (1) Dataset preparation

Runs:

```
prepare_dataset.py
```

Using:

* `prepare_dataset.yaml`
* Raw data from `data/raw/`

Creates time-series samples and stores train and validation sets for each feature in:

```
data/met/
data/emissions/
```

Each contains `train_<feature_name>.npy` and `val_<feature_name>.npy`.

---

## 🔹 (2) Model training

Runs:

```
train.py
```

Using:

* `train.yaml`

Outputs:

* Model checkpoints

  ```
  experiments/demo/checkpoints/*.pt
  ```
* Training logs

  ```
  experiments/demo/logs/
  ```

---

## 🔹 (3) Inference on validation set

Runs:

```
infer.py
```

Using:

* `infer.yaml`

Outputs:

```
experiments/demo/infer/val.npy
```

(predictions on validation data)

---

## 🔹 (4) Evaluation

Runs:

```
eval.py
```

Using:

* `eval.yaml`

Outputs:

### Whole-domain metrics

```
experiments/demo/eval_results/val_domain.csv
```

### City-wise metrics

```
experiments/demo/eval_results/cities/*.csv
```

---


# Demo Run Complete

After successful execution, you should have:

* trained model checkpoints
* validation predictions
* domain-level evaluation CSV
* city-wise evaluation CSVs

inside:

```
experiments/demo/
```

---
