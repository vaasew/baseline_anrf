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

  stored in the form - 
  `APRIL_2016_HOURLY_<feature_name>.npy`

* Normalization file(for pre-processing):

  * `min_max.mat`

---

# 3️⃣ Place Files in Correct Directories

### 🔹 Feature files

Place all features of the form -

```
APRIL_2016_HOURLY_<feature_name>.npy
```

inside:

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

Before running the demo, **ensure your environment is properly set up**
(required packages installed, correct Python environment activated, CUDA configured if using a GPU, paths updated if needed).

Both options below execute the **entire pipeline end-to-end**.

---

## ✅ Option A: Run locally

Activate your environment, then run:

```bash
bash run.sh
```

---

## ✅ Option B: Run on an HPC cluster

Submit the job script:

```bash
qsub run_job.pbs
```

Before submitting, **edit `run_job.pbs` as per your system setup**
(module loads, environment activation, paths, GPU settings, etc.).
The provided file is only a **boilerplate template**.

---




# 5️⃣ What the Pipeline Does

Running `run.sh` or `run_job.pbs` will automatically execute the following stages:

**dataset preparation → training → inference → evaluation**

---

## 🔹 (1) Dataset preparation

Runs:

```
prepare_dataset.py
```

Using:

* Configuration from:

  ```
  prepare_dataset.yaml
  ```
* Raw feature files from:

  ```
  data/raw/
  ```

Creates time-series samples and stores **training and validation sets for each feature** in:

```
data/met/
data/emissions/
```

Each directory contains:

```
train_<feature_name>.npy
val_<feature_name>.npy
```

---

## 🔹 (2) Model training

Runs:

```
train.py
```

Using:

* Configuration from:

  ```
  train.yaml
  ```
* Training and validation samples from:

  ```
  data/met/
  data/emissions/
  ```

Outputs:

* Model checkpoints:

  ```
  experiments/demo/checkpoints/*.pt
  ```
* Training logs:

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

* Configuration from:

  ```
  infer.yaml
  ```
* Trained model checkpoint:

  ```
  experiments/demo/checkpoints/demo_model_ep49.pt
  ```

Outputs:

```
experiments/demo/infer/val.npy
```

(Stored validation set predictions)

---

## 🔹 (4) Evaluation

Runs:

```
eval.py
```

Using:

* Configuration from:

  ```
  eval.yaml
  ```
* Model predictions on the validation set (first 10 hours):

  ```
  experiments/demo/infer/val.npy
  ```
* Reference validation targets (last 16 hours):

  ```
  data/met/val_pm25.npy
  ```

This step computes error metrics and generates the final evaluation CSV files.

---

### ✔️ Whole-domain metrics

```
experiments/demo/eval_results/val_domain.csv
```

### ✔️ City-wise metrics

```
experiments/demo/eval_results/cities/*.csv
```

Each city has a different csv. Ex-

```
experiments/demo/eval_results/cities/delhi.csv
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
