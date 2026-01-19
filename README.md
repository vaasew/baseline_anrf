# 🌍 Air Quality Forecasting Challenge — Baseline Repository

This repository provides the official **baseline code structure, data pipeline, and evaluation interface** for the Air Quality Forecasting Competition.

Participants are expected to build upon this repository to develop their own models while preserving the overall project structure and inference interface.

---

# 📌 Problem Overview

The objective of this challenge is to **forecast PM2.5 concentration fields for the next 16 hours**, given the **first 10 hours of spatio-temporal input data**.

Each sample contains:

* **10 hours of input**
* **16 hours of target output**

Regardless of your internal rollout strategy (single-step, autoregressive, multi-step, etc.), **final evaluation will always be done on 16 hours of prediction.**

---

# 🧩 Available Input Features

The dataset provides three groups of variables. **All are optional.**
Participants may use any subset by editing the lists in `train.yaml` and `infer.yaml`.

---

## 🌦 Meteorological Variables (`met_variables`)

```
['pm25','q2','t2','u10','swdown','pblh','v10','psfc','rain']
```

* `pm25`   – PM2.5 concentration
* `q2`     – 2 m specific humidity
* `t2`     – 2 m temperature
* `u10`    – 10 m zonal wind
* `v10`    – 10 m meridional wind
* `swdown` – downward shortwave radiation
* `pblh`   – planetary boundary layer height
* `psfc`   – surface pressure
* `rain`   – combined convective + non-convective rainfall

> Rain is internally constructed from `rainc` and `rainnc`.

---

## 🏭 Emission Variables (`emission_variables`)

Each emission input consists of **two components that are summed before normalization**.

```
[
  ['PM25_e','PM25_finn'],
  ['NH3_e','NH3_finn'],
  ['SO2_e','SO2_finn'],
  ['NOx_e','NOx_finn']
]
```

---

## 🌱 Single Emission Variables (`single_variables`)

```
['NMVOC_e','NMVOC_finn','bio']
```

---

# 📊 Dataset Description

##  Training Data

Training samples are constructed from the following months in **2016**:

* April 2016
* July 2016
* October 2016
* December 2016

---

## Test Data

Evaluation will be performed on **data from multiple months from 2017**.

There are **two test sets**:

### 🔓 Public Test Set

* Smaller subset
* Participants can see their evaluation scores
* Feeds into a **public leaderboard**
* Leaderboard updates once every 1–2 days

### 🔒 Private Hidden Test Set

* Never accessible during the competition
* Final rankings are based **only** on this set
* Dataset and results will be released **after the competition ends**

---

# Repository Structure

Participants must preserve the repository structure:

```
configs/
data/
experiments/
scripts/
src/
```

Your implementation may modify or extend internal files, but **the inference entrypoint must remain compatible.**

---

# Expected Submission Format

Participants must submit their **full project directory**, based on this repository.

Your submission **must include**:

* Trained model weights
* Updated `infer.py`
* Updated `infer.yaml`
* Model definitions
* Full preprocessing pipeline

We will **not** retrain models.

We will:

1. Modify only **data-related parameters** inside `infer.yaml`
2. Run:

   ```
   python scripts/infer.py
   ```
3. Evaluate the produced output file.

---

# Inference Interface (Strict Requirement)

Your inference must obey the interface defined in `infer.yaml`.

Example:

```yaml
paths:
  savepath_emissions: data/emissions/
  savepath_met: data/met/
  output_file: experiments/baseline/eval/val.npy

data:
  dataset: val     
  ntest: 528
  total_time: 26
  time_input: 10
  time_out: 16
```

We will change:

* `dataset`
* `ntest`
* `output_file`
* data paths

Your code **must work without modification.**

---

# Expected Output

Your `infer.py` must generate a NumPy file:

```
(output_file).npy
```

with shape:

```
(N, H, W, 16)
```

where:

* `N` = number of samples
* `H, W` = spatial dimensions
* `16` = forecast horizon

This file will be directly passed to the evaluation pipeline.

---

#  Forecasting Rule (Important)

Every test sample:

* always contains **10 input hours**
* always requires **16 output hours**

Even if your model predicts step-by-step internally, the **final submitted output must contain 16 hours.**

---

# 🧪 Feature Selection

You are **not required to use all features.**

You may remove any variables simply by editing:

```yaml
met_variables
emission_variables
single_variables
```

in both:

* `configs/train.yaml`
* `configs/infer.yaml`

The data loader will automatically adapt.

---

# 📚 FAQ

Please refer to the **FAQ section** for:

* Data loading behavior
* Normalization logic
* Rain construction
* Emission preprocessing
* Multi-step inference
* Evaluation format
* Common runtime issues

---

# 🏁 Final Notes

* You may restructure internal modules freely
* The only strict contract is:

> `infer.yaml` and `infer.py` must run correctly and produce the required output file.

---

Good luck, and we look forward to your solutions.

evironment compatibility?

standardise docker environment - listing libraries allowed + a forum where participants can raise requests to include certain libraries which will be reviewed and decided on inclusion
or
test sets given to input time data given to participants and take output time data as deliverable and report results(only for test1)
