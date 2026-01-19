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



# Feature Set, Data Interface, and Evaluation Rules

This section describes the available input features, how feature selection is controlled, how preprocessing may be modified, how test data is formatted, and how inference is performed during official evaluation.

---

## Feature Selection and Control

Participants are free to keep, drop, or reorder features by editing only the following fields in both `train.yaml` and `infer.yaml`:

features:
  V: 16  
  met_variables: [...]  
  emission_variables: [...]  
  single_variables: [...]  

The default data loader is fully driven by these lists. Removing any feature from these lists automatically removes it from the model input. No other data-loading code changes are required.

---

## Preprocessing and Normalization

The baseline code uses per-feature min-max normalization based on:

data/stats/min_max.mat

Participants are completely free to change normalization or standardization, apply custom preprocessing, add feature engineering, or use learned or statistical scaling. The only requirement is that your `infer.py` must correctly load and preprocess the official test data format.

---

## Test Data Format

All test datasets will always be distributed such that each feature is stored in a separate NumPy file. Meteorological variables are stored in `savepath_met` and emission variables are stored in `savepath_emissions`.

File naming format:

<dataset_name>_<feature_name>.npy

Example:

val_pm25.npy  
val_q2.npy  
val_PM25_e.npy  
val_PM25_finn.npy  

The dataset_name is controlled from:

data:
  dataset: val

in `infer.yaml`.

During official evaluation, only this field will be changed (for example: `test1`, `test2`). Your inference pipeline must rely on this naming convention.

---

## Inference Interface and Evaluation

During evaluation, the organizers will modify only the following fields in `infer.yaml`:

paths:
  savepath_emissions: data/emissions/  
  savepath_met: data/met/  
  output_file: experiments/baseline/eval/preds.npy  

data:
  dataset: val  
  ntest: 528  
  total_time: 26  
  time_input: 10  
  time_out: 16  

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

Regardless of your model design (single-step, multi-step, autoregressive, etc.), evaluation is always performed on 16-hour forecasts.


---

## Submission Requirement

Participants must submit their full project directory in the same structure as this repository. The following must be synchronized with your best model:

- infer.py  
- infer.yaml  
- model definitions  
- preprocessing pipeline  

During evaluation, organizers will replace only dataset-related paths and sizes in `infer.yaml` and then directly execute:

python scripts/infer.py

All predictions will be generated exclusively through this interface.

---

## Important Notes

* Test sets will always contain exactly 10 hours of input and Final evaluation is always on 16-hour predictions. 

* The final results are based solely on the hidden test set which would be made public along with the results after the competition deadline.

* You may restructure internal modules freely, the only strict contract is:

> `infer.yaml` and `infer.py` must run correctly and produce the required output file.

* Please refer to the FAQ section for data-loading clarifications, testing rules, and evaluation policies.


---

Good luck, and we look forward to your solutions.

evironment compatibility?

standardise docker environment - listing libraries allowed + a forum where participants can raise requests to include certain libraries which will be reviewed and decided on inclusion

---
or
---

test sets with input time data given to participants and take output time data as deliverable and report results(only for test1)
