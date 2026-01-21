# Country-Level PM2.5 Concentration Forecasting

This repository provides the official **baseline code structure, raw dataset format, data pipeline, and evaluation interface** for the Air Quality Forecasting Competition.

Participants are expected to build upon this repository to develop their own models while **preserving the overall project structure and inference interface.**

The objective of the competition is to design scientific machine learning models that can accurately forecast short-term **PM2.5 concentration fields over India**.

---

# 1. Problem Overview

This section introduces the scientific motivation and forecasting objective.

Particulate Matter (PM2.5) is a major public-health concern in India, where concentrations frequently exceed national standards by more than an order of magnitude. PM2.5 particles penetrate deep into the lungs and are linked to serious diseases including stroke, ischemic heart disease, and Chronic Obstructive Pulmonary Disease (COPD).

PM2.5 enters the atmosphere through:
- **Direct emissions** (combustion, dust storms, biomass burning)
- **Secondary chemical formation** (e.g., SO₂ forming sulfate aerosols)

Once emitted, PM is strongly influenced by **meteorology** through transport, vertical mixing, chemical transformation, and wet deposition.

The ability to forecast PM2.5 concentrations enables early warnings, public-health protection, and policy planning. Surrogate AI models for chemical transport simulations can drastically reduce inference time while retaining physical relevance.

The goal of this competition is to forecast **future PM2.5 concentration fields** given multi-source spatio-temporal inputs.

---

# 2. Core AI Tasks

This section clarifies the modeling directions encouraged in the competition.

Participants may adopt any modeling paradigm. However, emphasis is placed on modern scientific ML approaches, including:

- **Neural operator models** (FNO, AFNO, UNO, GNO, etc.)
- **Physics-informed or physics-guided models**
- **Fine-tuning large pretrained Earth-system models** (Aurora, ClimaX, foundation weather models, operator learners)

Models post-2020 are preferred, particularly operator-learning approaches.

Models may be single-step or multi-step, autoregressive or direct multi-horizon, deterministic or probabilistic, purely data-driven or physics-augmented.

---

# 3. Dataset Overview and Organization

This section explains what data is provided and how it is structured.

The dataset consists of **hourly gridded fields over India**, aligned on the same spatial grid and timestamps. The data is provided as **raw feature arrays**, not pre-constructed ML samples.

## 3.1 Raw Data Format

- Each feature is stored independently for each month as a NumPy array:
```
APRIL_16_HOURLY_pm25.npy  
APRIL_16_HOURLY_t2.npy  
APRIL_16_HOURLY_rainc.npy  
<Month>_<YY>_HOURLY_<feature_name>.npy  
```
- Each month also includes:
```
APRIL_16_HOURLY_times.npy
<Month>_<YY>_HOURLY_times.npy  
```
  which contains the timestamp for each hourly sample.

- All features and time arrays are hourly, aligned in time, and aligned on the same spatial grid. 

- Latitude and longitude grids are provided separately in:
```
lat_long.npy
```
  which spans for the (140,124) spatial grid of the samples stating the latitude and longitude of each grid point.

## 3.2 Constructing Training Samples

Participants are expected to construct their own **training and validation time-series samples** from the raw hourly arrays.

A reference pipeline is provided:
```
prepare_dataset.py  
prepare_dataset.yaml  
```
This demonstrates discarding spin-up hours, building sliding windows, generating train/validation splits, and saving time-series samples for model training.

---

# 4. Input Features

The dataset provides three broad input groups:

1. **PM2.5 concentrations**  
2. **Meteorological variables**  
3. **Emission variables**

All features are optional. Participants may use any subset or engineered combinations.

## 4.1 PM2.5 concentrations

| Feature | Description |
|--------|-------------|
| pm25 | Surface PM2.5 concentration (µg/m³). Used both as an input context variable and the forecast target. |

This is the only prediction target.

## 4.2 Meteorological Variables

met_variables = ['q2','t2','u10','v10','swdown','pblh','psfc','rainc','rainnc']

These variables control transport, chemistry, dilution, and wet removal.

| Feature  | Description                                                            |
| -------- | ---------------------------------------------------------------------- |
| `q2`     | 2-m specific humidity (kg/kg).                                         |
| `t2`     | 2-m air temperature (K).                                               |
| `u10`    | 10-m zonal wind (m/s). Controls horizontal transport.                  |
| `v10`    | 10-m meridional wind (m/s). Controls vertical transport.               |
| `swdown` | Downward shortwave radiation (W/m²). Proxy for photochemical activity. |
| `pblh`   | Planetary boundary layer height (m). Governs vertical mixing.          |
| `psfc`   | Surface pressure (Pa).                                                 |
| `rainc`  | Accumulated convective precipitation (mm).                             |
| `rainnc` | Accumulated non-convective precipitation (mm).                         |


Recommended: rain = rainc + rainnc


## 4.3 Emission Variables

This section describes the emission-related inputs provided in the dataset. These variables represent **surface emission fluxes** that directly control the formation and variability of PM2.5 through primary release and secondary chemical pathways.

The emission inventory is split into two major source categories:

- **EDGAR-based anthropogenic emissions (`*_e`)**  
  Derived from the *Emissions Database for Global Atmospheric Research (EDGAR)*, these represent emissions from human activities such as transportation, industry, power generation, residential combustion, and agriculture.

- **FINN-based biomass burning emissions (`*_finn`)**  
  Derived from the *Fire INventory from NCAR (FINN)*, these represent emissions from open biomass burning, including forest fires, crop residue burning, and grassland fires.


---

### Emission components provided

**Primary PM2.5**
- `PM25_e`    – Anthropogenic primary PM2.5 emissions (EDGAR)  
- `PM25_finn` – Biomass burning primary PM2.5 emissions (FINN)

**Ammonia (NH₃)**
- `NH3_e`     – Anthropogenic ammonia emissions (EDGAR; agriculture, livestock, waste, industry)  
- `NH3_finn`  – Biomass burning ammonia emissions (FINN)

**Sulfur dioxide (SO₂)**
- `SO2_e`     – Anthropogenic sulfur dioxide emissions (EDGAR; power plants, refineries, industry)  
- `SO2_finn`  – Biomass burning sulfur dioxide emissions (FINN)

**Nitrogen oxides (NOₓ)**
- `NOx_e`     – Anthropogenic nitrogen oxides emissions (EDGAR; transport, combustion, industry)  
- `NOx_finn`  – Biomass burning nitrogen oxides emissions (FINN)

**Non-methane volatile organic compounds (NMVOCs)**
- `NMVOC_e`    – Anthropogenic NMVOCs (EDGAR; solvents, fuels, industry)  
- `NMVOC_finn` – Biomass burning NMVOCs (FINN)

**Biogenic**
- `bio` – Biogenic isoprene emissions from natural vegetation (provided independently, not from EDGAR or FINN)

---

### Recommended emission construction

For most modeling setups, it is recommended to aggregate anthropogenic and biomass burning sources into unified chemical drivers:

```
PM25 = PM25_e + PM25_finn
NH3 = NH3_e + NH3_finn
SO2 = SO2_e + SO2_finn
NOx = NOx_e + NOx_finn
```


NMVOC emissions are recommended to be **kept separate** (`NMVOC_e`, `NMVOC_finn`) to allow the model to distinguish between anthropogenic and fire-driven chemical regimes.

Biogenic isoprene (`bio`) is provided independently and should be treated as a separate natural emission driver.


Emission variables are **major controls on PM2.5 formation and variability**, governing both direct particulate release and secondary aerosol production. Incorporating these features is strongly recommended.


# 5. Training Data

Training data is provided from 2016 WRF-Chem Simulation Data for the months:

April 2016  
July 2016  
October 2016  
December 2016  

Only this data may be used for training and validation.

- Raw data is hourly  
- First 48 hours of each month should be discarded(spin-up time)  
- All features share a common grid or spatial distribution of shape (140,124)
- Lazy loading is recommended  

Dataset helper notebook:

notebooks/dataset_helper.ipynb

---

# 6. Test Sets and Evaluation Protocol

This section explains how evaluation is conducted.

-Evaluation is performed on certain months of **2017** WRF Chem Simulation data. 
-Two test sets(with all features) are released in **input-only form**.

## Test Set 1 (Public)

- Inputs are released  
- Participants upload predictions  
- Evaluation scores are returned  
- Feeds into a **public leaderboard**  
- Updates once per day  
- Submission limit: once daily  
- Intended as a proxy validation set 

## Test Set 2 (Final)

- Inputs are released  
- Participants upload predictions  
- **No evaluation feedback** during the competition  
- Final rankings computed after the competition ends  
- Determines official standings 

---

## Anti-Leakage Policy for the Test Sets

- This section explains test-time protection mechanisms against training on the Test Set inputs.
- Both test sets contain intentionally corrupted samples excluded from evaluation. 
- Any attempt to train, fine-tune, self-supervise, or adapt on test inputs will propagate corrupted signals and severely degrade performance.(Particularly pertaining to participants training smaller horizon autoregressive models) 
- Rankings are computed only on a clean undisclosed subset of the corresponding Test Sets.

---

# 7. Submission Format

Participants must submit:

preds.npy

with shape:

(N, H, W, 16)

Each sample has 10 hours of input context. Evaluation is always on 16 future hours.

Evaluation code and pipeline(will give you an idea of how the metrics are computed for the spatial-temporal distribution) can be found in:

```
src/utils/metrics.py  
scripts/eval.py  
configs/eval.yaml  
```

---

# 8. Baseline Training and Demo Run

## 8.1 Baseline Compute Configuration and Usage

GPU: NVIDIA A100 (40 GB)  
Peak GPU reserved: ~3.3 GB  
Peak GPU allocated: ~2.8 GB  
CPU: Intel Xeon Platinum 8358  
Peak RAM: ~82 GB  

Each feature ~4.5 GB. Full training + validation set with runtime tensors ~82 GB.

## 8.2 Baseline Training Details

Total trainable parameters: 0.23M
Batch size: 4  
Epochs: 500  
Loss: time-weighted relative L2 norm
Average epoch time: ~37 s  
Total training time: ~5.1 h  

Code:

models/baseline_model.py  
scripts/train.py  
configs/train.yaml  

## 8.3 Baseline Results

Baseline benchmark results: (link)

All reported numbers correspond only to the provided baseline implementation as given in the `main` branch of this repository.


## 8.4 Running a Small End-to-End Demo

If you would like to run a **lightweight demonstration of the complete pipeline** — covering dataset preparation, model training, inference, and evaluation — using the **same architecture as the baseline**, but trained on a **smaller subset of data and fewer epochs**, please refer to: `run_demo.md`.


# 9. Helper Notebooks

``notebooks/training_tips.ipynb``

* Some generic(maybe not so generic) training tips.



---
