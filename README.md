 FINAL UPDATED README (READY TO PASTE)
# RDKD 2026 Project: Large-Scale Time Series Forecasting with Clustering

##  Project Overview

This project investigates whether **clustering households improves time series forecasting performance** for large-scale energy consumption data.

We use daily electricity consumption for **17,547 households** across:

- 2023 → Training data (365 days)**
- **2024 → Evaluation data (366 days, leap year)**

##  Objectives

1. Cluster households based on consumption behavior  
2. Build forecasting models:
   - Global model (baseline)
   - Cluster-based models (proposed)
3. Compare performance using **household-level MAE**


##  Important Constraint

All models follow a strict **no data leakage setup**:

- Training uses **only 2023 data**
- Clustering uses **only 2023 features**
- Forecasting uses **recursive prediction**
- 2024 data is used **only for evaluation**


## Repository Structure

📦 RDKD-2026-timeseries-forecasting
├── data/
│   ├── raw/                ← sample_23.csv, sample_24.csv
│   └── processed/          ← engineered features
│
├── src/
│   ├── config.py
│   ├── feature_engineering_daily.py
│   ├── forecasting.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_forecasting.ipynb   ← FINAL MODEL
│
├── outputs/ (ignored in Git)
│
├── report/
├── presentation/
├── diary/
│
├── README.md
├── requirements.txt
└── environment.yml

##  Methodology

### 1. Feature Engineering (Clustering)

From each household's 2023 time series, we extract statistical features:

- Level: mean
- Variability: coefficient of variation
- Shape: skewness, kurtosis
- Seasonality: summer/winter peaks
- Weekly behavior: weekend ratio
- Peaks: number and intensity
- Zeros: inactivity patterns

Reduced to **13 key features** for clustering.

### 2. Clustering (Task 1)

- Algorithm: **K-Means**
- Optimal clusters selected using:
  - Elbow method
  - Silhouette score

Result: **5 clusters of households**

Clusters represent distinct consumption behaviors.

### 3. Forecasting (Task 2)

We compare two approaches:

###  Global Model (Baseline)
- One model trained on all households

###  Cluster-Based Model (Proposed)
- One model per cluster
- Small clusters use global fallback

---

##  Feature Engineering for Forecasting

Each time series is transformed into supervised learning features:

### Lag Features
- lag_1, lag_2, lag_3, lag_5, lag_7, lag_10, lag_14, lag_21, lag_28, lag_30

### Rolling Statistics
- mean (7, 14, 30 days)
- std (7, 14, 30 days)

### Rolling Min/Max
- captures peaks and drops

### Exponential Weighted Average
- emphasizes recent values

### Calendar Features
- day of week, month, week, quarter, weekend

### Cyclical Encoding
- sine/cosine of time features

### Household-Level Features (2023 only)
- yearly mean, std, min, max
- weekday/weekend behavior
- variability metrics

 Total: **41 features**


##  Forecasting Strategy

We use **recursive forecasting**:

- Predict day 1 of 2024
- Feed prediction into history
- Predict day 2
- Continue for all 366 days

 No actual 2024 values are used during prediction.

##  Model Selection

Tested models:
- LightGBM
- XGBoost
- CatBoost
- Ridge
- HistGradientBoosting

 Final model:

**HistGradientBoostingRegressor**

Reason:
- Best MAE performance
- Efficient on large tabular data

##  Performance Optimization

- Implemented **batched forecasting**
- Predictions computed per cluster in parallel
- Reduced runtime from hours → minutes

##  Final Results

###  Main Metric: MAE (Mean Absolute Error)

| Model                     | MAE   |
|--------------------------|--------|
| Global HistGB            | 3.8947 |
| Cluster HistGB (Final)   | 3.7443 |

 **Improvement: +3.86%**
 
##  Per-Cluster MAE

| Cluster | Households | MAE   |
|--------|------------|--------|
| 0      | 5,733      | 6.4462 |
| 1      | 11,561     | 2.4580 |
| 2      | 84         | 1.9913 |
| 3      | 1          | 2.3147 |
| 4      | 168        | 0.9469 |


 Insight:
- Cluster 0 = hardest to predict
- Other clusters more stable

## Key Insights

- Clustering improves forecasting accuracy
- Feature engineering is critical
- Household-level statistics significantly boost performance
- Recursive forecasting is realistic but challenging
- Not all clusters are equally predictable

  
##  Experiments That Did NOT Help

- Longer seasonal lags (35, 42, 56 days)
- These increased MAE → removed from final model

##  Evaluation Method

- MAE computed per household
- Final score = average MAE across households

 Matches assignment requirement

##  How to Run

### Setup


conda env create -f environment.yml
conda activate kdd_ts


### Run notebooks

jupyter notebook notebooks/04_forecasting.ipynb

## 📦 Dataset

Download:
[https://ucloud.univie.ac.at/index.php/s/o5295C8mQo6Jg6m](https://ucloud.univie.ac.at/index.php/s/o5295C8mQo6Jg6m)

Place in: data/raw/

##  Final Conclusion

Cluster-based forecasting using HistGradientBoosting improves performance over a global model, achieving:

 **Final MAE: 3.7443**

This demonstrates that grouping households by behavior enables more accurate time series prediction.

## Author

Fahad Ali Abbasi
University of Vienna – Masters in Computer Science




