# RDKD 2026 Project: Large-Scale Time Series Forecasting with Clustering

## 📋 Project Overview

This project explores clustering as a preprocessing step for large-scale time series forecasting. Using daily energy consumption data for 17,547 households over two years (2023-2024), we:

1. **Cluster households** based on their consumption patterns using 13 extracted features
2. **Train forecasting models** at both global and cluster levels
3. **Compare performance** to demonstrate the value of clustering

The dataset contains daily energy consumption for 2023 (training) and 2024 (testing, leap year with 366 days).
## 🗂️ Repository Structure
├── 📁 data/
│ ├── 📁 raw/ ← Place sample_23.csv and sample_24.csv here
│ └── 📁 processed/ ← Cleaned data and extracted features
│
├── 📁 src/ ← Python modules
│ ├── config.py ← Configuration and paths
│ ├── data_loader.py ← Data loading functions
│ ├── preprocessing.py ← Data cleaning and normalization
│ ├── feature_engineering.py ← Feature extraction from time series
│ └── utils.py ← Utility functions
│
├── 📁 notebooks/ ← Jupyter notebooks for analysis
│ ├── 01_data_exploration.ipynb
│ ├── 02_feature_extraction.ipynb
│ ├── 03_clustering.ipynb
│ └── 04_forecasting.ipynb
│
├── 📁 outputs/
│ ├── 📁 clustering/ ← Cluster assignments and models
│ ├── 📁 forecasting/ ← Trained forecasting models
│ ├── 📁 evaluation/ ← Performance metrics
│ └── 📁 figures/ ← Generated plots
│
├── 📁 report/ ← Final report (PDF)
├── 📁 presentation/ ← 10-minute presentation
├── 📁 diary/ ← Individual research diaries
│
├── .gitignore
├── environment.yml ← Conda environment
├── README.md ← This file
└── requirements.txt ← Python dependencies

## Getting Started

### Prerequisites
- Python 3.11+
- Conda (recommended) or pip
- Git
- 
### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/[YOUR_USERNAME]/RDKD-2026-timeseries-forecasting.git
   cd RDKD-2026-timeseries-forecasting
   
## Download the dataset:
Download sample_23.csv and sample_24.csv from this link
Place both files in data/raw/
## Set up the environment:
Using Conda (recommended)
bash
conda env create -f environment.yml
conda activate kdd_ts
## pip install -r requirements.txt
Verify setup:
bash
python src/test_setup.py

## Features Extracted
I extracted 26 features per household, reduced to 13 key features for clustering:
Category	               Features	                                         Description
Level	                    mean	                                          Average daily consumption
Variability	               cv	                                            Coefficient of variation (std/mean)
Shape	                  skewness,                                         kurtosis	Distribution shape
Weekly                 	weekend_ratio	                                    Weekend vs weekday consumption
Seasonal	              summer_peak, winter_peak	                        Seasonal patterns
Trend	                  trend_slope	                                      Year-long trend
Peaks                  	n_peaks,peak_height_ratio	                        Peak behavior
Zeros	                  zero_percentage, avg_zero_run	                    Absence patterns

## PCA analysis shows 10 components capture 95% of variance, confirming feature quality.

## Methodology
Phase 1: Data Exploration & Preprocessing

Load and inspect 17,547 households × 365 days
Handle problematic series (zeros, constant values)
Normalize time series for fair comparison

Phase 2: Feature Extraction
Extract 26 statistical, temporal, peak, and zero features
Handle missing values via median imputation
Select 13 non-redundant features for clustering

Phase 3: Clustering (Task 1)
Determine optimal k using elbow method and silhouette score
Apply K-means clustering
Profile and interpret clusters

Phase 4: Forecasting (Task 2)
Global model: Single model trained on all households
Cluster models: Separate models per cluster
Evaluate using MAE on 2024 data
Compare performance improvements

## Key Findings (To Be Updated)
Finding	                          Value
Total households	                17,547
Problematic series	              610 (3.5%)
Features extracted              	26
Optimal PCA components	          10 (95% variance)
Selected features	                13

## Results from clustering and forecasting will be added here.

## Usage
## Run Data Exploration
jupyter notebook notebooks/01_data_exploration.ipynb

## Extract Features
jupyter notebook notebooks/02_feature_extraction.ipynb

## Perform Clustering
jupyter notebook notebooks/03_clustering.ipynb

