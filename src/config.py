# =============================================================================
# config.py
# Configuration file for the Large-Scale Time Series Forecasting project
# =============================================================================

import os
from pathlib import Path

# =============================================================================
# IMPORTANT:  project root path here
# =============================================================================
YOUR_PROJECT_PATH = '/Users/fahad/Documents/MY#Documents/FAHAD ALI/Uni Wien/6th Semester/RDKD/ProjectWORK'

# Convert to Path object
PROJECT_ROOT = Path(YOUR_PROJECT_PATH)

# Verify the path exists
if not PROJECT_ROOT.exists():
    print(f"⚠️  Warning: Path {PROJECT_ROOT} does not exist!")
    print("Please update YOUR_PROJECT_PATH in config.py")
else:
    print(f"✅ Project root set to: {PROJECT_ROOT}")

# =============================================================================
# Data paths
# =============================================================================
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

# Input files
FILE_2023 = RAW_DATA_DIR / 'sample_23.csv'
FILE_2024 = RAW_DATA_DIR / 'sample_24.csv'

# =============================================================================
# Output directories
# =============================================================================
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
CLUSTERING_OUTPUT_DIR = OUTPUT_DIR / 'clustering'
FORECASTING_OUTPUT_DIR = OUTPUT_DIR / 'forecasting'
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / 'evaluation'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# =============================================================================
# Create all directories automatically
# =============================================================================
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 CLUSTERING_OUTPUT_DIR, FORECASTING_OUTPUT_DIR, 
                 EVALUATION_OUTPUT_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created/Verified: {dir_path}")

# =============================================================================
# Project parameters
# =============================================================================
RANDOM_SEED = 42
N_DAYS_2023 = 365
N_DAYS_2024 = 366  # Leap year

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
FIGURE_DPI = 150

# Data quality thresholds
ZERO_THRESHOLD = 0.3  # Max 30% zeros allowed
CONSTANT_THRESHOLD = 0.8  # If 80% of values are same, consider constant

# Clustering parameters
MIN_CLUSTERS = 2
MAX_CLUSTERS = 20
CLUSTERING_METHODS = ['kmeans', 'agglomerative', 'dbscan']

# Feature selection
FEATURES_TO_KEEP = [
    'mean', 'cv', 'skewness', 'kurtosis',
    'weekend_ratio', 'summer_peak', 'winter_peak',
    'trend_slope', 'seasonal_strength_weekly',
    'n_peaks', 'peak_height_ratio',
    'zero_percentage', 'avg_zero_run'
]

print("\n✅ Configuration loaded successfully!")
print(f"📁 Data will be read from: {RAW_DATA_DIR}")
print(f"📁 Outputs will be saved to: {OUTPUT_DIR}")