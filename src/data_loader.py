# =============================================================================
# data_loader.py
# Data loading and initial exploration functions
# Team [6]
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to Python path
# This allows imports to work regardless of where script is run from
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now we can import config
from src.config import (
    FILE_2023, FILE_2024, FIGURES_DIR, RANDOM_SEED,
    N_DAYS_2023, N_DAYS_2024, PLOT_STYLE, FIGURE_DPI
)

# Set plotting style
plt.style.use(PLOT_STYLE)


def load_data(verbose=True):
    """
    Load the 2023 and 2024 datasets.
    
    Parameters:
    -----------
    verbose : bool, whether to print loading information
    
    Returns:
    --------
    df_2023, df_2024 : pandas DataFrames
    """
    if verbose:
        print("="*60)
        print("LOADING DATA")
        print("="*60)
    
    # Check if files exist
    if not FILE_2023.exists():
        raise FileNotFoundError(f"❌ File not found: {FILE_2023}\n"
                               f"Please place sample_23.csv in: {FILE_2023.parent}")
    
    if not FILE_2024.exists():
        raise FileNotFoundError(f"❌ File not found: {FILE_2024}\n"
                               f"Please place sample_24.csv in: {FILE_2024.parent}")
    
    # Load 2023 data
    df_2023 = pd.read_csv(FILE_2023)
    if verbose:
        print(f"\n✅ Loaded 2023 data:")
        print(f"   - Shape: {df_2023.shape}")
        print(f"   - Number of households: {df_2023.shape[0]}")
        print(f"   - Number of days (2023): {df_2023.shape[1] - 1}")
    
    # Load 2024 data
    df_2024 = pd.read_csv(FILE_2024)
    if verbose:
        print(f"\n✅ Loaded 2024 data:")
        print(f"   - Shape: {df_2024.shape}")
        print(f"   - Number of households: {df_2024.shape[0]}")
        print(f"   - Number of days (2024 leap year): {df_2024.shape[1] - 1}")
    
    return df_2023, df_2024


def quick_check(df_2023, df_2024):
    """
    Quick check of data integrity.
    
    Returns:
    --------
    dict with basic info
    """
    info = {
        'n_households_2023': df_2023.shape[0],
        'n_days_2023': df_2023.shape[1] - 1,
        'n_households_2024': df_2024.shape[0],
        'n_days_2024': df_2024.shape[1] - 1,
        'missing_2023': df_2023.isnull().sum().sum(),
        'missing_2024': df_2024.isnull().sum().sum(),
        'ids_match': set(df_2023['ID']) == set(df_2024['ID'])
    }
    
    return info


# Test the module when run directly
if __name__ == "__main__":
    print("Testing data_loader.py...")
    print(f"Project root: {project_root}")
    print(f"Data directory: {FILE_2023.parent}")
    
    try:
        df_23, df_24 = load_data()
        info = quick_check(df_23, df_24)
        print("\n📊 Quick Check Results:")
        for key, value in info.items():
            print(f"   - {key}: {value}")
        print("\n✅ data_loader.py is working correctly!")
    except Exception as e:
        print(f"❌ Error: {e}")