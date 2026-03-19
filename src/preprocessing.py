# =============================================================================
# preprocessing.py
# Data preprocessing functions (missing values, normalization)
# Team [6]
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import PROCESSED_DATA_DIR, FIGURES_DIR, RANDOM_SEED, FIGURE_DPI


def identify_problematic_series(df, id_col='ID', zero_threshold=0.5, constant_threshold=0.8):
    """
    Identify problematic time series (all zeros, constant, too many zeros).
    
    Parameters:
    -----------
    df : pandas DataFrame
    id_col : str, name of ID column
    zero_threshold : float, maximum fraction of zeros allowed
    constant_threshold : float, minimum fraction of values equal to mode to be considered constant
    
    Returns:
    --------
    problematic_ids : list, IDs of problematic series
    report : dict, detailed report
    """
    print("\n" + "="*60)
    print("IDENTIFYING PROBLEMATIC SERIES")
    print("="*60)
    
    data_cols = [col for col in df.columns if col != id_col]
    n_days = len(data_cols)
    
    problematic_ids = []
    report = {
        'all_zeros': [],
        'mostly_zeros': [],
        'constant': [],
        'negative_values': []
    }
    
    for idx in df.index:
        household_id = df.loc[idx, id_col]
        series = df.loc[idx, data_cols].values
        
        # Check for negative values
        if np.any(series < 0):
            report['negative_values'].append(household_id)
            problematic_ids.append(household_id)
            continue
        
        # Check if all zeros
        if np.all(series == 0):
            report['all_zeros'].append(household_id)
            problematic_ids.append(household_id)
            continue
        
        # Check if too many zeros
        zero_fraction = np.sum(series == 0) / n_days
        if zero_fraction > zero_threshold:
            report['mostly_zeros'].append(household_id)
            problematic_ids.append(household_id)
            continue
        
        # Check if constant (all values equal or nearly equal)
        # Using mode frequency as a simple measure
        unique_vals, counts = np.unique(series.round(2), return_counts=True)
        mode_fraction = np.max(counts) / n_days
        
        if mode_fraction > constant_threshold:
            report['constant'].append(household_id)
            problematic_ids.append(household_id)
            continue
    
    # Print report
    print(f"\n📊 Problematic series summary:")
    print(f"   - All zeros: {len(report['all_zeros'])} households")
    print(f"   - Mostly zeros (>={zero_threshold*100}% zeros): {len(report['mostly_zeros'])} households")
    print(f"   - Constant/near constant: {len(report['constant'])} households")
    print(f"   - Negative values: {len(report['negative_values'])} households")
    print(f"   - TOTAL problematic: {len(problematic_ids)} households")
    
    return problematic_ids, report


def impute_missing_values(df, id_col='ID', method='linear'):
    """
    Impute missing values in time series.
    
    Parameters:
    -----------
    df : pandas DataFrame
    id_col : str, name of ID column
    method : str, imputation method ('linear', 'ffill', 'bfill', 'mean')
    
    Returns:
    --------
    df_imputed : pandas DataFrame with imputed values
    """
    print(f"\n{'='*60}")
    print(f"IMPUTING MISSING VALUES (method: {method})")
    print(f"{'='*60}")
    
    df_imputed = df.copy()
    data_cols = [col for col in df.columns if col != id_col]
    
    missing_before = df[data_cols].isnull().sum().sum()
    print(f"   - Missing values before imputation: {missing_before}")
    
    if method == 'linear':
        # Linear interpolation for each household
        df_imputed[data_cols] = df_imputed[data_cols].interpolate(method='linear', axis=1, limit_direction='both')
    elif method == 'ffill':
        # Forward fill then backward fill
        df_imputed[data_cols] = df_imputed[data_cols].fillna(method='ffill', axis=1)
        df_imputed[data_cols] = df_imputed[data_cols].fillna(method='bfill', axis=1)
    elif method == 'mean':
        # Fill with household mean
        for idx in df_imputed.index:
            series = df_imputed.loc[idx, data_cols]
            household_mean = series.mean()
            df_imputed.loc[idx, data_cols] = series.fillna(household_mean)
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    missing_after = df_imputed[data_cols].isnull().sum().sum()
    print(f"   - Missing values after imputation: {missing_after}")
    
    return df_imputed


def normalize_series(df, id_col='ID', method='zscore'):
    """
    Normalize time series to remove scale differences.
    
    Parameters:
    -----------
    df : pandas DataFrame
    id_col : str, name of ID column
    method : str, normalization method ('zscore', 'minmax', 'robust')
    
    Returns:
    --------
    df_norm : pandas DataFrame with normalized values
    norm_params : dict, normalization parameters for inverse transform
    """
    print(f"\n{'='*60}")
    print(f"NORMALIZING TIME SERIES (method: {method})")
    print(f"{'='*60}")
    
    df_norm = df.copy()
    data_cols = [col for col in df.columns if col != id_col]
    norm_params = {}
    
    for idx in df.index:
        household_id = df.loc[idx, id_col]
        series = df.loc[idx, data_cols].values
        
        if method == 'zscore':
            mean = np.mean(series)
            std = np.std(series)
            if std == 0:
                # Constant series, set to 0 after normalization
                norm_series = np.zeros_like(series)
            else:
                norm_series = (series - mean) / std
            norm_params[household_id] = {'method': 'zscore', 'mean': mean, 'std': std}
            
        elif method == 'minmax':
            min_val = np.min(series)
            max_val = np.max(series)
            if max_val == min_val:
                norm_series = np.zeros_like(series)
            else:
                norm_series = (series - min_val) / (max_val - min_val)
            norm_params[household_id] = {'method': 'minmax', 'min': min_val, 'max': max_val}
            
        elif method == 'robust':
            median = np.median(series)
            q75, q25 = np.percentile(series, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                norm_series = np.zeros_like(series)
            else:
                norm_series = (series - median) / iqr
            norm_params[household_id] = {'method': 'robust', 'median': median, 'iqr': iqr}
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        df_norm.loc[idx, data_cols] = norm_series
    
    print(f"   - Normalized {len(df)} households")
    return df_norm, norm_params


def plot_preprocessing_comparison(df_original, df_processed, household_id, save=True):
    """
    Plot original vs processed time series for a household.
    
    Parameters:
    -----------
    df_original : pandas DataFrame, original data
    df_processed : pandas DataFrame, processed data
    household_id : int, household ID to plot
    save : bool, whether to save the figure
    """
    # Get the series
    original_series = df_original[df_original['ID'] == household_id].iloc[:, 1:].values.flatten()
    processed_series = df_processed[df_processed['ID'] == household_id].iloc[:, 1:].values.flatten()
    
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Original
    axes[0].plot(dates, original_series, linewidth=1, color='blue')
    axes[0].set_title(f'Household {household_id} - Original')
    axes[0].set_ylabel('Energy Consumption')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Processed
    axes[1].plot(dates, processed_series, linewidth=1, color='green')
    axes[1].set_title(f'Household {household_id} - Processed')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Normalized Consumption')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save:
        save_path = FIGURES_DIR / f'preprocessing_comparison_{household_id}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Quick test
    print("✅ preprocessing.py loaded successfully!")