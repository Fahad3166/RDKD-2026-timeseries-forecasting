# =============================================================================
# feature_engineering.py
# Extract features from time series for clustering
# 
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def extract_statistical_features(series):
    """
    Extract basic statistical features from a time series.
    
    Parameters:
    -----------
    series : array-like, time series values
    
    Returns:
    --------
    dict of statistical features
    """
    features = {}
    
    # Central tendency
    features['mean'] = np.mean(series)
    features['median'] = np.median(series)
    
    # Variability
    features['std'] = np.std(series)
    # Handle division by zero for CV
    if features['mean'] > 0:
        features['cv'] = features['std'] / features['mean']
    else:
        features['cv'] = 0
    
    features['iqr'] = np.percentile(series, 75) - np.percentile(series, 25)
    features['range'] = np.max(series) - np.min(series)
    
    # Shape
    features['skewness'] = stats.skew(series)
    features['kurtosis'] = stats.kurtosis(series)
    
    # Quantiles
    features['q25'] = np.percentile(series, 25)
    features['q75'] = np.percentile(series, 75)
    features['q90'] = np.percentile(series, 90)
    features['q95'] = np.percentile(series, 95)
    
    return features


def extract_temporal_features(series, dates):
    """
    Extract temporal patterns from time series.
    
    Parameters:
    -----------
    series : array-like, time series values
    dates : array-like of datetime, corresponding dates
    
    Returns:
    --------
    dict of temporal features
    """
    features = {}
    
    # Convert to pandas Series for easier manipulation
    s = pd.Series(series, index=dates)
    
    # Day of week patterns
    dow_avg = s.groupby(s.index.dayofweek).mean()
    weekday_avg = dow_avg[0:5].mean()
    if weekday_avg > 0:
        features['weekend_ratio'] = dow_avg[5:7].mean() / weekday_avg
    else:
        features['weekend_ratio'] = 1.0
    
    # Monthly patterns
    monthly_avg = s.groupby(s.index.month).mean()
    overall_mean = monthly_avg.mean()
    
    if overall_mean > 0:
        # Summer months (June-August)
        summer_months = [6, 7, 8]
        summer_avg = monthly_avg[monthly_avg.index.isin(summer_months)].mean()
        features['summer_peak'] = summer_avg / overall_mean
        
        # Winter months (December-February)
        winter_months = [12, 1, 2]
        winter_avg = monthly_avg[monthly_avg.index.isin(winter_months)].mean()
        features['winter_peak'] = winter_avg / overall_mean
    else:
        features['summer_peak'] = 1.0
        features['winter_peak'] = 1.0
    
    # Trend (using linear regression)
    x = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
    features['trend_slope'] = slope
    features['trend_r2'] = r_value**2
    
    # Seasonality strength - with fallback
    try:
        from statsmodels.tsa.stattools import acf
        acf_values = acf(series, nlags=7, fft=True)
        features['seasonal_strength_weekly'] = np.max(acf_values[1:8]) if len(acf_values) > 1 else 0
    except:
        # Simple lag-1 autocorrelation as fallback
        if len(series) > 1 and np.std(series) > 0:
            features['seasonal_strength_weekly'] = np.corrcoef(series[:-1], series[1:])[0, 1]
        else:
            features['seasonal_strength_weekly'] = 0
    
    return features


def extract_peak_features(series):
    """
    Extract features related to peaks in the time series.
    
    Parameters:
    -----------
    series : array-like, time series values
    
    Returns:
    --------
    dict of peak-related features
    """
    features = {}
    
    # Find peaks (local maxima)
    threshold = np.mean(series) + np.std(series)
    peaks, properties = find_peaks(series, height=threshold, distance=7)
    
    features['n_peaks'] = len(peaks)
    features['peak_density'] = len(peaks) / len(series) * 365  # peaks per year
    
    if len(peaks) > 0:
        features['avg_peak_height'] = np.mean(properties['peak_heights'])
        features['max_peak_height'] = np.max(properties['peak_heights'])
        if np.mean(series) > 0:
            features['peak_height_ratio'] = features['max_peak_height'] / np.mean(series)
        else:
            features['peak_height_ratio'] = 0
    else:
        features['avg_peak_height'] = 0
        features['max_peak_height'] = 0
        features['peak_height_ratio'] = 0
    
    return features


def extract_zero_features(series):
    """
    Extract features related to zero values.
    
    Parameters:
    -----------
    series : array-like, time series values
    
    Returns:
    --------
    dict of zero-related features
    """
    features = {}
    
    n_zeros = np.sum(series == 0)
    n_days = len(series)
    
    features['zero_percentage'] = n_zeros / n_days * 100
    
    # Find zero runs (consecutive zeros)
    if n_zeros > 0:
        # Create mask of zeros
        zero_mask = (series == 0).astype(int)
        
        # Find runs of zeros
        runs = []
        current_run = 0
        for val in zero_mask:
            if val == 1:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                    current_run = 0
        if current_run > 0:
            runs.append(current_run)
        
        features['max_zero_run'] = max(runs) if runs else 0
        features['avg_zero_run'] = np.mean(runs) if runs else 0
    else:
        features['max_zero_run'] = 0
        features['avg_zero_run'] = 0
    
    return features


def extract_all_features(df, id_col='ID'):
    """
    Extract all features for all households.
    
    Parameters:
    -----------
    df : pandas DataFrame with time series
    id_col : str, name of ID column
    
    Returns:
    --------
    features_df : DataFrame with extracted features
    """
    print("="*60)
    print("EXTRACTING FEATURES FROM TIME SERIES")
    print("="*60)
    
    data_cols = [col for col in df.columns if col != id_col]
    dates = pd.to_datetime(data_cols)
    
    all_features = []
    
    for idx in df.index:
        household_id = df.loc[idx, id_col]
        series = df.loc[idx, data_cols].values.astype(float)
        
        # Extract all feature groups
        features = {'ID': household_id}
        
        # Statistical features
        features.update(extract_statistical_features(series))
        
        # Temporal features
        features.update(extract_temporal_features(series, dates))
        
        # Peak features
        features.update(extract_peak_features(series))
        
        # Zero features
        features.update(extract_zero_features(series))
        
        all_features.append(features)
        
        if (idx + 1) % 1000 == 0:
            print(f"   Processed {idx + 1} households...")
    
    features_df = pd.DataFrame(all_features)
    print(f"\n✅ Extracted {len(features_df.columns) - 1} features for {len(features_df)} households")
    
    return features_df


def normalize_features(features_df, id_col='ID', method='zscore'):
    """
    Normalize features for clustering.
    
    Parameters:
    -----------
    features_df : DataFrame with features
    id_col : str, name of ID column
    method : str, normalization method
    
    Returns:
    --------
    normalized_df : DataFrame with normalized features
    scaler : scaler object for inverse transform
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    print(f"\n{'='*60}")
    print(f"NORMALIZING FEATURES (method: {method})")
    print(f"{'='*60}")
    
    feature_cols = [col for col in features_df.columns if col != id_col]
    
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    normalized_values = scaler.fit_transform(features_df[feature_cols])
    
    normalized_df = pd.DataFrame(
        normalized_values,
        columns=feature_cols,
        index=features_df.index
    )
    normalized_df[id_col] = features_df[id_col].values
    
    print(f"✅ Normalized {len(feature_cols)} features")
    
    return normalized_df, scaler


if __name__ == "__main__":
    print("✅ feature_engineering.py loaded successfully!")