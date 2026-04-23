# ============================================================================
# src/feature_engineering_daily.py
# Time-based features for forecasting (with enhancements)
# ============================================================================

import pandas as pd
import numpy as np
from pathlib import Path

def create_forecasting_features(df: pd.DataFrame, include_outliers: bool = True) -> pd.DataFrame:
    """
    Create time-based features for forecasting.
    
    Parameters:
    -----------
    df : DataFrame
        Can be wide format (ID + date columns) or long format (ID, day, consumption)
    include_outliers : bool
        If False, removes outlier household (Cluster 3, single household with all zeros)
    
    Returns:
    --------
    df_long : DataFrame with features
    """
    
    # ──────────────────────────────────────────────────────────────────────
    # 1. Convert to long format if needed
    # ──────────────────────────────────────────────────────────────────────
    if "day" in df.columns and "consumption" in df.columns:
        # Already long format
        df_long = df.copy()
        df_long["day"] = pd.to_datetime(df_long["day"])
    else:
        # Wide format → melt
        df_long = df.melt(id_vars=["ID"], var_name="day", value_name="consumption")
        df_long["day"] = pd.to_datetime(df_long["day"])
    
    # ──────────────────────────────────────────────────────────────────────
    # 2. Remove outlier household if needed (Cluster 3 with 1 household)
    # ──────────────────────────────────────────────────────────────────────
    if not include_outliers:
        # Identify household with all zeros (or very low consumption)
        household_means = df_long.groupby('ID')['consumption'].mean()
        outlier_ids = household_means[household_means < 0.1].index
        df_long = df_long[~df_long['ID'].isin(outlier_ids)]
        print(f"✅ Removed {len(outlier_ids)} outlier households")
    
    # Sort by household and date
    df_long = df_long.sort_values(["ID", "day"])
    
    # ──────────────────────────────────────────────────────────────────────
    # 3. Calendar features
    # ──────────────────────────────────────────────────────────────────────
    df_long["day_of_week"] = df_long["day"].dt.dayofweek
    df_long["month"] = df_long["day"].dt.month
    df_long["is_weekend"] = df_long["day_of_week"].isin([5, 6]).astype(int)
    df_long["season"] = ((df_long["month"] % 12 + 3) // 3).astype(int)
    
    # Day of year (1-365/366) with sin/cos encoding for seasonality
    df_long["day_of_year"] = df_long["day"].dt.dayofyear
    df_long["doy_sin"] = np.sin(2 * np.pi * df_long["day_of_year"] / 366)
    df_long["doy_cos"] = np.cos(2 * np.pi * df_long["day_of_year"] / 366)
    
    # ──────────────────────────────────────────────────────────────────────
    # 4. Lag features
    # ──────────────────────────────────────────────────────────────────────
    for lag in [1, 7, 14, 30]:
        df_long[f"lag_{lag}"] = df_long.groupby("ID")["consumption"].shift(lag)
    
    # ──────────────────────────────────────────────────────────────────────
    # 5. Rolling window features
    # ──────────────────────────────────────────────────────────────────────
    for window in [7, 14, 30]:
        # Rolling mean
        df_long[f"rolling_mean_{window}"] = (
            df_long.groupby("ID")["consumption"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        
        # Rolling standard deviation
        df_long[f"rolling_std_{window}"] = (
            df_long.groupby("ID")["consumption"]
            .rolling(window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
    
    # ──────────────────────────────────────────────────────────────────────
    # 6. Exponential weighted moving average (captures recent trends)
    # ──────────────────────────────────────────────────────────────────────
    df_long["ewm_alpha_0.3"] = (
        df_long.groupby("ID")["consumption"]
        .transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
    )
    
    # ──────────────────────────────────────────────────────────────────────
    # 7. Remove rows with NaN (first few days of each household)
    # ──────────────────────────────────────────────────────────────────────
    df_long = df_long.dropna()
    
    print(f"✅ Features created: {df_long.shape}")
    print(f"   - Rows: {df_long.shape[0]:,}")
    print(f"   - Columns: {df_long.shape[1]}")
    print(f"   - Households: {df_long['ID'].nunique()}")
    
    return df_long


def add_cluster_labels(df_features: pd.DataFrame, cluster_assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Add cluster labels to the feature dataframe for cluster-specific forecasting.
    
    Parameters:
    -----------
    df_features : DataFrame with daily features (must have 'ID' column)
    cluster_assignments : DataFrame with 'ID' and 'cluster' columns
    
    Returns:
    --------
    df_with_clusters : DataFrame with added 'cluster' column
    """
    df_with_clusters = df_features.merge(
        cluster_assignments[['ID', 'cluster']], 
        on='ID', 
        how='left'
    )
    return df_with_clusters


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Load your cleaned 2023 data
    from pathlib import Path
    DATA_DIR = Path("../data/processed")
    
    # Load data
    df_2023 = pd.read_csv(DATA_DIR / "household_features_clean.csv")
    
    # For testing, we need the raw daily data, not features
    # You'll need to load the raw 2023 data (sample_23.csv)
    RAW_DIR = Path("../data/raw")
    df_raw = pd.read_csv(RAW_DIR / "sample_23.csv", index_col=0)
    
    # Create features
    features_df = create_forecasting_features(df_raw, include_outliers=False)
    
    # Save to parquet (efficient format)
    features_df.to_parquet(DATA_DIR / "forecasting_features.parquet", index=False)
    print(f"✅ Saved to {DATA_DIR / 'forecasting_features.parquet'}")