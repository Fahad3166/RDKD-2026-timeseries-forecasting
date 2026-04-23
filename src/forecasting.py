# =============================================================================
# forecasting.py
# Time series forecasting with recursive prediction (NO DATA LEAKAGE!)
# Team [X] - KDDSS 2026 Project
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import PROCESSED_DATA_DIR, FORECASTING_OUTPUT_DIR, FIGURES_DIR, RANDOM_SEED


# =============================================================================
# CALENDAR UTILITIES (Handles 2023: 365 days, 2024: 366 days)
# =============================================================================

def get_days_in_year(year):
    """
    Return number of days in a year (handles leap years).
    
    Parameters:
    -----------
    year : int (2023 or 2024)
    
    Returns:
    --------
    int : number of days
    """
    if year == 2024:
        return 366  # Leap year
    else:
        return 365  # 2023 and others


def get_month_and_day(day_of_year, year):
    """
    Convert day of year to month and day.
    
    Parameters:
    -----------
    day_of_year : int (1-365 for 2023, 1-366 for 2024)
    year : int (2023 or 2024)
    
    Returns:
    --------
    month : int (1-12), day : int
    """
    # Days per month for each year
    if year == 2024:
        month_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    cumulative = 0
    for month, days in enumerate(month_days, 1):
        if day_of_year <= cumulative + days:
            day = day_of_year - cumulative
            return month, day
        cumulative += days
    
    return 12, 31  # fallback


def get_day_of_week(day_of_year, year):
    """
    Get day of week (0=Monday, 6=Sunday).
    
    Jan 1, 2023 = Sunday (day 6)
    Jan 1, 2024 = Monday (day 0)
    """
    if year == 2023:
        # Jan 1, 2023 was Sunday = 6
        return (day_of_year + 5) % 7
    else:  # 2024
        # Jan 1, 2024 was Monday = 0
        return (day_of_year - 1) % 7


# =============================================================================
# FEATURE ENGINEERING FOR FORECASTING (NO LEAKAGE!)
# =============================================================================

def create_recursive_features(history, day_of_year=None, year=2024):
    """
    Create features from historical values ONLY (no future leakage).
    
    Parameters:
    -----------
    history : list or array, past consumption values (from before prediction day)
    day_of_year : int, current day in year (1-366 for 2024, 1-365 for 2023)
    year : int, year being predicted (2023 or 2024)
    
    Returns:
    --------
    dict of features
    """
    features = {}
    n = len(history)
    
    # ===== Lag Features (only from available history) =====
    features['lag_1'] = history[-1] if n >= 1 else 0
    features['lag_2'] = history[-2] if n >= 2 else 0
    features['lag_3'] = history[-3] if n >= 3 else 0
    features['lag_7'] = history[-7] if n >= 7 else 0
    features['lag_14'] = history[-14] if n >= 14 else 0
    features['lag_21'] = history[-21] if n >= 21 else 0
    features['lag_28'] = history[-28] if n >= 28 else 0
    
    # ===== Rolling Statistics (only from available history) =====
    if n >= 7:
        features['rolling_mean_7'] = np.mean(history[-7:])
        features['rolling_std_7'] = np.std(history[-7:])
    else:
        features['rolling_mean_7'] = np.mean(history) if n > 0 else 0
        features['rolling_std_7'] = np.std(history) if n > 1 else 0
    
    if n >= 14:
        features['rolling_mean_14'] = np.mean(history[-14:])
        features['rolling_std_14'] = np.std(history[-14:])
    else:
        features['rolling_mean_14'] = features['rolling_mean_7']
        features['rolling_std_14'] = features['rolling_std_7']
    
    if n >= 30:
        features['rolling_mean_30'] = np.mean(history[-30:])
    else:
        features['rolling_mean_30'] = features['rolling_mean_14']
    
    # ===== Exponential Weighted Average =====
    if n >= 7:
        weights = np.exp(np.linspace(-1, 0, min(7, n)))
        weights = weights / weights.sum()
        features['ewma_7'] = np.sum(history[-min(7, n):] * weights[-min(7, n):])
    else:
        features['ewma_7'] = features['rolling_mean_7']
    
    # ===== Trend Features =====
    if n >= 7:
        x = np.arange(min(7, n))
        y = history[-min(7, n):]
        if len(x) > 1 and np.std(x) > 0:
            slope = np.polyfit(x, y, 1)[0]
            features['trend_7'] = slope
        else:
            features['trend_7'] = 0
    else:
        features['trend_7'] = 0
    
    # ===== Volatility =====
    if n >= 7:
        features['range_7'] = np.max(history[-7:]) - np.min(history[-7:])
    else:
        features['range_7'] = 0
    
    # ===== Time Features (known in advance - NO LEAKAGE!) =====
    if day_of_year is not None:
        days_in_year = get_days_in_year(year)
        
        # Cyclical encoding for day of year
        features['doy_sin'] = np.sin(2 * np.pi * day_of_year / days_in_year)
        features['doy_cos'] = np.cos(2 * np.pi * day_of_year / days_in_year)
        
        # Day of week
        features['day_of_week'] = get_day_of_week(day_of_year, year)
        features['is_weekend'] = 1 if features['day_of_week'] >= 5 else 0
        
        # Month
        month, _ = get_month_and_day(day_of_year, year)
        features['month'] = month
        features['is_summer'] = 1 if month in [6, 7, 8] else 0
        features['is_winter'] = 1 if month in [12, 1, 2] else 0
        
        # Quarter
        features['quarter'] = (month - 1) // 3 + 1
    
    return features


# =============================================================================
# RECURSIVE FORECASTING ENGINE
# =============================================================================

class RecursiveForecaster:
    """
    Recursive forecasting model that predicts day by day without leakage.
    """
    
    def __init__(self, model=None, window_size=30, feature_names=None):
        self.model = model
        self.window_size = window_size
        self.feature_names = feature_names
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def _prepare_training_data(self, time_series, year=2023):
        """
        Prepare training data from a single time series (NO LEAKAGE).
        Each sample uses only past values to predict the next value.
        """
        X_list = []
        y_list = []
        
        n = len(time_series)
        
        for i in range(self.window_size, n):
            # Past values (history) - only from before day i
            history = time_series[i - self.window_size:i].tolist()
            
            # Day of year for the prediction target (1-indexed)
            day_of_year = i + 1
            
            # Create features from past only
            features = create_recursive_features(history, day_of_year, year)
            X_list.append(features)
            y_list.append(time_series[i])
        
        return pd.DataFrame(X_list), np.array(y_list)
    
    def fit(self, df_2023):
        """
        Train model on all 2023 data (365 days, NO LEAKAGE).
        
        Parameters:
        -----------
        df_2023 : DataFrame, households × days (365 days)
        """
        print("="*60)
        print("TRAINING RECURSIVE FORECASTER ON 2023 DATA")
        print("="*60)
        print(f"   Training years: 2023 (365 days)")
        print(f"   Households: {len(df_2023)}")
        print(f"   Window size: {self.window_size} days")
        
        X_all = []
        y_all = []
        
        # Collect training data from all households
        for household_id in tqdm(df_2023.index, desc="Processing households"):
            series = df_2023.loc[household_id].values
            X, y = self._prepare_training_data(series, year=2023)
            X_all.append(X)
            y_all.append(y)
        
        X_train = pd.concat(X_all, ignore_index=True)
        y_train = np.concatenate(y_all)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        print(f"\n   Training samples: {len(X_train):,}")
        print(f"   Features: {len(self.feature_names)}")
        
        # Normalize features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        
        # Normalize target
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # Train model
        if self.model is None:
            self.model = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                verbose=-1,
                n_jobs=-1
            )
        
        print(f"\n   Training LightGBM model...")
        self.model.fit(X_train_scaled, y_train_scaled)
        
        print(f"✅ Training complete!")
        return self
    
    def predict_household(self, train_series, n_days=366, year=2024):
        """
        Recursively predict future values for a single household.
        
        Parameters:
        -----------
        train_series : array, historical data (2023, 365 days)
        n_days : int, number of days to predict (366 for 2024)
        year : int, year being predicted (2024)
        
        Returns:
        --------
        predictions : list of predicted values
        """
        # Initialize with last window_size days of training data
        history = train_series[-self.window_size:].tolist()
        predictions = []
        
        for day in range(n_days):
            # Day of year for this prediction (1-indexed)
            day_of_year = day + 1
            
            # Create features from history only
            features = create_recursive_features(history, day_of_year, year)
            X = pd.DataFrame([features])[self.feature_names]
            
            # Scale and predict
            X_scaled = self.scaler_X.transform(X)
            pred_scaled = self.model.predict(X_scaled)[0]
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            
            predictions.append(max(0, pred))  # No negative consumption
            
            # Update history (drop oldest, add prediction)
            history.append(pred)
            history = history[-self.window_size:]
        
        return predictions
    
    def predict_all(self, df_2023, df_2024_actual=None):
        """
        Predict 2024 for all households.
        
        Parameters:
        -----------
        df_2023 : DataFrame, training data (365 days)
        df_2024_actual : DataFrame (optional), actual values for evaluation
        
        Returns:
        --------
        predictions_df : DataFrame, predictions for all households
        mae_dict : dict, MAE per household (if actuals provided)
        """
        print("\n" + "="*60)
        print("RECURSIVE FORECASTING FOR 2024")
        print("="*60)
        print(f"   Predicting: 366 days (2024 leap year)")
        print(f"   Households: {len(df_2023)}")
        
        all_predictions = {}
        all_mae = {}
        
        for household_id in tqdm(df_2023.index, desc="Forecasting households"):
            train_series = df_2023.loc[household_id].values
            predictions = self.predict_household(train_series, n_days=366, year=2024)
            all_predictions[household_id] = predictions
            
            if df_2024_actual is not None and household_id in df_2024_actual.index:
                actuals = df_2024_actual.loc[household_id].values
                if len(actuals) == 366:  # Ensure correct length
                    mae = mean_absolute_error(actuals, predictions)
                    all_mae[household_id] = mae
        
        # Convert to DataFrame
        dates_2024 = pd.date_range(start='2024-01-01', periods=366, freq='D')
        predictions_df = pd.DataFrame(all_predictions).T
        predictions_df.columns = dates_2024.strftime('%Y-%m-%d')
        predictions_df.index.name = 'ID'
        
        print(f"\n✅ Forecasted {len(predictions_df)} households for 366 days")
        
        if all_mae:
            avg_mae = np.mean(list(all_mae.values()))
            print(f"📊 Average MAE: {avg_mae:.4f}")
        
        return predictions_df, all_mae


# =============================================================================
# CLUSTER-BASED FORECASTING
# =============================================================================

def train_cluster_forecasters(df_2023, cluster_assignments, n_clusters):
    """
    Train separate forecasting models for each cluster.
    
    Parameters:
    -----------
    df_2023 : DataFrame, training data (365 days)
    cluster_assignments : DataFrame with 'ID' and 'cluster' columns
    n_clusters : int, number of clusters
    
    Returns:
    --------
    forecasters : dict, trained models per cluster
    """
    print("\n" + "="*60)
    print(f"TRAINING CLUSTER-SPECIFIC MODELS")
    print("="*60)
    print(f"   Number of clusters: {n_clusters}")
    
    forecasters = {}
    
    for cluster_id in range(n_clusters):
        # Get households in this cluster
        cluster_households = cluster_assignments[
            cluster_assignments['cluster'] == cluster_id
        ]['ID'].values
        
        if len(cluster_households) < 10:
            print(f"   ⚠️ Cluster {cluster_id}: Only {len(cluster_households)} households (skipping)")
            continue
        
        # Subset data for this cluster
        df_cluster = df_2023[df_2023.index.isin(cluster_households)]
        
        print(f"   Cluster {cluster_id}: {len(df_cluster)} households")
        
        # Train forecaster on this cluster only
        forecaster = RecursiveForecaster(window_size=30)
        forecaster.fit(df_cluster)
        forecasters[cluster_id] = forecaster
    
    return forecasters


def forecast_by_cluster(df_2023, cluster_assignments, forecasters, df_2024=None):
    """
    Forecast using cluster-specific models.
    """
    print("\n" + "="*60)
    print("FORECASTING BY CLUSTER")
    print("="*60)
    
    all_predictions = {}
    all_mae = {}
    
    for cluster_id, forecaster in forecasters.items():
        cluster_households = cluster_assignments[
            cluster_assignments['cluster'] == cluster_id
        ]['ID'].values
        
        print(f"\n📌 Cluster {cluster_id}: {len(cluster_households)} households")
        
        for household_id in tqdm(cluster_households, desc=f"  Cluster {cluster_id}", leave=False):
            if household_id not in df_2023.index:
                continue
                
            train_series = df_2023.loc[household_id].values
            predictions = forecaster.predict_household(train_series, n_days=366, year=2024)
            all_predictions[household_id] = predictions
            
            if df_2024 is not None and household_id in df_2024.index:
                actuals = df_2024.loc[household_id].values
                if len(actuals) == 366:
                    mae = mean_absolute_error(actuals, predictions)
                    all_mae[household_id] = mae
    
    # Convert to DataFrame
    dates_2024 = pd.date_range(start='2024-01-01', periods=366, freq='D')
    predictions_df = pd.DataFrame(all_predictions).T
    predictions_df.columns = dates_2024.strftime('%Y-%m-%d')
    predictions_df.index.name = 'ID'
    
    if all_mae:
        avg_mae = np.mean(list(all_mae.values()))
        print(f"\n📊 Average MAE (cluster-based): {avg_mae:.4f}")
    
    return predictions_df, all_mae


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_forecasts(predictions_df, df_2024):
    """
    Evaluate forecasts against actual 2024 data.
    
    Returns:
    --------
    metrics : dict with MAE, RMSE, MAPE per household and overall
    """
    print("\n" + "="*60)
    print("EVALUATING FORECASTS AGAINST ACTUAL 2024 DATA")
    print("="*60)
    
    common_ids = predictions_df.index.intersection(df_2024.index)
    
    mae_list = []
    rmse_list = []
    mape_list = []
    
    for household_id in common_ids:
        pred = predictions_df.loc[household_id].values
        actual = df_2024.loc[household_id].values
        
        # Ensure both have 366 days
        if len(pred) != 366 or len(actual) != 366:
            continue
        
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        
        # MAPE (avoid division by zero)
        non_zero_mask = actual > 0.01
        if non_zero_mask.any():
            mape = np.mean(np.abs((actual[non_zero_mask] - pred[non_zero_mask]) / actual[non_zero_mask])) * 100
        else:
            mape = np.nan
        
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
    
    metrics = {
        'MAE': np.mean(mae_list),
        'MAE_std': np.std(mae_list),
        'RMSE': np.mean(rmse_list),
        'MAPE': np.nanmean(mape_list),
        'n_households': len(mae_list),
        'per_household': {
            'mae': dict(zip(common_ids, mae_list)),
            'rmse': dict(zip(common_ids, rmse_list))
        }
    }
    
    print(f"\n📊 Overall Metrics (based on {metrics['n_households']} households):")
    print(f"   MAE:  {metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}")
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    
    return metrics


# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("FORECASTING MODULE TEST")
    print("="*60)
    print(f"\n✅ forecasting.py loaded successfully!")
    print(f"\n📅 Calendar Info:")
    print(f"   2023: {get_days_in_year(2023)} days")
    print(f"   2024: {get_days_in_year(2024)} days (leap year)")
    
    # Test day of week calculations
    print(f"\n📅 Day of Week Test (2024):")
    print(f"   Jan 1, 2024 (day 1) should be Monday (0): {get_day_of_week(1, 2024)}")
    print(f"   Jan 7, 2024 (day 7) should be Sunday (6): {get_day_of_week(7, 2024)}")
    
    print(f"\n📅 Day of Week Test (2023):")
    print(f"   Jan 1, 2023 (day 1) should be Sunday (6): {get_day_of_week(1, 2023)}")
    print(f"   Jan 2, 2023 (day 2) should be Monday (0): {get_day_of_week(2, 2023)}")