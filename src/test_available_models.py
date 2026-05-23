import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

def create_time_series_features(df):
    """Add lag and rolling features"""
    df['demand_lag_1'] = df['demand'].shift(1)
    df['demand_lag_7'] = df['demand'].shift(7)
    df['demand_rolling_mean_7'] = df['demand'].rolling(window=7).mean()
    df['demand_rolling_std_7'] = df['demand'].rolling(window=7).std()
    return df

def test_model_performance(filename, component_name):
    """Test ML model performance on dataset"""
    print(f"\n{'='*80}")
    print(f"TESTING: {component_name.upper()} ({filename})")
    print(f"{'='*80}")
    
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Rename the target column to 'demand' for consistency
    if component_name in df.columns:
        df = df.rename(columns={component_name: 'demand'})
    elif f'{component_name}_demand' in df.columns:
        df = df.rename(columns={f'{component_name}_demand': 'demand'})
    
    # Add time series features
    df = create_time_series_features(df)
    
    # Drop rows with NaN values (from lag features)
    df = df.dropna()
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != 'demand']
    X = df[feature_cols]
    y = df['demand']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # Test Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results['RandomForest'] = {
        'R2': r2_score(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'MAE': mean_absolute_error(y_test, rf_pred)
    }
    
    # Test XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    results['XGBoost'] = {
        'R2': r2_score(y_test, xgb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
        'MAE': mean_absolute_error(y_test, xgb_pred)
    }
    
    # Test LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    results['LightGBM'] = {
        'R2': r2_score(y_test, lgb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lgb_pred)),
        'MAE': mean_absolute_error(y_test, lgb_pred)
    }
    
    # Print results
    print(f"\nResults for {component_name.upper()}:")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
    
    return results

print("="*80)
print("TESTING BLOODNET AI PREDICTION MODELS")
print("="*80)

# Test with available datasets
rbc_results = test_model_performance('rbc_demand.csv', 'rbc')
platelet_results = test_model_performance('platelet_demand.csv', 'platelet')
plasma_results = test_model_performance('plasma_demand.csv', 'plasma')

# Summary
print("\n" + "="*80)
print("SUMMARY: BEST R² SCORES FOR EACH COMPONENT")
print("="*80)

for component, results in [('RBC', rbc_results), ('Platelet', platelet_results), ('Plasma', plasma_results)]:
    best_model = max(results.items(), key=lambda x: x[1]['R2'])
    print(f"\n{component}:")
    print(f"  Best Model: {best_model[0]}")
    print(f"  R²: {best_model[1]['R2']:.4f}")
    print(f"  RMSE: {best_model[1]['RMSE']:.4f}")

print("\nBloodNet AI model testing completed!")
