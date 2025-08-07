#!/usr/bin/env python3
"""
F1 Race Prediction Model
========================

This module implements a comprehensive Formula 1 race prediction system that:
1. Collects historical F1 data from multiple sources
2. Engineers features for machine learning
3. Trains multiple models (Random Forest, XGBoost, LightGBM)
4. Makes predictions for upcoming races
5. Provides detailed analysis and visualization

Author: F1 Prediction System
Date: 2025
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F1DataCollector:
    """Handles data collection from various F1 data sources."""
    
    def __init__(self):
        self.base_url = "http://ergast.com/api/f1"
        self.current_season = datetime.now().year
        
    def get_race_results(self, year: int, limit: int = 1000) -> pd.DataFrame:
        """Fetch race results for a given year."""
        url = f"{self.base_url}/{year}/results.json?limit={limit}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            races_data = []
            for race in data['MRData']['RaceTable']['Races']:
                for result in race['Results']:
                    races_data.append({
                        'season': int(race['season']),
                        'round': int(race['round']),
                        'race_name': race['raceName'],
                        'circuit_name': race['Circuit']['circuitName'],
                        'date': race['date'],
                        'driver_id': result['Driver']['driverId'],
                        'driver_name': result['Driver']['givenName'] + ' ' + result['Driver']['familyName'],
                        'constructor_id': result['Constructor']['constructorId'],
                        'constructor_name': result['Constructor']['name'],
                        'grid': int(result['grid']) if result['grid'].isdigit() else 20,
                        'position': int(result['position']) if 'position' in result else 20,
                        'points': float(result['points']),
                        'status': result['status'],
                        'fastest_lap': result.get('FastestLap', {}).get('Time', {}).get('time', None),
                        'fastest_lap_rank': int(result.get('FastestLap', {}).get('@rank', 0)) if result.get('FastestLap', {}).get('@rank', '0').isdigit() else 0
                    })
            
            return pd.DataFrame(races_data)
        except Exception as e:
            logger.error(f"Error fetching race results for {year}: {e}")
            return pd.DataFrame()
    
    def get_qualifying_results(self, year: int, limit: int = 1000) -> pd.DataFrame:
        """Fetch qualifying results for a given year."""
        url = f"{self.base_url}/{year}/qualifying.json?limit={limit}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            qualifying_data = []
            for race in data['MRData']['RaceTable']['Races']:
                for result in race['QualifyingResults']:
                    qualifying_data.append({
                        'season': int(race['season']),
                        'round': int(race['round']),
                        'race_name': race['raceName'],
                        'driver_id': result['Driver']['driverId'],
                        'constructor_id': result['Constructor']['constructorId'],
                        'quali_position': int(result['position']),
                        'q1': result.get('Q1', None),
                        'q2': result.get('Q2', None),
                        'q3': result.get('Q3', None)
                    })
            
            return pd.DataFrame(qualifying_data)
        except Exception as e:
            logger.error(f"Error fetching qualifying results for {year}: {e}")
            return pd.DataFrame()
    
    def get_constructor_standings(self, year: int) -> pd.DataFrame:
        """Fetch constructor standings for a given year."""
        url = f"{self.base_url}/{year}/constructorStandings.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            standings_data = []
            for standing_list in data['MRData']['StandingsTable']['StandingsLists']:
                for standing in standing_list['ConstructorStandings']:
                    standings_data.append({
                        'season': int(standing_list['season']),
                        'round': int(standing_list['round']),
                        'constructor_id': standing['Constructor']['constructorId'],
                        'constructor_name': standing['Constructor']['name'],
                        'constructor_position': int(standing['position']),
                        'constructor_points': float(standing['points']),
                        'constructor_wins': int(standing['wins'])
                    })
            
            return pd.DataFrame(standings_data)
        except Exception as e:
            logger.error(f"Error fetching constructor standings for {year}: {e}")
            return pd.DataFrame()
    
    def get_driver_standings(self, year: int) -> pd.DataFrame:
        """Fetch driver standings for a given year."""
        url = f"{self.base_url}/{year}/driverStandings.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            standings_data = []
            for standing_list in data['MRData']['StandingsTable']['StandingsLists']:
                for standing in standing_list['DriverStandings']:
                    standings_data.append({
                        'season': int(standing_list['season']),
                        'round': int(standing_list['round']),
                        'driver_id': standing['Driver']['driverId'],
                        'driver_name': standing['Driver']['givenName'] + ' ' + standing['Driver']['familyName'],
                        'driver_position': int(standing['position']),
                        'driver_points': float(standing['points']),
                        'driver_wins': int(standing['wins'])
                    })
            
            return pd.DataFrame(standings_data)
        except Exception as e:
            logger.error(f"Error fetching driver standings for {year}: {e}")
            return pd.DataFrame()

class F1FeatureEngineer:
    """Handles feature engineering for the F1 prediction model."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create historical performance features."""
        df = df.copy()
        df = df.sort_values(['driver_id', 'season', 'round'])
        
        # Driver historical features
        df['driver_avg_position_last_5'] = df.groupby('driver_id')['position'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        df['driver_avg_points_last_5'] = df.groupby('driver_id')['points'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        df['driver_wins_last_10'] = df.groupby('driver_id')['position'].transform(
            lambda x: (x == 1).rolling(10, min_periods=1).sum().shift(1)
        )
        df['driver_podiums_last_10'] = df.groupby('driver_id')['position'].transform(
            lambda x: (x <= 3).rolling(10, min_periods=1).sum().shift(1)
        )
        
        # Constructor historical features
        df['constructor_avg_position_last_5'] = df.groupby('constructor_id')['position'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        df['constructor_avg_points_last_5'] = df.groupby('constructor_id')['points'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        
        # Circuit-specific features
        df['driver_circuit_avg_position'] = df.groupby(['driver_id', 'circuit_name'])['position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['constructor_circuit_avg_position'] = df.groupby(['constructor_id', 'circuit_name'])['position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        return df
    
    def create_season_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create season-based features."""
        df = df.copy()
        
        # Season progress
        df['season_progress'] = df['round'] / df.groupby('season')['round'].transform('max')
        
        # Championship position features (approximate)
        df['driver_championship_position'] = df.groupby(['season', 'round'])['driver_points'].rank(
            ascending=False, method='min'
        )
        df['constructor_championship_position'] = df.groupby(['season', 'round'])['constructor_points'].rank(
            ascending=False, method='min'
        )
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        categorical_columns = ['driver_id', 'constructor_id', 'circuit_name']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: self.label_encoders[col].transform([str(x)])[0] 
                    if str(x) in self.label_encoders[col].classes_ 
                    else -1
                )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between driver and constructor."""
        df = df.copy()
        
        # Driver-Constructor combination strength
        df['driver_constructor_combo_avg'] = df.groupby(['driver_id', 'constructor_id'])['position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Grid position vs historical performance
        df['grid_vs_avg_position'] = df['grid'] - df['driver_avg_position_last_5']
        
        return df

class F1PredictionModel:
    """Main F1 prediction model class."""
    
    def __init__(self):
        self.data_collector = F1DataCollector()
        self.feature_engineer = F1FeatureEngineer()
        self.models = {}
        self.feature_importance = {}
        
    def collect_data(self, start_year: int = 2018, end_year: int = None) -> pd.DataFrame:
        """Collect comprehensive F1 data."""
        if end_year is None:
            end_year = datetime.now().year
        
        logger.info(f"Collecting F1 data from {start_year} to {end_year}")
        
        all_race_data = []
        all_qualifying_data = []
        all_constructor_standings = []
        all_driver_standings = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Collecting data for {year}")
            
            # Race results
            race_data = self.data_collector.get_race_results(year)
            if not race_data.empty:
                all_race_data.append(race_data)
            
            # Qualifying results
            quali_data = self.data_collector.get_qualifying_results(year)
            if not quali_data.empty:
                all_qualifying_data.append(quali_data)
            
            # Constructor standings
            constructor_standings = self.data_collector.get_constructor_standings(year)
            if not constructor_standings.empty:
                all_constructor_standings.append(constructor_standings)
            
            # Driver standings
            driver_standings = self.data_collector.get_driver_standings(year)
            if not driver_standings.empty:
                all_driver_standings.append(driver_standings)
        
        # Combine all data
        if all_race_data:
            race_df = pd.concat(all_race_data, ignore_index=True)
        else:
            race_df = pd.DataFrame()
        
        if all_qualifying_data:
            quali_df = pd.concat(all_qualifying_data, ignore_index=True)
        else:
            quali_df = pd.DataFrame()
        
        if all_constructor_standings:
            constructor_df = pd.concat(all_constructor_standings, ignore_index=True)
        else:
            constructor_df = pd.DataFrame()
        
        if all_driver_standings:
            driver_df = pd.concat(all_driver_standings, ignore_index=True)
        else:
            driver_df = pd.DataFrame()
        
        # Merge datasets
        if not race_df.empty and not quali_df.empty:
            merged_df = race_df.merge(
                quali_df[['season', 'round', 'driver_id', 'quali_position']],
                on=['season', 'round', 'driver_id'],
                how='left'
            )
        else:
            merged_df = race_df
        
        if not merged_df.empty and not constructor_df.empty:
            merged_df = merged_df.merge(
                constructor_df[['season', 'round', 'constructor_id', 'constructor_points', 'constructor_wins']],
                on=['season', 'round', 'constructor_id'],
                how='left'
            )
        
        if not merged_df.empty and not driver_df.empty:
            merged_df = merged_df.merge(
                driver_df[['season', 'round', 'driver_id', 'driver_points', 'driver_wins']],
                on=['season', 'round', 'driver_id'],
                how='left'
            )
        
        logger.info(f"Collected {len(merged_df)} race records")
        return merged_df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning."""
        logger.info("Engineering features...")
        
        # Create historical features
        df = self.feature_engineer.create_historical_features(df)
        
        # Create season features
        df = self.feature_engineer.create_season_features(df)
        
        # Encode categorical features
        df = self.feature_engineer.encode_categorical_features(df)
        
        # Create interaction features
        df = self.feature_engineer.create_interaction_features(df)
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        return df
    
    def train_models(self, df: pd.DataFrame, target_col: str = 'position') -> Dict:
        """Train multiple prediction models."""
        logger.info("Training prediction models...")
        
        # Prepare features
        feature_columns = [
            'grid', 'quali_position', 'driver_avg_position_last_5', 'driver_avg_points_last_5',
            'driver_wins_last_10', 'driver_podiums_last_10', 'constructor_avg_position_last_5',
            'constructor_avg_points_last_5', 'season_progress', 'driver_circuit_avg_position',
            'constructor_circuit_avg_position', 'driver_constructor_combo_avg',
            'grid_vs_avg_position', 'driver_id_encoded', 'constructor_id_encoded', 'circuit_name_encoded'
        ]
        
        # Filter available columns
        available_columns = [col for col in feature_columns if col in df.columns]
        X = df[available_columns].fillna(0)
        y = df[target_col]
        
        # Remove rows with invalid target values
        valid_mask = (y >= 1) & (y <= 20) & (~y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.feature_engineer.scaler.fit_transform(X_train)
        X_test_scaled = self.feature_engineer.scaler.transform(X_test)
        
        models_performance = {}
        
        # Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred)
        
        self.models['random_forest'] = rf_model
        models_performance['random_forest'] = {'mse': rf_mse, 'rmse': np.sqrt(rf_mse)}
        
        # XGBoost
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        
        self.models['xgboost'] = xgb_model
        models_performance['xgboost'] = {'mse': xgb_mse, 'rmse': np.sqrt(xgb_mse)}
        
        # LightGBM
        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_mse = mean_squared_error(y_test, lgb_pred)
        
        self.models['lightgbm'] = lgb_model
        models_performance['lightgbm'] = {'mse': lgb_mse, 'rmse': np.sqrt(lgb_mse)}
        
        # Store feature importance
        self.feature_importance['random_forest'] = dict(zip(available_columns, rf_model.feature_importances_))
        self.feature_importance['xgboost'] = dict(zip(available_columns, xgb_model.feature_importances_))
        self.feature_importance['lightgbm'] = dict(zip(available_columns, lgb_model.feature_importances_))
        
        logger.info("Model training completed!")
        return models_performance
    
    def predict_race(self, race_data: pd.DataFrame, model_name: str = 'xgboost') -> pd.DataFrame:
        """Make predictions for a race."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet!")
        
        model = self.models[model_name]
        
        # Prepare features (same as training)
        feature_columns = [
            'grid', 'quali_position', 'driver_avg_position_last_5', 'driver_avg_points_last_5',
            'driver_wins_last_10', 'driver_podiums_last_10', 'constructor_avg_position_last_5',
            'constructor_avg_points_last_5', 'season_progress', 'driver_circuit_avg_position',
            'constructor_circuit_avg_position', 'driver_constructor_combo_avg',
            'grid_vs_avg_position', 'driver_id_encoded', 'constructor_id_encoded', 'circuit_name_encoded'
        ]
        
        available_columns = [col for col in feature_columns if col in race_data.columns]
        X = race_data[available_columns].fillna(0)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Create results dataframe
        results = race_data[['driver_name', 'constructor_name', 'grid']].copy()
        results['predicted_position'] = predictions
        results['predicted_position'] = results['predicted_position'].round().astype(int)
        results['predicted_position'] = results['predicted_position'].clip(1, 20)
        
        # Sort by predicted position
        results = results.sort_values('predicted_position')
        
        return results
    
    def visualize_feature_importance(self, model_name: str = 'xgboost'):
        """Visualize feature importance for a given model."""
        if model_name not in self.feature_importance:
            logger.error(f"Feature importance not available for {model_name}")
            return
        
        importance = self.feature_importance[model_name]
        features = list(importance.keys())
        values = list(importance.values())
        
        plt.figure(figsize=(12, 8))
        indices = np.argsort(values)[-15:]  # Top 15 features
        plt.barh(range(len(indices)), [values[i] for i in indices])
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importance - {model_name.title()}')
        plt.tight_layout()
        plt.show()
    
    def analyze_model_performance(self, performance: Dict):
        """Analyze and visualize model performance."""
        models = list(performance.keys())
        rmse_values = [performance[model]['rmse'] for model in models]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, rmse_values)
        plt.ylabel('RMSE')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        print("\nModel Performance Summary:")
        print("=" * 50)
        for model, metrics in performance.items():
            print(f"{model.upper()}:")
            print(f"  RMSE: {metrics['rmse']:.3f}")
            print(f"  MSE:  {metrics['mse']:.3f}")
            print()

def main():
    """Main function to run the F1 prediction system."""
    print("🏎️  F1 Race Prediction System")
    print("=" * 50)
    
    # Initialize the prediction model
    f1_model = F1PredictionModel()
    
    # Collect data
    print("📊 Collecting F1 data...")
    df = f1_model.collect_data(start_year=2020, end_year=2024)
    
    if df.empty:
        print("❌ No data collected. Please check your internet connection.")
        return
    
    # Prepare features
    print("🔧 Engineering features...")
    df = f1_model.prepare_features(df)
    
    # Train models
    print("🤖 Training machine learning models...")
    performance = f1_model.train_models(df)
    
    # Analyze performance
    print("📈 Analyzing model performance...")
    f1_model.analyze_model_performance(performance)
    
    # Visualize feature importance
    print("📊 Visualizing feature importance...")
    f1_model.visualize_feature_importance('xgboost')
    
    print("\n✅ F1 Prediction Model is ready!")
    print("📝 To make predictions for the next race, prepare race data with driver grid positions.")
    
    # Save the model for future use
    import pickle
    with open('f1_prediction_model.pkl', 'wb') as f:
        pickle.dump(f1_model, f)
    print("💾 Model saved as 'f1_prediction_model.pkl'")

if __name__ == "__main__":
    main()
