#!/usr/bin/env python3
"""
Enhanced F1 Race Prediction Model
=================================

Comprehensive F1 prediction system integrating:
- Enhanced data collection (FastF1, weather, circuit characteristics)
- Advanced feature engineering (60+ features)
- Multiple ML models with ensemble methods
- Real-time prediction capabilities
- Detailed analysis and visualization

Author: Enhanced F1 Prediction System
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import os

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Enhanced modules
from enhanced_data_collector import EnhancedF1DataCollector
from advanced_feature_engineering import AdvancedF1FeatureEngineer

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedF1PredictionModel:
    """Enhanced F1 prediction model with advanced features and ensemble methods."""
    
    def __init__(self, enable_fastf1=True, enable_weather=False):
        self.data_collector = EnhancedF1DataCollector(enable_fastf1, enable_weather)
        self.feature_engineer = AdvancedF1FeatureEngineer()
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = {}
        self.model_performance = {}
        self.scaler = StandardScaler()
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'num_leaves': 31,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 150,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'random_state': 42
                }
            }
        }
    
    def collect_comprehensive_data(self, start_year: int = 2020, end_year: int = None) -> pd.DataFrame:
        """Collect comprehensive F1 data with all enhancements."""
        if end_year is None:
            end_year = datetime.now().year
            
        logger.info(f"Collecting comprehensive F1 data from {start_year} to {end_year}")
        
        all_data = []
        reliability_data = None
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Processing {year} data...")
            
            # Get enhanced race data
            year_data = self.data_collector.get_enhanced_race_data(year)
            if not year_data.empty:
                all_data.append(year_data)
        
        if not all_data:
            logger.error("No data collected!")
            return pd.DataFrame()
        
        # Combine all years
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Get reliability data
        reliability_data = self.data_collector.get_reliability_data(start_year)
        
        # Apply advanced feature engineering
        logger.info("Applying advanced feature engineering...")
        enhanced_data = self.feature_engineer.create_all_advanced_features(
            combined_data, 
            reliability_data=reliability_data
        )
        
        logger.info(f"Comprehensive data collection complete: {len(enhanced_data)} records with {len(enhanced_data.columns)} features")
        return enhanced_data
    
    def prepare_model_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for model training with intelligent feature selection."""
        
        # Define core feature categories
        core_features = [
            # Grid and qualifying
            'grid', 'quali_position',
            # Historical performance  
            'driver_avg_position_last_5', 'driver_avg_points_last_5',
            'driver_wins_last_10', 'driver_podiums_last_10',
            'constructor_avg_position_last_5', 'constructor_avg_points_last_5',
            # Circuit specific
            'driver_circuit_avg_position', 'constructor_circuit_avg_position',
            'driver_constructor_combo_avg',
            # Season context
            'season_progress', 'driver_championship_position', 'constructor_championship_position',
            # Enhanced features
            'grid_vs_avg_position', 'recent_form', 'consistency', 'position_trend'
        ]
        
        # Weather features
        weather_features = [
            'race_temperature', 'race_humidity', 'rainfall', 'weather_condition_encoded',
            'temp_very_hot', 'temp_hot', 'temp_moderate', 'wet_race', 'driver_wet_performance'
        ]
        
        # Circuit features
        circuit_features = [
            'circuit_length', 'circuit_corners', 'circuit_type_encoded', 'overtaking_difficulty',
            'drs_zones', 'corner_density', 'difficult_overtaking', 'easy_overtaking'
        ]
        
        # Strategic features
        strategic_features = [
            'front_row_start', 'top_10_start', 'back_of_grid', 'strong_qualifier',
            'weak_qualifier', 'race_day_improver', 'title_contender'
        ]
        
        # Reliability features
        reliability_features = [
            'driver_mechanical_failure', 'constructor_mechanical_failure',
            'driver_accident', 'constructor_accident', 'dnf_risk_grid'
        ]
        
        # Categorical encodings
        categorical_features = [
            'driver_id_encoded', 'constructor_id_encoded', 'circuit_name_encoded'
        ]
        
        # Combine all feature categories
        all_feature_categories = [
            core_features, weather_features, circuit_features, 
            strategic_features, reliability_features, categorical_features
        ]
        
        selected_features = []
        for category in all_feature_categories:
            for feature in category:
                if feature in df.columns:
                    selected_features.append(feature)
        
        logger.info(f"Selected {len(selected_features)} features for modeling")
        return df[selected_features], selected_features
    
    def train_enhanced_models(self, df: pd.DataFrame, target_col: str = 'position') -> Dict:
        """Train enhanced models with hyperparameter tuning."""
        logger.info("Training enhanced models...")
        
        # Prepare features
        X, feature_names = self.prepare_model_features(df)
        y = df[target_col]
        
        # Remove invalid targets
        valid_mask = (y >= 1) & (y <= 20) & (~y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Fill missing values
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        performance_metrics = {}
        trained_models = []
        
        # Train individual models
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name}...")
            
            try:
                model = config['model'](**config['params'])
                
                # Use scaled features for some models, original for tree-based
                if model_name in ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                else:
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                cv_rmse = np.sqrt(-cv_scores.mean())
                
                performance_metrics[model_name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_rmse': cv_rmse
                }
                
                self.models[model_name] = model
                trained_models.append((model_name, model))
                
                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(zip(feature_names, model.feature_importances_))
                
                logger.info(f"{model_name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        # Create ensemble model
        if len(trained_models) >= 2:
            logger.info("Creating ensemble model...")
            try:
                ensemble_estimators = [(name, model) for name, model in trained_models[:3]]  # Use top 3 models
                self.ensemble_model = VotingRegressor(estimators=ensemble_estimators)
                self.ensemble_model.fit(X_train, y_train)
                
                ensemble_predictions = self.ensemble_model.predict(X_test)
                ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions))
                ensemble_mae = mean_absolute_error(y_test, ensemble_predictions)
                ensemble_r2 = r2_score(y_test, ensemble_predictions)
                
                performance_metrics['ensemble'] = {
                    'mse': mean_squared_error(y_test, ensemble_predictions),
                    'rmse': ensemble_rmse,
                    'mae': ensemble_mae,
                    'r2': ensemble_r2,
                    'cv_rmse': ensemble_rmse  # Approximation
                }
                
                logger.info(f"Ensemble: RMSE={ensemble_rmse:.3f}, MAE={ensemble_mae:.3f}, R2={ensemble_r2:.3f}")
                
            except Exception as e:
                logger.error(f"Error creating ensemble model: {e}")
        
        self.model_performance = performance_metrics
        self.feature_names = feature_names
        
        return performance_metrics
    
    def predict_race_enhanced(self, race_data: pd.DataFrame, model_name: str = 'ensemble', 
                            confidence_interval: bool = True) -> pd.DataFrame:
        """Make enhanced predictions with confidence intervals."""
        
        model = self.ensemble_model if model_name == 'ensemble' and self.ensemble_model else self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not available!")
        
        # Prepare features
        X, _ = self.prepare_model_features(race_data)
        X = X.fillna(X.median())
        
        # Make predictions
        predictions = model.predict(X)
        
        # Create results dataframe
        results = race_data[['driver_name', 'constructor_name', 'grid']].copy()
        results['predicted_position'] = np.clip(np.round(predictions), 1, 20).astype(int)
        
        # Add confidence intervals if requested
        if confidence_interval:
            if model_name != 'ensemble':
                # For individual models, use prediction intervals
                if hasattr(model, 'estimators_'):  # Random Forest
                    individual_predictions = np.array([estimator.predict(X) for estimator in model.estimators_])
                    prediction_std = np.std(individual_predictions, axis=0)
                    
                    results['prediction_lower'] = np.clip(predictions - 1.96 * prediction_std, 1, 20)
                    results['prediction_upper'] = np.clip(predictions + 1.96 * prediction_std, 1, 20)
                    results['confidence'] = 1 - (prediction_std / predictions.mean())
                else:
                    # Simple confidence based on model performance
                    model_rmse = self.model_performance.get(model_name, {}).get('rmse', 3.0)
                    results['prediction_lower'] = np.clip(predictions - model_rmse, 1, 20)
                    results['prediction_upper'] = np.clip(predictions + model_rmse, 1, 20)
                    results['confidence'] = np.clip(1 - (model_rmse / 10), 0.1, 0.9)
            else:
                # For ensemble, use individual model variation
                if len(self.models) >= 2:
                    model_predictions = []
                    for model_name_individual, model_individual in self.models.items():
                        try:
                            pred = model_individual.predict(X)
                            model_predictions.append(pred)
                        except:
                            continue
                    
                    if model_predictions:
                        model_predictions = np.array(model_predictions)
                        prediction_std = np.std(model_predictions, axis=0)
                        
                        results['prediction_lower'] = np.clip(predictions - prediction_std, 1, 20)
                        results['prediction_upper'] = np.clip(predictions + prediction_std, 1, 20)
                        results['confidence'] = np.clip(1 - (prediction_std / 5), 0.2, 0.95)
        
        # Sort by predicted position
        results = results.sort_values('predicted_position').reset_index(drop=True)
        results.index = results.index + 1  # 1-based indexing for positions
        
        return results
    
    def create_enhanced_visualizations(self):
        """Create comprehensive visualizations of model performance and insights."""
        
        if not self.model_performance:
            logger.warning("No model performance data available for visualization")
            return
        
        # 1. Model Performance Comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE Comparison', 'R² Score Comparison', 'MAE Comparison', 'Feature Importance Top 10'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        models = list(self.model_performance.keys())
        rmse_values = [self.model_performance[model]['rmse'] for model in models]
        r2_values = [self.model_performance[model]['r2'] for model in models]
        mae_values = [self.model_performance[model]['mae'] for model in models]
        
        # RMSE comparison
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='lightblue'),
            row=1, col=1
        )
        
        # R² comparison
        fig.add_trace(
            go.Bar(x=models, y=r2_values, name='R²', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # MAE comparison
        fig.add_trace(
            go.Bar(x=models, y=mae_values, name='MAE', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Feature importance (using best model)
        if self.feature_importance:
            best_model = min(self.model_performance.keys(), 
                           key=lambda x: self.model_performance[x]['rmse'])
            if best_model in self.feature_importance:
                importance = self.feature_importance[best_model]
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                
                fig.add_trace(
                    go.Bar(x=[f[1] for f in top_features], y=[f[0] for f in top_features], 
                          name='Importance', marker_color='gold', orientation='h'),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Enhanced F1 Model Performance Analysis")
        fig.show()
        
        # 2. Feature Importance by Category
        if self.feature_importance:
            self._plot_feature_categories()
    
    def _plot_feature_categories(self):
        """Plot feature importance grouped by categories."""
        categories = self.feature_engineer.get_feature_importance_categories()
        
        if not self.feature_importance:
            return
            
        best_model = min(self.model_performance.keys(), 
                        key=lambda x: self.model_performance[x]['rmse'])
        
        if best_model not in self.feature_importance:
            return
            
        importance = self.feature_importance[best_model]
        
        # Group features by category
        category_importance = {}
        for category, features in categories.items():
            cat_importance = sum(importance.get(feature, 0) for feature in features if feature in importance)
            if cat_importance > 0:
                category_importance[category] = cat_importance
        
        if category_importance:
            plt.figure(figsize=(12, 6))
            categories_sorted = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
            
            plt.bar([cat[0] for cat in categories_sorted], [cat[1] for cat in categories_sorted])
            plt.title(f'Feature Importance by Category ({best_model.title()})')
            plt.xlabel('Feature Category')
            plt.ylabel('Total Importance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    def save_model(self, filename: str = 'enhanced_f1_model.pkl'):
        """Save the enhanced model."""
        model_data = {
            'models': self.models,
            'ensemble_model': self.ensemble_model,
            'feature_engineer': self.feature_engineer,
            'data_collector': self.data_collector,
            'scaler': self.scaler,
            'feature_names': getattr(self, 'feature_names', []),
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Enhanced model saved to {filename}")
    
    @classmethod
    def load_model(cls, filename: str = 'enhanced_f1_model.pkl'):
        """Load the enhanced model."""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls()
        instance.models = model_data['models']
        instance.ensemble_model = model_data['ensemble_model']
        instance.feature_engineer = model_data['feature_engineer']
        instance.data_collector = model_data['data_collector']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data.get('feature_names', [])
        instance.model_performance = model_data['model_performance']
        instance.feature_importance = model_data['feature_importance']
        
        logger.info(f"Enhanced model loaded from {filename}")
        return instance


def main():
    """Main function to run the enhanced F1 prediction system."""
    print("🏎️  Enhanced F1 Race Prediction System")
    print("=" * 60)
    
    # Initialize the enhanced model
    f1_model = EnhancedF1PredictionModel(enable_fastf1=True, enable_weather=False)
    
    # Collect comprehensive data
    print("📊 Collecting comprehensive F1 data...")
    try:
        df = f1_model.collect_comprehensive_data(start_year=2022, end_year=2024)
        
        if df.empty:
            print("❌ No data collected. Please check your setup and internet connection.")
            return
            
        print(f"✅ Collected {len(df)} records with {len(df.columns)} features")
        
        # Train enhanced models
        print("\n🤖 Training enhanced machine learning models...")
        performance = f1_model.train_enhanced_models(df)
        
        # Show performance summary
        print("\n📈 Model Performance Summary:")
        print("=" * 50)
        for model, metrics in performance.items():
            print(f"{model.upper()}:")
            print(f"  RMSE: {metrics['rmse']:.3f}")
            print(f"  MAE:  {metrics['mae']:.3f}")
            print(f"  R²:   {metrics['r2']:.3f}")
            if 'cv_rmse' in metrics:
                print(f"  CV-RMSE: {metrics['cv_rmse']:.3f}")
            print()
        
        # Create visualizations
        print("📊 Creating enhanced visualizations...")
        f1_model.create_enhanced_visualizations()
        
        # Save the model
        print("💾 Saving enhanced model...")
        f1_model.save_model('enhanced_f1_model.pkl')
        
        print("\n✅ Enhanced F1 Prediction Model is ready!")
        print("📝 Use the enhanced model to make predictions with confidence intervals.")
        print("🎯 Features include weather, circuit characteristics, tire strategy, and more!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")
        print("📋 Please check the error logs and ensure all dependencies are installed.")


if __name__ == "__main__":
    main()
