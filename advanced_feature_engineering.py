#!/usr/bin/env python3
"""
Advanced F1 Feature Engineering
===============================

Enhanced feature engineering with tire strategy, weather impacts, circuit characteristics,
reliability metrics, and advanced statistical features for F1 race prediction.

Author: Enhanced F1 Prediction System
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

# Original feature engineer
from f1_predictor import F1FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedF1FeatureEngineer(F1FeatureEngineer):
    """Advanced feature engineering for F1 prediction with enhanced features."""
    
    def __init__(self):
        super().__init__()
        self.weather_encoders = {}
        self.circuit_type_encoder = LabelEncoder()
        
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-based features."""
        df = df.copy()
        
        # Temperature-based features
        if 'race_temperature' in df.columns:
            # Temperature ranges that affect tire performance
            df['temp_very_hot'] = (df['race_temperature'] >= 30).astype(int)
            df['temp_hot'] = ((df['race_temperature'] >= 25) & (df['race_temperature'] < 30)).astype(int)
            df['temp_moderate'] = ((df['race_temperature'] >= 15) & (df['race_temperature'] < 25)).astype(int)
            df['temp_cold'] = (df['race_temperature'] < 15).astype(int)
            
            # Track temperature differential (affects tire degradation)
            if 'track_temperature' in df.columns:
                df['track_air_temp_diff'] = df['track_temperature'] - df['race_temperature']
        
        # Humidity effects (affects downforce and engine performance)
        if 'race_humidity' in df.columns:
            df['high_humidity'] = (df['race_humidity'] >= 80).astype(int)
            df['moderate_humidity'] = ((df['race_humidity'] >= 50) & (df['race_humidity'] < 80)).astype(int)
            df['low_humidity'] = (df['race_humidity'] < 50).astype(int)
        
        # Wind effects
        if 'race_wind_speed' in df.columns:
            df['windy_conditions'] = (df['race_wind_speed'] >= 20).astype(int)
            df['calm_conditions'] = (df['race_wind_speed'] < 10).astype(int)
        
        # Rain impact
        if 'rainfall' in df.columns:
            df['wet_race'] = df['rainfall'].astype(int)
            
            # Historical wet weather performance
            df['driver_wet_performance'] = df.groupby('driver_id').apply(
                lambda x: (x[x['rainfall'] == 1]['position'].mean() if x['rainfall'].sum() > 0 else x['position'].mean())
            ).reindex(df.index, level=1)
        
        # Weather condition encoding
        if 'weather_condition' in df.columns:
            if 'weather_condition' not in self.weather_encoders:
                self.weather_encoders['weather_condition'] = LabelEncoder()
                df['weather_condition_encoded'] = self.weather_encoders['weather_condition'].fit_transform(
                    df['weather_condition'].fillna('Clear').astype(str)
                )
            else:
                df['weather_condition_encoded'] = df['weather_condition'].apply(
                    lambda x: self.weather_encoders['weather_condition'].transform([str(x) if pd.notna(x) else 'Clear'])[0]
                    if (str(x) if pd.notna(x) else 'Clear') in self.weather_encoders['weather_condition'].classes_
                    else -1
                )
        
        return df
    
    def create_circuit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create circuit-specific features."""
        df = df.copy()
        
        # Circuit type encoding
        if 'circuit_type' in df.columns:
            df['circuit_type_encoded'] = self.circuit_type_encoder.fit_transform(
                df['circuit_type'].fillna('permanent')
            )
            
            # Circuit type indicators
            df['street_circuit'] = (df['circuit_type'] == 'street').astype(int)
            df['permanent_circuit'] = (df['circuit_type'] == 'permanent').astype(int)
        
        # Track characteristics impact on performance
        if 'circuit_length' in df.columns:
            df['long_circuit'] = (df['circuit_length'] >= 5.5).astype(int)
            df['short_circuit'] = (df['circuit_length'] <= 4.0).astype(int)
        
        if 'circuit_corners' in df.columns:
            df['corner_density'] = df['circuit_corners'] / df.get('circuit_length', 5.0)
            df['high_corner_density'] = (df['corner_density'] >= 4.0).astype(int)
        
        if 'overtaking_difficulty' in df.columns:
            df['difficult_overtaking'] = (df['overtaking_difficulty'] >= 7.0).astype(int)
            df['easy_overtaking'] = (df['overtaking_difficulty'] <= 3.0).astype(int)
        
        if 'circuit_elevation_change' in df.columns:
            df['elevation_challenging'] = (df['circuit_elevation_change'] >= 50).astype(int)
        
        if 'drs_zones' in df.columns:
            df['multiple_drs'] = (df['drs_zones'] >= 2).astype(int)
        
        # Historical circuit performance by driver characteristics
        circuit_features = ['circuit_length', 'circuit_corners', 'overtaking_difficulty']
        for feature in circuit_features:
            if feature in df.columns:
                # Driver preference for circuit type
                df[f'driver_{feature}_preference'] = df.groupby('driver_id')[feature].transform(
                    lambda x: (df.loc[x.index, 'position'] * df.loc[x.index, feature]).corr(df.loc[x.index, feature]) 
                    if len(x) > 3 else 0
                )
        
        return df
    
    def create_tire_strategy_features(self, df: pd.DataFrame, tire_data: Optional[Dict] = None) -> pd.DataFrame:
        """Create tire strategy and compound features."""
        df = df.copy()
        
        # If we have detailed tire data from FastF1
        if tire_data:
            # Add tire strategy features based on historical data
            for driver_id in df['driver_id'].unique():
                if driver_id in tire_data:
                    strategy = tire_data[driver_id]
                    # Add features like number of stops, compound preferences, etc.
                    df.loc[df['driver_id'] == driver_id, 'typical_stops'] = len(strategy)
                    # More tire strategy features can be added here
        
        # Grid position impact on tire strategy
        if 'grid' in df.columns:
            df['front_row_start'] = (df['grid'] <= 2).astype(int)
            df['top_10_start'] = (df['grid'] <= 10).astype(int)
            df['back_of_grid'] = (df['grid'] >= 15).astype(int)
        
        # Historical tire performance by track temperature
        if 'track_temperature' in df.columns:
            df['hot_track'] = (df['track_temperature'] >= 45).astype(int)
            df['cold_track'] = (df['track_temperature'] <= 25).astype(int)
        
        return df
    
    def create_reliability_features(self, df: pd.DataFrame, reliability_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create reliability-based features."""
        df = df.copy()
        
        if reliability_data is not None and not reliability_data.empty:
            # Calculate reliability metrics
            driver_reliability = reliability_data.groupby('driver_id').agg({
                'mechanical_failure': 'mean',
                'accident': 'mean',
                'season': 'count'
            }).rename(columns={'season': 'total_races'})
            
            constructor_reliability = reliability_data.groupby('constructor_id').agg({
                'mechanical_failure': 'mean',
                'accident': 'mean',
                'season': 'count'
            }).rename(columns={'season': 'total_races'})
            
            # Merge reliability data
            df = df.merge(
                driver_reliability.add_prefix('driver_'),
                left_on='driver_id',
                right_index=True,
                how='left'
            )
            
            df = df.merge(
                constructor_reliability.add_prefix('constructor_'),
                left_on='constructor_id',
                right_index=True,
                how='left'
            )
            
            # Fill missing reliability data
            reliability_cols = [col for col in df.columns if 'reliability' in col or 'mechanical_failure' in col or 'accident' in col]
            for col in reliability_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
        
        # DNF risk based on grid position (higher risk for back markers)
        if 'grid' in df.columns:
            df['dnf_risk_grid'] = np.log(df['grid']) / np.log(20)  # Normalized log scale
        
        return df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum and form-based features."""
        df = df.copy()
        df = df.sort_values(['driver_id', 'season', 'round'])
        
        # Recent form (last 3 races)
        df['recent_form'] = df.groupby('driver_id')['position'].transform(
            lambda x: x.rolling(3, min_periods=1).apply(
                lambda r: (r <= 10).sum() / len(r)
            ).shift(1)
        )
        
        # Consistency measure (standard deviation of recent positions)
        df['consistency'] = df.groupby('driver_id')['position'].transform(
            lambda x: x.rolling(5, min_periods=2).std().shift(1)
        )
        
        # Improving/declining trend
        df['position_trend'] = df.groupby('driver_id')['position'].transform(
            lambda x: x.rolling(4, min_periods=2).apply(
                lambda r: stats.linregress(range(len(r)), r)[0] if len(r) >= 2 else 0
            ).shift(1)
        )
        
        # Hot streak (consecutive good results)
        df['consecutive_points'] = df.groupby('driver_id').apply(
            lambda x: x['points'].gt(0).groupby((x['points'].le(0)).cumsum()).cumsum()
        ).reset_index(level=0, drop=True)
        
        # Championship battle pressure
        if 'driver_championship_position' in df.columns:
            df['title_contender'] = (df['driver_championship_position'] <= 3).astype(int)
            df['midfield_battle'] = ((df['driver_championship_position'] > 3) & 
                                   (df['driver_championship_position'] <= 10)).astype(int)
        
        return df
    
    def create_competitor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on competitor analysis."""
        df = df.copy()
        
        # Teammate comparison
        df['teammate_avg_position'] = df.groupby(['constructor_id', 'season', 'round'])['position'].transform(
            lambda x: x.sum() / len(x) if len(x) == 2 else x.iloc[0]
        )
        
        # Constructor strength at circuit
        circuit_wins = df.groupby(['constructor_id', 'circuit_name']).apply(
            lambda x: (x['position'] == 1).sum()
        ).reset_index(name='constructor_circuit_wins')
        
        df = df.merge(
            circuit_wins,
            on=['constructor_id', 'circuit_name'],
            how='left'
        )
        df['constructor_circuit_wins'] = df['constructor_circuit_wins'].fillna(0)
        
        # Driver vs constructor performance
        df['driver_constructor_synergy'] = (
            df['driver_avg_position_last_5'] - df['constructor_avg_position_last_5']
        )
        
        # Field strength (average competitiveness of grid)
        df['grid_competitiveness'] = df.groupby(['season', 'round'])['driver_avg_position_last_5'].transform('std')
        
        return df
    
    def create_strategic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create strategy-related features."""
        df = df.copy()
        
        # Qualifying vs race performance gap
        if 'quali_position' in df.columns:
            df['quali_race_gap'] = df['quali_position'] - df['position']
            df['quali_race_gap_avg'] = df.groupby('driver_id')['quali_race_gap'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            )
        
        # Saturday performance indicator
        if 'quali_position' in df.columns:
            df['strong_qualifier'] = (df['quali_position'] <= 5).astype(int)
            df['weak_qualifier'] = (df['quali_position'] >= 15).astype(int)
        
        # Race day performance
        race_day_stats = df.groupby('driver_id').apply(
            lambda x: ((x['position'] < x['grid']).sum() / len(x) if 'grid' in x.columns else 0)
        ).to_dict()
        
        df['race_day_improver'] = df['driver_id'].map(race_day_stats).fillna(0)
        
        return df
    
    def create_advanced_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced interaction features."""
        df = df.copy()
        
        # Weather × Circuit interactions
        weather_circuit_features = []
        if 'rainfall' in df.columns and 'overtaking_difficulty' in df.columns:
            df['wet_difficult_overtaking'] = df['rainfall'] * df['overtaking_difficulty']
            weather_circuit_features.append('wet_difficult_overtaking')
        
        if 'race_temperature' in df.columns and 'circuit_length' in df.columns:
            df['temp_length_interaction'] = df['race_temperature'] * df['circuit_length']
            weather_circuit_features.append('temp_length_interaction')
        
        # Driver × Circuit specialization
        if 'driver_circuit_avg_position' in df.columns and 'circuit_type' in df.columns:
            # Encode circuit type preferences for each driver
            for circuit_type in df['circuit_type'].unique():
                if pd.notna(circuit_type):
                    # Calculate driver performance at specific circuit type
                    circuit_type_data = df[df['circuit_type'] == circuit_type]
                    driver_circuit_perf = circuit_type_data.groupby('driver_id')['position'].mean()
                    
                    # Map back to main dataframe
                    df[f'driver_prefers_{circuit_type}'] = df['driver_id'].map(
                        driver_circuit_perf < 10
                    ).fillna(False).astype(int)
        
        # Grid position × Weather
        if 'grid' in df.columns and 'rainfall' in df.columns:
            df['grid_wet_interaction'] = df['grid'] * df['rainfall']
        
        # Constructor × Circuit type
        if 'constructor_id' in df.columns and 'circuit_type' in df.columns:
            df['constructor_circuit_specialty'] = df.groupby(['constructor_id', 'circuit_type'])['position'].transform(
                lambda x: (x.mean() <= 8) if len(x) > 2 else False
            ).astype(int)
        
        return df
    
    def create_all_advanced_features(self, df: pd.DataFrame, 
                                   tire_data: Optional[Dict] = None,
                                   reliability_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create all advanced features in the correct order."""
        logger.info("Creating advanced features...")
        
        # Start with base features from parent class
        df = self.create_historical_features(df)
        df = self.create_season_features(df)
        df = self.encode_categorical_features(df)
        df = self.create_interaction_features(df)
        
        # Add advanced features
        logger.info("Adding weather features...")
        df = self.create_weather_features(df)
        
        logger.info("Adding circuit features...")
        df = self.create_circuit_features(df)
        
        logger.info("Adding tire strategy features...")
        df = self.create_tire_strategy_features(df, tire_data)
        
        logger.info("Adding reliability features...")
        df = self.create_reliability_features(df, reliability_data)
        
        logger.info("Adding momentum features...")
        df = self.create_momentum_features(df)
        
        logger.info("Adding competitor features...")
        df = self.create_competitor_features(df)
        
        logger.info("Adding strategic features...")
        df = self.create_strategic_features(df)
        
        logger.info("Adding advanced interaction features...")
        df = self.create_advanced_interaction_features(df)
        
        # Fill missing values
        logger.info("Handling missing values...")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df
    
    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """Categorize features for better analysis."""
        return {
            'Historical Performance': [
                'driver_avg_position_last_5', 'driver_avg_points_last_5', 'driver_wins_last_10',
                'driver_podiums_last_10', 'constructor_avg_position_last_5', 'constructor_avg_points_last_5'
            ],
            'Weather & Conditions': [
                'race_temperature', 'race_humidity', 'rainfall', 'weather_condition_encoded',
                'temp_very_hot', 'temp_hot', 'temp_moderate', 'temp_cold', 'wet_race',
                'track_air_temp_diff', 'windy_conditions'
            ],
            'Circuit Characteristics': [
                'circuit_length', 'circuit_corners', 'circuit_type_encoded', 'overtaking_difficulty',
                'drs_zones', 'corner_density', 'difficult_overtaking', 'easy_overtaking',
                'street_circuit', 'permanent_circuit'
            ],
            'Grid & Qualifying': [
                'grid', 'quali_position', 'grid_vs_avg_position', 'strong_qualifier',
                'weak_qualifier', 'front_row_start', 'top_10_start', 'back_of_grid'
            ],
            'Reliability & Risk': [
                'driver_mechanical_failure', 'constructor_mechanical_failure', 'driver_accident',
                'constructor_accident', 'dnf_risk_grid'
            ],
            'Form & Momentum': [
                'recent_form', 'consistency', 'position_trend', 'consecutive_points',
                'race_day_improver', 'title_contender'
            ],
            'Strategic & Competitive': [
                'season_progress', 'driver_championship_position', 'constructor_championship_position',
                'teammate_avg_position', 'driver_constructor_synergy', 'grid_competitiveness'
            ]
        }


def main():
    """Test the advanced feature engineering."""
    print("🔧 Advanced F1 Feature Engineering Test")
    print("=" * 50)
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'season': [2024] * 10,
        'round': [1] * 10,
        'driver_id': ['hamilton', 'verstappen', 'leclerc'] * 3 + ['norris'],
        'constructor_id': ['mercedes', 'red_bull', 'ferrari'] * 3 + ['mclaren'],
        'position': [3, 1, 2, 5, 2, 4, 7, 3, 6, 8],
        'grid': [2, 1, 3, 6, 1, 5, 8, 4, 7, 9],
        'points': [15, 25, 18, 10, 18, 12, 6, 15, 8, 4],
        'driver_points': [150, 250, 180, 100, 180, 120, 60, 150, 80, 40],  # Season points
        'constructor_points': [300, 400, 350, 200, 350, 240, 120, 300, 160, 80],  # Season points
        'circuit_name': ['Bahrain'] * 10,
        'race_temperature': [28.5] * 10,
        'race_humidity': [65] * 10,
        'rainfall': [0] * 10,
        'circuit_length': [5.412] * 10,
        'circuit_corners': [15] * 10,
        'circuit_type': ['permanent'] * 10,
        'overtaking_difficulty': [3.5] * 10,
        'drs_zones': [3] * 10
    })
    
    # Initialize feature engineer
    feature_engineer = AdvancedF1FeatureEngineer()
    
    # Test feature creation
    print("\n🔧 Testing Advanced Feature Engineering...")
    try:
        enhanced_data = feature_engineer.create_all_advanced_features(sample_data)
        
        print(f"✅ Original features: {len(sample_data.columns)}")
        print(f"✅ Enhanced features: {len(enhanced_data.columns)}")
        print(f"✅ New features added: {len(enhanced_data.columns) - len(sample_data.columns)}")
        
        # Show feature categories
        categories = feature_engineer.get_feature_importance_categories()
        print(f"\n📊 Feature Categories:")
        for category, features in categories.items():
            available_features = [f for f in features if f in enhanced_data.columns]
            print(f"  {category}: {len(available_features)} features")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
    
    print("\n✅ Advanced Feature Engineering ready!")


if __name__ == "__main__":
    main()
