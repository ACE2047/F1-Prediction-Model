#!/usr/bin/env python3
"""
F1 Race Prediction Demo
=======================

This demo shows how the F1 prediction system works using sample data.
It demonstrates the complete workflow of data processing, feature engineering,
model training, and race prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def create_sample_f1_data():
    """Create realistic sample F1 data for demonstration."""
    
    # Sample drivers and constructors
    drivers = [
        'max_verstappen', 'lewis_hamilton', 'charles_leclerc', 'lando_norris',
        'oscar_piastri', 'carlos_sainz', 'george_russell', 'sergio_perez',
        'fernando_alonso', 'lance_stroll', 'pierre_gasly', 'esteban_ocon',
        'alex_albon', 'logan_sargeant', 'yuki_tsunoda', 'daniel_ricciardo',
        'nico_hulkenberg', 'kevin_magnussen', 'valtteri_bottas', 'zhou_guanyu'
    ]
    
    driver_names = [
        'Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc', 'Lando Norris',
        'Oscar Piastri', 'Carlos Sainz Jr', 'George Russell', 'Sergio Pérez',
        'Fernando Alonso', 'Lance Stroll', 'Pierre Gasly', 'Esteban Ocon',
        'Alex Albon', 'Logan Sargeant', 'Yuki Tsunoda', 'Daniel Ricciardo',
        'Nico Hülkenberg', 'Kevin Magnussen', 'Valtteri Bottas', 'Zhou Guanyu'
    ]
    
    constructors = [
        'red_bull', 'red_bull', 'ferrari', 'mclaren',
        'mclaren', 'ferrari', 'mercedes', 'red_bull',
        'aston_martin', 'aston_martin', 'alpine', 'alpine',
        'williams', 'williams', 'alphatauri', 'alphatauri',
        'haas', 'haas', 'kick_sauber', 'kick_sauber'
    ]
    
    constructor_names = [
        'Red Bull Racing', 'Red Bull Racing', 'Ferrari', 'McLaren',
        'McLaren', 'Ferrari', 'Mercedes', 'Red Bull Racing',
        'Aston Martin', 'Aston Martin', 'Alpine', 'Alpine',
        'Williams', 'Williams', 'AlphaTauri', 'AlphaTauri',
        'Haas F1', 'Haas F1', 'Kick Sauber', 'Kick Sauber'
    ]
    
    circuits = [
        'bahrain', 'saudi_arabia', 'australia', 'japan', 'china',
        'miami', 'imola', 'monaco', 'canada', 'spain',
        'austria', 'silverstone', 'hungary', 'spa', 'netherlands',
        'monza', 'singapore', 'austin', 'mexico', 'brazil', 'las_vegas', 'qatar', 'abu_dhabi'
    ]
    
    circuit_names = [
        'Bahrain International Circuit', 'Jeddah Corniche Circuit', 'Albert Park Circuit',
        'Suzuka International Racing Course', 'Shanghai International Circuit',
        'Miami International Autodrome', 'Autodromo Enzo e Dino Ferrari', 'Circuit de Monaco',
        'Circuit Gilles Villeneuve', 'Circuit de Barcelona-Catalunya',
        'Red Bull Ring', 'Silverstone Circuit', 'Hungaroring', 'Circuit de Spa-Francorchamps',
        'Circuit Zandvoort', 'Autodromo Nazionale di Monza', 'Marina Bay Street Circuit',
        'Circuit of the Americas', 'Autódromo Hermanos Rodríguez', 'Interlagos',
        'Las Vegas Strip Circuit', 'Losail International Circuit', 'Yas Marina Circuit'
    ]
    
    # Team performance tiers (affects base performance)
    team_performance = {
        'red_bull': 1,
        'ferrari': 2,
        'mclaren': 2,
        'mercedes': 3,
        'aston_martin': 4,
        'alpine': 5,
        'williams': 6,
        'alphatauri': 6,
        'haas': 7,
        'kick_sauber': 8
    }
    
    # Driver skill levels (affects performance within team)
    driver_skill = {
        'max_verstappen': 1.0, 'charles_leclerc': 0.95, 'lewis_hamilton': 0.98,
        'lando_norris': 0.90, 'oscar_piastri': 0.85, 'carlos_sainz': 0.88,
        'george_russell': 0.87, 'sergio_perez': 0.82, 'fernando_alonso': 0.92,
        'lance_stroll': 0.75, 'pierre_gasly': 0.80, 'esteban_ocon': 0.78,
        'alex_albon': 0.79, 'logan_sargeant': 0.65, 'yuki_tsunoda': 0.77,
        'daniel_ricciardo': 0.81, 'nico_hulkenberg': 0.83, 'kevin_magnussen': 0.76,
        'valtteri_bottas': 0.84, 'zhou_guanyu': 0.72
    }
    
    # Generate data for multiple seasons
    data = []
    seasons = [2022, 2023, 2024]
    
    for season in seasons:
        for round_num, (circuit, circuit_name) in enumerate(zip(circuits, circuit_names), 1):
            # Generate race results for this round
            race_results = []
            
            for i, (driver, driver_name, constructor, constructor_name) in enumerate(
                zip(drivers, driver_names, constructors, constructor_names)
            ):
                # Base performance based on team and driver skill
                base_performance = team_performance[constructor] + (1 - driver_skill[driver]) * 3
                
                # Add circuit-specific variation
                circuit_factor = np.random.normal(0, 0.5)
                
                # Add qualifying position (affects grid)
                quali_performance = base_performance + np.random.normal(0, 1.5)
                grid_position = max(1, min(20, int(quali_performance + np.random.normal(0, 1))))
                
                # Race position based on grid + race pace + randomness
                race_performance = (
                    base_performance * 0.6 +  # Team/driver skill
                    grid_position * 0.2 +     # Grid position effect
                    circuit_factor +          # Circuit-specific
                    np.random.normal(0, 2)    # Race randomness
                )
                
                position = max(1, min(20, int(race_performance)))
                
                # Points based on position
                points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
                points = points_map.get(position, 0)
                
                race_results.append({
                    'season': season,
                    'round': round_num,
                    'race_name': f'{circuit_name.split()[0]} Grand Prix',
                    'circuit_name': circuit_name,
                    'driver_id': driver,
                    'driver_name': driver_name,
                    'constructor_id': constructor,
                    'constructor_name': constructor_name,
                    'grid': grid_position,
                    'position': position,
                    'points': points,
                    'quali_position': grid_position
                })
            
            # Sort by position to ensure correct finishing order
            race_results.sort(key=lambda x: x['position'])
            for i, result in enumerate(race_results, 1):
                result['position'] = i
                points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
                result['points'] = points_map.get(i, 0)
            
            data.extend(race_results)
    
    return pd.DataFrame(data)

def create_features(df):
    """Create features for machine learning."""
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
    
    # Season progress
    df['season_progress'] = df['round'] / df.groupby('season')['round'].transform('max')
    
    # Encode categorical features
    le_driver = LabelEncoder()
    le_constructor = LabelEncoder()
    le_circuit = LabelEncoder()
    
    df['driver_id_encoded'] = le_driver.fit_transform(df['driver_id'])
    df['constructor_id_encoded'] = le_constructor.fit_transform(df['constructor_id'])
    df['circuit_name_encoded'] = le_circuit.fit_transform(df['circuit_name'])
    
    # Interaction features
    df['driver_constructor_combo_avg'] = df.groupby(['driver_id', 'constructor_id'])['position'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df['grid_vs_avg_position'] = df['grid'] - df['driver_avg_position_last_5']
    
    # Fill missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    return df, le_driver, le_constructor, le_circuit

def train_models(df):
    """Train prediction models."""
    print("🤖 Training machine learning models...")
    
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
    y = df['position']
    
    # Remove rows with invalid target values
    valid_mask = (y >= 1) & (y <= 20) & (~y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {}
    performance = {}
    
    # Random Forest
    print("  - Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    models['random_forest'] = rf_model
    performance['random_forest'] = rf_rmse
    
    # XGBoost
    print("  - Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    
    models['xgboost'] = xgb_model
    performance['xgboost'] = xgb_rmse
    
    return models, performance, available_columns

def predict_next_race(models, encoders, feature_columns):
    """Predict the next race results."""
    print("\n🏁 Predicting next race results...")
    
    le_driver, le_constructor, le_circuit = encoders
    
    # Sample next race data (Hungarian GP - Hungaroring)
    next_race_data = [
        {'driver_id': 'lando_norris', 'driver_name': 'Lando Norris', 'constructor_id': 'mclaren', 'constructor_name': 'McLaren', 'grid': 1},
        {'driver_id': 'oscar_piastri', 'driver_name': 'Oscar Piastri', 'constructor_id': 'mclaren', 'constructor_name': 'McLaren', 'grid': 2},
        {'driver_id': 'max_verstappen', 'driver_name': 'Max Verstappen', 'constructor_id': 'red_bull', 'constructor_name': 'Red Bull Racing', 'grid': 3},
        {'driver_id': 'charles_leclerc', 'driver_name': 'Charles Leclerc', 'constructor_id': 'ferrari', 'constructor_name': 'Ferrari', 'grid': 4},
        {'driver_id': 'lewis_hamilton', 'driver_name': 'Lewis Hamilton', 'constructor_id': 'mercedes', 'constructor_name': 'Mercedes', 'grid': 5},
        {'driver_id': 'carlos_sainz', 'driver_name': 'Carlos Sainz Jr', 'constructor_id': 'ferrari', 'constructor_name': 'Ferrari', 'grid': 6},
        {'driver_id': 'george_russell', 'driver_name': 'George Russell', 'constructor_id': 'mercedes', 'constructor_name': 'Mercedes', 'grid': 7},
        {'driver_id': 'sergio_perez', 'driver_name': 'Sergio Pérez', 'constructor_id': 'red_bull', 'constructor_name': 'Red Bull Racing', 'grid': 8},
        {'driver_id': 'fernando_alonso', 'driver_name': 'Fernando Alonso', 'constructor_id': 'aston_martin', 'constructor_name': 'Aston Martin', 'grid': 9},
        {'driver_id': 'daniel_ricciardo', 'driver_name': 'Daniel Ricciardo', 'constructor_id': 'alphatauri', 'constructor_name': 'AlphaTauri', 'grid': 10},
        {'driver_id': 'yuki_tsunoda', 'driver_name': 'Yuki Tsunoda', 'constructor_id': 'alphatauri', 'constructor_name': 'AlphaTauri', 'grid': 11},
        {'driver_id': 'lance_stroll', 'driver_name': 'Lance Stroll', 'constructor_id': 'aston_martin', 'constructor_name': 'Aston Martin', 'grid': 12},
        {'driver_id': 'alex_albon', 'driver_name': 'Alex Albon', 'constructor_id': 'williams', 'constructor_name': 'Williams', 'grid': 13},
        {'driver_id': 'pierre_gasly', 'driver_name': 'Pierre Gasly', 'constructor_id': 'alpine', 'constructor_name': 'Alpine', 'grid': 14},
        {'driver_id': 'esteban_ocon', 'driver_name': 'Esteban Ocon', 'constructor_id': 'alpine', 'constructor_name': 'Alpine', 'grid': 15},
        {'driver_id': 'nico_hulkenberg', 'driver_name': 'Nico Hülkenberg', 'constructor_id': 'haas', 'constructor_name': 'Haas F1', 'grid': 16},
        {'driver_id': 'kevin_magnussen', 'driver_name': 'Kevin Magnussen', 'constructor_id': 'haas', 'constructor_name': 'Haas F1', 'grid': 17},
        {'driver_id': 'valtteri_bottas', 'driver_name': 'Valtteri Bottas', 'constructor_id': 'kick_sauber', 'constructor_name': 'Kick Sauber', 'grid': 18},
        {'driver_id': 'logan_sargeant', 'driver_name': 'Logan Sargeant', 'constructor_id': 'williams', 'constructor_name': 'Williams', 'grid': 19},
        {'driver_id': 'zhou_guanyu', 'driver_name': 'Zhou Guanyu', 'constructor_id': 'kick_sauber', 'constructor_name': 'Kick Sauber', 'grid': 20}
    ]
    
    race_df = pd.DataFrame(next_race_data)
    
    # Add required features with realistic values
    race_df['season'] = 2025
    race_df['round'] = 13
    race_df['circuit_name'] = 'Hungaroring'
    race_df['quali_position'] = race_df['grid']
    race_df['season_progress'] = 0.54
    
    # Add historical performance estimates
    performance_estimates = {
        'max_verstappen': {'avg_pos': 2.1, 'avg_points': 18.5, 'wins': 8, 'podiums': 9},
        'lando_norris': {'avg_pos': 4.2, 'avg_points': 12.8, 'wins': 2, 'podiums': 6},
        'oscar_piastri': {'avg_pos': 5.1, 'avg_points': 10.2, 'wins': 1, 'podiums': 4},
        'charles_leclerc': {'avg_pos': 4.8, 'avg_points': 11.5, 'wins': 1, 'podiums': 5},
        'carlos_sainz': {'avg_pos': 6.2, 'avg_points': 8.9, 'wins': 0, 'podiums': 3},
        'lewis_hamilton': {'avg_pos': 6.8, 'avg_points': 8.1, 'wins': 0, 'podiums': 2},
        'george_russell': {'avg_pos': 7.1, 'avg_points': 7.8, 'wins': 0, 'podiums': 2},
        'sergio_perez': {'avg_pos': 8.2, 'avg_points': 6.5, 'wins': 0, 'podiums': 1},
        'fernando_alonso': {'avg_pos': 9.5, 'avg_points': 4.2, 'wins': 0, 'podiums': 0},
        'lance_stroll': {'avg_pos': 12.1, 'avg_points': 2.1, 'wins': 0, 'podiums': 0}
    }
    
    # Default values for drivers not in performance_estimates
    default_perf = {'avg_pos': 13.0, 'avg_points': 1.5, 'wins': 0, 'podiums': 0}
    
    for _, row in race_df.iterrows():
        driver = row['driver_id']
        perf = performance_estimates.get(driver, default_perf)
        
        race_df.loc[race_df['driver_id'] == driver, 'driver_avg_position_last_5'] = perf['avg_pos']
        race_df.loc[race_df['driver_id'] == driver, 'driver_avg_points_last_5'] = perf['avg_points']
        race_df.loc[race_df['driver_id'] == driver, 'driver_wins_last_10'] = perf['wins']
        race_df.loc[race_df['driver_id'] == driver, 'driver_podiums_last_10'] = perf['podiums']
    
    # Constructor averages (simplified)
    constructor_perf = {
        'red_bull': 3.5, 'mclaren': 4.8, 'ferrari': 6.2, 'mercedes': 7.5,
        'aston_martin': 11.2, 'alpine': 12.8, 'williams': 14.5, 'alphatauri': 15.2,
        'haas': 16.1, 'kick_sauber': 17.8
    }
    
    for constructor, avg_pos in constructor_perf.items():
        mask = race_df['constructor_id'] == constructor
        race_df.loc[mask, 'constructor_avg_position_last_5'] = avg_pos
        race_df.loc[mask, 'constructor_avg_points_last_5'] = max(0, 25 - avg_pos * 1.5)
    
    # Circuit-specific performance (Hungarian GP favorites - technical track rewards precision)
    hungarian_bonus = {
        'max_verstappen': -0.8,  # Strong in technical sections
        'charles_leclerc': -0.6,  # Ferrari historically good at Hungary
        'fernando_alonso': -1.0,  # Master of Hungary, 2-time winner
        'lewis_hamilton': -0.4,  # Multiple Hungary winner
        'daniel_ricciardo': -0.5,  # Won in 2021 from 3rd
        'carlos_sainz': -0.3,    # Consistent at technical tracks
        'lando_norris': -0.2     # McLaren improving at technical circuits
    }
    
    for driver, bonus in hungarian_bonus.items():
        if driver in race_df['driver_id'].values:
            current_avg = race_df[race_df['driver_id'] == driver]['driver_avg_position_last_5'].iloc[0]
            race_df.loc[race_df['driver_id'] == driver, 'driver_circuit_avg_position'] = current_avg + bonus
            race_df.loc[race_df['driver_id'] == driver, 'constructor_circuit_avg_position'] = current_avg + bonus * 0.5
    
    # Fill missing circuit data
    race_df['driver_circuit_avg_position'] = race_df['driver_circuit_avg_position'].fillna(race_df['driver_avg_position_last_5'])
    race_df['constructor_circuit_avg_position'] = race_df['constructor_circuit_avg_position'].fillna(race_df['constructor_avg_position_last_5'])
    
    # Driver-constructor combo
    race_df['driver_constructor_combo_avg'] = (race_df['driver_avg_position_last_5'] + race_df['constructor_avg_position_last_5']) / 2
    
    # Grid vs average
    race_df['grid_vs_avg_position'] = race_df['grid'] - race_df['driver_avg_position_last_5']
    
    # Encode categorical features
    race_df['driver_id_encoded'] = race_df['driver_id'].apply(
        lambda x: le_driver.transform([x])[0] if x in le_driver.classes_ else -1
    )
    race_df['constructor_id_encoded'] = race_df['constructor_id'].apply(
        lambda x: le_constructor.transform([x])[0] if x in le_constructor.classes_ else -1
    )
    race_df['circuit_name_encoded'] = race_df['circuit_name'].apply(
        lambda x: le_circuit.transform([x])[0] if x in le_circuit.classes_ else -1
    )
    
    # Prepare features for prediction
    X_pred = race_df[feature_columns].fillna(0)
    
    # Make predictions with XGBoost (best performing model)
    model = models['xgboost']
    predictions = model.predict(X_pred)
    
    # Create results
    results = race_df[['driver_name', 'constructor_name', 'grid']].copy()
    results['predicted_position'] = predictions
    results['predicted_position'] = results['predicted_position'].round().astype(int)
    results['predicted_position'] = results['predicted_position'].clip(1, 20)
    
    # Sort by predicted position
    results = results.sort_values('predicted_position')
    results = results.reset_index(drop=True)
    
    return results

def visualize_results(df, performance, predictions):
    """Create visualizations."""
    print("\n📊 Creating visualizations...")
    
    # 1. Model Performance Comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    models = list(performance.keys())
    rmse_values = list(performance.values())
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = plt.bar(models, rmse_values, color=colors)
    plt.ylabel('RMSE (Lower is Better)')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, rmse in zip(bars, rmse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rmse:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Constructor Performance Distribution
    plt.subplot(1, 3, 2)
    constructor_avg = df.groupby('constructor_name')['position'].mean().sort_values()
    
    colors_map = {
        'Red Bull Racing': '#0600EF', 'Ferrari': '#DC143C', 'McLaren': '#FF8700',
        'Mercedes': '#00D2BE', 'Aston Martin': '#006F62', 'Alpine': '#0090FF',
        'Williams': '#005AFF', 'AlphaTauri': '#2B4562', 'Haas F1': '#FFFFFF',
        'Kick Sauber': '#900000'
    }
    
    colors = [colors_map.get(team, '#888888') for team in constructor_avg.index]
    
    plt.barh(constructor_avg.index, constructor_avg.values, color=colors)
    plt.xlabel('Average Finishing Position')
    plt.title('Constructor Performance (2022-2024)')
    plt.gca().invert_yaxis()
    
    # 3. Next Race Prediction
    plt.subplot(1, 3, 3)
    y_pos = np.arange(len(predictions))
    
    # Color code by constructor
    pred_colors = []
    for _, row in predictions.iterrows():
        pred_colors.append(colors_map.get(row['constructor_name'], '#888888'))
    
    plt.barh(y_pos, predictions['predicted_position'], color=pred_colors)
    plt.yticks(y_pos, [f"{row['driver_name'].split()[-1]}" for _, row in predictions.iterrows()])
    plt.xlabel('Predicted Position')
    plt.title('Next Race Prediction')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    # 4. Driver Performance Over Time
    plt.figure(figsize=(15, 8))
    
    # Top drivers performance trend
    top_drivers = ['Max Verstappen', 'Lando Norris', 'Charles Leclerc', 'Oscar Piastri', 'Lewis Hamilton']
    
    for driver in top_drivers:
        driver_data = df[df['driver_name'] == driver].copy()
        if not driver_data.empty:
            driver_data['race_number'] = range(len(driver_data))
            driver_data['rolling_avg'] = driver_data['position'].rolling(5, min_periods=1).mean()
            
            plt.plot(driver_data['race_number'], driver_data['rolling_avg'], 
                    marker='o', linewidth=2, label=driver, markersize=4)
    
    plt.xlabel('Race Number')
    plt.ylabel('Average Position (5-race rolling)')
    plt.title('Driver Performance Trends (2022-2024)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    plt.show()

def main():
    """Main function to run the F1 prediction demo."""
    print("🏎️  F1 Race Prediction System - DEMO")
    print("=" * 50)
    
    # Create sample data
    print("📊 Generating sample F1 data...")
    df = create_sample_f1_data()
    print(f"   Generated {len(df)} race records from {len(df['season'].unique())} seasons")
    
    # Create features
    print("🔧 Engineering features...")
    df_features, le_driver, le_constructor, le_circuit = create_features(df)
    print(f"   Created {len(df_features.columns)} features")
    
    # Train models
    models, performance, feature_columns = train_models(df_features)
    
    # Print performance
    print("\n📈 Model Performance:")
    print("-" * 30)
    for model, rmse in performance.items():
        print(f"{model.upper():12} RMSE: {rmse:.3f}")
    
    # Predict next race
    predictions = predict_next_race(models, (le_driver, le_constructor, le_circuit), feature_columns)
    
    # Display results
    print(f"\n🏁 HUNGARIAN GRAND PRIX PREDICTION")
    print("=" * 60)
    print(f"{'Pos':<4} {'Driver':<25} {'Team':<20} {'Grid':<6}")
    print("-" * 60)
    
    for i, (_, row) in enumerate(predictions.iterrows(), 1):
        print(f"{i:<4} {row['driver_name']:<25} {row['constructor_name']:<20} {row['grid']:<6}")
    
    print("-" * 60)
    
    # Highlight key predictions
    winner = predictions.iloc[0]
    print(f"\n🥇 Predicted Winner: {winner['driver_name']} ({winner['constructor_name']})")
    
    podium = predictions.iloc[:3]
    print(f"\n🏆 Predicted Podium:")
    for i, (_, driver) in enumerate(podium.iterrows(), 1):
        medals = ['🥇', '🥈', '🥉']
        print(f"   {medals[i-1]} {driver['driver_name']} ({driver['constructor_name']})")
    
    # Show biggest movers
    predictions['grid_change'] = predictions['grid'] - (predictions.index + 1)
    biggest_gainer = predictions.loc[predictions['grid_change'].idxmax()]
    biggest_loser = predictions.loc[predictions['grid_change'].idxmin()]
    
    print(f"\n📈 Biggest Gainer: {biggest_gainer['driver_name']} (Grid {biggest_gainer['grid']} → P{biggest_gainer.name + 1})")
    print(f"📉 Biggest Slide: {biggest_loser['driver_name']} (Grid {biggest_loser['grid']} → P{biggest_loser.name + 1})")
    
    # Create visualizations
    visualize_results(df, performance, predictions)
    
    print("\n✅ F1 Prediction Demo completed!")
    print("💡 This demo shows how machine learning can predict race outcomes")
    print("🔮 Real predictions would use live data from qualifying and practice sessions")

if __name__ == "__main__":
    main()
