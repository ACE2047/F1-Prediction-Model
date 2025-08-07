#!/usr/bin/env python3
"""
Next F1 Race Predictor
======================

This script makes predictions for the next F1 race using the trained model.
It fetches the latest data and generates race predictions.

Usage:
    python predict_next_race.py
"""

import pandas as pd
import numpy as np
import requests
import pickle
from datetime import datetime
import logging
from f1_predictor import F1PredictionModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NextRacePredictor:
    """Handles predictions for the next F1 race."""
    
    def __init__(self, model_path: str = 'f1_prediction_model.pkl'):
        """Initialize with trained model."""
        try:
            with open(model_path, 'rb') as f:
                self.f1_model = pickle.load(f)
            logger.info("Loaded trained F1 model successfully!")
        except FileNotFoundError:
            logger.warning("No trained model found. Creating new model...")
            self.f1_model = F1PredictionModel()
            self._train_model()
    
    def _train_model(self):
        """Train a new model if none exists."""
        logger.info("Training new F1 prediction model...")
        
        # Collect data
        df = self.f1_model.collect_data(start_year=2020, end_year=2024)
        
        if df.empty:
            logger.error("No data collected. Cannot train model.")
            return
        
        # Prepare features and train
        df = self.f1_model.prepare_features(df)
        performance = self.f1_model.train_models(df)
        
        # Save the model
        with open('f1_prediction_model.pkl', 'wb') as f:
            pickle.dump(self.f1_model, f)
        
        logger.info("Model trained and saved successfully!")
    
    def get_next_race_info(self):
        """Get information about the next F1 race."""
        current_year = datetime.now().year
        url = f"http://ergast.com/api/f1/{current_year}.json"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            races = data['MRData']['RaceTable']['Races']
            now = datetime.now()
            
            # Find the next race
            for race in races:
                race_date = datetime.strptime(race['date'], '%Y-%m-%d')
                if race_date > now:
                    return {
                        'round': int(race['round']),
                        'race_name': race['raceName'],
                        'circuit_name': race['Circuit']['circuitName'],
                        'date': race['date'],
                        'country': race['Circuit']['Location']['country']
                    }
            
            # If no future races this year, check next year
            next_year = current_year + 1
            url = f"http://ergast.com/api/f1/{next_year}.json"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                races = data['MRData']['RaceTable']['Races']
                if races:
                    race = races[0]  # First race of next season
                    return {
                        'round': int(race['round']),
                        'race_name': race['raceName'],
                        'circuit_name': race['Circuit']['circuitName'],
                        'date': race['date'],
                        'country': race['Circuit']['Location']['country']
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching next race info: {e}")
            return None
    
    def get_current_drivers_and_teams(self):
        """Get current drivers and their teams."""
        current_year = datetime.now().year
        url = f"http://ergast.com/api/f1/{current_year}/drivers.json"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            drivers = []
            for driver in data['MRData']['DriverTable']['Drivers']:
                drivers.append({
                    'driver_id': driver['driverId'],
                    'driver_name': driver['givenName'] + ' ' + driver['familyName'],
                    'nationality': driver['nationality']
                })
            
            # Get constructor info
            url = f"http://ergast.com/api/f1/{current_year}/constructors.json"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            constructors = {}
            for constructor in data['MRData']['ConstructorTable']['Constructors']:
                constructors[constructor['constructorId']] = constructor['name']
            
            # Get driver-constructor pairings from recent results
            url = f"http://ergast.com/api/f1/{current_year}/results.json?limit=1000"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                driver_teams = {}
                
                # Get most recent pairings
                for race in reversed(data['MRData']['RaceTable']['Races']):
                    for result in race['Results']:
                        driver_id = result['Driver']['driverId']
                        constructor_id = result['Constructor']['constructorId']
                        if driver_id not in driver_teams:
                            driver_teams[driver_id] = constructor_id
                
                # Add constructor info to drivers
                for driver in drivers:
                    if driver['driver_id'] in driver_teams:
                        constructor_id = driver_teams[driver['driver_id']]
                        driver['constructor_id'] = constructor_id
                        driver['constructor_name'] = constructors.get(constructor_id, 'Unknown')
                    else:
                        driver['constructor_id'] = 'unknown'
                        driver['constructor_name'] = 'Unknown'
            
            return drivers
            
        except Exception as e:
            logger.error(f"Error fetching current drivers: {e}")
            return []
    
    def create_mock_grid_positions(self, drivers: list) -> dict:
        """Create mock grid positions based on recent performance."""
        # This is a simplified approach - in reality, you'd get qualifying results
        # For now, we'll use recent performance to estimate grid positions
        
        grid_positions = {}
        
        # Get recent race results to estimate grid positions
        current_year = datetime.now().year
        url = f"http://ergast.com/api/f1/{current_year}/results.json?limit=100"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Calculate average positions for each driver
            driver_avg_positions = {}
            driver_counts = {}
            
            for race in data['MRData']['RaceTable']['Races'][-5:]:  # Last 5 races
                for result in race['Results']:
                    driver_id = result['Driver']['driverId']
                    position = int(result['position']) if 'position' in result else 20
                    
                    if driver_id not in driver_avg_positions:
                        driver_avg_positions[driver_id] = 0
                        driver_counts[driver_id] = 0
                    
                    driver_avg_positions[driver_id] += position
                    driver_counts[driver_id] += 1
            
            # Calculate averages
            for driver_id in driver_avg_positions:
                if driver_counts[driver_id] > 0:
                    driver_avg_positions[driver_id] /= driver_counts[driver_id]
            
            # Sort drivers by average position and assign grid positions
            sorted_drivers = sorted(driver_avg_positions.keys(), 
                                  key=lambda x: driver_avg_positions.get(x, 20))
            
            for i, driver_id in enumerate(sorted_drivers[:20]):  # Top 20 drivers
                # Add some randomness to simulate qualifying variability
                grid_pos = min(20, max(1, i + 1 + np.random.randint(-2, 3)))
                grid_positions[driver_id] = grid_pos
            
            # Fill in any missing drivers
            current_grid = 1
            for driver in drivers:
                if driver['driver_id'] not in grid_positions:
                    while current_grid in grid_positions.values():
                        current_grid += 1
                    if current_grid <= 20:
                        grid_positions[driver['driver_id']] = current_grid
                        current_grid += 1
            
        except Exception as e:
            logger.error(f"Error creating mock grid positions: {e}")
            # Fallback: random grid positions
            available_positions = list(range(1, 21))
            np.random.shuffle(available_positions)
            for i, driver in enumerate(drivers[:20]):
                grid_positions[driver['driver_id']] = available_positions[i]
        
        return grid_positions
    
    def predict_next_race(self):
        """Make predictions for the next F1 race."""
        logger.info("🏁 Predicting next F1 race results...")
        
        # Get next race info
        next_race = self.get_next_race_info()
        if not next_race:
            logger.error("Could not find information about the next race.")
            return None
        
        print(f"\n🏎️  NEXT RACE PREDICTION")
        print("=" * 50)
        print(f"Race: {next_race['race_name']}")
        print(f"Circuit: {next_race['circuit_name']}")
        print(f"Country: {next_race['country']}")
        print(f"Date: {next_race['date']}")
        print("=" * 50)
        
        # Get current drivers and teams
        drivers = self.get_current_drivers_and_teams()
        if not drivers:
            logger.error("Could not fetch current drivers and teams.")
            return None
        
        # Create mock grid positions (in reality, you'd get qualifying results)
        grid_positions = self.create_mock_grid_positions(drivers)
        
        # Prepare race data for prediction
        race_data = []
        for driver in drivers:
            if driver['driver_id'] in grid_positions:
                race_data.append({
                    'season': datetime.now().year,
                    'round': next_race['round'],
                    'race_name': next_race['race_name'],
                    'circuit_name': next_race['circuit_name'],
                    'driver_id': driver['driver_id'],
                    'driver_name': driver['driver_name'],
                    'constructor_id': driver['constructor_id'],
                    'constructor_name': driver['constructor_name'],
                    'grid': grid_positions[driver['driver_id']],
                    'quali_position': grid_positions[driver['driver_id']],  # Approximation
                    'season_progress': next_race['round'] / 24,  # Approximate season length
                    # Add some default values for other features
                    'driver_avg_position_last_5': 10,
                    'driver_avg_points_last_5': 5,
                    'driver_wins_last_10': 0,
                    'driver_podiums_last_10': 1,
                    'constructor_avg_position_last_5': 10,
                    'constructor_avg_points_last_5': 10,
                    'driver_circuit_avg_position': 10,
                    'constructor_circuit_avg_position': 10,
                    'driver_constructor_combo_avg': 10,
                    'grid_vs_avg_position': 0,
                    'driver_points': 50,
                    'constructor_points': 100,
                    'driver_wins': 0,
                    'constructor_wins': 0
                })
        
        race_df = pd.DataFrame(race_data)
        
        # Prepare features for prediction
        race_df = self.f1_model.prepare_features(race_df)
        
        # Make predictions
        try:
            predictions = self.f1_model.predict_race(race_df, model_name='xgboost')
            
            print("\n🏆 PREDICTED RACE RESULTS:")
            print("-" * 60)
            print(f"{'Pos':<4} {'Driver':<25} {'Team':<20} {'Grid':<6}")
            print("-" * 60)
            
            for i, (_, row) in enumerate(predictions.iterrows(), 1):
                print(f"{i:<4} {row['driver_name']:<25} {row['constructor_name']:<20} {row['grid']:<6}")
            
            print("-" * 60)
            
            # Highlight key predictions
            winner = predictions.iloc[0]
            print(f"\n🥇 Predicted Winner: {winner['driver_name']} ({winner['constructor_name']})")
            
            podium = predictions.iloc[:3]
            print(f"🥈 Podium Finishers:")
            for i, (_, driver) in enumerate(podium.iterrows(), 1):
                print(f"   {i}. {driver['driver_name']} ({driver['constructor_name']})")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None

def main():
    """Main function to predict the next F1 race."""
    predictor = NextRacePredictor()
    predictions = predictor.predict_next_race()
    
    if predictions is not None:
        print("\n📊 Prediction completed successfully!")
        print("💡 Note: These predictions are based on historical data and machine learning.")
        print("🏁 Actual race results may vary due to many unpredictable factors!")
    else:
        print("\n❌ Failed to generate predictions. Please check your internet connection.")

if __name__ == "__main__":
    main()
