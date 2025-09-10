#!/usr/bin/env python3
"""
Enhanced F1 Data Collector
==========================

Extended data collection with FastF1, weather data, circuit characteristics,
and additional data sources for improved F1 race predictions.

Author: Enhanced F1 Prediction System
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
import time
import os

# FastF1 for telemetry data
try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    logging.warning("FastF1 not available. Install with: pip install fastf1")

# Original data collector
from f1_predictor import F1DataCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedF1DataCollector(F1DataCollector):
    """Enhanced F1 data collector with additional data sources."""
    
    def __init__(self, enable_fastf1=True, enable_weather=True):
        super().__init__()
        self.enable_fastf1 = enable_fastf1 and FASTF1_AVAILABLE
        self.enable_weather = enable_weather
        
        # Setup FastF1 cache
        if self.enable_fastf1:
            cache_dir = os.path.join(os.getcwd(), 'f1_cache')
            os.makedirs(cache_dir, exist_ok=True)
            fastf1.Cache.enable_cache(cache_dir)
            
        # Circuit characteristics database
        self.circuit_characteristics = self._load_circuit_data()
        
        # Weather API key (you'll need to get this from OpenWeatherMap)
        self.weather_api_key = os.getenv('OPENWEATHER_API_KEY')
        
    def _load_circuit_data(self) -> Dict:
        """Load circuit characteristics data."""
        return {
            'monaco': {
                'length_km': 3.337,
                'corners': 19,
                'type': 'street',
                'elevation_change': 42,
                'overtaking_difficulty': 9.5,
                'drs_zones': 1,
                'avg_speed_kph': 161,
                'coordinates': (43.7347, 7.4206)
            },
            'silverstone': {
                'length_km': 5.891,
                'corners': 18,
                'type': 'permanent',
                'elevation_change': 18,
                'overtaking_difficulty': 4.0,
                'drs_zones': 2,
                'avg_speed_kph': 233,
                'coordinates': (52.0786, -1.0169)
            },
            'monza': {
                'length_km': 5.793,
                'corners': 11,
                'type': 'permanent',
                'elevation_change': 26,
                'overtaking_difficulty': 2.0,
                'drs_zones': 3,
                'avg_speed_kph': 243,
                'coordinates': (45.6156, 9.2811)
            },
            'spa': {
                'length_km': 7.004,
                'corners': 20,
                'type': 'permanent',
                'elevation_change': 104,
                'overtaking_difficulty': 3.0,
                'drs_zones': 2,
                'avg_speed_kph': 234,
                'coordinates': (50.4372, 5.9714)
            },
            'interlagos': {
                'length_km': 4.309,
                'corners': 15,
                'type': 'permanent',
                'elevation_change': 40,
                'overtaking_difficulty': 4.5,
                'drs_zones': 2,
                'avg_speed_kph': 213,
                'coordinates': (-23.7036, -46.6997)
            },
            'suzuka': {
                'length_km': 5.807,
                'corners': 18,
                'type': 'permanent',
                'elevation_change': 40,
                'overtaking_difficulty': 6.0,
                'drs_zones': 1,
                'avg_speed_kph': 231,
                'coordinates': (34.8431, 136.5407)
            },
            'austin': {
                'length_km': 5.513,
                'corners': 20,
                'type': 'permanent',
                'elevation_change': 41,
                'overtaking_difficulty': 5.0,
                'drs_zones': 2,
                'avg_speed_kph': 217,
                'coordinates': (30.1328, -97.6411)
            },
            'melbourne': {
                'length_km': 5.278,
                'corners': 16,
                'type': 'street',
                'elevation_change': 9,
                'overtaking_difficulty': 7.0,
                'drs_zones': 2,
                'avg_speed_kph': 223,
                'coordinates': (-37.8497, 144.9681)
            },
            'bahrain': {
                'length_km': 5.412,
                'corners': 15,
                'type': 'permanent',
                'elevation_change': 32,
                'overtaking_difficulty': 3.5,
                'drs_zones': 3,
                'avg_speed_kph': 214,
                'coordinates': (26.0325, 50.5106)
            },
            'jeddah': {
                'length_km': 6.174,
                'corners': 27,
                'type': 'street',
                'elevation_change': 22,
                'overtaking_difficulty': 8.0,
                'drs_zones': 3,
                'avg_speed_kph': 252,
                'coordinates': (21.6319, 39.1044)
            }
        }
    
    def get_telemetry_data(self, year: int, race: str, session_type: str = 'R') -> Optional[Dict]:
        """Get detailed telemetry data using FastF1."""
        if not self.enable_fastf1:
            logger.warning("FastF1 not enabled or available")
            return None
            
        try:
            logger.info(f"Loading FastF1 data for {year} {race} {session_type}")
            session = fastf1.get_session(year, race, session_type)
            session.load(telemetry=True, weather=True, messages=False)
            
            # Process laps data
            laps = session.laps
            weather_data = session.weather_data
            
            # Create summary statistics
            telemetry_summary = {
                'fastest_lap_time': laps['LapTime'].min().total_seconds() if not laps['LapTime'].isna().all() else None,
                'avg_lap_time': laps['LapTime'].mean().total_seconds() if not laps['LapTime'].isna().all() else None,
                'tire_compounds': laps['Compound'].value_counts().to_dict(),
                'pit_stops': laps.groupby('Driver')['PitInTime'].count().to_dict(),
                'sector_times': {
                    'sector_1_avg': laps['Sector1Time'].mean().total_seconds() if not laps['Sector1Time'].isna().all() else None,
                    'sector_2_avg': laps['Sector2Time'].mean().total_seconds() if not laps['Sector2Time'].isna().all() else None,
                    'sector_3_avg': laps['Sector3Time'].mean().total_seconds() if not laps['Sector3Time'].isna().all() else None,
                }
            }
            
            # Weather summary
            if not weather_data.empty:
                weather_summary = {
                    'avg_air_temp': weather_data['AirTemp'].mean(),
                    'avg_track_temp': weather_data['TrackTemp'].mean(),
                    'avg_humidity': weather_data['Humidity'].mean(),
                    'avg_pressure': weather_data['Pressure'].mean(),
                    'rainfall': weather_data['Rainfall'].sum() > 0,
                    'wind_speed_avg': weather_data['WindSpeed'].mean()
                }
            else:
                weather_summary = {}
            
            return {
                'telemetry_summary': telemetry_summary,
                'weather_summary': weather_summary,
                'raw_laps': laps,
                'raw_weather': weather_data
            }
            
        except Exception as e:
            logger.error(f"Error getting telemetry data: {e}")
            return None
    
    def get_historical_weather(self, circuit_name: str, date: str) -> Optional[Dict]:
        """Get historical weather data for a circuit."""
        if not self.enable_weather or not self.weather_api_key:
            return None
            
        circuit_key = circuit_name.lower().replace(' ', '_')
        if circuit_key not in self.circuit_characteristics:
            logger.warning(f"Circuit {circuit_name} not found in characteristics database")
            return None
            
        coords = self.circuit_characteristics[circuit_key]['coordinates']
        
        try:
            # Convert date to timestamp
            dt = datetime.strptime(date, '%Y-%m-%d')
            timestamp = int(dt.timestamp())
            
            url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
            params = {
                'lat': coords[0],
                'lon': coords[1],
                'dt': timestamp,
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            current = data['current']
            return {
                'temperature': current['temp'],
                'humidity': current['humidity'],
                'pressure': current['pressure'],
                'wind_speed': current['wind_speed'],
                'weather_condition': current['weather'][0]['main'],
                'description': current['weather'][0]['description'],
                'rain': current.get('rain', {}).get('1h', 0)
            }
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return None
    
    def get_circuit_characteristics(self, circuit_name: str) -> Optional[Dict]:
        """Get characteristics for a specific circuit."""
        circuit_key = circuit_name.lower().replace(' ', '_')
        return self.circuit_characteristics.get(circuit_key)
    
    def get_tire_strategy_data(self, year: int, race: str) -> Optional[Dict]:
        """Get tire strategy data using FastF1."""
        if not self.enable_fastf1:
            return None
            
        try:
            session = fastf1.get_session(year, race, 'R')
            session.load()
            laps = session.laps
            
            # Analyze tire strategies
            tire_strategies = {}
            for driver in laps['Driver'].unique():
                driver_laps = laps[laps['Driver'] == driver]
                stint_data = []
                
                current_compound = None
                stint_start = 0
                
                for idx, lap in driver_laps.iterrows():
                    if lap['Compound'] != current_compound:
                        if current_compound is not None:
                            stint_data.append({
                                'compound': current_compound,
                                'stint_length': idx - stint_start,
                                'avg_lap_time': driver_laps.iloc[stint_start:idx]['LapTime'].mean()
                            })
                        current_compound = lap['Compound']
                        stint_start = idx
                
                tire_strategies[driver] = stint_data
                
            return tire_strategies
            
        except Exception as e:
            logger.error(f"Error getting tire strategy data: {e}")
            return None
    
    def get_reliability_data(self, start_year: int = 2020) -> pd.DataFrame:
        """Get reliability data (DNFs) for drivers and constructors."""
        reliability_data = []
        
        for year in range(start_year, datetime.now().year + 1):
            race_data = self.get_race_results(year)
            if not race_data.empty:
                # Analyze DNFs and their reasons
                dnf_data = race_data[race_data['status'] != 'Finished'].copy()
                
                for _, row in dnf_data.iterrows():
                    reliability_data.append({
                        'season': row['season'],
                        'driver_id': row['driver_id'],
                        'constructor_id': row['constructor_id'],
                        'dnf_reason': row['status'],
                        'mechanical_failure': 'Engine' in row['status'] or 'Gearbox' in row['status'] or 
                                           'Transmission' in row['status'] or 'Hydraulics' in row['status'],
                        'accident': 'Accident' in row['status'] or 'Collision' in row['status'],
                        'grid_position': row['grid']
                    })
        
        return pd.DataFrame(reliability_data)
    
    def get_practice_session_data(self, year: int, race: str) -> Optional[Dict]:
        """Get practice session data for additional insights."""
        if not self.enable_fastf1:
            return None
            
        practice_data = {}
        
        for session_type in ['FP1', 'FP2', 'FP3']:
            try:
                session = fastf1.get_session(year, race, session_type)
                session.load()
                
                if not session.laps.empty:
                    practice_data[session_type] = {
                        'fastest_lap': session.laps['LapTime'].min().total_seconds(),
                        'avg_lap_time': session.laps['LapTime'].mean().total_seconds(),
                        'lap_count': len(session.laps),
                        'drivers_participated': len(session.laps['Driver'].unique())
                    }
                    
            except Exception as e:
                logger.warning(f"Could not load {session_type} data: {e}")
                continue
        
        return practice_data if practice_data else None
    
    def get_enhanced_race_data(self, year: int) -> pd.DataFrame:
        """Get enhanced race data with additional features."""
        base_data = self.get_race_results(year)
        if base_data.empty:
            return base_data
        
        enhanced_data = []
        
        for _, race in base_data.groupby(['round', 'circuit_name']):
            race_info = race.iloc[0]
            circuit_name = race_info['circuit_name']
            race_date = race_info['date']
            
            # Get circuit characteristics
            circuit_chars = self.get_circuit_characteristics(circuit_name)
            
            # Get weather data
            weather_data = self.get_historical_weather(circuit_name, race_date)
            
            # Get telemetry data
            telemetry_data = self.get_telemetry_data(year, race_info['race_name'])
            
            for _, row in race.iterrows():
                enhanced_row = row.to_dict()
                
                # Add circuit characteristics
                if circuit_chars:
                    enhanced_row.update({
                        'circuit_length': circuit_chars['length_km'],
                        'circuit_corners': circuit_chars['corners'],
                        'circuit_type': circuit_chars['type'],
                        'circuit_elevation_change': circuit_chars['elevation_change'],
                        'overtaking_difficulty': circuit_chars['overtaking_difficulty'],
                        'drs_zones': circuit_chars['drs_zones'],
                        'avg_speed_kph': circuit_chars['avg_speed_kph']
                    })
                
                # Add weather data
                if weather_data:
                    enhanced_row.update({
                        'race_temperature': weather_data['temperature'],
                        'race_humidity': weather_data['humidity'],
                        'race_pressure': weather_data['pressure'],
                        'race_wind_speed': weather_data['wind_speed'],
                        'weather_condition': weather_data['weather_condition'],
                        'rainfall': weather_data['rain'] > 0
                    })
                
                # Add telemetry insights
                if telemetry_data and telemetry_data['weather_summary']:
                    weather_summary = telemetry_data['weather_summary']
                    enhanced_row.update({
                        'track_temperature': weather_summary.get('avg_track_temp'),
                        'air_temperature': weather_summary.get('avg_air_temp'),
                        'session_rainfall': weather_summary.get('rainfall', False)
                    })
                
                enhanced_data.append(enhanced_row)
        
        return pd.DataFrame(enhanced_data)


def main():
    """Test the enhanced data collector."""
    print("🏎️  Enhanced F1 Data Collector Test")
    print("=" * 50)
    
    collector = EnhancedF1DataCollector(enable_fastf1=True, enable_weather=False)
    
    # Test circuit characteristics
    print("\n📍 Circuit Characteristics:")
    monaco_chars = collector.get_circuit_characteristics("Monaco")
    if monaco_chars:
        print(f"Monaco: {monaco_chars}")
    
    # Test enhanced race data (limited to avoid long execution)
    print("\n📊 Testing Enhanced Race Data Collection...")
    try:
        enhanced_data = collector.get_enhanced_race_data(2024)
        if not enhanced_data.empty:
            print(f"✅ Collected {len(enhanced_data)} enhanced race records")
            print("New columns added:")
            new_columns = [col for col in enhanced_data.columns if col not in 
                          ['season', 'round', 'race_name', 'circuit_name', 'date', 
                           'driver_id', 'driver_name', 'constructor_id', 'constructor_name',
                           'grid', 'position', 'points', 'status']]
            for col in new_columns[:10]:  # Show first 10
                print(f"  - {col}")
        else:
            print("❌ No enhanced data collected")
    except Exception as e:
        print(f"❌ Error testing enhanced data collection: {e}")
    
    print("\n✅ Enhanced Data Collector ready!")


if __name__ == "__main__":
    main()
