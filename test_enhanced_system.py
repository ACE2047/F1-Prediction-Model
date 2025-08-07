#!/usr/bin/env python3
"""
Enhanced F1 System Test Script
==============================

Quick test script to verify that all enhanced components are working correctly.
This script tests data collection, feature engineering, and model training
without requiring external API keys.

Author: Enhanced F1 Prediction System
Date: 2025
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Core libraries
        import pandas as pd
        import numpy as np
        import sklearn
        print("  ✅ Core data science libraries")
        
        # ML libraries
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestRegressor
        print("  ✅ Machine learning libraries")
        
        # Visualization
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        print("  ✅ Visualization libraries")
        
        # FastF1 (optional)
        try:
            import fastf1
            print("  ✅ FastF1 library")
            fastf1_available = True
        except ImportError:
            print("  ⚠️  FastF1 not available (install with: pip install fastf1)")
            fastf1_available = False
        
        # Enhanced modules
        try:
            from enhanced_data_collector import EnhancedF1DataCollector
            print("  ✅ Enhanced data collector")
        except ImportError as e:
            print(f"  ❌ Enhanced data collector import failed: {e}")
            return False
            
        try:
            from advanced_feature_engineering import AdvancedF1FeatureEngineer
            print("  ✅ Advanced feature engineering")
        except ImportError as e:
            print(f"  ❌ Advanced feature engineering import failed: {e}")
            return False
            
        try:
            from enhanced_f1_predictor import EnhancedF1PredictionModel
            print("  ✅ Enhanced prediction model")
        except ImportError as e:
            print(f"  ❌ Enhanced prediction model import failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_data_collector():
    """Test the enhanced data collector."""
    print("\n📊 Testing enhanced data collector...")
    
    try:
        from enhanced_data_collector import EnhancedF1DataCollector
        
        # Test basic initialization
        collector = EnhancedF1DataCollector(enable_fastf1=False, enable_weather=False)
        print("  ✅ Data collector initialization")
        
        # Test circuit characteristics
        monaco_chars = collector.get_circuit_characteristics("Monaco")
        if monaco_chars:
            print("  ✅ Circuit characteristics database")
        else:
            print("  ❌ Circuit characteristics not found")
            
        # Test base data collection (limited to avoid long execution)
        try:
            race_data = collector.get_race_results(2024)
            if not race_data.empty:
                print(f"  ✅ Race data collection ({len(race_data)} records)")
            else:
                print("  ⚠️  No race data collected (may be expected if 2024 season not complete)")
        except Exception as e:
            print(f"  ⚠️  Race data collection failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data collector test failed: {e}")
        return False

def test_feature_engineering():
    """Test the advanced feature engineering."""
    print("\n🔧 Testing advanced feature engineering...")
    
    try:
        from advanced_feature_engineering import AdvancedF1FeatureEngineer
        
        # Create sample data with all required columns
        sample_data = pd.DataFrame({
            'season': [2024] * 20,
            'round': [1] * 20,
            'driver_id': ['hamilton', 'verstappen', 'leclerc', 'norris', 'sainz'] * 4,
            'constructor_id': ['mercedes', 'red_bull', 'ferrari', 'mclaren', 'ferrari'] * 4,
            'position': np.random.randint(1, 21, 20),
            'grid': np.random.randint(1, 21, 20),
            'points': np.random.randint(0, 26, 20),
            'circuit_name': ['Bahrain'] * 20,
            'race_temperature': [28.5] * 20,
            'race_humidity': [65] * 20,
            'rainfall': [0] * 20,
            'circuit_length': [5.412] * 20,
            'circuit_corners': [15] * 20,
            'circuit_type': ['permanent'] * 20,
            'overtaking_difficulty': [3.5] * 20,
            'drs_zones': [3] * 20,
            # Add missing columns that original feature engineering expects
            'driver_points': np.random.randint(0, 100, 20),
            'constructor_points': np.random.randint(0, 200, 20),
            'quali_position': np.random.randint(1, 21, 20)
        })
        
        # Test feature engineering
        feature_engineer = AdvancedF1FeatureEngineer()
        enhanced_data = feature_engineer.create_all_advanced_features(sample_data)
        
        print(f"  ✅ Feature engineering ({len(sample_data.columns)} → {len(enhanced_data.columns)} features)")
        
        # Test feature categories
        categories = feature_engineer.get_feature_importance_categories()
        print(f"  ✅ Feature categorization ({len(categories)} categories)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature engineering test failed: {e}")
        return False

def test_model_training():
    """Test model training with sample data."""
    print("\n🤖 Testing model training...")
    
    try:
        from enhanced_f1_predictor import EnhancedF1PredictionModel
        from advanced_feature_engineering import AdvancedF1FeatureEngineer
        
        # Create larger sample dataset for training
        np.random.seed(42)
        n_samples = 200
        
        drivers = ['hamilton', 'verstappen', 'leclerc', 'norris', 'sainz', 'russell', 'perez', 'alonso']
        constructors = ['mercedes', 'red_bull', 'ferrari', 'mclaren', 'aston_martin']
        circuits = ['bahrain', 'saudi_arabia', 'australia', 'imola', 'monaco']
        
        sample_data = pd.DataFrame({
            'season': np.random.choice([2022, 2023, 2024], n_samples),
            'round': np.random.randint(1, 24, n_samples),
            'driver_id': np.random.choice(drivers, n_samples),
            'constructor_id': np.random.choice(constructors, n_samples),
            'position': np.random.randint(1, 21, n_samples),
            'grid': np.random.randint(1, 21, n_samples),
            'points': np.random.randint(0, 26, n_samples),
            'circuit_name': np.random.choice(circuits, n_samples),
            'race_temperature': np.random.uniform(15, 35, n_samples),
            'race_humidity': np.random.uniform(40, 80, n_samples),
            'rainfall': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'circuit_length': np.random.uniform(3.0, 7.0, n_samples),
            'circuit_corners': np.random.randint(10, 25, n_samples),
            'circuit_type': np.random.choice(['permanent', 'street'], n_samples),
            'overtaking_difficulty': np.random.uniform(2.0, 9.0, n_samples),
            'drs_zones': np.random.randint(1, 4, n_samples),
            # Add missing columns
            'driver_points': np.random.randint(0, 300, n_samples),
            'constructor_points': np.random.randint(0, 600, n_samples),
            'quali_position': np.random.randint(1, 21, n_samples)
        })
        
        # Apply feature engineering
        feature_engineer = AdvancedF1FeatureEngineer()
        enhanced_data = feature_engineer.create_all_advanced_features(sample_data)
        
        # Test model training
        model = EnhancedF1PredictionModel(enable_fastf1=False, enable_weather=False)
        
        # Train with limited models for testing
        original_configs = model.model_configs.copy()
        model.model_configs = {
            'random_forest': original_configs['random_forest'],
            'xgboost': original_configs['xgboost']
        }
        
        # Reduce model complexity for faster testing
        model.model_configs['random_forest']['params']['n_estimators'] = 50
        model.model_configs['xgboost']['params']['n_estimators'] = 50
        
        performance = model.train_enhanced_models(enhanced_data)
        
        if performance:
            print(f"  ✅ Model training successful ({len(performance)} models)")
            for model_name, metrics in performance.items():
                print(f"    {model_name}: RMSE={metrics['rmse']:.3f}")
        else:
            print("  ❌ No models trained successfully")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ Model training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction():
    """Test prediction functionality."""
    print("\n🎯 Testing prediction functionality...")
    
    try:
        from enhanced_f1_predictor import EnhancedF1PredictionModel
        from advanced_feature_engineering import AdvancedF1FeatureEngineer
        
        # Create training data
        np.random.seed(42)
        n_samples = 100
        
        sample_data = pd.DataFrame({
            'season': [2024] * n_samples,
            'round': np.random.randint(1, 10, n_samples),
            'driver_id': np.random.choice(['hamilton', 'verstappen', 'leclerc', 'norris'], n_samples),
            'constructor_id': np.random.choice(['mercedes', 'red_bull', 'ferrari', 'mclaren'], n_samples),
            'position': np.random.randint(1, 21, n_samples),
            'grid': np.random.randint(1, 21, n_samples),
            'points': np.random.randint(0, 26, n_samples),
            'circuit_name': ['bahrain'] * n_samples,
            'race_temperature': [28.5] * n_samples,
            'race_humidity': [65] * n_samples,
            'rainfall': [0] * n_samples,
            'circuit_length': [5.412] * n_samples,
            'circuit_corners': [15] * n_samples,
            'circuit_type': ['permanent'] * n_samples,
            'overtaking_difficulty': [3.5] * n_samples,
            'drs_zones': [3] * n_samples,
            # Add missing columns
            'driver_points': np.random.randint(0, 200, n_samples),
            'constructor_points': np.random.randint(0, 400, n_samples),
            'quali_position': np.random.randint(1, 21, n_samples)
        })
        
        # Quick training
        feature_engineer = AdvancedF1FeatureEngineer()
        enhanced_data = feature_engineer.create_all_advanced_features(sample_data)
        
        model = EnhancedF1PredictionModel(enable_fastf1=False, enable_weather=False)
        model.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {'n_estimators': 20, 'random_state': 42, 'n_jobs': -1}
            }
        }
        
        from sklearn.ensemble import RandomForestRegressor
        performance = model.train_enhanced_models(enhanced_data)
        
        # Test prediction with new data
        prediction_data = sample_data.iloc[:4].copy()  # Use first 4 rows for prediction
        prediction_data['driver_name'] = ['L. Hamilton', 'M. Verstappen', 'C. Leclerc', 'L. Norris']
        prediction_data['constructor_name'] = ['Mercedes', 'Red Bull Racing', 'Ferrari', 'McLaren']
        
        enhanced_prediction_data = feature_engineer.create_all_advanced_features(prediction_data)
        results = model.predict_race_enhanced(enhanced_prediction_data, model_name='random_forest', confidence_interval=True)
        
        if not results.empty:
            print(f"  ✅ Prediction successful ({len(results)} drivers predicted)")
            print("    Top 3 predictions:")
            for i, row in results.head(3).iterrows():
                conf = row.get('confidence', 'N/A')
                print(f"      P{i}: {row['driver_name']} (Grid: {row['grid']}) - Confidence: {conf}")
        else:
            print("  ❌ Prediction failed - no results")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests."""
    print("🏎️  Enhanced F1 Prediction System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Collector Test", test_data_collector),
        ("Feature Engineering Test", test_feature_engineering),
        ("Model Training Test", test_model_training),
        ("Prediction Test", test_prediction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your enhanced F1 system is ready to use.")
        print("\nNext steps:")
        print("  1. Run 'python enhanced_f1_predictor.py' to train the full model")
        print("  2. Check the data_sources_guide.md for additional data sources")
        print("  3. Consider setting up weather API keys for enhanced features")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\nCommon issues:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - FastF1 installation: pip install fastf1")
        print("  - Internet connectivity for data collection")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
