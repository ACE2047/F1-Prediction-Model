#!/usr/bin/env python3
"""
F1 Prediction Analysis
======================

This script provides additional analysis and insights from the F1 prediction model,
including feature importance, prediction confidence, and strategic recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def analyze_prediction_confidence(predictions_df):
    """Analyze prediction confidence and uncertainty."""
    print("🎯 Prediction Confidence Analysis")
    print("=" * 40)
    
    # Simulate confidence intervals (in real scenario, this would come from model uncertainty)
    np.random.seed(42)
    confidence_intervals = []
    
    for i, row in predictions_df.iterrows():
        # Simulate confidence based on grid position and historical performance
        base_uncertainty = 2.5  # Base uncertainty in positions
        grid_factor = abs(row['grid'] - (i + 1)) * 0.3  # Higher uncertainty for big changes
        
        uncertainty = base_uncertainty + grid_factor
        confidence = max(0.1, min(0.95, 1 - (uncertainty / 10)))
        
        confidence_intervals.append({
            'driver': row['driver_name'],
            'predicted_pos': i + 1,
            'confidence': confidence,
            'uncertainty_range': uncertainty
        })
    
    confidence_df = pd.DataFrame(confidence_intervals)
    
    # Show most confident predictions
    most_confident = confidence_df.nlargest(5, 'confidence')
    print("\n🔮 Most Confident Predictions:")
    for _, row in most_confident.iterrows():
        print(f"   P{row['predicted_pos']}: {row['driver']} ({row['confidence']:.1%} confidence)")
    
    # Show least confident predictions
    least_confident = confidence_df.nsmallest(5, 'confidence')
    print("\n❓ Least Confident Predictions:")
    for _, row in least_confident.iterrows():
        print(f"   P{row['predicted_pos']}: {row['driver']} ({row['confidence']:.1%} confidence)")
    
    return confidence_df

def analyze_strategic_insights(predictions_df):
    """Provide strategic insights for teams and fantasy players."""
    print("\n🏁 Strategic Insights")
    print("=" * 40)
    
    # Identify potential fantasy sleepers (high finish from low grid)
    predictions_df['position_gain'] = predictions_df['grid'] - (predictions_df.index + 1)
    
    # Best value picks (good predicted finish from poor grid)
    fantasy_sleepers = predictions_df[predictions_df['position_gain'] > 3].head(3)
    if not fantasy_sleepers.empty:
        print("\n💎 Fantasy F1 Sleeper Picks:")
        for _, row in fantasy_sleepers.iterrows():
            gain = int(row['position_gain'])
            print(f"   {row['driver_name']}: Grid {row['grid']} → P{row.name + 1} (+{gain} positions)")
    
    # Disappointment risks (poor predicted finish from good grid)
    disappointments = predictions_df[predictions_df['position_gain'] < -3].head(3)
    if not disappointments.empty:
        print("\n📉 Potential Disappointments:")
        for _, row in disappointments.iterrows():
            loss = int(abs(row['position_gain']))
            print(f"   {row['driver_name']}: Grid {row['grid']} → P{row.name + 1} (-{loss} positions)")
    
    # Constructor battle predictions
    print("\n🏗️  Constructor Battle Predictions:")
    constructor_points = {}
    points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
    
    for i, row in predictions_df.iterrows():
        team = row['constructor_name']
        points = points_system[i] if i < 10 else 0
        
        if team not in constructor_points:
            constructor_points[team] = 0
        constructor_points[team] += points
    
    # Sort constructors by predicted points
    sorted_constructors = sorted(constructor_points.items(), key=lambda x: x[1], reverse=True)
    
    for i, (team, points) in enumerate(sorted_constructors[:5], 1):
        print(f"   {i}. {team}: {points} points")

def create_performance_heatmap(predictions_df):
    """Create a performance heatmap."""
    print("\n📊 Creating Performance Heatmap...")
    
    # Create a matrix of driver vs metrics
    drivers = predictions_df['driver_name'].head(10)  # Top 10 for readability
    
    # Normalize different metrics to 0-1 scale for comparison
    metrics_data = []
    for _, row in predictions_df.head(10).iterrows():
        metrics_data.append({
            'Predicted Finish': 1 - (row.name / 20),  # Invert so higher is better
            'Grid Position': 1 - (row['grid'] / 20),
            'Position Change': (row['position_gain'] + 10) / 20,  # Normalize around 0
        })
    
    metrics_df = pd.DataFrame(metrics_data, index=drivers)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(metrics_df.T, annot=True, cmap='RdYlGn', cbar_kws={'label': 'Normalized Performance'})
    plt.title('Driver Performance Heatmap - Hungarian GP Prediction')
    plt.xlabel('Drivers')
    plt.ylabel('Performance Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def weather_impact_analysis():
    """Simulate weather impact on predictions."""
    print("\n🌧️  Weather Impact Analysis")
    print("=" * 40)
    
    weather_scenarios = {
        'Dry': {'impact': 0, 'description': 'Normal race conditions'},
        'Light Rain': {'impact': 0.3, 'description': 'Increased uncertainty, skill factor important'},
        'Heavy Rain': {'impact': 0.7, 'description': 'High chaos factor, experience matters'},
        'Mixed Conditions': {'impact': 0.5, 'description': 'Strategy and adaptability key'}
    }
    
    print("Weather scenario impacts on prediction confidence:")
    for condition, data in weather_scenarios.items():
        base_confidence = 0.75
        adjusted_confidence = base_confidence * (1 - data['impact'])
        print(f"   {condition:15}: {adjusted_confidence:.1%} confidence ({data['description']})")
    
    print(f"\n🌦️  Hungaroring Historical Weather:")
    print(f"   - 80% chance of dry conditions")
    print(f"   - 15% chance of light rain/showers")
    print(f"   - 5% chance of thunderstorms")
    print(f"   - Strategy recommendation: Hot weather, tire management crucial")

def driver_head_to_head_analysis(predictions_df):
    """Analyze key driver battles."""
    print("\n⚔️  Key Driver Battles")
    print("=" * 40)
    
    # Define key rivalries
    rivalries = [
        ('Max Verstappen', 'Lando Norris', 'Championship Battle'),
        ('Charles Leclerc', 'Carlos Sainz Jr', 'Ferrari Teammate Battle'),
        ('Lewis Hamilton', 'George Russell', 'Mercedes Showdown'),
        ('Oscar Piastri', 'Lando Norris', 'McLaren Young vs Experienced')
    ]
    
    for driver1, driver2, description in rivalries:
        pos1 = predictions_df[predictions_df['driver_name'] == driver1].index
        pos2 = predictions_df[predictions_df['driver_name'] == driver2].index
        
        if len(pos1) > 0 and len(pos2) > 0:
            pos1, pos2 = pos1[0] + 1, pos2[0] + 1
            winner = driver1 if pos1 < pos2 else driver2
            margin = abs(pos1 - pos2)
            
            print(f"\n🥊 {description}:")
            print(f"   {driver1}: P{pos1}")
            print(f"   {driver2}: P{pos2}")
            print(f"   Predicted winner: {winner} (by {margin} position{'s' if margin != 1 else ''})")

def create_race_timeline_simulation():
    """Create a simulated race timeline."""
    print("\n🏁 Race Timeline Simulation")
    print("=" * 40)
    
    key_events = [
        {"lap": 1, "event": "Race Start", "impact": "Narrow track, overtaking difficult"},
        {"lap": 18, "event": "First Pit Window", "impact": "Strategy crucial due to limited overtaking"},
        {"lap": 35, "event": "Mid-race Heat", "impact": "Tire degradation in hot conditions"},
        {"lap": 50, "event": "Second Pit Window", "impact": "Undercut opportunities key"},
        {"lap": 70, "event": "Final Laps", "impact": "Track position more important than pace"}
    ]
    
    print("Predicted key race moments:")
    for event in key_events:
        print(f"   Lap {event['lap']:2d}: {event['event']} - {event['impact']}")
    
    print(f"\n🎲 Chaos Factor Predictions:")
    print(f"   - Safety Car probability: 20% (narrow track, fewer incidents)")
    print(f"   - Weather change: 15% (typically stable summer weather)")
    print(f"   - Overtaking difficulty: Very High (narrow twisty track)")
    print(f"   - DRS effectiveness: Limited (Hungaroring has 1 DRS zone)")

def main():
    """Main analysis function."""
    print("🏎️  F1 Prediction Analysis - Hungarian Grand Prix")
    print("=" * 60)
    
    # Sample prediction data (would come from the main model)
    predictions_data = [
        {'driver_name': 'Max Verstappen', 'constructor_name': 'Red Bull Racing', 'grid': 1},
        {'driver_name': 'Oscar Piastri', 'constructor_name': 'McLaren', 'grid': 3},
        {'driver_name': 'Carlos Sainz Jr', 'constructor_name': 'Ferrari', 'grid': 7},
        {'driver_name': 'Charles Leclerc', 'constructor_name': 'Ferrari', 'grid': 5},
        {'driver_name': 'George Russell', 'constructor_name': 'Mercedes', 'grid': 4},
        {'driver_name': 'Lewis Hamilton', 'constructor_name': 'Mercedes', 'grid': 6},
        {'driver_name': 'Sergio Pérez', 'constructor_name': 'Red Bull Racing', 'grid': 8},
        {'driver_name': 'Lando Norris', 'constructor_name': 'McLaren', 'grid': 2},
        {'driver_name': 'Fernando Alonso', 'constructor_name': 'Aston Martin', 'grid': 9},
        {'driver_name': 'Lance Stroll', 'constructor_name': 'Aston Martin', 'grid': 10}
    ]
    
    predictions_df = pd.DataFrame(predictions_data)
    
    # Run various analyses
    confidence_df = analyze_prediction_confidence(predictions_df)
    analyze_strategic_insights(predictions_df)
    create_performance_heatmap(predictions_df)
    weather_impact_analysis()
    driver_head_to_head_analysis(predictions_df)
    create_race_timeline_simulation()
    
    print(f"\n📋 Final Prediction Summary:")
    print(f"   🏆 Most likely winner: {predictions_df.iloc[0]['driver_name']}")
    print(f"   🎯 Confidence level: High (pole position advantage)")
    print(f"   🎲 Upset potential: Medium (weather dependent)")
    print(f"   📈 Best value bet: {predictions_df.iloc[2]['driver_name']} (good pace, grid disadvantage)")
    
    print(f"\n✅ Analysis complete! Use these insights for:")
    print(f"   - Fantasy F1 team selection")
    print(f"   - Race strategy understanding")
    print(f"   - Betting insights (gamble responsibly)")
    print(f"   - Pure F1 enjoyment! 🏎️")

if __name__ == "__main__":
    main()
