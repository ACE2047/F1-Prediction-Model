# 🏎️ F1 Race Prediction Model

A comprehensive Formula 1 race prediction system using machine learning to predict race outcomes, analyze driver performance, and provide strategic insights.

## 🚀 Features

- **Real-time Data Collection**: Fetches live F1 data from the Ergast API
- **Advanced Feature Engineering**: Creates 26+ features including historical performance, circuit-specific data, and driver-constructor combinations
- **Multiple ML Models**: Implements Random Forest, XGBoost, and LightGBM for robust predictions
- **Comprehensive Analysis**: Provides confidence intervals, strategic insights, and weather impact analysis
- **Beautiful Visualizations**: Creates performance heatmaps, trend charts, and prediction graphics
- **Race Timeline Simulation**: Predicts key race moments and potential chaos factors
- **Completed Race Results**: View detailed results from finished races with podium finishers, lap times, and race statistics
- **Interactive Dashboard**: Modern React-based UI with race modals, driver stats, and team analysis

## 📁 Project Structure

```
f1-prediction-model/
├── f1_predictor.py         # Main prediction model with data collection and training
├── predict_next_race.py    # Script to predict upcoming race results
├── f1_demo.py             # Demo version with sample data (works offline)
├── f1_analysis.py         # Advanced analysis and strategic insights
├── requirements.txt       # Python dependencies
├── src/
│   ├── components/
│   │   └── EnhancedF1Dashboard.js  # React dashboard component
│   └── data/
│       └── raceData.js            # F1 2025 race data and results
└── README.md             # This file
```

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Required packages:**
   - pandas, numpy (data manipulation)
   - scikit-learn (machine learning)
   - xgboost, lightgbm (advanced ML models)
   - matplotlib, seaborn, plotly (visualizations)
   - requests, beautifulsoup4 (data collection)

## 🎯 Usage

### Quick Start (Demo Mode)
Run the demo with sample data (no internet required):
```bash
python f1_demo.py
```

### Full Prediction System
Train the model with real F1 data:
```bash
python f1_predictor.py
```

### Predict Next Race
Generate predictions for the upcoming race:
```bash
python predict_next_race.py
```

### Advanced Analysis
Get detailed insights and strategic recommendations:
```bash
python f1_analysis.py
```

## 🔧 How It Works

### 1. Data Collection
- **Race Results**: Historical finishing positions, points, and lap times
- **Qualifying Data**: Grid positions and qualifying times
- **Championship Standings**: Driver and constructor points throughout seasons
- **Circuit Information**: Track-specific performance data

### 2. Feature Engineering
- **Historical Performance**: Rolling averages of positions and points
- **Circuit Specialization**: Driver/team performance at specific tracks
- **Season Context**: Championship position and season progress
- **Team Dynamics**: Constructor performance and driver-team combinations
- **Weather Factors**: Historical weather patterns and impact

### 3. Machine Learning Models
- **Random Forest**: Ensemble method for robust predictions
- **XGBoost**: Gradient boosting for high accuracy
- **LightGBM**: Fast training with competitive performance
- **Model Selection**: Automatically chooses best performing model

### 4. Prediction Output
- **Race Positions**: Predicted finishing order for all drivers
- **Confidence Intervals**: Reliability estimates for each prediction
- **Strategic Insights**: Fantasy picks, upset potential, and team battles
- **Weather Impact**: How different conditions affect predictions

## 📊 Sample Output

### Race Predictions
```
🏁 BRITISH GRAND PRIX PREDICTION
============================================================
Pos  Driver                    Team                 Grid  
------------------------------------------------------------
1    Max Verstappen            Red Bull Racing      1     
2    Oscar Piastri             McLaren              3     
3    Carlos Sainz Jr           Ferrari              7     
4    Charles Leclerc           Ferrari              5     
5    George Russell            Mercedes             4     

🥇 Predicted Winner: Max Verstappen (Red Bull Racing)

🏆 Predicted Podium:
   🥇 Max Verstappen (Red Bull Racing)
   🥈 Oscar Piastri (McLaren)
   🥉 Carlos Sainz Jr (Ferrari)

📈 Biggest Gainer: Carlos Sainz Jr (Grid 7 → P3)
📉 Biggest Slide: Lando Norris (Grid 2 → P8)
```

### Completed Race Results
```
🏆 AUSTRALIAN GRAND PRIX RESULTS
============================================================
Pos  Driver                    Team                 Time
------------------------------------------------------------
1    Lando Norris              McLaren              1:20:32.456
2    Oscar Piastri             McLaren              +5.234s
3    Max Verstappen            Red Bull Racing      +12.891s
4    George Russell            Mercedes             +18.567s
5    Lewis Hamilton            Ferrari              +25.123s

🎆 Race Winner: Lando Norris (McLaren)
🏆 Podium: Norris, Piastri, Verstappen
⚡ Fastest Lap: Lando Norris (1:18.234)
🌤️ Weather: Sunny, 24°C
```

## 🎨 Visualizations

The system generates several types of visualizations:

1. **Model Performance Comparison**: RMSE scores for different algorithms
2. **Constructor Performance**: Average finishing positions by team
3. **Driver Trends**: Performance evolution over time
4. **Prediction Confidence**: Heatmaps showing certainty levels
5. **Feature Importance**: Which factors matter most for predictions

## 🧠 Model Performance

Based on historical data testing:
- **Random Forest RMSE**: ~4.1 positions
- **XGBoost RMSE**: ~4.5 positions
- **Prediction Accuracy**: ~75% within 3 positions
- **Podium Prediction**: ~65% accuracy for top 3 finishers

## 📈 Strategic Applications

### Fantasy F1
- **Sleeper Picks**: Drivers likely to outperform their grid position
- **Value Bets**: Good performance potential at low cost
- **Avoid List**: Drivers at risk of underperforming

### Race Analysis
- **Constructor Battles**: Points predictions for team championships
- **Driver Rivalries**: Head-to-head matchup predictions
- **Weather Scenarios**: How conditions affect race outcomes

### Entertainment
- **Race Timeline**: Predicted key moments during the race
- **Upset Potential**: Likelihood of surprising results
- **Statistical Insights**: Data-driven race narratives

## 🌧️ Weather Impact

The system considers weather scenarios:
- **Dry Conditions**: 75% prediction confidence
- **Light Rain**: 52% confidence (skill factor increases)
- **Heavy Rain**: 22% confidence (high chaos factor)
- **Mixed Conditions**: 37% confidence (strategy crucial)

## 🔮 Limitations

- **Mechanical Failures**: Cannot predict unexpected car breakdowns
- **Race Incidents**: Crashes and safety cars add unpredictability
- **Strategy Variables**: Team decisions during the race
- **Weather Changes**: Real-time conditions during race weekend
- **Driver Form**: Day-of performance variations

## 📚 Data Sources

- **Ergast API**: Primary source for F1 historical data
- **Real-time APIs**: For current season standings and results
- **Weather Services**: Historical weather patterns at circuits
- **Manual Curation**: Circuit characteristics and driver ratings
- **2025 F1 Season Data**: Complete driver lineups, team changes, and race calendar
- **Race Results Database**: Detailed results from completed 2025 races with timing data

## 🤝 Contributing

Want to improve the model? Consider:
- Adding more data sources (telemetry, practice times)
- Implementing ensemble methods
- Creating web interface
- Adding real-time updates during race weekends
- Expanding to other motorsports

## ⚠️ Disclaimer

This model is for educational and entertainment purposes. Formula 1 races involve many unpredictable factors. Always gamble responsibly if using for betting purposes.

## 🎆 Recent Updates & Features

### ✅ Completed (v2.0)
- **Interactive React Dashboard**: Modern web interface with race predictions and results
- **Completed Race Results**: View detailed results from finished 2025 races
- **Enhanced Data Structure**: Comprehensive race data with podium results, lap times, and statistics
- **Modal Race Details**: Click any race for detailed predictions or results
- **Driver & Team Statistics**: Updated 2025 F1 season data with all teams and drivers
- **Circuit Information**: Track characteristics, lap records, and historical data

### 🕰️ Future Enhancements
- **Real-time Updates**: Live predictions during race weekend
- **Mobile App**: Predictions on your phone
- **Social Features**: Share predictions with friends
- **Expanded Coverage**: Other racing series (F2, IndyCar, etc.)
- **Live Race Commentary**: AI-powered race analysis during events

---

**Enjoy the races! 🏎️💨**

*May your predictions be accurate and your favorite drivers podium!*
