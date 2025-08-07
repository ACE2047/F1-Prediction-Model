# 🏎️ F1 Data Sources Guide

This comprehensive guide covers all available data sources and how to integrate them with your F1 prediction model.

## 📊 **Primary Data Sources**

### **1. Ergast API (Already Integrated)**
- **URL**: `http://ergast.com/api/f1`
- **Coverage**: 1950-present
- **Data Types**: Race results, qualifying, standings, circuits, drivers, constructors
- **Format**: JSON/XML
- **Rate Limits**: None (reasonable usage)
- **Integration**: ✅ Already in `f1_predictor.py`

**Example Usage:**
```python
from enhanced_data_collector import EnhancedF1DataCollector

collector = EnhancedF1DataCollector()
race_data = collector.get_race_results(2024)
```

### **2. FastF1 Library (Integrated)**
- **Installation**: `pip install fastf1`
- **Coverage**: 2018-present with telemetry
- **Data Types**: Lap times, telemetry, weather, tire data
- **Features**: Session data, car telemetry, timing data
- **Integration**: ✅ In `enhanced_data_collector.py`

**Setup:**
```python
import fastf1
fastf1.Cache.enable_cache('cache_directory')

# Get session data
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load(telemetry=True, weather=True)
```

## 🌤️ **Weather Data Sources**

### **3. OpenWeatherMap API**
- **URL**: `https://openweathermap.org/api`
- **Coverage**: Historical and current weather
- **Data Types**: Temperature, humidity, wind, precipitation
- **Cost**: Free tier available (1000 calls/day)
- **Setup Required**: API key needed

**Setup Steps:**
1. Sign up at [OpenWeatherMap](https://openweathermap.org/api)
2. Get your API key
3. Set environment variable:
   ```bash
   $env:OPENWEATHER_API_KEY="your_api_key_here"
   ```

### **4. Weather API Alternatives**
- **WeatherAPI.com**: 1M calls/month free
- **Visual Crossing**: Historical weather data
- **AccuWeather**: Professional weather data
- **NOAA**: Free government weather data

## 🏁 **Track and Circuit Data**

### **5. Built-in Circuit Database (Integrated)**
- **Coverage**: Major F1 circuits
- **Data**: Length, corners, elevation, DRS zones, coordinates
- **Integration**: ✅ In `enhanced_data_collector.py`

**Available Circuits:**
- Monaco, Silverstone, Monza, Spa-Francorchamps
- Interlagos, Suzuka, Austin, Melbourne
- Bahrain, Jeddah, and more...

### **6. Additional Circuit Data Sources**
- **F1Technical.net**: Technical circuit information
- **RaceFans.net**: Circuit analysis and data
- **Individual circuit websites**: Official data

## 📈 **Advanced Data Sources**

### **7. RapidAPI F1 Services**
Multiple F1 APIs available on RapidAPI platform:
- **F1 Live Motorsport Data**
- **API-SPORTS F1**
- **Formula 1 World Championship Stats**

**Setup:**
1. Sign up at [RapidAPI](https://rapidapi.com)
2. Subscribe to F1 APIs
3. Use provided API keys

### **8. Sports Data Providers**
- **The Sports DB**: Free sports data
- **SportRadar**: Professional sports data
- **Sportradar**: Real-time sports data

### **9. Official F1 Data**
- **F1 Live Timing**: `https://livetiming.formula1.com/signalr`
- **F1 API**: Official Formula 1 API (limited access)
- **FIA Results**: Official race documents

## 🔧 **Setup Instructions**

### **Environment Variables**
Create a `.env` file in your project directory:

```bash
# Weather API
OPENWEATHER_API_KEY=your_openweather_key

# RapidAPI Keys (optional)
RAPIDAPI_KEY=your_rapidapi_key

# Other API keys as needed
WEATHER_API_KEY=your_weather_api_key
SPORTS_API_KEY=your_sports_api_key
```

### **Enhanced Requirements**
Update your `requirements.txt`:

```text
# Existing requirements
pandas
numpy
scikit-learn
matplotlib
seaborn
requests
beautifulsoup4
xgboost
lightgbm
plotly
jupyter

# Enhanced requirements
fastf1
scipy
python-dateutil
lxml
openpyxl
python-dotenv
aiohttp
asyncio
```

### **Installation Commands**
```bash
# Install all requirements
pip install -r requirements.txt

# Install specific enhancements
pip install fastf1 scipy python-dotenv aiohttp

# For development
pip install jupyter notebook ipykernel
```

## 🚀 **Quick Start Guide**

### **1. Basic Enhanced Setup**
```python
from enhanced_data_collector import EnhancedF1DataCollector
from advanced_feature_engineering import AdvancedF1FeatureEngineer
from enhanced_f1_predictor import EnhancedF1PredictionModel

# Initialize with FastF1 (no weather API key needed for basic setup)
model = EnhancedF1PredictionModel(enable_fastf1=True, enable_weather=False)

# Collect and train
data = model.collect_comprehensive_data(start_year=2022, end_year=2024)
performance = model.train_enhanced_models(data)
```

### **2. Full Setup with Weather Data**
```python
import os
os.environ['OPENWEATHER_API_KEY'] = 'your_key_here'

# Initialize with all features
model = EnhancedF1PredictionModel(enable_fastf1=True, enable_weather=True)

# Full data collection
data = model.collect_comprehensive_data(start_year=2020, end_year=2024)
performance = model.train_enhanced_models(data)
```

### **3. Custom Data Integration**
```python
# Extend the data collector for custom sources
class CustomF1DataCollector(EnhancedF1DataCollector):
    def get_custom_data_source(self):
        # Your custom data integration
        pass

# Use custom collector
model = EnhancedF1PredictionModel()
model.data_collector = CustomF1DataCollector()
```

## 📊 **Available Features**

### **Base Features (26+)**
- Historical performance (position, points, wins, podiums)
- Circuit-specific performance
- Season context and championship position
- Constructor performance
- Qualifying data

### **Enhanced Features (40+)**
- **Weather**: Temperature, humidity, rainfall, conditions
- **Circuit**: Length, corners, type, elevation, overtaking difficulty
- **Tire Strategy**: Compound data, strategy patterns
- **Reliability**: DNF history, mechanical failures
- **Form**: Recent performance, consistency, trends
- **Strategic**: Grid effects, qualifying vs race performance

### **Advanced Features (20+)**
- **Momentum**: Win streaks, form trends
- **Competitive**: Teammate comparisons, field strength
- **Interaction**: Weather×Circuit, Driver×Track combinations
- **Risk**: DNF probability, reliability metrics

## 🔄 **Data Update Strategies**

### **Manual Updates**
```python
# Update for specific race weekend
collector = EnhancedF1DataCollector()
latest_data = collector.get_enhanced_race_data(2024)
```

### **Automated Updates**
```python
import schedule
import time

def update_f1_data():
    model = EnhancedF1PredictionModel.load_model('enhanced_f1_model.pkl')
    # Update logic here
    model.save_model('enhanced_f1_model.pkl')

# Schedule updates
schedule.every().friday.at("10:00").do(update_f1_data)  # Practice day
schedule.every().sunday.at("20:00").do(update_f1_data)  # Post-race

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

## 🎯 **Best Practices**

### **Data Quality**
1. **Validate data sources** before integration
2. **Handle missing data** gracefully
3. **Check for data consistency** across sources
4. **Monitor API rate limits**

### **Performance Optimization**
1. **Cache FastF1 data** to avoid repeated downloads
2. **Use batch requests** when possible
3. **Implement retry logic** for API failures
4. **Store processed data** locally

### **Security**
1. **Never commit API keys** to version control
2. **Use environment variables** for secrets
3. **Rotate API keys** regularly
4. **Monitor API usage**

## 🆘 **Troubleshooting**

### **Common Issues**

**FastF1 Installation Issues:**
```bash
# If FastF1 fails to install
pip install --upgrade pip
pip install fastf1 --no-cache-dir
```

**API Rate Limits:**
```python
import time
import requests

def make_request_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 429:  # Rate limited
                time.sleep(60)  # Wait 1 minute
                continue
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(5)
```

**Memory Issues with Large Datasets:**
```python
# Process data in chunks
def process_large_dataset(years):
    all_data = []
    for year in years:
        year_data = collector.get_enhanced_race_data(year)
        if not year_data.empty:
            # Process immediately to save memory
            processed_data = feature_engineer.create_all_advanced_features(year_data)
            all_data.append(processed_data)
    return pd.concat(all_data, ignore_index=True)
```

## 📞 **Support Resources**

- **FastF1 Documentation**: https://docs.fastf1.dev/
- **Ergast API Documentation**: http://ergast.com/mrd/
- **OpenWeatherMap API Docs**: https://openweathermap.org/api
- **F1 Technical Forum**: https://www.f1technical.net/forum/
- **GitHub Issues**: Create issues in the project repository

---

**Happy F1 Data Analysis! 🏁**

*Remember: The more data sources you integrate, the better your predictions become. Start with the basics and gradually add more sophisticated data sources as you become comfortable with the system.*
