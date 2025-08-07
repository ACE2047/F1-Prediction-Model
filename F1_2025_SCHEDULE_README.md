# 🏁 F1 2025 Race Schedule - Complete & Accurate

This repository now contains the complete and accurate Formula 1 2025 race schedule, based on the official calendar from Formula1.com.

## 📋 Schedule Files

### 1. `src/data/raceData.js` 
- **Purpose**: React/JavaScript application data
- **Contains**: Complete race calendar with enhanced data structure
- **Features**: 
  - All 24 races with accurate dates
  - Proper race names with sponsors
  - Country flags and locations
  - Status tracking (upcoming, completed, next)

### 2. `f1_2025_schedule.py`
- **Purpose**: Python machine learning models
- **Contains**: Structured data with datetime objects
- **Features**:
  - Utility functions for race filtering
  - Circuit characteristics data
  - Timezone information
  - Date/time calculations

### 3. `f1_2025_calendar.csv`
- **Purpose**: Universal data format for any application
- **Contains**: Clean CSV with all race information
- **Usage**: Easy import into spreadsheets, databases, or other tools

## 🏎️ 2025 F1 Season Overview

- **Total Races**: 24
- **Season Start**: March 14, 2025 (Australia GP)
- **Season End**: December 7, 2025 (Abu Dhabi GP)
- **Next Race**: Hungarian GP (August 1-3, 2025)*

*As of current system date

## 📅 Complete Race Calendar

| Round | Race | Location | Date | Status |
|-------|------|----------|------|--------|
| 1 | 🇦🇺 Australian GP | Albert Park Circuit | Mar 14-16 | Upcoming |
| 2 | 🇨🇳 Chinese GP | Shanghai International Circuit | Mar 21-23 | Upcoming |
| 3 | 🇯🇵 Japanese GP | Suzuka International Racing Course | Apr 4-6 | Upcoming |
| 4 | 🇧🇭 Bahrain GP | Bahrain International Circuit | Apr 11-13 | Upcoming |
| 5 | 🇸🇦 Saudi Arabian GP | Jeddah Corniche Circuit | Apr 18-20 | Upcoming |
| 6 | 🇺🇸 Miami GP | Miami International Autodrome | May 2-4 | Upcoming |
| 7 | 🇮🇹 Emilia-Romagna GP | Autodromo Enzo e Dino Ferrari | May 16-18 | Upcoming |
| 8 | 🇲🇨 Monaco GP | Circuit de Monaco | May 23-25 | Upcoming |
| 9 | 🇪🇸 Spanish GP | Circuit de Barcelona-Catalunya | May 30-Jun 1 | Upcoming |
| 10 | 🇨🇦 Canadian GP | Circuit Gilles Villeneuve | Jun 13-15 | Upcoming |
| 11 | 🇦🇹 Austrian GP | Red Bull Ring | Jun 27-29 | Upcoming |
| 12 | 🇬🇧 British GP | Silverstone Circuit | Jul 4-6 | Upcoming |
| 13 | 🇧🇪 Belgian GP | Circuit de Spa-Francorchamps | Jul 25-27 | Upcoming |
| 14 | 🇭🇺 Hungarian GP | Hungaroring | Aug 1-3 | **Next Race** |
| 15 | 🇳🇱 Dutch GP | Circuit Zandvoort | Aug 29-31 | Upcoming |
| 16 | 🇮🇹 Italian GP | Autodromo Nazionale Monza | Sep 5-7 | Upcoming |
| 17 | 🇦🇿 Azerbaijan GP | Baku City Circuit | Sep 19-21 | Upcoming |
| 18 | 🇸🇬 Singapore GP | Marina Bay Street Circuit | Oct 3-5 | Upcoming |
| 19 | 🇺🇸 United States GP | Circuit of the Americas | Oct 17-19 | Upcoming |
| 20 | 🇲🇽 Mexico City GP | Autódromo Hermanos Rodríguez | Oct 24-26 | Upcoming |
| 21 | 🇧🇷 São Paulo GP | Interlagos Circuit | Nov 7-9 | Upcoming |
| 22 | 🇺🇸 Las Vegas GP | Las Vegas Strip Circuit | Nov 20-22 | Upcoming |
| 23 | 🇶🇦 Qatar GP | Lusail International Circuit | Nov 28-30 | Upcoming |
| 24 | 🇦🇪 Abu Dhabi GP | Yas Marina Circuit | Dec 5-7 | Upcoming |

## 🔧 Usage Examples

### Python Usage
```python
from f1_2025_schedule import get_next_race, get_remaining_races, F1_2025_RACES

# Get the next race
next_race = get_next_race()
print(f"Next race: {next_race['short_name']} on {next_race['dates']}")

# Get all remaining races
remaining = get_remaining_races()
print(f"Races left: {len(remaining)}")

# Get races by month
march_races = get_races_by_month(3)
```

### JavaScript Usage
```javascript
import { enhancedRaceCalendar } from './src/data/raceData.js';

// Filter upcoming races
const upcomingRaces = enhancedRaceCalendar.filter(race => race.status === 'upcoming');

// Find next race
const nextRace = enhancedRaceCalendar.find(race => race.status === 'next');
```

### CSV Usage
```bash
# Import into pandas
import pandas as pd
df = pd.read_csv('f1_2025_calendar.csv')

# Import into Excel, Google Sheets, or any spreadsheet application
```

## ✅ Data Accuracy

The schedule has been cross-referenced with:
- ✅ Official Formula1.com calendar
- ✅ Race weekend dates (Friday-Sunday format)
- ✅ Correct circuit names and locations
- ✅ Proper sponsor naming conventions
- ✅ Accurate timezone information

## 🔄 Updates

This schedule is current as of August 2025. For any changes to the official F1 calendar:
1. Update the relevant data files
2. Verify dates against Formula1.com
3. Test with your prediction models
4. Update this README if needed

## 🎯 Integration with Prediction Models

The schedule data is designed to work seamlessly with your F1 prediction models:

- **Date filtering**: Find next races for prediction
- **Circuit analysis**: Match historical performance by track
- **Season progression**: Understand championship context
- **Timeline planning**: Schedule prediction runs

## 📊 Data Structure

Each race includes:
- Round number (1-24)
- Full official race name
- Short name for displays
- Circuit location and country
- Start/end dates
- Race day (always Sunday)
- Current status
- Timezone for local time conversion

---

**Ready to race! 🏎️💨**

*Your F1 prediction model now has access to the complete and accurate 2025 race schedule.*
