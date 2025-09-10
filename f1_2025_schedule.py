# F1 2025 Race Schedule - Official Calendar
# Updated with accurate dates and venues from Formula1.com

from datetime import datetime

# Complete F1 2025 Season Calendar
F1_2025_RACES = [
    {
        "round": 0,
        "name": "Pre-Season Testing",
        "location": "Sakhir Circuit",
        "country": "Bahrain",
        "dates": "Feb 26-28, 2025",
        "start_date": datetime(2025, 2, 26),
        "end_date": datetime(2025, 2, 28),
        "status": "testing",
        "type": "testing"
    },
    {
        "round": 1,
        "name": "Formula 1 Louis Vuitton Australian Grand Prix",
        "short_name": "Australian GP",
        "location": "Albert Park Circuit",
        "country": "Australia",
        "city": "Melbourne",
        "dates": "Mar 14-16, 2025",
        "start_date": datetime(2025, 3, 14),
        "end_date": datetime(2025, 3, 16),
        "race_date": datetime(2025, 3, 16),
        "status": "upcoming",
        "timezone": "Australia/Melbourne"
    },
    {
        "round": 2,
        "name": "Formula 1 Heineken Chinese Grand Prix", 
        "short_name": "Chinese GP",
        "location": "Shanghai International Circuit",
        "country": "China",
        "city": "Shanghai",
        "dates": "Mar 21-23, 2025",
        "start_date": datetime(2025, 3, 21),
        "end_date": datetime(2025, 3, 23),
        "race_date": datetime(2025, 3, 23),
        "status": "upcoming",
        "timezone": "Asia/Shanghai"
    },
    {
        "round": 3,
        "name": "Formula 1 Lenovo Japanese Grand Prix",
        "short_name": "Japanese GP", 
        "location": "Suzuka International Racing Course",
        "country": "Japan",
        "city": "Suzuka",
        "dates": "Apr 4-6, 2025",
        "start_date": datetime(2025, 4, 4),
        "end_date": datetime(2025, 4, 6),
        "race_date": datetime(2025, 4, 6),
        "status": "upcoming",
        "timezone": "Asia/Tokyo"
    },
    {
        "round": 4,
        "name": "Formula 1 Gulf Air Bahrain Grand Prix",
        "short_name": "Bahrain GP",
        "location": "Bahrain International Circuit", 
        "country": "Bahrain",
        "city": "Sakhir",
        "dates": "Apr 11-13, 2025",
        "start_date": datetime(2025, 4, 11),
        "end_date": datetime(2025, 4, 13),
        "race_date": datetime(2025, 4, 13),
        "status": "upcoming",
        "timezone": "Asia/Bahrain"
    },
    {
        "round": 5,
        "name": "Formula 1 STC Saudi Arabian Grand Prix",
        "short_name": "Saudi Arabian GP",
        "location": "Jeddah Corniche Circuit",
        "country": "Saudi Arabia", 
        "city": "Jeddah",
        "dates": "Apr 18-20, 2025",
        "start_date": datetime(2025, 4, 18),
        "end_date": datetime(2025, 4, 20),
        "race_date": datetime(2025, 4, 20),
        "status": "upcoming",
        "timezone": "Asia/Riyadh"
    },
    {
        "round": 6,
        "name": "Formula 1 Crypto.com Miami Grand Prix",
        "short_name": "Miami GP",
        "location": "Miami International Autodrome",
        "country": "USA",
        "city": "Miami",
        "dates": "May 2-4, 2025", 
        "start_date": datetime(2025, 5, 2),
        "end_date": datetime(2025, 5, 4),
        "race_date": datetime(2025, 5, 4),
        "status": "upcoming",
        "timezone": "America/New_York"
    },
    {
        "round": 7,
        "name": "Formula 1 AWS Gran Premio del Made in Italy e dell'Emilia-Romagna",
        "short_name": "Emilia-Romagna GP",
        "location": "Autodromo Enzo e Dino Ferrari",
        "country": "Italy",
        "city": "Imola", 
        "dates": "May 16-18, 2025",
        "start_date": datetime(2025, 5, 16),
        "end_date": datetime(2025, 5, 18),
        "race_date": datetime(2025, 5, 18),
        "status": "upcoming",
        "timezone": "Europe/Rome"
    },
    {
        "round": 8,
        "name": "Formula 1 Grand Prix de Monaco",
        "short_name": "Monaco GP",
        "location": "Circuit de Monaco",
        "country": "Monaco",
        "city": "Monte Carlo",
        "dates": "May 23-25, 2025",
        "start_date": datetime(2025, 5, 23),
        "end_date": datetime(2025, 5, 25),
        "race_date": datetime(2025, 5, 25),
        "status": "upcoming",
        "timezone": "Europe/Monaco"
    },
    {
        "round": 9,
        "name": "Formula 1 Aramco Gran Premio de España",
        "short_name": "Spanish GP",
        "location": "Circuit de Barcelona-Catalunya",
        "country": "Spain",
        "city": "Barcelona",
        "dates": "May 30 - Jun 1, 2025",
        "start_date": datetime(2025, 5, 30),
        "end_date": datetime(2025, 6, 1),
        "race_date": datetime(2025, 6, 1),
        "status": "upcoming", 
        "timezone": "Europe/Madrid"
    },
    {
        "round": 10,
        "name": "Formula 1 Pirelli Grand Prix du Canada",
        "short_name": "Canadian GP",
        "location": "Circuit Gilles Villeneuve",
        "country": "Canada",
        "city": "Montreal",
        "dates": "Jun 13-15, 2025",
        "start_date": datetime(2025, 6, 13),
        "end_date": datetime(2025, 6, 15),
        "race_date": datetime(2025, 6, 15),
        "status": "upcoming",
        "timezone": "America/Montreal"
    },
    {
        "round": 11,
        "name": "Formula 1 MSC Cruises Austrian Grand Prix",
        "short_name": "Austrian GP",
        "location": "Red Bull Ring",
        "country": "Austria",
        "city": "Spielberg",
        "dates": "Jun 27-29, 2025",
        "start_date": datetime(2025, 6, 27),
        "end_date": datetime(2025, 6, 29),
        "race_date": datetime(2025, 6, 29),
        "status": "upcoming",
        "timezone": "Europe/Vienna"
    },
    {
        "round": 12,
        "name": "Formula 1 Qatar Airways British Grand Prix",
        "short_name": "British GP",
        "location": "Silverstone Circuit", 
        "country": "United Kingdom",
        "city": "Silverstone",
        "dates": "Jul 4-6, 2025",
        "start_date": datetime(2025, 7, 4),
        "end_date": datetime(2025, 7, 6),
        "race_date": datetime(2025, 7, 6),
        "status": "upcoming",
        "timezone": "Europe/London"
    },
    {
        "round": 13,
        "name": "Formula 1 Moet & Chandon Belgian Grand Prix",
        "short_name": "Belgian GP",
        "location": "Circuit de Spa-Francorchamps",
        "country": "Belgium",
        "city": "Spa",
        "dates": "Jul 25-27, 2025",
        "start_date": datetime(2025, 7, 25),
        "end_date": datetime(2025, 7, 27),
        "race_date": datetime(2025, 7, 27),
        "status": "upcoming",
        "timezone": "Europe/Brussels"
    },
    {
        "round": 14,
        "name": "Formula 1 Lenovo Hungarian Grand Prix", 
        "short_name": "Hungarian GP",
        "location": "Hungaroring",
        "country": "Hungary",
        "city": "Budapest",
        "dates": "Aug 1-3, 2025",
        "start_date": datetime(2025, 8, 1),
        "end_date": datetime(2025, 8, 3),
        "race_date": datetime(2025, 8, 3),
        "status": "next_race", 
        "timezone": "Europe/Budapest"
    },
    {
        "round": 15,
        "name": "Formula 1 Heineken Dutch Grand Prix",
        "short_name": "Dutch GP", 
        "location": "Circuit Zandvoort",
        "country": "Netherlands",
        "city": "Zandvoort", 
        "dates": "Aug 29-31, 2025",
        "start_date": datetime(2025, 8, 29),
        "end_date": datetime(2025, 8, 31),
        "race_date": datetime(2025, 8, 31),
        "status": "upcoming",
        "timezone": "Europe/Amsterdam"
    },
    {
        "round": 16,
        "name": "Formula 1 Pirelli Gran Premio d'Italia",
        "short_name": "Italian GP",
        "location": "Autodromo Nazionale Monza",
        "country": "Italy",
        "city": "Monza",
        "dates": "Sep 5-7, 2025",
        "start_date": datetime(2025, 9, 5),
        "end_date": datetime(2025, 9, 7),
        "race_date": datetime(2025, 9, 7),
        "status": "upcoming",
        "timezone": "Europe/Rome"
    },
    {
        "round": 17,
        "name": "Formula 1 Qatar Airways Azerbaijan Grand Prix",
        "short_name": "Azerbaijan GP",
        "location": "Baku City Circuit", 
        "country": "Azerbaijan",
        "city": "Baku",
        "dates": "Sep 19-21, 2025",
        "start_date": datetime(2025, 9, 19),
        "end_date": datetime(2025, 9, 21),
        "race_date": datetime(2025, 9, 21),
        "status": "upcoming",
        "timezone": "Asia/Baku"
    },
    {
        "round": 18,
        "name": "Formula 1 Singapore Airlines Singapore Grand Prix",
        "short_name": "Singapore GP",
        "location": "Marina Bay Street Circuit",
        "country": "Singapore",
        "city": "Singapore",
        "dates": "Oct 3-5, 2025",
        "start_date": datetime(2025, 10, 3),
        "end_date": datetime(2025, 10, 5),
        "race_date": datetime(2025, 10, 5),
        "status": "upcoming", 
        "timezone": "Asia/Singapore"
    },
    {
        "round": 19,
        "name": "Formula 1 MSC Cruises United States Grand Prix",
        "short_name": "United States GP",
        "location": "Circuit of the Americas",
        "country": "USA",
        "city": "Austin",
        "dates": "Oct 17-19, 2025",
        "start_date": datetime(2025, 10, 17),
        "end_date": datetime(2025, 10, 19),
        "race_date": datetime(2025, 10, 19),
        "status": "upcoming",
        "timezone": "America/Chicago"
    },
    {
        "round": 20,
        "name": "Formula 1 Gran Premio de la Ciudad de México",
        "short_name": "Mexico City GP",
        "location": "Autódromo Hermanos Rodríguez",
        "country": "Mexico",
        "city": "Mexico City",
        "dates": "Oct 24-26, 2025",
        "start_date": datetime(2025, 10, 24),
        "end_date": datetime(2025, 10, 26),
        "race_date": datetime(2025, 10, 26),
        "status": "upcoming",
        "timezone": "America/Mexico_City"
    },
    {
        "round": 21,
        "name": "Formula 1 MSC Cruises Grande Prêmio de São Paulo",
        "short_name": "São Paulo GP",
        "location": "Interlagos Circuit",
        "country": "Brazil",
        "city": "São Paulo",
        "dates": "Nov 7-9, 2025",
        "start_date": datetime(2025, 11, 7),
        "end_date": datetime(2025, 11, 9),
        "race_date": datetime(2025, 11, 9),
        "status": "upcoming",
        "timezone": "America/Sao_Paulo"
    },
    {
        "round": 22,
        "name": "Formula 1 Heineken Las Vegas Grand Prix",
        "short_name": "Las Vegas GP", 
        "location": "Las Vegas Strip Circuit",
        "country": "USA",
        "city": "Las Vegas",
        "dates": "Nov 20-22, 2025",
        "start_date": datetime(2025, 11, 20),
        "end_date": datetime(2025, 11, 22),
        "race_date": datetime(2025, 11, 22),
        "status": "upcoming",
        "timezone": "America/Los_Angeles"
    },
    {
        "round": 23,
        "name": "Formula 1 Qatar Airways Qatar Grand Prix",
        "short_name": "Qatar GP",
        "location": "Lusail International Circuit",
        "country": "Qatar", 
        "city": "Lusail",
        "dates": "Nov 28-30, 2025",
        "start_date": datetime(2025, 11, 28),
        "end_date": datetime(2025, 11, 30),
        "race_date": datetime(2025, 11, 30),
        "status": "upcoming",
        "timezone": "Asia/Qatar"
    },
    {
        "round": 24,
        "name": "Formula 1 Etihad Airways Abu Dhabi Grand Prix", 
        "short_name": "Abu Dhabi GP",
        "location": "Yas Marina Circuit",
        "country": "UAE",
        "city": "Abu Dhabi",
        "dates": "Dec 5-7, 2025",
        "start_date": datetime(2025, 12, 5),
        "end_date": datetime(2025, 12, 7),
        "race_date": datetime(2025, 12, 7),
        "status": "upcoming",
        "timezone": "Asia/Dubai"
    }
]

# Useful functions for working with the schedule
def get_next_race():
    """Get the next upcoming race"""
    current_date = datetime.now()
    for race in F1_2025_RACES:
        if race.get("type") != "testing" and race["start_date"] > current_date:
            return race
    return None

def get_race_by_round(round_number):
    """Get race by round number"""
    for race in F1_2025_RACES:
        if race.get("round") == round_number:
            return race
    return None

def get_races_by_month(month, year=2025):
    """Get all races in a specific month"""
    races_in_month = []
    for race in F1_2025_RACES:
        if race["start_date"].month == month and race["start_date"].year == year:
            races_in_month.append(race)
    return races_in_month

def get_remaining_races():
    """Get all remaining races in the season"""
    current_date = datetime.now()
    remaining = []
    for race in F1_2025_RACES:
        if race.get("type") != "testing" and race["start_date"] > current_date:
            remaining.append(race)
    return remaining

def get_completed_races():
    """Get all completed races"""
    current_date = datetime.now()
    completed = []
    for race in F1_2025_RACES:
        if race.get("type") != "testing" and race["end_date"] < current_date:
            completed.append(race)
    return completed

# Circuit characteristics for prediction models
CIRCUIT_CHARACTERISTICS = {
    "Albert Park Circuit": {
        "type": "street",
        "length_km": 5.278,
        "laps": 58,
        "characteristics": ["medium_speed_corners", "tight_sections", "drs_zones"],
        "surface": "asphalt",
        "direction": "clockwise"
    },
    "Shanghai International Circuit": {
        "type": "permanent",
        "length_km": 5.451,
        "laps": 56,
        "characteristics": ["long_straight", "hairpin", "high_speed_corners"],
        "surface": "asphalt", 
        "direction": "clockwise"
    },
    "Suzuka International Racing Course": {
        "type": "permanent",
        "length_km": 5.807,
        "laps": 53,
        "characteristics": ["figure_eight", "challenging_corners", "elevation_changes"],
        "surface": "asphalt",
        "direction": "clockwise"
    },
    # Add more circuits as needed...
}

# Export commonly used data
TOTAL_RACES = 24
SEASON_START = datetime(2025, 3, 14)  # Australia GP
SEASON_END = datetime(2025, 12, 7)    # Abu Dhabi GP

if __name__ == "__main__":
    # Example usage
    print("F1 2025 Season Overview:")
    print(f"Total races: {TOTAL_RACES}")
    print(f"Season: {SEASON_START.strftime('%B %d')} - {SEASON_END.strftime('%B %d, %Y')}")
    
    next_race = get_next_race()
    if next_race:
        print(f"\nNext race: {next_race['short_name']}")
        print(f"Date: {next_race['dates']}")
        print(f"Location: {next_race['location']}")
    
    remaining = get_remaining_races()
    print(f"\nRemaining races this season: {len(remaining)}")
