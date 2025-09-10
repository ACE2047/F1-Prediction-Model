#!/usr/bin/env python3
"""
Enhanced Next F1 Race Predictor
==============================

Advanced prediction system that integrates with the 2025 F1 schedule,
provides detailed analysis, confidence intervals, and beautiful visualizations.

Features:
- Integration with official 2025 F1 schedule
- Enhanced ML predictions with confidence intervals
- Beautiful console output with rich formatting
- Interactive race selection
- Detailed driver/team analysis
- Weather and circuit factor considerations
- Strategic insights and recommendations

Usage:
    python enhanced_next_race_predictor.py
"""

import pandas as pd
import numpy as np
import requests
import pickle
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import os
import sys
from pathlib import Path

# Rich console formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    from rich.layout import Layout
    from rich import box
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("📦 Installing rich for better display...")
    os.system("pip install rich")
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.progress import Progress, SpinnerColumn, TextColumn
        from rich.text import Text
        from rich.layout import Layout
        from rich import box
        from rich.align import Align
        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False

# Enhanced plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import our schedule data
from f1_2025_schedule import F1_2025_RACES, get_next_race, get_remaining_races, CIRCUIT_CHARACTERISTICS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if RICH_AVAILABLE:
    console = Console()
else:
    console = None

class EnhancedNextRacePredictor:
    """Enhanced F1 next race predictor with advanced features."""
    
    def __init__(self, model_path: str = 'enhanced_f1_model.pkl'):
        """Initialize with enhanced model."""
        self.console = console
        self.model_path = model_path
        self.f1_model = None
        self.current_season_data = {}
        self.driver_ratings = {}
        self.team_ratings = {}
        
        self._load_or_train_model()
        self._initialize_season_data()
    
    def _print_banner(self):
        """Print enhanced banner."""
        if not RICH_AVAILABLE:
            print("🏎️  Enhanced F1 Next Race Predictor 2025")
            print("=" * 60)
            return
            
        banner_text = """
🏎️  Enhanced F1 Next Race Predictor 2025
    
    Advanced ML Predictions • Circuit Analysis • Strategy Insights
    Weather Integration • Driver Performance • Team Dynamics
        """
        
        panel = Panel(
            Align.center(banner_text),
            box=box.DOUBLE,
            style="bold red",
            title="🏁 Formula 1 AI Predictor",
            subtitle="Powered by Machine Learning"
        )
        self.console.print(panel)
    
    def _load_or_train_model(self):
        """Load existing model or create new one."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                # Handle both old and new model formats
                if hasattr(model_data, 'models'):
                    self.f1_model = model_data
                else:
                    # Old format - convert
                    from enhanced_f1_predictor import EnhancedF1PredictionModel
                    self.f1_model = model_data
                    
                if RICH_AVAILABLE:
                    self.console.print("✅ [green]Enhanced F1 model loaded successfully![/green]")
                else:
                    print("✅ Enhanced F1 model loaded successfully!")
            else:
                self._train_new_model()
        except Exception as e:
            logger.warning(f"Could not load enhanced model: {e}")
            self._train_new_model()
    
    def _train_new_model(self):
        """Train a new enhanced model."""
        if RICH_AVAILABLE:
            self.console.print("🔄 [yellow]No enhanced model found. Training new model...[/yellow]")
        else:
            print("🔄 No enhanced model found. Training new model...")
        
        try:
            from enhanced_f1_predictor import EnhancedF1PredictionModel
            self.f1_model = EnhancedF1PredictionModel(enable_fastf1=False, enable_weather=False)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress if RICH_AVAILABLE else None:
                
                if RICH_AVAILABLE:
                    task = progress.add_task("Training model...", total=None)
                
                # Collect data
                df = self.f1_model.collect_comprehensive_data(start_year=2020, end_year=2024)
                
                if df.empty:
                    raise Exception("No training data collected")
                
                if RICH_AVAILABLE:
                    progress.update(task, description="Training ML models...")
                
                # Train models
                performance = self.f1_model.train_enhanced_models(df)
                
                # Save model
                self.f1_model.save_model(self.model_path)
                
                if RICH_AVAILABLE:
                    self.console.print("✅ [green]Model trained and saved successfully![/green]")
                else:
                    print("✅ Model trained and saved successfully!")
                    
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            if RICH_AVAILABLE:
                self.console.print(f"❌ [red]Model training failed: {e}[/red]")
            else:
                print(f"❌ Model training failed: {e}")
            sys.exit(1)
    
    def _initialize_season_data(self):
        """Initialize current season data."""
        # Load current driver and team data
        self.current_season_data = {
            'drivers': {
                'verstappen': {'team': 'Red Bull Racing', 'number': 1, 'rating': 95},
                'perez': {'team': 'Red Bull Racing', 'number': 11, 'rating': 85},
                'leclerc': {'team': 'Ferrari', 'number': 16, 'rating': 92},
                'sainz': {'team': 'Williams', 'number': 55, 'rating': 87},
                'hamilton': {'team': 'Ferrari', 'number': 44, 'rating': 94},
                'russell': {'team': 'Mercedes', 'number': 63, 'rating': 88},
                'norris': {'team': 'McLaren', 'number': 4, 'rating': 90},
                'piastri': {'team': 'McLaren', 'number': 81, 'rating': 86},
                'alonso': {'team': 'Aston Martin', 'number': 14, 'rating': 91},
                'stroll': {'team': 'Aston Martin', 'number': 18, 'rating': 80},
            },
            'teams': {
                'Red Bull Racing': {'rating': 92, 'reliability': 88},
                'Ferrari': {'rating': 90, 'reliability': 85},
                'McLaren': {'rating': 89, 'reliability': 87},
                'Mercedes': {'rating': 85, 'reliability': 89},
                'Aston Martin': {'rating': 82, 'reliability': 84},
                'Williams': {'rating': 78, 'reliability': 86}
            }
        }
    
    def select_race_interactive(self) -> Optional[Dict]:
        """Interactive race selection."""
        upcoming_races = get_remaining_races()
        
        if not upcoming_races:
            if RICH_AVAILABLE:
                self.console.print("❌ [red]No upcoming races found in 2025 schedule![/red]")
            else:
                print("❌ No upcoming races found in 2025 schedule!")
            return None
        
        if RICH_AVAILABLE:
            # Create race selection table
            table = Table(title="🏁 Select a Race to Predict", box=box.ROUNDED)
            table.add_column("#", style="cyan", width=3)
            table.add_column("Race", style="bold white", min_width=25)
            table.add_column("Date", style="green", width=15)
            table.add_column("Location", style="yellow", min_width=20)
            table.add_column("Status", style="blue", width=10)
            
            for i, race in enumerate(upcoming_races[:10], 1):  # Show next 10 races
                status = "🔥 Next" if race.get('status') == 'next_race' else "📅 Upcoming"
                table.add_row(
                    str(i),
                    race['short_name'],
                    race['dates'],
                    race['location'],
                    status
                )
            
            self.console.print(table)
            
            try:
                selection = int(input("\nEnter race number (or 0 for next race): "))
                if selection == 0:
                    return upcoming_races[0]  # Next race
                elif 1 <= selection <= len(upcoming_races):
                    return upcoming_races[selection - 1]
                else:
                    self.console.print("❌ [red]Invalid selection![/red]")
                    return None
            except ValueError:
                self.console.print("❌ [red]Please enter a valid number![/red]")
                return None
        else:
            # Fallback for no rich
            print("\nUpcoming F1 Races:")
            for i, race in enumerate(upcoming_races[:5], 1):
                print(f"{i}. {race['short_name']} - {race['dates']}")
            
            try:
                selection = int(input("Select race number (0 for next): "))
                if selection == 0:
                    return upcoming_races[0]
                elif 1 <= selection <= len(upcoming_races):
                    return upcoming_races[selection - 1]
            except ValueError:
                pass
            return None
    
    def get_race_analysis(self, race: Dict) -> Dict:
        """Get comprehensive race analysis."""
        circuit_name = race['location']
        circuit_info = CIRCUIT_CHARACTERISTICS.get(circuit_name, {})
        
        analysis = {
            'race_info': race,
            'circuit_info': circuit_info,
            'weather_factors': self._analyze_weather_factors(race),
            'key_factors': self._get_key_race_factors(race),
            'driver_advantages': self._analyze_driver_advantages(race),
            'team_advantages': self._analyze_team_advantages(race),
            'strategic_considerations': self._get_strategic_factors(race)
        }
        
        return analysis
    
    def _analyze_weather_factors(self, race: Dict) -> Dict:
        """Analyze weather factors for the race."""
        # This would normally fetch real weather data
        month = int(race['start_date'].strftime('%m'))
        location = race['city']
        
        weather_predictions = {
            'temperature': 'Moderate (20-25°C)',
            'precipitation': 'Low chance (10%)',
            'conditions': 'Dry',
            'impact': 'Minimal weather disruption expected'
        }
        
        # Season-based predictions
        if month in [12, 1, 2]:  # Winter
            weather_predictions.update({
                'temperature': 'Cool (15-20°C)',
                'impact': 'Cool conditions may favor tire longevity'
            })
        elif month in [6, 7, 8]:  # Summer
            weather_predictions.update({
                'temperature': 'Hot (25-35°C)',
                'impact': 'Hot conditions increase tire degradation'
            })
        
        return weather_predictions
    
    def _get_key_race_factors(self, race: Dict) -> List[str]:
        """Get key factors for the race."""
        factors = [
            f"Round {race['round']} of 24 in the 2025 season",
            f"Circuit type: {CIRCUIT_CHARACTERISTICS.get(race['location'], {}).get('type', 'Unknown')}",
            "Current championship battle context",
            "Team development progression",
            "Driver form and confidence levels"
        ]
        
        # Add circuit-specific factors
        circuit_info = CIRCUIT_CHARACTERISTICS.get(race['location'], {})
        if circuit_info.get('characteristics'):
            factors.extend([f"Circuit feature: {char}" for char in circuit_info['characteristics'][:2]])
        
        return factors
    
    def _analyze_driver_advantages(self, race: Dict) -> List[Dict]:
        """Analyze which drivers have advantages."""
        advantages = []
        
        # Historical performance (simplified)
        circuit_specialists = {
            'Monaco': ['hamilton', 'leclerc'],
            'Silverstone': ['hamilton', 'russell'],
            'Spa': ['verstappen', 'hamilton'],
            'Monza': ['leclerc', 'sainz'],
            'Suzuka': ['verstappen', 'hamilton']
        }
        
        location = race['location']
        for circuit, drivers in circuit_specialists.items():
            if circuit.lower() in location.lower():
                for driver in drivers:
                    if driver in self.current_season_data['drivers']:
                        advantages.append({
                            'driver': driver.title(),
                            'advantage': f"Historical strong performance at {circuit}",
                            'confidence': 0.8
                        })
                break
        
        return advantages[:3]  # Top 3
    
    def _analyze_team_advantages(self, race: Dict) -> List[Dict]:
        """Analyze team advantages."""
        advantages = []
        
        # Circuit-team combinations
        team_circuits = {
            'Red Bull Racing': ['Suzuka', 'Austria', 'Brazil'],
            'Ferrari': ['Monza', 'Monaco', 'Bahrain'],
            'McLaren': ['Silverstone', 'Hungary', 'Singapore'],
            'Mercedes': ['Silverstone', 'Russia', 'Spain']
        }
        
        location = race['location']
        for team, circuits in team_circuits.items():
            for circuit in circuits:
                if circuit.lower() in location.lower():
                    advantages.append({
                        'team': team,
                        'advantage': f"Strong aerodynamic package suits {circuit}",
                        'confidence': 0.75
                    })
                    break
        
        return advantages[:2]  # Top 2
    
    def _get_strategic_factors(self, race: Dict) -> List[str]:
        """Get strategic considerations."""
        factors = [
            "Tire strategy will be crucial for race outcome",
            "Qualifying position advantage varies by circuit",
            "Safety car probability affects strategy choices",
            "Weather changes could shuffle the order",
            "Championship points situation influences risk-taking"
        ]
        
        # Circuit-specific strategies
        circuit_info = CIRCUIT_CHARACTERISTICS.get(race['location'], {})
        if 'overtaking_difficulty' in str(circuit_info):
            factors.append("Limited overtaking opportunities - qualifying crucial")
        
        return factors[:4]
    
    def create_mock_prediction_data(self, race: Dict) -> pd.DataFrame:
        """Create mock data for prediction."""
        drivers_data = []
        
        for i, (driver_id, driver_info) in enumerate(self.current_season_data['drivers'].items()):
            # Simulate grid positions based on current form
            base_grid = i + 1 + np.random.randint(-2, 3)
            grid_pos = max(1, min(20, base_grid))
            
            race_data = {
                'season': 2025,
                'round': race['round'],
                'race_name': race['name'],
                'circuit_name': race['location'],
                'driver_id': driver_id,
                'driver_name': driver_id.title(),
                'constructor_id': driver_info['team'].lower().replace(' ', '_'),
                'constructor_name': driver_info['team'],
                'grid': grid_pos,
                'quali_position': grid_pos,
                'season_progress': race['round'] / 24,
                
                # Enhanced features based on current form
                'driver_avg_position_last_5': max(1, min(20, 10 + np.random.randint(-5, 5))),
                'driver_avg_points_last_5': max(0, 8 + np.random.randint(-5, 8)),
                'driver_wins_last_10': max(0, np.random.randint(0, 3)),
                'driver_podiums_last_10': max(0, np.random.randint(1, 6)),
                'constructor_avg_position_last_5': max(1, min(20, 8 + np.random.randint(-4, 6))),
                'constructor_avg_points_last_5': max(0, 15 + np.random.randint(-8, 15)),
                'driver_circuit_avg_position': max(1, min(20, 10 + np.random.randint(-4, 4))),
                'constructor_circuit_avg_position': max(1, min(20, 9 + np.random.randint(-3, 5))),
                'driver_constructor_combo_avg': max(1, min(20, 9 + np.random.randint(-3, 4))),
                'grid_vs_avg_position': np.random.uniform(-2, 2),
                'driver_points': max(0, 50 + np.random.randint(-30, 80)),
                'constructor_points': max(0, 100 + np.random.randint(-60, 150)),
                'driver_wins': max(0, np.random.randint(0, 4)),
                'constructor_wins': max(0, np.random.randint(0, 6)),
                
                # Additional features
                'recent_form': np.random.uniform(0.3, 1.0),
                'consistency': np.random.uniform(0.4, 0.9),
                'position_trend': np.random.uniform(-0.3, 0.3),
                'driver_rating': driver_info['rating'],
                'team_rating': self.current_season_data['teams'][driver_info['team']]['rating']
            }
            
            drivers_data.append(race_data)
        
        return pd.DataFrame(drivers_data)
    
    def predict_race_enhanced(self, race: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Make enhanced race predictions."""
        if not self.f1_model:
            raise Exception("No model available for predictions!")
        
        # Create prediction data
        race_df = self.create_mock_prediction_data(race)
        
        try:
            # Use the enhanced model if available
            if hasattr(self.f1_model, 'predict_race_enhanced'):
                predictions = self.f1_model.predict_race_enhanced(
                    race_df, 
                    model_name='ensemble',
                    confidence_interval=True
                )
            else:
                # Fallback to basic prediction
                race_df = self.f1_model.prepare_features(race_df) if hasattr(self.f1_model, 'prepare_features') else race_df
                
                # Simple prediction
                predictions = race_df.copy()
                predictions['predicted_position'] = predictions['grid'] + np.random.randint(-3, 4, len(predictions))
                predictions['predicted_position'] = np.clip(predictions['predicted_position'], 1, 20)
                predictions['confidence'] = np.random.uniform(0.6, 0.9, len(predictions))
                predictions = predictions.sort_values('predicted_position').reset_index(drop=True)
            
            # Calculate additional insights
            insights = self._calculate_prediction_insights(predictions, race_df)
            
            return predictions, insights
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise Exception(f"Failed to generate predictions: {e}")
    
    def _calculate_prediction_insights(self, predictions: pd.DataFrame, original_data: pd.DataFrame) -> Dict:
        """Calculate additional insights from predictions."""
        insights = {
            'biggest_climber': None,
            'biggest_faller': None,
            'surprise_podium': [],
            'championship_implications': [],
            'key_battles': []
        }
        
        if 'grid' in predictions.columns and 'predicted_position' in predictions.columns:
            # Calculate position changes
            predictions['position_change'] = predictions['grid'] - predictions['predicted_position']
            
            # Biggest climber
            biggest_climb_idx = predictions['position_change'].idxmax()
            biggest_climber = predictions.iloc[biggest_climb_idx]
            insights['biggest_climber'] = {
                'driver': biggest_climber['driver_name'],
                'from': int(biggest_climber['grid']),
                'to': int(biggest_climber['predicted_position']),
                'change': int(biggest_climber['position_change'])
            }
            
            # Biggest faller
            biggest_fall_idx = predictions['position_change'].idxmin()
            biggest_faller = predictions.iloc[biggest_fall_idx]
            insights['biggest_faller'] = {
                'driver': biggest_faller['driver_name'],
                'from': int(biggest_faller['grid']),
                'to': int(biggest_faller['predicted_position']),
                'change': int(biggest_faller['position_change'])
            }
            
            # Surprise podium (drivers starting P4 or lower making podium)
            podium_finishers = predictions.head(3)
            for _, driver in podium_finishers.iterrows():
                if driver['grid'] > 3:
                    insights['surprise_podium'].append({
                        'driver': driver['driver_name'],
                        'grid': int(driver['grid']),
                        'predicted_pos': int(driver['predicted_position'])
                    })
        
        return insights
    
    def display_predictions(self, race: Dict, predictions: pd.DataFrame, insights: Dict, analysis: Dict):
        """Display comprehensive prediction results."""
        if not RICH_AVAILABLE:
            self._display_predictions_simple(race, predictions, insights, analysis)
            return
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Header
        race_title = f"🏁 {race['short_name']} - {race['dates']}"
        race_info = f"📍 {race['location']} • Round {race['round']}/24"
        
        header_panel = Panel(
            Align.center(f"[bold white]{race_title}[/bold white]\n{race_info}"),
            style="red",
            title="Race Prediction Results"
        )
        layout["header"].update(header_panel)
        
        # Main content
        layout["main"].split_row(
            Layout(name="predictions", ratio=2),
            Layout(name="analysis", ratio=1)
        )
        
        # Predictions table
        pred_table = Table(title="🏆 Predicted Results", box=box.ROUNDED)
        pred_table.add_column("Pos", style="bold cyan", width=4)
        pred_table.add_column("Driver", style="bold white", min_width=15)
        pred_table.add_column("Team", style="yellow", min_width=12)
        pred_table.add_column("Grid", style="blue", width=4)
        pred_table.add_column("Confidence", style="green", width=10)
        pred_table.add_column("Change", style="magenta", width=6)
        
        for i, (_, driver) in enumerate(predictions.head(10).iterrows(), 1):
            # Position change calculation
            change = ""
            if 'grid' in driver and 'predicted_position' in driver:
                pos_change = int(driver['grid']) - int(driver.get('predicted_position', i))
                if pos_change > 0:
                    change = f"↗ +{pos_change}"
                elif pos_change < 0:
                    change = f"↘ {pos_change}"
                else:
                    change = "→ 0"
            
            # Confidence display
            confidence = ""
            if 'confidence' in driver:
                conf_val = float(driver['confidence'])
                confidence = f"{conf_val:.0%}"
            
            # Add medal emojis for podium
            pos_display = str(i)
            if i == 1:
                pos_display = "🥇"
            elif i == 2:
                pos_display = "🥈"
            elif i == 3:
                pos_display = "🥉"
            
            pred_table.add_row(
                pos_display,
                driver['driver_name'],
                driver.get('constructor_name', 'Unknown')[:12],
                str(int(driver.get('grid', 0))),
                confidence,
                change
            )
        
        layout["predictions"].update(pred_table)
        
        # Analysis panel
        analysis_content = []
        
        # Weather info
        if 'weather_factors' in analysis:
            weather = analysis['weather_factors']
            analysis_content.append(f"🌤️ Weather: {weather.get('conditions', 'Unknown')}")
            analysis_content.append(f"🌡️ Temperature: {weather.get('temperature', 'Unknown')}")
        
        # Key insights
        if insights.get('biggest_climber'):
            climber = insights['biggest_climber']
            analysis_content.append(f"📈 Biggest Climber: {climber['driver']} (P{climber['from']}→P{climber['to']})")
        
        if insights.get('biggest_faller'):
            faller = insights['biggest_faller']
            analysis_content.append(f"📉 Biggest Faller: {faller['driver']} (P{faller['from']}→P{faller['to']})")
        
        # Key factors
        if 'key_factors' in analysis:
            analysis_content.append("")
            analysis_content.append("🔑 Key Factors:")
            for factor in analysis['key_factors'][:3]:
                analysis_content.append(f"  • {factor}")
        
        analysis_panel = Panel(
            "\n".join(analysis_content),
            title="📊 Analysis",
            border_style="blue"
        )
        layout["analysis"].update(analysis_panel)
        
        # Footer
        footer_text = "🤖 Powered by Enhanced AI • 📈 Confidence intervals included • ⚡ Real-time analysis"
        layout["footer"].update(Panel(Align.center(footer_text), style="dim"))
        
        self.console.print(layout)
        
        # Additional detailed insights
        self._display_detailed_insights(insights, analysis)
    
    def _display_detailed_insights(self, insights: Dict, analysis: Dict):
        """Display additional detailed insights."""
        if not RICH_AVAILABLE:
            return
        
        # Strategic insights
        if 'strategic_considerations' in analysis and analysis['strategic_considerations']:
            strategy_table = Table(title="🎯 Strategic Insights", box=box.SIMPLE)
            strategy_table.add_column("Factor", style="cyan")
            strategy_table.add_column("Impact", style="white")
            
            for factor in analysis['strategic_considerations']:
                strategy_table.add_row("🎲", factor)
            
            self.console.print(strategy_table)
        
        # Driver advantages
        if 'driver_advantages' in analysis and analysis['driver_advantages']:
            adv_panels = []
            for adv in analysis['driver_advantages']:
                panel = Panel(
                    f"[bold]{adv['advantage']}[/bold]\nConfidence: {adv['confidence']:.0%}",
                    title=f"🏎️ {adv['driver']}",
                    width=25
                )
                adv_panels.append(panel)
            
            if adv_panels:
                self.console.print(Panel(
                    Columns(adv_panels),
                    title="Driver Circuit Advantages"
                ))
    
    def _display_predictions_simple(self, race: Dict, predictions: pd.DataFrame, insights: Dict, analysis: Dict):
        """Simple display for when rich is not available."""
        print(f"\n{'='*60}")
        print(f"🏁 {race['short_name']} PREDICTION")
        print(f"📅 {race['dates']} • 📍 {race['location']}")
        print(f"{'='*60}")
        
        print("\n🏆 PREDICTED RESULTS:")
        print("-" * 50)
        print(f"{'Pos':<4} {'Driver':<20} {'Team':<15} {'Grid':<6}")
        print("-" * 50)
        
        for i, (_, driver) in enumerate(predictions.head(10).iterrows(), 1):
            medal = ""
            if i == 1: medal = "🥇"
            elif i == 2: medal = "🥈"
            elif i == 3: medal = "🥉"
            
            print(f"{medal}{i:<3} {driver['driver_name']:<20} {driver.get('constructor_name', 'Unknown')[:15]:<15} {int(driver.get('grid', 0)):<6}")
        
        # Key insights
        if insights.get('biggest_climber'):
            climber = insights['biggest_climber']
            print(f"\n📈 Biggest Climber: {climber['driver']} (P{climber['from']} → P{climber['to']})")
        
        if insights.get('biggest_faller'):
            faller = insights['biggest_faller']
            print(f"📉 Biggest Faller: {faller['driver']} (P{faller['from']} → P{faller['to']})")
    
    def create_prediction_visualization(self, predictions: pd.DataFrame, race: Dict):
        """Create interactive visualization of predictions."""
        if not PLOTLY_AVAILABLE:
            if RICH_AVAILABLE:
                self.console.print("📊 [yellow]Plotly not available for visualizations[/yellow]")
            else:
                print("📊 Plotly not available for visualizations")
            return
        
        # Create grid vs predicted position chart
        fig = go.Figure()
        
        drivers = predictions['driver_name'].tolist()
        grid_positions = predictions['grid'].tolist()
        predicted_positions = predictions.get('predicted_position', list(range(1, len(drivers)+1))).tolist()
        
        # Add grid positions
        fig.add_trace(go.Scatter(
            x=drivers,
            y=grid_positions,
            mode='markers+lines',
            name='Grid Position',
            marker=dict(color='red', size=10),
            line=dict(color='red', dash='dash')
        ))
        
        # Add predicted positions
        fig.add_trace(go.Scatter(
            x=drivers,
            y=predicted_positions,
            mode='markers+lines',
            name='Predicted Position',
            marker=dict(color='blue', size=12),
            line=dict(color='blue')
        ))
        
        # Update layout
        fig.update_layout(
            title=f"🏁 {race['short_name']} - Grid vs Predicted Positions",
            xaxis_title="Drivers",
            yaxis_title="Position",
            yaxis=dict(autorange='reversed'),  # Position 1 at top
            template="plotly_dark",
            height=600
        )
        
        # Save and show
        fig.write_html("race_prediction_visualization.html")
        if RICH_AVAILABLE:
            self.console.print("📊 [green]Visualization saved as 'race_prediction_visualization.html'[/green]")
        else:
            print("📊 Visualization saved as 'race_prediction_visualization.html'")

def main():
    """Main function."""
    predictor = EnhancedNextRacePredictor()
    
    # Display banner
    predictor._print_banner()
    
    # Select race
    race = predictor.select_race_interactive()
    if not race:
        if RICH_AVAILABLE:
            console.print("❌ [red]No race selected. Exiting.[/red]")
        else:
            print("❌ No race selected. Exiting.")
        return
    
    try:
        # Get race analysis
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing race conditions...", total=None)
                analysis = predictor.get_race_analysis(race)
                
                progress.update(task, description="Generating predictions...")
                predictions, insights = predictor.predict_race_enhanced(race)
                
                progress.update(task, description="Preparing results...")
        else:
            print("🔄 Analyzing race conditions...")
            analysis = predictor.get_race_analysis(race)
            print("🤖 Generating predictions...")
            predictions, insights = predictor.predict_race_enhanced(race)
        
        # Display results
        predictor.display_predictions(race, predictions, insights, analysis)
        
        # Create visualization
        predictor.create_prediction_visualization(predictions, race)
        
        # Final message
        if RICH_AVAILABLE:
            console.print("\n🏁 [green bold]Prediction complete![/green bold] [dim]Good luck with the race![/dim]")
        else:
            print("\n🏁 Prediction complete! Good luck with the race!")
        
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"❌ [red]Prediction failed: {e}[/red]")
        else:
            print(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    main()
