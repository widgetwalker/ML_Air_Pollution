from typing import Optional
import sys

from .prediction_service import PredictionService
from .context_enrichment import ContextEnrichment
from .llm_service import LLMService
from .user_profile import UserProfile
from .database import Database
from .display_utils import (
    console, print_header, print_menu, print_predictions_table,
    print_overall_aqi, print_trends, print_advice, print_user_profile,
    print_error, print_success, print_warning, print_info,
    get_spinner_progress, clear_screen, print_divider
)
from .config import settings

class AirQualityCLI:
    def __init__(self):
        self.prediction_service = None
        self.context_enricher = ContextEnrichment()
        self.llm_service = None
        self.user_profile_manager = UserProfile()
        self.database = Database()
        self.current_user_profile = None
        self.current_enriched_data = None
        self._initialize_services()
    
    def _initialize_services(self):
        with get_spinner_progress() as progress:
            task = progress.add_task("[cyan]Initializing services...", total=None)
            try:
                self.prediction_service = PredictionService()
                
                self.llm_service = LLMService()
                
                self.user_profile_manager.load_default_profiles()
                
                self.current_user_profile = self.user_profile_manager.get_profile(
                    settings.default_user_profile
                )
                
                progress.update(task, completed=True)
            
            except Exception as e:
                print_error(f"Failed to initialize services: {e}")
                sys.exit(1)
    
    def run(self):
        clear_screen()
        print_header()
        if not settings.validate_api_key():
            print_warning(
                f"LLM API key not configured for {settings.llm_provider}. "
                "You'll receive rule-based advice instead of AI-generated advice."
            )
            print_info("To configure, edit the .env file with your API key.")
            console.input("\nPress Enter to continue...")
        
        while True:
            clear_screen()
            print_header()
            
            if self.current_user_profile:
                console.print(f"[dim]Current Profile: {self.current_user_profile['profile_id']}[/dim]\n")
            
            menu_options = [
                "Get Current Air Quality Prediction",
                "Get Personalized Health Advice",
                "Interactive Chat with AI Advisor",
                "View Prediction History",
                "Manage User Profile",
                "System Settings",
                "Exit"
            ]
            
            print_menu(menu_options)
            
            choice = console.input("[cyan]Select an option (1-7):[/cyan] ").strip()
            
            if choice == "1":
                self._get_prediction()
            elif choice == "2":
                self._get_health_advice()
            elif choice == "3":
                self._interactive_chat()
            elif choice == "4":
                self._view_history()
            elif choice == "5":
                self._manage_profile()
            elif choice == "6":
                self._system_settings()
            elif choice == "7":
                self._exit()
                break
            else:
                print_error("Invalid option. Please select 1-7.")
                console.input("\nPress Enter to continue...")
    
    def _get_prediction(self):
        clear_screen()
        console.print("[bold cyan]═══ Air Quality Prediction ═══[/bold cyan]\n")
        console.print("Enter current sensor readings (or press Enter for sample data):\n")
        
        try:
            pm25 = console.input("PM2.5 (μg/m³) [default: 35.0]: ").strip()
            pm25 = float(pm25) if pm25 else 35.0
            
            pm10 = console.input("PM10 (μg/m³) [default: 68.0]: ").strip()
            pm10 = float(pm10) if pm10 else 68.0
            
            co2 = console.input("CO2 (ppm) [default: 850]: ").strip()
            co2 = float(co2) if co2 else 850.0
            
            tvoc = console.input("TVOC (ppb) [default: 180]: ").strip()
            tvoc = float(tvoc) if tvoc else 180.0
            
            temp = console.input("Temperature (°C) [default: 24.5]: ").strip()
            temp = float(temp) if temp else 24.5
            
            humidity = console.input("Humidity (%) [default: 55.0]: ").strip()
            humidity = float(humidity) if humidity else 55.0
            
            pressure = console.input("Pressure (hPa) [default: 1013.0]: ").strip()
            pressure = float(pressure) if pressure else 1013.0
            
            console.print()
            
            with get_spinner_progress() as progress:
                task = progress.add_task("[cyan]Generating predictions...", total=None)
                
                sensor_data = {
                    'pm25': pm25,
                    'pm10': pm10,
                    'co2': co2,
                    'tvoc': tvoc,
                    'temperature': temp,
                    'humidity': humidity,
                    'pressure': pressure,
                }
                
                predictions = self.prediction_service.predict(sensor_data)
                
                self.current_enriched_data = self.context_enricher.enrich_predictions(
                    sensor_data,
                    predictions.get('predicted', {}),
                    self.current_user_profile
                )
                
                progress.update(task, completed=True)
            
            console.print()
            print_overall_aqi(self.current_enriched_data)
            print_predictions_table(self.current_enriched_data)
            print_trends(self.current_enriched_data)
            
            self.database.save_prediction(self.current_enriched_data)
            
        except ValueError as e:
            print_error(f"Invalid input: {e}")
        except Exception as e:
            print_error(f"Error generating prediction: {e}")
        
        console.input("\n[dim]Press Enter to continue...[/dim]")
    
    def _get_health_advice(self):
        clear_screen()
        console.print("[bold cyan]═══ Personalized Health Advice ═══[/bold cyan]\n")
        if not self.current_enriched_data:
            print_warning("No prediction data available. Please get a prediction first.")
            console.input("\n[dim]Press Enter to continue...[/dim]")
            return
        
        console.print("Are you planning a specific activity? (optional)")
        activity = console.input("Activity (or press Enter to skip): ").strip()
        activity = activity if activity else None
        
        console.print()
        
        with get_spinner_progress() as progress:
            task = progress.add_task("[cyan]Generating personalized advice...", total=None)
            
            advice, from_cache = self.llm_service.generate_advice(
                self.current_enriched_data,
                activity=activity
            )
            
            progress.update(task, completed=True)
        
        console.print()
        print_advice(advice, from_cache)
        
        console.input("\n[dim]Press Enter to continue...[/dim]")
    
    def _interactive_chat(self):
        clear_screen()
        console.print("[bold cyan]═══ Interactive AI Advisor ═══[/bold cyan]\n")
        if not self.current_enriched_data:
            print_warning("No prediction data available. Please get a prediction first.")
            console.input("\n[dim]Press Enter to continue...[/dim]")
            return
        
        
        console.print("[dim]Type your questions about air quality and health.\\nType 'exit' or 'quit' to return to main menu.[/dim]\\n")
        
        print_divider()
        
        while True:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                break
            
            if not question:
                continue
            
            with get_spinner_progress() as progress:
                task = progress.add_task("[cyan]Thinking...", total=None)
                
                advice, from_cache = self.llm_service.generate_advice(
                    self.current_enriched_data,
                    custom_question=question
                )
                
                progress.update(task, completed=True)
            
            from rich.markup import escape
            safe_advice = escape(advice)
            console.print(f"\n[bold green]AI Advisor:[/bold green] {safe_advice}\n")
            print_divider()
            
            self.database.save_conversation(
                self.current_user_profile.get('profile_id', 'unknown'),
                question,
                advice,
                self.current_enriched_data
            )
    
    def _view_history(self):
        clear_screen()
        console.print("[bold cyan]═══ Prediction History ═══[/bold cyan]\n")
        predictions = self.database.get_recent_predictions(limit=10)
        
        if not predictions:
            print_info("No prediction history available.")
        else:
            from rich.table import Table
            
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Date/Time", width=20)
            table.add_column("AQI Category", width=25)
            table.add_column("Primary Pollutant", width=15)
            table.add_column("PM2.5", justify="right", width=10)
            
            for pred in predictions:
                from datetime import datetime
                timestamp = pred.get('created_at', '')
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    time_str = timestamp[:16]
                
                category = pred.get('aqi_category', 'Unknown')
                from .display_utils import AQI_COLORS
                color = AQI_COLORS.get(category, 'white')
                
                table.add_row(
                    time_str,
                    f"[{color}]{category}[/{color}]",
                    pred.get('primary_pollutant', 'N/A'),
                    f"{pred.get('pm25', 0):.1f}"
                )
            
            console.print(table)
        
        console.input("\n[dim]Press Enter to continue...[/dim]")
    
    def _manage_profile(self):
        clear_screen()
        console.print("[bold cyan]═══ User Profile Management ═══[/bold cyan]\n")
        profile_menu = [
            "View Current Profile",
            "Switch Profile",
            "Create New Profile",
            "List All Profiles",
            "Back to Main Menu"
        ]
        
        print_menu(profile_menu)
        
        choice = console.input("[cyan]Select an option (1-5):[/cyan] ").strip()
        
        if choice == "1":
            console.print()
            if self.current_user_profile:
                print_user_profile(self.current_user_profile)
            else:
                print_warning("No profile selected.")
        
        elif choice == "2":
            self._switch_profile()
        
        elif choice == "3":
            self._create_profile()
        
        elif choice == "4":
            self._list_profiles()
        
        console.input("\n[dim]Press Enter to continue...[/dim]")
    
    def _switch_profile(self):
        console.print("\n[bold]Available Profiles:[/bold]\n")
        profiles = self.user_profile_manager.list_profiles()
        for i, profile in enumerate(profiles, 1):
            console.print(f"  {i}. {profile['profile_id']}")
        
        console.print()
        choice = console.input("Select profile number: ").strip()
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(profiles):
                self.current_user_profile = profiles[idx]
                print_success(f"Switched to profile: {self.current_user_profile['profile_id']}")
            else:
                print_error("Invalid profile number.")
        except ValueError:
            print_error("Invalid input.")
    
    def _create_profile(self):
        console.print("\n[bold]Create New Profile[/bold]\n")
        profile_id = console.input("Profile ID: ").strip()
        if not profile_id:
            print_error("Profile ID cannot be empty.")
            return
        
        age_group = console.input("Age Group (child/adult/senior) [adult]: ").strip() or "adult"
        
        resp_issues = console.input("Respiratory issues? (y/n) [n]: ").strip().lower() == 'y'
        cardio_issues = console.input("Cardiovascular issues? (y/n) [n]: ").strip().lower() == 'y'
        allergies = console.input("Allergies? (y/n) [n]: ").strip().lower() == 'y'
        
        activity = console.input("Activity level (sedentary/moderate/active) [moderate]: ").strip() or "moderate"
        
        try:
            profile = self.user_profile_manager.create_profile(
                profile_id=profile_id,
                age_group=age_group,
                respiratory_issues=resp_issues,
                cardiovascular_issues=cardio_issues,
                allergies=allergies,
                activity_level=activity
            )
            print_success(f"Profile '{profile_id}' created successfully!")
            self.current_user_profile = profile
        except Exception as e:
            print_error(f"Error creating profile: {e}")
    
    def _list_profiles(self):
        console.print("\n[bold]All Profiles:[/bold]\n")
        profiles = self.user_profile_manager.list_profiles()
        for profile in profiles:
            console.print(f"  • {profile['profile_id']} ({profile.get('age_group', 'N/A')})")
    
    def _system_settings(self):
        clear_screen()
        console.print("[bold cyan]═══ System Settings ═══[/bold cyan]\n")
        from rich.table import Table
        
        table = Table(show_header=False, box=None)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("LLM Provider", settings.llm_provider)
        table.add_row("LLM Model", settings.llm_model)
        table.add_row("API Key Configured", "✓" if settings.validate_api_key() else "✗")
        table.add_row("Cache Enabled", "✓" if settings.enable_cache else "✗")
        table.add_row("Models Directory", str(settings.models_dir))
        table.add_row("Database Path", str(settings.database_path))
        
        model_info = self.prediction_service.get_model_info()
        table.add_row("Loaded Models", f"{model_info['total_models']}/7")
        
        cache_stats = self.llm_service.get_cache_stats()
        table.add_row("Cache Size", str(cache_stats['cache_size']))
        
        console.print(table)
        
        console.input("\n[dim]Press Enter to continue...[/dim]")
    
    def _exit(self):
        clear_screen()
        console.print("\n[bold cyan]Thank you for using the Air Quality Advisory System![/bold cyan]")
        console.print("[dim]Stay safe and breathe easy! 🌍[/dim]\n")
        self.database.close()
