import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.prediction_service import PredictionService
from src.context_enrichment import ContextEnrichment
from src.llm_service import LLMService
from src.user_profile import UserProfile
from src.display_utils import (
    console, print_header, print_predictions_table, print_overall_aqi,
    print_trends, print_advice, print_user_profile, print_success,
    print_info, print_divider, get_spinner_progress
)

def demo():
    print_header()
    console.print("[bold cyan]🎯 Air Quality Advisory System Demo[/bold cyan]\n")
    console.print("This demo will showcase the system's capabilities:\n")
    console.print("  1. Load trained ML models")
    console.print("  2. Generate air quality predictions")
    console.print("  3. Provide personalized advice for different user profiles")
    console.print("  4. Demonstrate the interactive features\n")
    
    print_divider()
    console.input("\nPress Enter to start the demo...")
    
    console.print("\n[bold]Step 1: Initializing Services[/bold]\n")
    
    with get_spinner_progress() as progress:
        task = progress.add_task("[cyan]Loading ML models and services...", total=None)
        
        prediction_service = PredictionService()
        context_enricher = ContextEnrichment()
        llm_service = LLMService()
        user_manager = UserProfile()
        user_manager.load_default_profiles()
        
        progress.update(task, completed=True)
    
    print_success("All services initialized successfully!")
    console.input("\nPress Enter to continue...")
    
    scenarios = [
        {
            'name': 'Moderate Air Quality - Normal Adult',
            'sensor_data': {
                'pm25': 65.0,
                'pm10': 95.0,
                'co2': 850.0,
                'tvoc': 180.0,
                'temperature': 24.5,
                'humidity': 55.0,
                'pressure': 1013.0,
            },
            'profile_id': 'normal_adult',
        },
        {
            'name': 'Unhealthy Air Quality - Child with Asthma',
            'sensor_data': {
                'pm25': 165.0,
                'pm10': 220.0,
                'co2': 1200.0,
                'tvoc': 450.0,
                'temperature': 28.0,
                'humidity': 65.0,
                'pressure': 1010.0,
            },
            'profile_id': 'child_with_asthma',
        },
        {
            'name': 'Good Air Quality - Athlete',
            'sensor_data': {
                'pm25': 25.0,
                'pm10': 40.0,
                'co2': 600.0,
                'tvoc': 100.0,
                'temperature': 22.0,
                'humidity': 50.0,
                'pressure': 1015.0,
            },
            'profile_id': 'athlete',
        },
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        console.print(f"\n[bold]Step {i+1}: {scenario['name']}[/bold]\n")
        
        profile = user_manager.get_profile(scenario['profile_id'])
        console.print("[cyan]User Profile:[/cyan]")
        print_user_profile(profile)
        
        console.input("Press Enter to generate prediction...")
        
        with get_spinner_progress() as progress:
            task = progress.add_task("[cyan]Generating predictions...", total=None)
            
            predictions = prediction_service.predict(scenario['sensor_data'])
            enriched_data = context_enricher.enrich_predictions(
                scenario['sensor_data'],
                predictions.get('predicted', {}),
                profile
            )
            
            progress.update(task, completed=True)
        
        console.print()
        print_overall_aqi(enriched_data)
        print_predictions_table(enriched_data)
        print_trends(enriched_data)
        
        console.input("Press Enter to get personalized advice...")
        
        with get_spinner_progress() as progress:
            task = progress.add_task("[cyan]Generating AI advice...", total=None)
            
            advice, from_cache = llm_service.generate_advice(enriched_data)
            
            progress.update(task, completed=True)
        
        console.print()
        print_advice(advice, from_cache)
        
        if i < len(scenarios):
            print_divider()
            console.input("\nPress Enter for next scenario...")
    
    console.print("\n[bold green]✓ Demo Complete![/bold green]\n")
    console.print("The system demonstrated:")
    console.print("  ✓ Loading 7 trained ML models")
    console.print("  ✓ Generating air quality predictions")
    console.print("  ✓ Calculating AQI categories and trends")
    console.print("  ✓ Providing personalized advice based on user profiles")
    console.print("  ✓ Adapting recommendations to different air quality levels\n")
    
    console.print("[cyan]To use the full interactive system, run:[/cyan]")
    console.print("  [bold]python main.py[/bold]\n")

if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        console.print("\n\nDemo interrupted.")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
