from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Dict

console = Console()

AQI_COLORS = {
    'Good': 'green',
    'Moderate': 'yellow',
    'Unhealthy for Sensitive Groups': 'orange3',
    'Unhealthy': 'red',
    'Very Unhealthy': 'purple',
    'Hazardous': 'dark_red',
}

def print_header():
    header_text = Text()
    header_text.append("🌍 ", style="bold cyan")
    header_text.append("Air Quality Advisory System", style="bold cyan")
    header_text.append(" 🌍", style="bold cyan")
    console.print()
    console.print(Panel(
        header_text,
        border_style="cyan",
        padding=(1, 2)
    ))
    console.print()

def print_menu(options: list):
    console.print("[bold cyan]═══ Main Menu ═══[/bold cyan]\n")
    for i, option in enumerate(options, 1):
        console.print(f"  [cyan]{i}.[/cyan] {option}")
    console.print()
def print_predictions_table(enriched_data: Dict):
    table = Table(title="Current Air Quality Readings", show_header=True, header_style="bold cyan")
    table.add_column("Pollutant", style="cyan", width=15)
    table.add_column("Value", justify="right", width=12)
    table.add_column("Category", width=30)
    table.add_column("Status", justify="center", width=10)
    
    current = enriched_data.get('current', {})
    
    for pollutant, data in current.items():
        value = data.get('value', 0)
        category = data.get('category', 'Unknown')
        color = AQI_COLORS.get(category, 'white')
        
        if pollutant in ['temperature', 'humidity', 'pressure']:
            value_str = f"{value:.1f}"
        else:
            value_str = f"{value:.1f}"
        
        units = {
            'pm25': 'μg/m³',
            'pm10': 'μg/m³',
            'co2': 'ppm',
            'tvoc': 'ppb',
            'temperature': '°C',
            'humidity': '%',
            'pressure': 'hPa',
        }
        unit = units.get(pollutant, '')
        
        if category == 'Good':
            status = "✓"
            status_color = "green"
        elif category == 'Moderate':
            status = "⚠"
            status_color = "yellow"
        else:
            status = "⚠"
            status_color = "red"
        
        table.add_row(
            pollutant.upper(),
            f"{value_str} {unit}",
            f"[{color}]{category}[/{color}]",
            f"[{status_color}]{status}[/{status_color}]"
        )
    
    console.print(table)
    console.print()

def print_overall_aqi(enriched_data: Dict):
    overall = enriched_data.get('overall_aqi', {})
    category = overall.get('category', 'Unknown')
    primary = overall.get('primary_pollutant', 'N/A')
    color = AQI_COLORS.get(category, 'white')
    status_text = Text()
    status_text.append("Overall Air Quality: ", style="bold")
    status_text.append(category, style=f"bold {color}")
    status_text.append(f"\nPrimary Pollutant: {primary}", style="dim")
    
    console.print(Panel(
        status_text,
        border_style=color,
        padding=(1, 2),
        title="[bold]AQI Status[/bold]",
        title_align="left"
    ))
    console.print()

def print_trends(enriched_data: Dict):
    trends = enriched_data.get('trends', {})
    if not trends:
        return
    
    console.print("[bold cyan]Predicted Trends:[/bold cyan]")
    for pollutant, trend in trends.items():
        if 'improving' in trend:
            icon = "📉"
            color = "green"
        elif 'worsening' in trend:
            icon = "📈"
            color = "red"
        else:
            icon = "➡"
            color = "yellow"
        
        console.print(f"  {icon} [{color}]{pollutant.upper()}:[/{color}] {trend}")
    console.print()

def print_advice(advice_text: str, from_cache: bool = False):
    title = "💡 Personalized Health Advice"
    if from_cache:
        title += " (cached)"
    console.print(Panel(
        advice_text,
        border_style="green",
        padding=(1, 2),
        title=title,
        title_align="left"
    ))
    console.print()

def print_user_profile(profile: Dict):
    table = Table(title="User Profile", show_header=False, box=None)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    field_labels = {
        'profile_id': 'Profile ID',
        'age_group': 'Age Group',
        'activity_level': 'Activity Level',
        'outdoor_exposure': 'Outdoor Exposure',
        'sensitivity_level': 'Sensitivity Level',
    }
    
    for field, label in field_labels.items():
        if field in profile:
            table.add_row(label, str(profile[field]))
    
    conditions = []
    if profile.get('respiratory_issues'):
        conditions.append('Respiratory Issues')
    if profile.get('cardiovascular_issues'):
        conditions.append('Cardiovascular Issues')
    if profile.get('allergies'):
        conditions.append('Allergies')
    if profile.get('pregnancy'):
        conditions.append('Pregnancy')
    
    if conditions:
        table.add_row('Health Conditions', ', '.join(conditions))
    else:
        table.add_row('Health Conditions', 'None reported')
    
    console.print(table)
    console.print()

def print_error(message: str):
    from rich.markup import escape
    safe_message = escape(str(message))
    console.print(f"[bold red]✗ Error:[/bold red] {safe_message}")
    console.print()
def print_success(message: str):
    from rich.markup import escape
    safe_message = escape(str(message))
    console.print(f"[bold green]✓ Success:[/bold green] {safe_message}")
    console.print()
def print_warning(message: str):
    from rich.markup import escape
    safe_message = escape(str(message))
    console.print(f"[bold yellow]⚠ Warning:[/bold yellow] {safe_message}")
    console.print()
def print_info(message: str):
    from rich.markup import escape
    safe_message = escape(str(message))
    console.print(f"[cyan]ℹ[/cyan] {safe_message}")
    console.print()
def get_spinner_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )
def clear_screen():
    console.clear()
def print_divider():
    console.print("─" * console.width, style="dim")