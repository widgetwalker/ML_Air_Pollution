import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pollutant(df, sensor_name, pollutant):
    # Preprocessing
    if 'received_at' in df.columns:
        df['received_at'] = pd.to_datetime(df['received_at'], errors='coerce')
        df = df.sort_values('received_at')
        df.set_index('received_at', inplace=True)

    # Find the column dynamically
    pollutant_col = next((col for col in df.columns if pollutant.lower() in col.lower()), None)
    if not pollutant_col:
        print(f"{sensor_name}: Could not find column for {pollutant}")
        return

    # Resample to daily averages
    daily_df = df[[pollutant_col]].resample('D').mean()

    # Calculate summary statistics
    min_val = daily_df[pollutant_col].min()
    max_val = daily_df[pollutant_col].max()
    mean_val = daily_df[pollutant_col].mean()
    std_val = daily_df[pollutant_col].std()

    # Setup plotting style
    sns.set_theme(style="whitegrid")

    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=daily_df.index.strftime('%Y-%m-%d'), y=daily_df[pollutant_col], color='skyblue')
    plt.title(f'{sensor_name} - {pollutant.upper()} Levels', fontsize=16, fontweight='bold')
    plt.ylabel(f'{pollutant.upper()}', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add statistics text box on the graph
    stats_text = (f"Min: {min_val:.2f}\n"
                  f"Max: {max_val:.2f}\n"
                  f"Mean: {mean_val:.2f}\n"
                  f"Std Dev: {std_val:.2f}")
    plt.gcf().text(0.85, 0.5, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Save and show
    output_file = f"{sensor_name}_{pollutant}_graph.png"
    plt.savefig(output_file, dpi=300)
    print(f"{sensor_name}: Graph saved successfully as '{output_file}'")
    plt.show()

def main():
    print("Loading data...")
    try:
        df1 = pd.read_excel('Sensor1+24_mar_11_20.xlsx')
        df2 = pd.read_excel('sesnor2_24_mar_11_20.xlsx')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Ask user which pollutant to plot
    choice = input("Which graph do you want to generate? (co2, pm2_5, pm10, tvoc, humidity, temperature, pressure, light): ").strip().lower()

    # Switch-case style (Python 3.10+)
    match choice:
        case "co2" | "pm2_5" | "pm10" | "tvoc" | "humidity" | "temperature" | "pressure" | "light":
            plot_pollutant(df1, "Sensor1", choice)
            plot_pollutant(df2, "Sensor2", choice)
        case _:
            print("Invalid choice. Please select from the given options.")

if __name__ == "__main__":
    main()
