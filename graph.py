import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_pollutants():
    print("Loading data...")
    try:
        df1 = pd.read_excel('Sensor1+24_mar_11_20.xlsx')
        df2 = pd.read_excel('sesnor2_24_mar_11_20.xlsx')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Function to process and plot for each sensor
    def process_and_plot(df, sensor_name, output_file):
        # Preprocessing
        if 'received_at' in df.columns:
            df['received_at'] = pd.to_datetime(df['received_at'], errors='coerce')
            df = df.sort_values('received_at')
            df.set_index('received_at', inplace=True)

        # Identify columns (replace pm2_5 with tvoc)
        tvoc_col = next((col for col in df.columns if 'tvoc' in col.lower()), None)
        co2_col = next((col for col in df.columns if 'co2' in col.lower()), None)

        if not tvoc_col or not co2_col:
            print(f"{sensor_name}: Could not find columns. TVOC: {tvoc_col}, CO2: {co2_col}")
            return

        # Resample to daily averages
        daily_df = df[[tvoc_col, co2_col]].resample('D').mean()

        # Setup plotting style
        sns.set_theme(style="whitegrid")

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # Plot TVOC
        sns.barplot(x=daily_df.index.strftime('%Y-%m-%d'), y=daily_df[tvoc_col], ax=ax1, color='skyblue')
        ax1.set_title(f'{sensor_name} - Daily Average TVOC Levels', fontsize=16, fontweight='bold')
        ax1.set_ylabel('TVOC (ppb)', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)

        # Plot CO2
        sns.barplot(x=daily_df.index.strftime('%Y-%m-%d'), y=daily_df[co2_col], ax=ax2, color='lightgreen')
        ax2.set_title(f'{sensor_name} - Daily Average CO2 Levels', fontsize=16, fontweight='bold')
        ax2.set_ylabel('CO2 (ppm)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"{sensor_name}: Graph saved successfully as '{output_file}'")
        plt.show()

    # Process each sensor separately
    process_and_plot(df1, "Sensor 1", "sensor1_pollutants.png")
    process_and_plot(df2, "Sensor 2", "sensor2_pollutants.png")

if __name__ == "__main__":
    plot_pollutants()
