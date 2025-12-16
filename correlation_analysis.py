import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import sys


def load_sensor_data():
    """Load data from both sensors."""
    print("Loading sensor data...")
    try:
        df1 = pd.read_excel('Sensor1+24_mar_11_20.xlsx')
        df2 = pd.read_excel('sesnor2_24_mar_11_20.xlsx')
        print("✓ Data loaded successfully!\n")
        return df1, df2
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def preprocess_data(df, sensor_name):
    """Preprocess sensor data: parse timestamps and sort."""
    if 'received_at' in df.columns:
        df['received_at'] = pd.to_datetime(df['received_at'], errors='coerce')
        df = df.sort_values('received_at')
        df.set_index('received_at', inplace=True)
    
    print(f"{sensor_name} columns: {list(df.columns)}")
    return df


def find_pollutant_column(df, pollutant_name):
    """Find the column name for a given pollutant (case-insensitive)."""
    pollutant_lower = pollutant_name.lower()
    
    # Try exact match first
    for col in df.columns:
        if pollutant_lower == col.lower():
            return col
    
    # Try partial match
    for col in df.columns:
        if pollutant_lower in col.lower():
            return col
    
    return None


def calculate_correlation(df1, df2, pollutant):
    """Calculate correlation between sensor 1 and sensor 2 for a specific pollutant."""
    
    # Find the pollutant column in both dataframes
    col1 = find_pollutant_column(df1, pollutant)
    col2 = find_pollutant_column(df2, pollutant)
    
    if not col1:
        print(f"Error: Pollutant '{pollutant}' not found in Sensor 1 data.")
        print(f"Available columns in Sensor 1: {list(df1.columns)}")
        return None
    
    if not col2:
        print(f"Error: Pollutant '{pollutant}' not found in Sensor 2 data.")
        print(f"Available columns in Sensor 2: {list(df2.columns)}")
        return None
    
    print(f"\nAnalyzing correlation for: {pollutant}")
    print(f"  Sensor 1 column: {col1}")
    print(f"  Sensor 2 column: {col2}")
    
    # Extract the data and clean
    sensor1_data = df1[col1].dropna().reset_index(drop=True)
    sensor2_data = df2[col2].dropna().reset_index(drop=True)
    
    print(f"  Sensor 1 data points: {len(sensor1_data)}")
    print(f"  Sensor 2 data points: {len(sensor2_data)}")
    
    # Use the minimum length to align data sequentially
    min_len = min(len(sensor1_data), len(sensor2_data))
    
    if min_len == 0:
        print("Error: One or both sensors have no valid data.")
        return None
    
    # Align by taking the first min_len points from each sensor
    aligned_df = pd.DataFrame({
        'Sensor1': sensor1_data[:min_len].values,
        'Sensor2': sensor2_data[:min_len].values
    })
    
    print(f"  Number of aligned data points: {len(aligned_df)}")
    
    # Calculate correlation
    pearson_corr, pearson_pval = stats.pearsonr(aligned_df['Sensor1'], aligned_df['Sensor2'])
    spearman_corr, spearman_pval = stats.spearmanr(aligned_df['Sensor1'], aligned_df['Sensor2'])
    
    # Calculate statistics
    results = {
        'pollutant': pollutant,
        'col1': col1,
        'col2': col2,
        'n_points': len(aligned_df),
        'pearson_r': pearson_corr,
        'pearson_pval': pearson_pval,
        'spearman_r': spearman_corr,
        'spearman_pval': spearman_pval,
        'sensor1_mean': aligned_df['Sensor1'].mean(),
        'sensor1_std': aligned_df['Sensor1'].std(),
        'sensor2_mean': aligned_df['Sensor2'].mean(),
        'sensor2_std': aligned_df['Sensor2'].std(),
        'data': aligned_df
    }
    
    return results


def print_correlation_results(results):
    """Print correlation analysis results in a formatted way."""
    if not results:
        return
    
    print("\n" + "="*70)
    print(f"CORRELATION COEFFICIENT: {results['pollutant'].upper()}")
    print("="*70)
    
    # Show correlation prominently first
    print(f"\n🔗 CORRELATION (Sensor 1 vs Sensor 2): r = {results['pearson_r']:.4f}")
    
    # Interpret correlation strength
    abs_corr = abs(results['pearson_r'])
    if abs_corr >= 0.9:
        strength = "Very Strong"
    elif abs_corr >= 0.7:
        strength = "Strong"
    elif abs_corr >= 0.5:
        strength = "Moderate"
    elif abs_corr >= 0.3:
        strength = "Weak"
    else:
        strength = "Very Weak"
    
    direction = "Positive" if results['pearson_r'] > 0 else "Negative"
    
    print(f"   → Strength: {strength} {direction} Correlation")
    
    if results['pearson_pval'] < 0.05:
        print(f"   → Statistically Significant: YES (p < 0.05)")
    else:
        print(f"   → Statistically Significant: NO (p >= 0.05)")
    
    print(f"\n📊 Data Points Analyzed: {results['n_points']}")
    
    print("\n📈 Sensor Statistics:")
    print(f"   Sensor 1: Mean = {results['sensor1_mean']:.2f}, Std Dev = {results['sensor1_std']:.2f}")
    print(f"   Sensor 2: Mean = {results['sensor2_mean']:.2f}, Std Dev = {results['sensor2_std']:.2f}")
    
    print("\n📐 Additional Correlation Metrics:")
    print(f"   Pearson:  r = {results['pearson_r']:.4f}  (p = {results['pearson_pval']:.4e})")
    print(f"   Spearman: ρ = {results['spearman_r']:.4f}  (p = {results['spearman_pval']:.4e})")
    
    print("="*70 + "\n")


def plot_correlation(results, save_file=None):
    """Create visualization of the correlation between sensors."""
    if not results:
        return
    
    data = results['data']
    pollutant = results['pollutant']
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Scatter plot with regression line
    ax1 = plt.subplot(2, 2, 1)
    sns.regplot(x='Sensor1', y='Sensor2', data=data, ax=ax1, 
                scatter_kws={'alpha': 0.5, 's': 30}, 
                line_kws={'color': 'red', 'linewidth': 2})
    ax1.set_xlabel(f'Sensor 1 - {pollutant}', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'Sensor 2 - {pollutant}', fontsize=12, fontweight='bold')
    ax1.set_title(f'Scatter Plot with Regression Line\nr = {results["pearson_r"]:.4f}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Time series comparison
    ax2 = plt.subplot(2, 2, 2)
    sample_numbers = range(len(data))
    ax2.plot(sample_numbers, data['Sensor1'], label='Sensor 1', alpha=0.7, linewidth=1.5)
    ax2.plot(sample_numbers, data['Sensor2'], label='Sensor 2', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Sample Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'{pollutant}', fontsize=12, fontweight='bold')
    ax2.set_title('Sequential Data Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual plot
    ax3 = plt.subplot(2, 2, 3)
    residuals = data['Sensor2'] - data['Sensor1']
    ax3.scatter(data['Sensor1'], residuals, alpha=0.5, s=30)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel(f'Sensor 1 - {pollutant}', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Residuals (Sensor2 - Sensor1)', fontsize=12, fontweight='bold')
    ax3.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution comparison
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(data['Sensor1'], bins=30, alpha=0.5, label='Sensor 1', color='blue', edgecolor='black')
    ax4.hist(data['Sensor2'], bins=30, alpha=0.5, label='Sensor 2', color='orange', edgecolor='black')
    ax4.set_xlabel(f'{pollutant}', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle(f'Correlation Analysis: {pollutant.upper()} (Sensor 1 vs Sensor 2)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved as '{save_file}'")
    
    plt.show()


def main():
    """Main function to run correlation analysis."""
    print("\n" + "="*70)
    print("AIR POLLUTANT CORRELATION ANALYSIS")
    print("Sensor 1 vs Sensor 2")
    print("="*70 + "\n")
    
    # Load data
    df1, df2 = load_sensor_data()
    
    # Preprocess
    df1 = preprocess_data(df1, "Sensor 1")
    df2 = preprocess_data(df2, "Sensor 2")
    
    print("\n" + "-"*70)
    
    # Get user input for pollutant
    print("\nCommon pollutants: pm25, pm10, co2, tvoc, temperature, humidity, pressure")
    pollutant = input("\nEnter the air pollutant to analyze: ").strip()
    
    if not pollutant:
        print("Error: No pollutant specified.")
        sys.exit(1)
    
    # Calculate correlation
    results = calculate_correlation(df1, df2, pollutant)
    
    if results:
        # Print results
        print_correlation_results(results)
        
        # Plot visualization
        save_file = f"correlation_{pollutant.lower()}_sensor1_vs_sensor2.png"
        plot_correlation(results, save_file)
        
        print(f"\n✓ Analysis complete!")
    else:
        print("\n✗ Analysis failed. Please check the pollutant name and try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
