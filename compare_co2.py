import pandas as pd

def load_sensor_data(filepath: str) -> pd.DataFrame:
    """Load sensor data from an Excel file.
    Returns a DataFrame with the data or raises an informative error.
    """
    try:
        df = pd.read_excel(filepath)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")

def compute_co2_stats(df: pd.DataFrame) -> float:
    """Return the mean CO₂ value from the DataFrame.
    If the CO₂ column is missing, raise a ValueError.
    """
    # Accept common variations of the column name
    possible_names = [col for col in df.columns if col.lower() == "co2" or "co2" in col.lower()]
    if not possible_names:
        raise ValueError("CO2 column not found in the dataset.")
    # Use the first matching column
    co2_col = possible_names[0]
    return df[co2_col].mean()

def main():
    # Filenames as used in the repository
    sensor1_file = "Sensor1+24_mar_11_20.xlsx"
    sensor2_file = "sesnor2_24_mar_11_20.xlsx"

    df1 = load_sensor_data(sensor1_file)
    df2 = load_sensor_data(sensor2_file)

    try:
        avg_co2_s1 = compute_co2_stats(df1)
        avg_co2_s2 = compute_co2_stats(df2)
    except ValueError as ve:
        print(f"Error: {ve}")
        return

    diff = avg_co2_s2 - avg_co2_s1
    print("CO₂ Comparison (Sensor 1 = outside, Sensor 2 = inside)")
    print(f"  Average CO₂ (outside): {avg_co2_s1:.2f}")
    print(f"  Average CO₂ (inside):  {avg_co2_s2:.2f}")
    print(f"  Difference (inside - outside): {diff:.2f}")

if __name__ == "__main__":
    main()
