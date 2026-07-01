import sys
from pathlib import Path
import pandas as pd

from nowcast import (
    Nowcaster, 
    detect_flares_rolling_sigma, 
    detect_flares_derivative,
    load_and_preprocess_solexs
)
from data_pipeline import load_solexs_fits_table

def test_inference():
    # Paths to saved metadata
    metadata_path = Path("nowcast_metadata.json")
    
    print("Loading Nowcaster...")
    # Initialize Nowcaster (model_path is ignored by the statistical implementation)
    nowcaster = Nowcaster(metadata_path=metadata_path)
    
    # Path to test data directory
    data_dir = Path("Data/solexs_2026Jun29T175402251/AL1_SLX_L1_20240507_v1.0")
    if not data_dir.exists():
        print(f"Error: Test data directory not found at {data_dir}")
        sys.exit(1)
        
    print(f"Loading test data from {data_dir}...")
    df = load_and_preprocess_solexs(data_dir)
    print(f"Total rows loaded: {len(df)}")
    
    # Find a flare region to test
    mean_counts = df['COUNTS'].mean()
    std_counts = df['COUNTS'].std()
    threshold = mean_counts + 3.0 * std_counts
    
    flare_indices = df[df['COUNTS'] > threshold].index
    
    # Quiet window
    quiet_idx = 100
    quiet_df = df.iloc[quiet_idx : quiet_idx + 60]
    
    print("\n--- Testing Nowcaster Predict on Quiet Window ---")
    print(f"Counts range in window: {quiet_df['COUNTS'].min():.2f} - {quiet_df['COUNTS'].max():.2f} counts/s")
    prob_quiet = nowcaster.predict(quiet_df)
    print(f"Predicted Flare Probability/Severity: {prob_quiet * 100.0:.4f}%")
    
    # Flare window
    if len(flare_indices) > 0:
        flare_idx = max(0, flare_indices[0] - 59)
        flare_df = df.iloc[flare_idx : flare_idx + 60]
        
        print("\n--- Testing Nowcaster Predict on Flare Onset Window ---")
        print(f"Counts range in window: {flare_df['COUNTS'].min():.2f} - {flare_df['COUNTS'].max():.2f} counts/s")
        prob_flare = nowcaster.predict(flare_df)
        print(f"Predicted Flare Probability/Severity: {prob_flare * 100.0:.4f}%")
    else:
        print("\nNo flare timestamps found in the dataset using the threshold.")
        
    # --- Test Statistical Change-Detection Algorithms ---
    print("\n====================================================")
    print("Testing Statistical Algorithms on Entire Day of Data")
    print("====================================================")
    
    print("\nRunning Adaptive Rolling Sigma Nowcaster (k=3.5)...")
    sigma_flares = detect_flares_rolling_sigma(df, k=3.5, bg_window_size=600, min_duration_sec=5)
    print(f"Detected {len(sigma_flares)} events:")
    if not sigma_flares.empty:
        for idx, row in sigma_flares.iterrows():
            print(f"  Event {idx+1}:")
            print(f"    Start time : {row['start_time']}")
            print(f"    Peak time  : {row['peak_time']} (Peak Rate: {row['peak_rate']:.2f} cts/s)")
            print(f"    End time   : {row['end_time']}")
            print(f"    Total counts: {row['total_counts']:.2f}")
            
    print("\nRunning Rate-of-Rise Derivative Nowcaster (rise_threshold=10.0)...")
    deriv_flares = detect_flares_derivative(df, rise_threshold=10.0, min_duration_sec=3)
    print(f"Detected {len(deriv_flares)} events:")
    if not deriv_flares.empty:
        for idx, row in deriv_flares.iterrows():
            print(f"  Event {idx+1}:")
            print(f"    Start time : {row['start_time']}")
            print(f"    Peak time  : {row['peak_time']} (Peak Rate: {row['peak_rate']:.2f} cts/s)")
            print(f"    End time   : {row['end_time']}")
            print(f"    Total counts: {row['total_counts']:.2f}")

if __name__ == "__main__":
    test_inference()
