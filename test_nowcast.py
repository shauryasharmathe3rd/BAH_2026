import sys
from pathlib import Path
import pandas as pd

from nowcast import Nowcaster
from data_pipeline import load_solexs_fits_table

def test_inference():
    # Paths to saved model and metadata
    model_path = Path("nowcast_model.pt")
    metadata_path = Path("nowcast_metadata.json")
    
    if not model_path.exists() or not metadata_path.exists():
        print("Error: Model weights or metadata files do not exist yet. Please wait for training to complete at least one epoch.")
        sys.exit(1)
        
    print("Loading Nowcaster...")
    nowcaster = Nowcaster(model_path, metadata_path)
    
    # Path to test data
    data_file = Path(r"C:\Users\USER\Documents\AL1_SLX_L1_20240507_v1.0\AL1_SLX_L1_20240507_v1.0\SDD2\AL1_SOLEXS_20240507_SDD2_L1.lc.gz")
    if not data_file.exists():
        print(f"Error: Test data file not found at {data_file}")
        sys.exit(1)
        
    print(f"Loading test data from {data_file}...")
    df = load_solexs_fits_table(data_file)
    print(f"Total rows loaded: {len(df)}")
    
    # Clean nulls
    df = df.dropna(subset=["TIME", "COUNTS"]).reset_index(drop=True)
    
    # Find a flare region to test
    mean_counts = df['COUNTS'].mean()
    std_counts = df['COUNTS'].std()
    threshold = mean_counts + 3.0 * std_counts
    
    flare_indices = df[df['COUNTS'] > threshold].index
    
    # Quiet window
    quiet_idx = 100
    quiet_df = df.iloc[quiet_idx : quiet_idx + 60]
    
    print("\n--- Testing Quiet Window ---")
    print(f"Counts range in window: {quiet_df['COUNTS'].min():.2f} - {quiet_df['COUNTS'].max():.2f} counts/s")
    prob_quiet = nowcaster.predict(quiet_df)
    print(f"Predicted Flare Probability: {prob_quiet * 100.0:.4f}%")
    
    # Flare window
    if len(flare_indices) > 0:
        # We start the window 59 steps before the first flare index, so the very last step in the window is the flare onset
        flare_idx = max(0, flare_indices[0] - 59)
        flare_df = df.iloc[flare_idx : flare_idx + 60]
        
        print("\n--- Testing Flare Onset Window ---")
        print(f"Counts range in window: {flare_df['COUNTS'].min():.2f} - {flare_df['COUNTS'].max():.2f} counts/s")
        prob_flare = nowcaster.predict(flare_df)
        print(f"Predicted Flare Probability: {prob_flare * 100.0:.4f}%")
    else:
        print("\nNo flare timestamps found in the dataset using the threshold.")

if __name__ == "__main__":
    test_inference()
