"""
Verification script for combined SoLEXS and HEL1OS nowcasting using statistical algorithms.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from nowcast import (
    detect_flares_rolling_sigma, 
    detect_flares_derivative,
    build_master_catalogue,
    load_and_preprocess_solexs
)
from data_pipeline import load_hel1os_fits_table

def load_preprocess_hel1os(file_path: Path) -> pd.DataFrame:
    """Load a HEL1OS light curve file and preprocess it for nowcasting."""
    print(f"Loading HEL1OS file: {file_path.name}...")
    df = load_hel1os_fits_table(file_path).dropna(subset=["ISOT", "CTR"])
    
    # Decode byte strings in ISOT
    if df['ISOT'].dtype == object:
        isot_col = df['ISOT'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    else:
        isot_col = df['ISOT']
        
    # Convert to UNIX epoch seconds
    df['TIME'] = pd.to_datetime(isot_col).astype('datetime64[ns]').astype('int64') // 10**9
    df['COUNTS'] = df['CTR'].astype(np.float32)
    
    # Sort and clean
    df = df.sort_values(by="TIME").reset_index(drop=True)
    df = df[["TIME", "COUNTS"]].copy()
    df["instrument"] = "hel1os"
    return df

def run_verification():
    # Paths to datasets
    solexs_dir = Path("Data/solexs_2026Jun29T175402251/AL1_SLX_L1_20240507_v1.0")
    hel1os_file = Path("Data/hel1os_2026Jun29T180143795/HLS_20240507_000006_26239sec_lev1_V111/2024/05/07/HLS_20240507_000006_26239sec_lev1_V111/czt/lightcurve_czt1.fits")
    
    if not solexs_dir.exists():
        print(f"Error: SoLEXS directory not found at {solexs_dir}")
        sys.exit(1)
    if not hel1os_file.exists():
        print(f"Error: HEL1OS CZT1 lightcurve not found at {hel1os_file}")
        sys.exit(1)
        
    print("====================================================")
    print("STEP 1: Load and Preprocess SoLEXS & HEL1OS Data")
    print("====================================================")
    
    # Load SoLEXS (Soft X-rays)
    solexs_df = load_and_preprocess_solexs(solexs_dir)
    print(f"SoLEXS: Loaded {len(solexs_df)} records.")
    
    # Load HEL1OS (Hard X-rays)
    hel1os_df = load_preprocess_hel1os(hel1os_file)
    print(f"HEL1OS: Loaded {len(hel1os_df)} records.")
    
    print("\n====================================================")
    print("STEP 2: Generate Independent Event Catalogues")
    print("====================================================")
    
    # Generate Soft X-ray (SoLEXS) event catalogue
    print("\nRunning Rolling Sigma on SoLEXS (Soft X-ray)...")
    solexs_catalogue = detect_flares_rolling_sigma(
        solexs_df, 
        k=3.5, 
        bg_window_size=600, 
        min_duration_sec=5
    )
    print(f"Detected {len(solexs_catalogue)} Soft X-ray events.")
    
    # Generate Hard X-ray (HEL1OS) event catalogue
    print("\nRunning Rolling Sigma on HEL1OS CZT1 (Hard X-ray)...")
    # Apply rolling mean smoothing to handle the spiky non-thermal counts
    hel1os_df_smoothed = hel1os_df.copy()
    hel1os_df_smoothed['COUNTS'] = hel1os_df['COUNTS'].rolling(window=10, center=True, min_periods=1).mean()
    
    # Run rolling sigma nowcasting on the smoothed data
    hel1os_catalogue = detect_flares_rolling_sigma(
        hel1os_df_smoothed, 
        k=3.0, 
        bg_window_size=300, 
        min_duration_sec=3
    )
    # Rename event ID suffix to identify HEL1OS events
    if not hel1os_catalogue.empty:
        hel1os_catalogue["event_id"] = hel1os_catalogue["event_id"].str.replace("FL-SIG-", "FL-HLS-")
    print(f"Detected {len(hel1os_catalogue)} Hard X-ray events.")
    
    print("\n====================================================")
    print("STEP 3: Merge Catalogues (The Neupert Effect)")
    print("====================================================")
    
    # Merge catalogues using Neupert Effect logic
    master_catalogue = build_master_catalogue(solexs_catalogue, hel1os_catalogue)
    print(f"Generated Master Catalogue containing {len(master_catalogue)} entries:")
    
    if not master_catalogue.empty:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        # Display joint events where both SXR and HXR signatures match
        joint_events = master_catalogue[master_catalogue["source_type"] == "joint"]
        print(f"\n--- Associated Joint Events (Neupert Effect Delay) ---")
        if not joint_events.empty:
            for idx, row in joint_events.iterrows():
                print(f"Joint Event {idx+1}:")
                print(f"  Start Time        : {row['start_time']}")
                print(f"  Peak Time (Soft)  : {row['peak_time_soft']}")
                print(f"  Peak Time (Hard)  : {row['peak_time_hard']}")
                print(f"  End Time          : {row['end_time']}")
                print(f"  Peak Rate (Soft)  : {row['peak_rate_soft']:.2f} cts/s")
                print(f"  Peak Rate (Hard)  : {row['peak_rate_hard']:.2f} cts/s")
                print(f"  Neupert Delay     : {row['neupert_delay_sec']:.2f} seconds ({row['neupert_delay_sec']/60:.2f} minutes)")
        else:
            print("No associated joint events found.")
            
        # Display other single-instrument events
        soft_only = master_catalogue[master_catalogue["source_type"] == "soft_only"]
        hard_only = master_catalogue[master_catalogue["source_type"] == "hard_only"]
        print(f"\n--- Soft-Only Events Count: {len(soft_only)} | Hard-Only Events Count: {len(hard_only)} ---")
        
        # Save master catalogue
        out_file = Path("master_flare_catalogue.csv")
        master_catalogue.to_csv(out_file, index=False)
        print(f"\n[SUCCESS] Master Catalogue successfully saved to {out_file.absolute()}")
    else:
        print("Master catalogue is empty.")

if __name__ == "__main__":
    run_verification()
