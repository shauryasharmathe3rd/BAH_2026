"""
Solar Flare Nowcasting Pipeline using Statistical Change-Detection.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from data_pipeline import load_solexs_fits_table, load_hel1os_fits_table


def get_solexs_lc_files(data_dir: Path) -> list[Path]:
    """Find all SoLEXS .lc and .lc.gz files recursively under data_dir."""
    files = []
    # Search for files with suffix .lc or .lc.gz
    for file_path in data_dir.rglob("*"):
        if file_path.name.endswith(".lc") or file_path.name.endswith(".lc.gz"):
            files.append(file_path)
    return sorted(files)


def load_and_preprocess_solexs(data_dir: Path) -> pd.DataFrame:
    """
    Load SoLEXS light curve files, prioritizing SDD2 if SDD1 is saturated,
    or using whatever is available.
    """
    lc_files = get_solexs_lc_files(data_dir)
    if not lc_files:
        raise FileNotFoundError(f"No SoLEXS .lc or .lc.gz files found in {data_dir}")

    # Group files by date (e.g., AL1_SOLEXS_YYYYMMDD_SDD*_L1.lc.gz)
    # We want to pair SDD1 and SDD2 for the same day if they both exist
    day_files = {}
    for f in lc_files:
        filename = f.name
        # Extract date from name (usually YYYYMMDD at index 11)
        parts = filename.split('_')
        date_str = None
        for part in parts:
            if part.isdigit() and len(part) == 8:
                date_str = part
                break
        if not date_str:
            date_str = "unknown"
        
        if date_str not in day_files:
            day_files[date_str] = {}
        
        if "sdd1" in filename.lower():
            day_files[date_str]["SDD1"] = f
        elif "sdd2" in filename.lower():
            day_files[date_str]["SDD2"] = f
        else:
            # Fallback
            day_files[date_str]["SDD2"] = f

    dfs = []
    for date_str, paths in day_files.items():
        sdd1_path = paths.get("SDD1")
        sdd2_path = paths.get("SDD2")

        if sdd1_path and sdd2_path:
            print(f"Loading paired SoLEXS SDD1 & SDD2 data for {date_str}...")
            df_sdd1 = load_solexs_fits_table(sdd1_path).dropna(subset=["TIME", "COUNTS"]).sort_values(by="TIME")
            df_sdd2 = load_solexs_fits_table(sdd2_path).dropna(subset=["TIME", "COUNTS"]).sort_values(by="TIME")
            
            # Align by TIME
            df_merged = pd.merge(df_sdd1, df_sdd2, on="TIME", suffixes=("_sdd1", "_sdd2"))
            
            # Switch logic: if SDD1 counts > 15000 (saturation threshold), use SDD2 scaled, else SDD1.
            # SDD1 area is 7.106 mm2, SDD2 area is 0.106 mm2. Scale factor = 7.106 / 0.106 ~ 67.0
            scale_factor = 67.0
            counts_final = []
            for _, row in df_merged.iterrows():
                if row["COUNTS_sdd1"] >= 15000.0:
                    counts_final.append(row["COUNTS_sdd2"] * scale_factor)
                else:
                    counts_final.append(row["COUNTS_sdd1"])
            
            df_day = pd.DataFrame({
                "TIME": df_merged["TIME"],
                "COUNTS": counts_final,
                "instrument": "solexs"
            })
            dfs.append(df_day)
        elif sdd2_path:
            print(f"Loading SoLEXS SDD2 data for {date_str}...")
            df_day = load_solexs_fits_table(sdd2_path).dropna(subset=["TIME", "COUNTS"]).sort_values(by="TIME")
            df_day = df_day[["TIME", "COUNTS"]].copy()
            df_day["instrument"] = "solexs"
            dfs.append(df_day)
        elif sdd1_path:
            print(f"Loading SoLEXS SDD1 data for {date_str}...")
            df_day = load_solexs_fits_table(sdd1_path).dropna(subset=["TIME", "COUNTS"]).sort_values(by="TIME")
            df_day = df_day[["TIME", "COUNTS"]].copy()
            df_day["instrument"] = "solexs"
            dfs.append(df_day)

    if not dfs:
        raise ValueError("Failed to load any valid SoLEXS data.")

    df = pd.concat(dfs, ignore_index=True).sort_values(by="TIME").reset_index(drop=True)
    return df


# ============================================================================
# NOWCASTING ALGORITHMS
# ============================================================================

def detect_flares_rolling_sigma(
    df: pd.DataFrame, 
    time_col: str = "TIME", 
    count_col: str = "COUNTS", 
    bg_window_size: int = 600, 
    k: float = 3.5, 
    min_duration_sec: int = 5
) -> pd.DataFrame:
    """
    Algorithm 1: Adaptive Rolling Sigma-Threshold.
    Triggers when counts exceed background mean + k * std for a sustained duration.
    """
    rates = df[count_col].values
    times = df[time_col].values
    n = len(rates)
    
    if n == 0:
        return pd.DataFrame()

    # Calculate rolling background statistics
    df_temp = pd.DataFrame({count_col: rates})
    rolling = df_temp[count_col].rolling(window=bg_window_size, min_periods=50)
    roll_mean = rolling.mean().values
    roll_std = rolling.std().values
    
    # Fallback for initial NaN values
    global_mean = np.nanmean(rates)
    global_std = np.nanstd(rates)
    if global_std == 0:
        global_std = 1.0
        
    roll_mean = np.nan_to_num(roll_mean, nan=global_mean)
    roll_std = np.nan_to_num(roll_std, nan=global_std)
    roll_std[roll_std == 0] = 1.0
    
    thresholds = roll_mean + k * roll_std
    is_above = rates > thresholds
    
    events = []
    in_flare = False
    start_idx = 0
    
    for i in range(n):
        if is_above[i] and not in_flare:
            # Lookahead to ensure trigger is sustained
            if i + min_duration_sec < n and all(is_above[i : i + min_duration_sec]):
                in_flare = True
                start_idx = i
        elif in_flare:
            # End trigger: drops below mean + 1.5 * std for min_duration_sec
            lower_thresh = roll_mean[i] + 1.5 * roll_std[i]
            if rates[i] <= lower_thresh or i == n - 1:
                # Sustained end check
                if i == n - 1 or all(rates[i : i + min_duration_sec] <= (roll_mean[i:i + min_duration_sec] + 1.5 * roll_std[i:i + min_duration_sec])):
                    in_flare = False
                    end_idx = i
                    
                    # Extract statistics
                    event_rates = rates[start_idx : end_idx + 1]
                    event_times = times[start_idx : end_idx + 1]
                    peak_idx_local = np.argmax(event_rates)
                    
                    events.append({
                        "event_id": f"FL-SIG-{int(times[start_idx])}",
                        "start_time": float(times[start_idx]),
                        "peak_time": float(event_times[peak_idx_local]),
                        "end_time": float(times[end_idx]),
                        "peak_rate": float(event_rates[peak_idx_local]),
                        "total_counts": float(np.sum(event_rates))
                    })
                    
    return pd.DataFrame(events)


def detect_flares_derivative(
    df: pd.DataFrame, 
    time_col: str = "TIME", 
    count_col: str = "COUNTS", 
    smooth_window: int = 5, 
    rise_threshold: float = 10.0, 
    min_duration_sec: int = 3
) -> pd.DataFrame:
    """
    Algorithm 2: Smoothed First-Derivative Thresholding.
    Triggers on sudden steep increase in counts.
    """
    if len(df) < smooth_window:
        return pd.DataFrame()

    # Apply rolling smoothing to suppress high-frequency noise
    smoothed_rates = df[count_col].rolling(window=smooth_window, center=True, min_periods=1).mean().values
    times = df[time_col].values
    rates = df[count_col].values
    n = len(smoothed_rates)
    
    # Compute derivative dC/dt
    dt = np.diff(times)
    dt[dt == 0] = 1.0  # Avoid division by zero
    
    dC_dt = np.zeros(n)
    dC_dt[1:] = np.diff(smoothed_rates) / dt
    
    # Threshold check
    is_rising = dC_dt > rise_threshold
    
    events = []
    in_flare = False
    start_idx = 0
    
    # Maintain global mean/std for peak verification
    global_mean = np.mean(rates)
    global_std = np.std(rates)
    
    for i in range(1, n):
        if is_rising[i] and not in_flare:
            # Sustained slope increase
            if i + min_duration_sec < n and all(is_rising[i : i + min_duration_sec]):
                in_flare = True
                start_idx = i
        elif in_flare:
            # Slope crosses 0 (peak) or becomes negative, and counts drop back to baseline
            # We look for where the slope becomes stable or negative and counts drop below global mean + 1.5 * std
            lower_thresh = global_mean + 1.5 * global_std
            if (dC_dt[i] <= 0 and rates[i] <= lower_thresh) or i == n - 1:
                in_flare = False
                end_idx = i
                
                event_rates = rates[start_idx : end_idx + 1]
                event_times = times[start_idx : end_idx + 1]
                peak_idx_local = np.argmax(event_rates)
                
                events.append({
                    "event_id": f"FL-DER-{int(times[start_idx])}",
                    "start_time": float(times[start_idx]),
                    "peak_time": float(event_times[peak_idx_local]),
                    "end_time": float(times[end_idx]),
                    "peak_rate": float(event_rates[peak_idx_local]),
                    "total_counts": float(np.sum(event_rates))
                })
                
    return pd.DataFrame(events)


def build_master_catalogue(soft_df: pd.DataFrame, hard_df: pd.DataFrame, 
                           association_window_sec: int = 600) -> pd.DataFrame:
    """
    Combines independent Soft (SoLEXS) and Hard (HEL1OS) catalogues based on the Neupert Effect.
    HEL1OS peak should precede SoLEXS peak by up to association_window_sec.
    """
    master_events = []
    associated_hard_indices = set()
    
    if soft_df.empty and hard_df.empty:
        return pd.DataFrame()
    elif soft_df.empty:
        # All events are hard-only
        for idx, row in hard_df.iterrows():
            master_events.append({
                "source_type": "hard_only",
                "start_time": row["start_time"],
                "peak_time_soft": None,
                "peak_time_hard": row["peak_time"],
                "end_time": row["end_time"],
                "peak_rate_soft": None,
                "peak_rate_hard": row["peak_rate"],
                "neupert_delay_sec": None
            })
        return pd.DataFrame(master_events)
    elif hard_df.empty:
        # All events are soft-only
        for idx, row in soft_df.iterrows():
            master_events.append({
                "source_type": "soft_only",
                "start_time": row["start_time"],
                "peak_time_soft": row["peak_time"],
                "peak_time_hard": None,
                "end_time": row["end_time"],
                "peak_rate_soft": row["peak_rate"],
                "peak_rate_hard": None,
                "neupert_delay_sec": None
            })
        return pd.DataFrame(master_events)

    for s_idx, soft_event in soft_df.iterrows():
        t_soft_start = soft_event["start_time"]
        t_soft_peak = soft_event["peak_time"]
        
        # Hard peak is within soft start - 300s to soft peak + 60s
        matches = hard_df[
            (hard_df["peak_time"] >= t_soft_start - 300) & 
            (hard_df["peak_time"] <= t_soft_peak + 60)
        ]
        
        if not matches.empty:
            # Associate with the largest hard flare matching the window
            best_match_idx = matches["peak_rate"].idxmax()
            hard_event = hard_df.loc[best_match_idx]
            associated_hard_indices.add(best_match_idx)
            
            master_events.append({
                "source_type": "joint",
                "start_time": min(soft_event["start_time"], hard_event["start_time"]),
                "peak_time_soft": soft_event["peak_time"],
                "peak_time_hard": hard_event["peak_time"],
                "end_time": max(soft_event["end_time"], hard_event["end_time"]),
                "peak_rate_soft": soft_event["peak_rate"],
                "peak_rate_hard": hard_event["peak_rate"],
                "neupert_delay_sec": soft_event["peak_time"] - hard_event["peak_time"]
            })
        else:
            master_events.append({
                "source_type": "soft_only",
                "start_time": soft_event["start_time"],
                "peak_time_soft": soft_event["peak_time"],
                "peak_time_hard": None,
                "end_time": soft_event["end_time"],
                "peak_rate_soft": soft_event["peak_rate"],
                "peak_rate_hard": None,
                "neupert_delay_sec": None
            })
            
    # Add remaining hard X-ray events that had no soft X-ray counterpart
    for h_idx, hard_event in hard_df.iterrows():
        if h_idx not in associated_hard_indices:
            master_events.append({
                "source_type": "hard_only",
                "start_time": hard_event["start_time"],
                "peak_time_soft": None,
                "peak_time_hard": hard_event["peak_time"],
                "end_time": hard_event["end_time"],
                "peak_rate_soft": None,
                "peak_rate_hard": hard_event["peak_rate"],
                "neupert_delay_sec": None
            })
            
    return pd.DataFrame(master_events)


# ============================================================================
# COMPATIBILITY CLASS FOR INFERENCE TESTING
# ============================================================================

class Nowcaster:
    """Inference helper for real-time statistical solar flare nowcasting."""
    
    def __init__(self, model_path=None, metadata_path=None):
        # Model path is ignored as this is not a predictive model anymore
        self.k = 3.5
        self.bg_mean = 500.0
        self.bg_std = 100.0
        
        if metadata_path:
            try:
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
                    self.bg_mean = meta.get("feature_means", [0, 500])[1]
                    self.bg_std = meta.get("feature_stds", [1, 100])[1]
            except Exception:
                pass
        
    def predict(self, recent_df: pd.DataFrame) -> float:
        """
        Predict the flare probability using a statistical Z-score.
        Returns a probability float in range [0.0, 1.0].
        """
        if recent_df.empty:
            return 0.0
            
        counts = recent_df["COUNTS"].values
        latest = counts[-1]
        
        # Calculate dynamic statistics of the window (excluding the last few steps)
        if len(counts) > 10:
            mean = np.mean(counts[:-5])
            std = np.std(counts[:-5])
            if std == 0:
                std = 1.0
        else:
            mean = self.bg_mean
            std = self.bg_std
            
        z_score = (latest - mean) / std
        
        # Sigmoid function centered at trigger multiplier k
        prob = 1.0 / (1.0 + np.exp(-(z_score - self.k)))
        return float(prob)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aditya-L1 Statistical Nowcasting CLI")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\USER\Documents\AL1_SLX_L1_20240507_v1.0",
                        help="Path to SoLEXS FITS folder")
    parser.add_argument("--algo", type=str, choices=["sigma", "derivative", "both"], default="both",
                        help="Algorithm to run")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save catalogues")
    parser.add_argument("--k", type=float, default=3.5, help="Sigma threshold multiplier")
    parser.add_argument("--bg_window", type=int, default=600, help="Background window size in seconds")
    parser.add_argument("--min_duration", type=int, default=5, help="Min duration to trigger flare")
    parser.add_argument("--rise_threshold", type=float, default=10.0, help="Derivative rise threshold")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    print(f"Loading data from {data_path}...")
    
    try:
        df = load_and_preprocess_solexs(data_path)
    except Exception as e:
        print(f"Error loading SoLEXS data: {e}")
        sys.exit(1)
        
    print(f"Loaded {len(df)} records of light curve data.")
    
    # Calculate baseline mean and standard deviation for saving metadata
    mean_counts = df["COUNTS"].mean()
    std_counts = df["COUNTS"].std()
    
    # Save a metadata file for compatibility
    metadata = {
        "feature_cols": ["TIME", "COUNTS"],
        "feature_means": [df["TIME"].mean(), float(mean_counts)],
        "feature_stds": [df["TIME"].std(), float(std_counts)],
        "window_size": 60,
        "k": args.k,
        "bg_window": args.bg_window
    }
    metadata_path = Path(args.output_dir) / "nowcast_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved nowcasting metadata to {metadata_path}")
    
    soft_catalogue = pd.DataFrame()
    if args.algo in ["sigma", "both"]:
        print("\nRunning Adaptive Rolling Sigma Nowcaster...")
        soft_catalogue = detect_flares_rolling_sigma(
            df, 
            bg_window_size=args.bg_window, 
            k=args.k, 
            min_duration_sec=args.min_duration
        )
        print(f"Detected {len(soft_catalogue)} events using Sigma algorithm.")
        if not soft_catalogue.empty:
            print(soft_catalogue[["start_time", "peak_time", "end_time", "peak_rate"]])
            out_file = Path(args.output_dir) / "solexs_flares_sigma.csv"
            soft_catalogue.to_csv(out_file, index=False)
            print(f"Saved catalogue to {out_file}")
            
    if args.algo in ["derivative", "both"]:
        print("\nRunning Rate-of-Rise Derivative Nowcaster...")
        deriv_catalogue = detect_flares_derivative(
            df, 
            rise_threshold=args.rise_threshold, 
            min_duration_sec=args.min_duration
        )
        print(f"Detected {len(deriv_catalogue)} events using Derivative algorithm.")
        if not deriv_catalogue.empty:
            print(deriv_catalogue[["start_time", "peak_time", "end_time", "peak_rate"]])
            out_file = Path(args.output_dir) / "solexs_flares_derivative.csv"
            deriv_catalogue.to_csv(out_file, index=False)
            print(f"Saved catalogue to {out_file}")
            
    print("\nNowcasting finished successfully.")
