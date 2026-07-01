# Solar Flare Algorithmic Nowcasting: Methodologies and Implementation

This document outlines suitable algorithms for real-time detection (nowcasting) of solar flares using Aditya-L1 SoLEXS (soft X-ray) and HEL1OS (hard X-ray) light curves, along with a concrete strategy to build independent event catalogues and merge them into a single master catalogue as specified in [ProblemDescription.txt](../ProblemDescription.txt).

---

## 1. Suggested Algorithms for Nowcasting

Unlike predictive models (like LSTMs) which forecast future steps, **nowcasting** requires rapid, low-latency, statistical detection of events currently taking place. Three robust algorithms are recommended:

### Algorithm 1: Adaptive Rolling Sigma-Threshold (Statistical Change-Detection)
This algorithm defines a baseline for "quiet Sun" conditions and triggers a flare alert when count rates significantly deviate from this baseline.

*   **Mechanism**:
    1.  Maintain a rolling window of historical data representing the quiet background (e.g., the last $10$ to $30$ minutes).
    2.  Calculate the rolling mean ($\mu$) and standard deviation ($\sigma$) of the count rates.
    3.  Define the trigger threshold: $T_{\text{trigger}} = \mu + k \cdot \sigma$, where $k$ is a multiplier (typically between $3.0$ and $5.0$).
    4.  **Flare Start**: Triggered when the current count rate exceeds $T_{\text{trigger}}$ for a sustained duration (e.g., $\ge 5$ consecutive seconds) to filter out cosmic ray spikes.
    5.  **Flare End**: Flagged when the count rate drops back below $\mu + 1.5 \cdot \sigma$ (or baseline) for a sustained duration.
*   **Pros/Cons**: Highly adaptive to slowly drifting instrument baselines; however, it can be slow to trigger during very gradual flare rises.

---

### Algorithm 2: Smoothed First-Derivative Thresholding (Rate-of-Rise Detection)
Solar flares are characterized by a sudden, extremely steep increase in count rates. Monitoring the rate of change detects flares earlier than absolute thresholding.

*   **Mechanism**:
    1.  Apply a smoothing filter (e.g., Savitzky-Golay or a simple rolling Gaussian filter) to the count rate sequence to suppress high-frequency noise.
    2.  Compute the numerical first derivative:
        $$\frac{dC}{dt} \approx \frac{C_t - C_{t-\Delta t}}{\Delta t}$$
    3.  **Flare Start**: Triggered when $\frac{dC}{dt} > \theta_{\text{rise}}$ (a pre-configured slope threshold) for $\ge 3$ consecutive seconds.
    4.  **Flare Peak**: Identified when $\frac{dC}{dt}$ crosses $0$ (going from positive to negative) while the absolute counts are above the baseline.
    5.  **Flare End**: Identified when the slope stabilizes back near $0$ and absolute counts return to baseline.
*   **Pros/Cons**: Extremely low latency (triggers immediately at onset); sensitive to smoothing window parameters.

---

### Algorithm 3: Cumulative Sum (CUSUM) change-point detection
CUSUM is a sequential analysis technique used to detect small changes in the mean value of a time series.

*   **Mechanism**:
    1.  Compute normalized residuals: $s_t = \frac{C_t - \mu}{\sigma} - \delta$, where $\mu$ and $\sigma$ are background statistics and $\delta$ is an allowable drift parameter.
    2.  Calculate the cumulative sum: $S_t = \max(0, S_{t-1} + s_t)$ with $S_0 = 0$.
    3.  **Flare Start**: Triggered when $S_t > H$, where $H$ is a decision threshold.
*   **Pros/Cons**: High statistical rigor; highly resistant to false alarms.

---

## 2. Instrument-Specific Nowcasting Strategies

Because SoLEXS and HEL1OS measure different physical processes, their signals look very different (as documented in [data_assessment.md](../data_assessment.md)):

```
                     Rise Phase             Peak Flux             Decay Phase
SoLEXS (Soft)        Gradual (Minutes)      Very High (Thermal)   Slow (Thermal cooling)
HEL1OS (Hard)        Impulsive (Seconds)    Spiky (Non-Thermal)   Rapid (Ends quickly)
```

### SoLEXS (Soft X-Ray) Pipeline
*   **Physical Focus**: Thermal plasma heating in the corona.
*   **Instrument Handling**:
    -   Must check for **SDD1 saturation** (aperture 7.1 mm²). If SDD1 count rates are saturated or near limit, dynamically switch the input feed to **SDD2** (aperture 0.1 mm²).
    -   Apply **Algorithm 1 (Rolling Sigma)** with $k=3.5$ on the `COUNTS` column.
    -   Apply a temporal median filter (window size = $5\text{s}$) to avoid false triggers from instrument transients.

### HEL1OS (Hard X-Ray) Pipeline
*   **Physical Focus**: Non-thermal electron acceleration.
*   **Instrument Handling**:
    -   Since HEL1OS is split into **CdTe** (lower energy, $\ge 9.5$ keV) and **CZT** (higher energy, $\ge 35$ keV), sum the count rates across CdTe or CZT channels to create a high-SNR hard X-ray light curve.
    -   Apply **Algorithm 2 (Derivative Rise)** to CZT or CdTe light curves, since non-thermal emission rises rapidly and impulsively.

---

## 3. Implementation Blueprint

Below is the conceptual architecture of how these algorithms are implemented and combined.

### Step 1: Independent Detection Catalogues
Each detector channel operates its own instance of a detection pipeline. When a flare is detected, an entry is added to its local catalogue.

**Event Schema**:
*   `event_id`: Unique identifier (e.g. `SLX-20240507-001`).
*   `start_time`: Timestamp of trigger onset.
*   `peak_time`: Timestamp of peak count rate.
*   `end_time`: Timestamp of flare termination.
*   `peak_rate`: Peak counts per second recorded.
*   `total_counts`: Integrated counts over the flare duration.

---

### Step 2: Catalogue Combination & The Neupert Effect
To build the **Master Catalogue**, we must combine the soft and hard X-ray catalogues. Because of the **Neupert Effect**:
- The non-thermal hard X-ray peak (HEL1OS) corresponds to the rate of energy injection, which often peaks *during the rise phase* of the thermal soft X-ray peak (SoLEXS).
- Therefore, the HEL1OS peak will systematically **precede** the SoLEXS peak by $1$ to $10$ minutes.

**Association Logic**:
1. Loop through all SoLEXS events.
2. For each SoLEXS event, search for any HEL1OS events whose `peak_time` falls within $[t_{\text{SoLEXS\_start}} - 5\text{ min}, t_{\text{SoLEXS\_peak}}]$.
3. If a match is found:
   - Create a combined event.
   - The master event `start_time` is the **earliest** of the two (typically HEL1OS).
   - The master event `peak_time` includes both `peak_time_soft` and `peak_time_hard`.
   - The master event `end_time` is the **latest** (typically SoLEXS).
4. If no match is found, classify the event as a soft-only or hard-only event in the database.

---

## 4. Python Implementation Draft

Here is a clean Python implementation showing how the nowcasting algorithm and merging pipeline can be coded.

```python
import numpy as np
import pandas as pd

def detect_flares_sigma_threshold(df: pd.DataFrame, time_col: str, count_col: str, 
                                 bg_window_size: int = 600, k: float = 3.5, 
                                 min_duration_sec: int = 5) -> pd.DataFrame:
    """
    Detects flares using rolling background mean + std thresholding.
    
    Args:
        df: DataFrame containing light curve data
        time_col: Name of the timestamp column
        count_col: Name of the count rate column
        bg_window_size: Size of background window (in seconds/timesteps)
        k: Sigma threshold multiplier
        min_duration_sec: Minimum duration in seconds to trigger a flare
    """
    rates = df[count_col].values
    times = df[time_col].values
    n = len(rates)
    
    # Calculate rolling statistics
    df_temp = pd.DataFrame({count_col: rates})
    rolling = df_temp[count_col].rolling(window=bg_window_size, min_periods=50)
    roll_mean = rolling.mean().values
    roll_std = rolling.std().values
    
    # Fill initial nan values with global mean/std
    roll_mean = np.nan_to_num(roll_mean, nan=np.nanmean(rates))
    roll_std = np.nan_to_num(roll_std, nan=np.nanstd(rates))
    roll_std[roll_std == 0] = 1.0
    
    thresholds = roll_mean + k * roll_std
    is_above = rates > thresholds
    
    events = []
    in_flare = False
    start_idx = 0
    
    for i in range(n):
        if is_above[i] and not in_flare:
            # Lookahead to verify trigger is sustained
            if i + min_duration_sec < n and all(is_above[i:i + min_duration_sec]):
                in_flare = True
                start_idx = i
        elif in_flare:
            # Flare ends when count drops below a lower threshold (e.g. mean + 1.5 * std)
            lower_thresh = roll_mean[i] + 1.5 * roll_std[i]
            if rates[i] <= lower_thresh or i == n - 1:
                in_flare = False
                end_idx = i
                
                # Extract event statistics
                event_window_rates = rates[start_idx : end_idx + 1]
                event_window_times = times[start_idx : end_idx + 1]
                peak_idx_local = np.argmax(event_window_rates)
                
                events.append({
                    "start_time": float(times[start_idx]),
                    "peak_time": float(event_window_times[peak_idx_local]),
                    "end_time": float(times[end_idx]),
                    "peak_rate": float(event_window_rates[peak_idx_local]),
                    "total_counts": float(np.sum(event_window_rates))
                })
                
    return pd.DataFrame(events)


def build_master_catalogue(soft_df: pd.DataFrame, hard_df: pd.DataFrame, 
                           association_window_sec: int = 600) -> pd.DataFrame:
    """
    Combines independent soft and hard X-ray catalogues into a master catalogue,
    associating events based on temporal overlap and the Neupert Effect.
    """
    master_events = []
    associated_hard_indices = set()
    
    for s_idx, soft_event in soft_df.iterrows():
        # Match hard events where hard peak occurs near soft start/peak window
        # Allow hard peak to precede soft peak by up to association_window_sec (Neupert Effect)
        t_soft_start = soft_event["start_time"]
        t_soft_peak = soft_event["peak_time"]
        
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
            # Soft X-ray only event
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
```
