# SoLEXS and HEL1OS Data & Manuals Assessment

This document presents a comprehensive assessment of the Solar Low Energy X-ray Spectrometer (SoLEXS) and High Energy L1 Orbiting X-ray Spectrometer (HEL1OS) instruments onboard the Aditya-L1 spacecraft, based on their official user manuals and the local Level-1 datasets (covering May 7 to May 10, 2024).

---

## 1. Instrument Specifications & Overview

Derived from the [SoLEXS-UserManual.pdf](file:///home/shaurya/Documents/Antigravity/BAH_Project/Manuals/SoLEXS-UserManual.pdf) and [HEL1OS_UserManual.pdf](file:///home/shaurya/Documents/Antigravity/BAH_Project/Manuals/HEL1OS_UserManual.pdf):

| Parameter | SoLEXS (Soft X-rays) | HEL1OS (Hard X-rays) |
| :--- | :--- | :--- |
| **Energy Range** | 2 – 22 keV (spectral fitting: 2.8 – 22 keV) | 8 – 150 keV (spectral fitting: CdTe $\ge$ 9.5 keV, CZT $\ge$ 35 keV) |
| **Detectors** | 2 x Silicon Drift Detectors (SDD1 & SDD2) | 2 x CdTe (8-70 keV) and 2 x CZT (20-150 keV) |
| **Geometric Area** | SDD1: 7.106 mm², SDD2: 0.106 mm² | CdTe: 0.5 cm² total, CZT: 32 cm² total |
| **Energy Resolution** | ~170 eV @ 5.9 keV | CdTe: ~1 keV @ 14 keV, CZT: ~7 keV @ 60 keV |
| **Time Cadence** | 1 second (Light Curve & Spectra) | 1 second (Light Curve), 20 seconds (Type-II Spectra) |
| **Channels** | 340 channels (grouped beyond channel 168) | CdTe: 511 channels, CZT: 341 channels |
| **Target Flares** | Wide dynamic range (A-class to X-class) | Hard X-ray temporal/spectral diagnostics |

---

## 2. Dataset Structure & Schema

### SoLEXS Data (`solexs_2026Jun29T054518533/`)
Organized day-wise as `AL1_SLX_L1_YYYYMMDD_v1.0/` folders containing:
- **GTI Files (`.gti`)**: Good Time Intervals.
- **Light Curves (`.lc`)**: 1-second cadence rates in `RATE` extension (`TIME`, `COUNTS` columns).
- **Spectral Files (`.pi`)**: Type-II Pulse Invariant file (`SPECTRUM` extension containing `TSTART`, `TELAPSE`, `SPEC_NUM`, `CHANNEL` [340], `COUNTS` [340], `EXPOSURE`).

### HEL1OS Data (`hel1os_2026Jun29T054426034/`)
Organized by telemetry dumps as `HLS_YYYYMMDD_hhmmss_XXXXXsec_lev1_V111/` containing:
- **`cdte/`**: Spectra (`hel1os_cdte_spectra_cdte*.fits`) and Light Curves (`lightcurve_cdte*.fits`) divided into energy bands (e.g. 5-20 keV, 20-30 keV, etc.).
- **`czt/`**: Spectra (`hel1os_czt_spectra_czt*.fits`) and Light Curves (`lightcurve_czt*.fits`) divided into bands (20-40 keV, 40-60 keV, etc.).
- **`events/`**: Raw photon events list (`evt.fits`).
- **`aux/`**: Housekeeping (`hk.fits`) and Good Time Intervals (`gticdte*.fits`, `gticzt*.fits`).

---

## 3. Solar Flare Detections (May 7 – May 10, 2024)

By scanning the light curves of both instruments, we have identified and mapped the following major solar flare events:

### Event Summary Table
| Date | Flare Event | Soft X-ray (SoLEXS SDD2) Peak Time | Hard X-ray (HEL1OS CZT1 18-160 keV) Peak Time | SoLEXS Peak Rate (cts/s) | HEL1OS Peak Rate (cts/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2024-05-07** | **Flare 1** | 06:15:12 UTC | 06:05:06 UTC (CZT1) / 06:13:07 UTC (CdTe1) | 3,802.0 | 440.0 (CZT1) / 1,514.0 (CdTe1) |
| **2024-05-07** | **Flare 2** | 16:30:28 UTC | 16:28:51 UTC | 6,659.0 | 2,771.0 |
| **2024-05-08** | *Quiet Day* | None detected | None detected | Baseline (~900) | Baseline |
| **2024-05-09** | **Flare 3** | 09:13:59 UTC | 09:04:26 UTC | 14,555.0 | 1,067.0 |
| **2024-05-10** | **Flare 4** | 06:50:45 UTC | N/A (Telemetry missing) | 19,937.0 | N/A |

---

## 4. Physical Insights & Cross-Correlation

A physical cross-correlation of the flare peaks reveals the **Neupert Effect**:

1. **May 7th (16:30 Flare)**:
   - **HEL1OS CZT1 (20-40 keV)**: Peaks at **16:28:54 UTC** (non-thermal particle acceleration).
   - **SoLEXS SDD2 (2-22 keV)**: Peaks at **16:30:28 UTC** (thermal response of heated coronal plasma).
   - *Time Delay*: ~1.5 minutes.
2. **May 9th (09:14 Flare)**:
   - **HEL1OS CZT1 (20-40 keV)**: Peaks at **09:04:26 UTC**.
   - **SoLEXS SDD2 (2-22 keV)**: Peaks at **09:13:59 UTC**.
   - *Time Delay*: ~9.5 minutes (indicates prolonged energy release/heating).

```mermaid
chronology
    title Flare 2 Peak Sequence (May 7, 2024)
    16:28:48 : HEL1OS CZT 40-60 keV Peak
    16:28:51 : HEL1OS CZT 18-160 keV Peak
    16:28:54 : HEL1OS CZT 20-40 keV Peak
    16:30:00 : HEL1OS CdTe 5-20 keV Peak
    16:30:28 : SoLEXS SDD2 Soft X-ray Peak
```

---

## 5. Spectral Fitting Guidelines

Derived from the analysis guides in the user manuals:

> [!IMPORTANT]
> **Energy Selection Limits**
> - **SoLEXS**: Exclude data below 2.8 keV because the instrument response from 2.0 to 2.8 keV is currently uncalibrated. Use the range **2.8 – 12 keV** for fitting.
> - **HEL1OS CdTe**: Use energy channels $\ge$ 9.5 keV.
> - **HEL1OS CZT**: Use energy channels $\ge$ 35 keV.

> [!WARNING]
> **Detector Saturation**
> During peak flare phases, **SoLEXS SDD1** (geometric area 7.1 mm²) frequently saturates due to high photon flux. Spectral analysis during high flux should utilize the smaller aperture **SDD2** (0.1 mm²).

> [!TIP]
> **Systematic Errors**
> The Level-1 data contains systematic errors (`SYS_ERR`) initialized to zero. For robust spectral fitting in XSPEC or Sherpa, a systematic error of **4%** must be applied manually to avoid over-fitting.
