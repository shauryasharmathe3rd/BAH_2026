# Solar Flare Dashboard API Server Analysis

This report provides an analysis of the functionality in [api_server.py](../api_server.py) and documents the path adjustments made across the codebase to align with the local workspace directory structure.

---

## 1. What is `api_server.py` doing?

[api_server.py](../api_server.py) is a lightweight Python-based HTTP web server that serves both the frontend web dashboard and the backend API endpoints. It is implemented using Python's standard `http.server.SimpleHTTPRequestHandler` class.

Key functionalities include:

1.  **Frontend Static Server**:
    *   Serves static web files (such as `index.html`, Javascript, and CSS files) from the project directory.
    *   Defaults requests for `/` or empty paths to `index.html`.
2.  **API Endpoint - `/api/stream` (Light Curve Data)**:
    *   Queries params: `day` (e.g. `20240507`) and `instrument` (e.g. `solexs` or `hel1os`).
    *   Reads and processes the target dataset dynamically.
    *   **Downsampling**: If the dataset has more than $5,000$ points, it downsamples the data to $3,000$ points to prevent frontend chart rendering lag in the browser.
    *   Streams the structured time series back as a JSON response: `{"time": UNIX_timestamp, "counts": count_rate}`.
3.  **API Endpoint - `/api/catalogue` (Master Catalogue)**:
    *   Serves the combined Soft and Hard X-Ray master event catalogue.
    *   If `master_flare_catalogue.csv` is missing from the directory, it executes [verify_combined_instruments.py](../verify_combined_instruments.py) using a subprocess to generate it in real-time, then serves the result as JSON.

---

## 2. Updated Project Path Structure

To ensure that the API server and all test/verification pipelines execute successfully on this Linux system, several hardcoded Windows-specific paths (pointing to `C:\Users\USER\...`) were updated to use the relative paths in this project's `Data/` directory.

The following changes were made:

### A. [api_server.py](../api_server.py)
Updated the `DATA_DIRS` lookup dictionary:
```python
DATA_DIRS = {
    "20240507": {
        "solexs": Path("Data/solexs_2026Jun29T175402251/AL1_SLX_L1_20240507_v1.0"),
        "hel1os": Path("Data/hel1os_2026Jun29T180143795/HLS_20240507_000006_26239sec_lev1_V111/2024/05/07/HLS_20240507_000006_26239sec_lev1_V111/czt/lightcurve_czt1.fits")
    },
    "20240509": {
        "solexs": Path("Data/solexs_2026Jun29T175809476/AL1_SLX_L1_20240509_v1.0")
    },
    "20240510": {
        "solexs": Path("Data/solexs_2026Jun29T175954834/AL1_SLX_L1_20240510_v1.0")
    }
}
```

### B. [verify_combined_instruments.py](../verify_combined_instruments.py)
Updated the datasets paths inside the `run_verification` function:
```python
    solexs_dir = Path("Data/solexs_2026Jun29T175402251/AL1_SLX_L1_20240507_v1.0")
    hel1os_file = Path("Data/hel1os_2026Jun29T180143795/HLS_20240507_000006_26239sec_lev1_V111/2024/05/07/HLS_20240507_000006_26239sec_lev1_V111/czt/lightcurve_czt1.fits")
```

### C. [test_nowcast.py](../test_nowcast.py)
Updated the test path inside the `test_inference` function:
```python
    data_dir = Path("Data/solexs_2026Jun29T175402251/AL1_SLX_L1_20240507_v1.0")
```

### D. [nowcast.py](../nowcast.py)
Updated the CLI default `--data_dir` argument:
```python
    parser.add_argument("--data_dir", type=str, default="Data/solexs_2026Jun29T175402251/AL1_SLX_L1_20240507_v1.0",
```

---

## 3. Verification Run Results

All scripts were ran locally to confirm correctness:
*   [verify_combined_instruments.py](../verify_combined_instruments.py) successfully loaded the data, processed nowcast event catalogues, performed Neupert Effect matching, and outputted the master catalogue `master_flare_catalogue.csv` containing $56$ joint and single-instrument events.
*   [test_nowcast.py](../test_nowcast.py) ran all tests successfully, outputting both Algorithm 1 (Sigma) and Algorithm 2 (Derivative) detections.
*   [api_server.py](../api_server.py) compiled cleanly and is ready to stream light curves to the web dashboard.
