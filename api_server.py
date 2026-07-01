"""
API and Static File Web Server for Solar Flare Dashboard.
Serves the HTML dashboard and endpoints for light curve data streaming.
"""

import http.server
import json
import urllib.parse
import os
from pathlib import Path
import pandas as pd
import numpy as np

from nowcast import load_and_preprocess_solexs
from verify_combined_instruments import load_preprocess_hel1os

PORT = 8000
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

class DashboardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    
    def log_message(self, format, *args):
        # Suppress request logs to keep terminal output clean
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # 1. API: Stream Light Curve Data
        if path == "/api/stream":
            self.handle_api_stream(query_params)
        # 2. API: Get Master Event Catalogue
        elif path == "/api/catalogue":
            self.handle_api_catalogue()
        # 3. Serve Frontend static files
        else:
            # Default to index.html
            if path == "/" or path == "":
                self.path = "/index.html"
            super().do_GET()

    def handle_api_stream(self, query_params):
        day = query_params.get("day", ["20240507"])[0]
        instrument = query_params.get("instrument", ["solexs"])[0]
        
        if day not in DATA_DIRS:
            self.send_json_error(404, f"Day {day} not found in available datasets.")
            return
            
        inst_paths = DATA_DIRS[day]
        if instrument not in inst_paths:
            self.send_json_error(404, f"Instrument {instrument} not available for day {day}.")
            return
            
        data_path = inst_paths[instrument]
        if not data_path.exists():
            self.send_json_error(404, f"Data path not found: {data_path}")
            return
            
        try:
            if instrument == "solexs":
                df = load_and_preprocess_solexs(data_path)
            else:
                df = load_preprocess_hel1os(data_path)
                
            # Clean and downsample data to prevent browser chart lag
            df = df.dropna(subset=["TIME", "COUNTS"]).sort_values(by="TIME")
            
            # Downsampling: if more than 5000 records, take every Nth record
            n_records = len(df)
            if n_records > 5000:
                step = n_records // 3000
                df = df.iloc[::step]
            
            # Prepare JSON payload
            data_list = []
            for _, row in df.iterrows():
                data_list.append({
                    "time": int(row["TIME"]),
                    "counts": float(row["COUNTS"])
                })
                
            self.send_json_response({
                "day": day,
                "instrument": instrument,
                "total_points": len(data_list),
                "data": data_list
            })
            
        except Exception as e:
            self.send_json_error(500, f"Error processing data: {str(e)}")

    def handle_api_catalogue(self):
        cat_file = Path("master_flare_catalogue.csv")
        if not cat_file.exists():
            # If master catalogue doesn't exist, try to run verify script to generate it
            try:
                import subprocess
                subprocess.run(["python", "verify_combined_instruments.py"], check=True)
            except Exception:
                pass
                
        if not cat_file.exists():
            self.send_json_response({"events": []})
            return
            
        try:
            df = pd.read_csv(cat_file)
            # Convert NaNs to None for JSON conversion
            df = df.replace({np.nan: None})
            events = df.to_dict(orient="records")
            self.send_json_response({"events": events})
        except Exception as e:
            self.send_json_error(500, f"Error loading master catalogue: {str(e)}")

    def send_json_response(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def send_json_error(self, status_code, message):
        self.send_json_response({"error": message}, status_code)


def start_server():
    server_address = ('', PORT)
    httpd = http.server.HTTPServer(server_address, DashboardHTTPRequestHandler)
    print(f"\n====================================================")
    print(f"[SUCCESS] Dashboard Server successfully started on port {PORT}")
    print(f"URL: http://localhost:{PORT}")
    print(f"====================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Dashboard Server...")
        httpd.server_close()

if __name__ == "__main__":
    start_server()