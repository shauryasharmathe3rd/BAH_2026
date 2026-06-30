import urllib.request
import json
import sys

def test():
    # Make a dummy 60-step time series payload
    history = [{"TIME": float(1715083264 + i), "COUNTS": float(500 + i % 10)} for i in range(60)]
    payload = {"history": history}
    
    # 1. Test /nowcast
    print("Testing /nowcast...")
    req = urllib.request.Request(
        "http://localhost:8001/nowcast",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            print(json.loads(r.read().decode()))
    except Exception as e:
        print(f"Error testing /nowcast: {e}")
        
    # 2. Test /forecast
    print("\nTesting /forecast...")
    req = urllib.request.Request(
        "http://localhost:8001/forecast",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            print(json.loads(r.read().decode()))
    except Exception as e:
        print(f"Error testing /forecast: {e}")

if __name__ == "__main__":
    test()
