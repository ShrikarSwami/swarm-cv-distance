#!/usr/bin/env python3
"""Test Poly Haven API for overcast HDRI assets"""

import json
from urllib.request import urlopen, Request
from urllib.error import URLError

# Try different API endpoints to find overcast HDRIs
api_endpoints = [
    "https://api.polyhaven.com/assets?t=hdri",
    "https://api.polyhaven.com/assets?c=skies",
    "https://api.polyhaven.com/assets?t=overcast",
    "https://api.polyhaven.com/assets?c=skies&t=overcast",
    "https://api.polyhaven.com/assets?c=hdri&t=overcast",
]

print("=== Testing Poly Haven API Endpoints ===\n")

for endpoint in api_endpoints:
    print(f"Endpoint: {endpoint}")
    try:
        req = Request(endpoint)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            print(f"  Status: {response.status}")
            print(f"  Response type: {type(data)}")
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())[:10]}")
                if 'assets' in data:
                    assets = data['assets']
                    print(f"  Number of assets: {len(assets)}")
                    if assets:
                        print(f"  First few assets: {list(assets.keys())[:5]}")
            elif isinstance(data, list):
                print(f"  Number of items: {len(data)}")
                if data:
                    print(f"  First item: {data[0]}")
    except URLError as e:
        print(f"  Error: {e}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
