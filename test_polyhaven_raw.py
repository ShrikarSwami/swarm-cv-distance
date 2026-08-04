#!/usr/bin/env python3
"""Check raw Poly Haven API response"""

import json
from urllib.request import urlopen, Request

# Get all assets
api_url = "https://api.polyhaven.com/assets"
req = Request(api_url)
req.add_header('User-Agent', 'Mozilla/5.0')

with urlopen(req, timeout=30) as response:
    data = json.loads(response.read())
    print(f"Response type: {type(data)}")
    print(f"Keys: {list(data.keys())[:20]}")

    # Check if there's an 'assets' key
    if 'assets' in data:
        assets = data['assets']
        print(f"Number of assets: {len(assets)}")
        # Show first few
        for i, (key, value) in enumerate(list(assets.items())[:5]):
            print(f"  {key}: {value}")
    else:
        print("No 'assets' key found")
        # Print first 500 chars of response
        print(f"Response preview: {str(data)[:500]}")
