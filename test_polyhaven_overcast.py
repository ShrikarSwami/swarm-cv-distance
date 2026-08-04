#!/usr/bin/env python3
"""Search Poly Haven skies category for overcast assets"""

import json
from urllib.request import urlopen, Request

# Get all skies
api_url = "https://api.polyhaven.com/assets?c=skies"
req = Request(api_url)
req.add_header('User-Agent', 'Mozilla/5.0')

with urlopen(req, timeout=30) as response:
    data = json.loads(response.read())
    assets = data.get('assets', {})
    print(f"Total skies: {len(assets)}")

    # Search for overcast-related names
    overcast_keywords = ['overcast', 'cloudy', 'cloud', 'gray', 'grey', 'diffuse', 'soft']
    print(f"\nSearching for overcast-related assets:")
    for asset_id in sorted(assets.keys()):
        for keyword in overcast_keywords:
            if keyword in asset_id.lower():
                print(f"  {asset_id}")
                break

    # Also search for assets with "sky" in the name
    print(f"\nSearching for sky assets:")
    sky_assets = [a for a in assets.keys() if 'sky' in a.lower()]
    for asset_id in sorted(sky_assets)[:20]:  # Show first 20
        print(f"  {asset_id}")
