#!/usr/bin/env python3
"""Search Poly Haven for HDRI sky assets"""

import json
from urllib.request import urlopen, Request

# Get all assets
api_url = "https://api.polyhaven.com/assets"
req = Request(api_url)
req.add_header('User-Agent', 'Mozilla/5.0')

with urlopen(req, timeout=60) as response:
    data = json.loads(response.read())
    print(f"Total assets: {len(data)}")

    # Filter for HDRI assets (type=0 is HDRI)
    hdri_assets = {}
    for asset_id, asset_info in data.items():
        if asset_info.get('type') == 0:  # Type 0 = HDRI
            hdri_assets[asset_id] = asset_info

    print(f"HDRI assets: {len(hdri_assets)}")

    # Search for overcast-related names
    overcast_keywords = ['overcast', 'cloudy', 'cloud', 'gray', 'grey', 'diffuse', 'soft']
    print(f"\nSearching for overcast-related HDRI assets:")
    found_overcast = []
    for asset_id in sorted(hdri_assets.keys()):
        for keyword in overcast_keywords:
            if keyword in asset_id.lower():
                found_overcast.append(asset_id)
                print(f"  {asset_id}")
                break

    if not found_overcast:
        print("  None found with keywords in name")

    # Also check tags for overcast
    print(f"\nSearching for HDRI assets with overcast in tags:")
    for asset_id, asset_info in sorted(hdri_assets.items()):
        tags = asset_info.get('tags', [])
        if any(keyword in tag.lower() for tag in tags for keyword in overcast_keywords):
            print(f"  {asset_id}: {tags}")
