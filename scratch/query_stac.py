import requests
import json
from datetime import datetime

STAC_API_URL = "https://earth-search.aws.element84.com/v1/search"

def search_stac(bbox, datetime_range, limit=5, cloud_cover=20):
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": datetime_range,
        "query": {
            "eo:cloud_cover": {"lt": cloud_cover}
        },
        "limit": limit
    }
    resp = requests.post(STAC_API_URL, json=payload)
    if resp.status_code != 200:
        print(f"Error: {resp.status_code}")
        return []
    
    data = resp.json()
    return data.get("features", [])

# Case A: Dixie Fire (Started July 13, 2021, burned till October)
# Location roughly around Lake Almanor, CA: [-121.2, 40.1, -121.0, 40.3]
print("--- Case A: Dixie Fire (California, 2021) ---")
bbox_a = [-121.2, 40.1, -121.0, 40.3]
pre_fire = search_stac(bbox_a, "2021-06-01T00:00:00Z/2021-07-10T00:00:00Z")
post_fire = search_stac(bbox_a, "2021-09-01T00:00:00Z/2021-10-15T00:00:00Z")

if pre_fire and post_fire:
    print(f"Pre-fire T1 ID: {pre_fire[0]['id']} | Date: {pre_fire[0]['properties']['datetime']} | Clouds: {pre_fire[0]['properties']['eo:cloud_cover']}")
    print(f"Post-fire T2 ID: {post_fire[0]['id']} | Date: {post_fire[0]['properties']['datetime']} | Clouds: {post_fire[0]['properties']['eo:cloud_cover']}")
    print(f"T1 Assets: {list(pre_fire[0]['assets'].keys())}")

# Case B: Stable Forest (Redwoods National Park area)
# Location roughly: [-124.0, 41.3, -123.9, 41.4]
print("\n--- Case B: Stable Forest (Redwoods, CA) ---")
bbox_b = [-124.0, 41.3, -123.9, 41.4]
stable_t1 = search_stac(bbox_b, "2021-06-01T00:00:00Z/2021-07-10T00:00:00Z")
stable_t2 = search_stac(bbox_b, "2021-09-01T00:00:00Z/2021-10-15T00:00:00Z")

if stable_t1 and stable_t2:
    print(f"Stable T1 ID: {stable_t1[0]['id']} | Date: {stable_t1[0]['properties']['datetime']} | Clouds: {stable_t1[0]['properties']['eo:cloud_cover']}")
    print(f"Stable T2 ID: {stable_t2[0]['id']} | Date: {stable_t2[0]['properties']['datetime']} | Clouds: {stable_t2[0]['properties']['eo:cloud_cover']}")

# Case C: Agricultural Seasonal Change (Central Valley, CA)
# Location roughly: [-120.0, 36.5, -119.8, 36.6]
print("\n--- Case C: Agriculture (Central Valley, CA) ---")
bbox_c = [-120.0, 36.5, -119.8, 36.6]
ag_t1 = search_stac(bbox_c, "2021-06-01T00:00:00Z/2021-07-10T00:00:00Z")
ag_t2 = search_stac(bbox_c, "2021-09-01T00:00:00Z/2021-10-15T00:00:00Z")

if ag_t1 and ag_t2:
    print(f"Ag T1 ID: {ag_t1[0]['id']} | Date: {ag_t1[0]['properties']['datetime']} | Clouds: {ag_t1[0]['properties']['eo:cloud_cover']}")
    print(f"Ag T2 ID: {ag_t2[0]['id']} | Date: {ag_t2[0]['properties']['datetime']} | Clouds: {ag_t2[0]['properties']['eo:cloud_cover']}")
