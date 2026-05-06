#!/usr/bin/env python3
# data/ingest_topology.py  (updated for live mode)
#
# Two modes:
#   python data/ingest_topology.py            → synthetic data (original behaviour)
#   python data/ingest_topology.py --live     → real RIPE RIS stream via ris_live.py
#
# The --live flag launches ris_live.py and also starts posting classified
# events to the local bridge (ws://localhost:8000) so the dashboard updates.

import os
import sys
import json
import math
import random
import asyncio
import argparse
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(".env.local")

TOKEN    = os.getenv("TINYBIRD_TOKEN")
BASE_URL = os.getenv("TINYBIRD_API_URL", "https://api.tinybird.co")
BRIDGE   = os.getenv("BRIDGE_URL", "http://localhost:8000")

# ── Node definitions (unchanged from original) ────────────────────────────────

NODES = {
    "NBO": {"city": "Nairobi",        "country": "KE", "lat": -1.286,  "lng": 36.817,  "gdp": 2100,  "fiber": 0.72, "ixp": 2, "latency": 42,  "gdp_per_capita": 2100,  "ixp_count": 2, "mean_latency_ms": 42},
    "LOS": {"city": "Lagos",          "country": "NG", "lat": 6.524,   "lng": 3.379,   "gdp": 2200,  "fiber": 0.65, "ixp": 1, "latency": 58,  "gdp_per_capita": 2200,  "ixp_count": 1, "mean_latency_ms": 58},
    "JNB": {"city": "Johannesburg",   "country": "ZA", "lat": -26.195, "lng": 28.034,  "gdp": 7400,  "fiber": 0.91, "ixp": 3, "latency": 28,  "gdp_per_capita": 7400,  "ixp_count": 3, "mean_latency_ms": 28},
    "CPT": {"city": "Cape Town",      "country": "ZA", "lat": -33.925, "lng": 18.424,  "gdp": 7100,  "fiber": 0.88, "ixp": 2, "latency": 31,  "gdp_per_capita": 7100,  "ixp_count": 2, "mean_latency_ms": 31},
    "ACC": {"city": "Accra",          "country": "GH", "lat": 5.556,   "lng": -0.197,  "gdp": 2300,  "fiber": 0.58, "ixp": 1, "latency": 67,  "gdp_per_capita": 2300,  "ixp_count": 1, "mean_latency_ms": 67},
    "DAR": {"city": "Dar es Salaam",  "country": "TZ", "lat": -6.792,  "lng": 39.208,  "gdp": 1100,  "fiber": 0.44, "ixp": 1, "latency": 89,  "gdp_per_capita": 1100,  "ixp_count": 1, "mean_latency_ms": 89},
    "ADD": {"city": "Addis Ababa",    "country": "ET", "lat": 9.025,   "lng": 38.747,  "gdp": 900,   "fiber": 0.38, "ixp": 0, "latency": 112, "gdp_per_capita": 900,   "ixp_count": 0, "mean_latency_ms": 112},
    "KIN": {"city": "Kinshasa",       "country": "CD", "lat": -4.322,  "lng": 15.322,  "gdp": 550,   "fiber": 0.21, "ixp": 0, "latency": 134, "gdp_per_capita": 550,   "ixp_count": 0, "mean_latency_ms": 134},
    "KMP": {"city": "Kampala",        "country": "UG", "lat": 0.347,   "lng": 32.582,  "gdp": 850,   "fiber": 0.41, "ixp": 0, "latency": 98,  "gdp_per_capita": 850,   "ixp_count": 0, "mean_latency_ms": 98},
    "LUS": {"city": "Lusaka",         "country": "ZM", "lat": -15.417, "lng": 28.283,  "gdp": 1200,  "fiber": 0.47, "ixp": 1, "latency": 76,  "gdp_per_capita": 1200,  "ixp_count": 1, "mean_latency_ms": 76},
}

TRANSIT_HUBS = {
    1273:  "London",
    174:   "Ashburn",
    3356:  "Dallas",
    6762:  "Frankfurt",
    5511:  "Paris",
    3257:  "Frankfurt",
    1299:  "Stockholm",
    2914:  "San Jose",
}

TROMBONE_THRESHOLD = 2.0

# ── Geodesic ──────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def classify_detour(src, dst, as_path):
    n1, n2 = NODES[src], NODES[dst]
    direct_km = haversine_km(n1["lat"], n1["lng"], n2["lat"], n2["lng"])
    hub_in_path = None
    for asn in as_path:
        if asn in TRANSIT_HUBS:
            hub_in_path = TRANSIT_HUBS[asn]
            break
    if not hub_in_path:
        return None
    actual_km = direct_km * random.uniform(2.1, 5.8)
    ratio = actual_km / direct_km
    if ratio < TROMBONE_THRESHOLD:
        return None
    wasted_ms = round((actual_km - direct_km) / 200)
    return {
        "src": src, "dst": dst, "via": hub_in_path,
        "direct_km": round(direct_km), "actual_km": round(actual_km),
        "ratio": round(ratio, 2), "wasted_ms": wasted_ms,
    }


# ── Tinybird ingest ───────────────────────────────────────────────────────────

def post_to_tinybird(datasource: str, rows: list):
    if not TOKEN:
        return
    ndjson = "\n".join(json.dumps(r) for r in rows)
    url = f"{BASE_URL}/v0/events?name={datasource}"
    try:
        r = requests.post(
            url, data=ndjson,
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/x-ndjson"},
            timeout=5,
        )
        if r.status_code != 202:
            print(f"[tinybird] {datasource} → HTTP {r.status_code}: {r.text[:80]}")
    except Exception as e:
        print(f"[tinybird] error: {e}")


def post_to_bridge(event: dict):
    """Also forward classified events to the local FastAPI bridge."""
    try:
        requests.post(f"{BRIDGE}/ingest", json=event, timeout=2)
    except Exception:
        pass  # bridge may not be running in synthetic mode


# ── Synthetic mode (original behaviour) ───────────────────────────────────────

def generate_synthetic_batch(batch_size=5):
    node_ids = list(NODES.keys())
    events = []
    for _ in range(batch_size):
        src = random.choice(node_ids)
        dst = random.choice([n for n in node_ids if n != src])
        n   = NODES[src]
        # Build a fake AS path that may contain transit hubs
        as_path = [random.randint(10000, 60000) for _ in range(random.randint(2, 5))]
        if random.random() < 0.35:
            as_path.insert(random.randint(0, len(as_path)), random.choice(list(TRANSIT_HUBS.keys())))

        detour = classify_detour(src, dst, as_path)

        event = {
            "ts":              datetime.now(timezone.utc).isoformat(),
            "src_node":        src,
            "src_asn":         as_path[-1],
            "src_op":          f"Synthetic-{src}",
            "dst_node":        dst,
            "prefix":          f"41.{random.randint(0,255)}.{random.randint(0,255)}.0/24",
            "as_path":         " ".join(str(a) for a in as_path),
            "path_len":        len(as_path),
            "collector":       "synthetic",
            "is_trombone":     detour is not None,
            "trombone_via":    detour["via"]       if detour else None,
            "trombone_ratio":  detour["ratio"]     if detour else None,
            "trombone_wasted_ms": detour["wasted_ms"] if detour else None,
            "direct_km":       detour["direct_km"] if detour else None,
            "actual_km":       detour["actual_km"] if detour else None,
        }
        events.append(event)
        if detour:
            print(f"[synthetic] TROMBONE {src}→{dst} via {detour['via']}  ratio={detour['ratio']}  +{detour['wasted_ms']}ms")
    return events


def run_synthetic(interval_s=10):
    print("[ingest] Running in SYNTHETIC mode — use --live for real BGP data")
    while True:
        batch = generate_synthetic_batch(batch_size=random.randint(3, 8))
        post_to_tinybird("bgp_events", batch)
        import time; time.sleep(interval_s)


# ── Live mode ─────────────────────────────────────────────────────────────────

def run_live():
    print("[ingest] Running in LIVE mode — connecting to RIPE RIS")
    print("[ingest] Make sure bridge.py is running: uvicorn data.bridge:app --port 8000")
    # Delegate to ris_live.py
    import subprocess, sys
    subprocess.run([sys.executable, "data/ris_live.py"])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Kijiji BGP ingest")
    parser.add_argument("--live", action="store_true", help="Use real RIPE RIS stream")
    parser.add_argument("--interval", type=int, default=10, help="Synthetic batch interval (s)")
    args = parser.parse_args()

    if args.live:
        run_live()
    else:
        run_synthetic(interval_s=args.interval)