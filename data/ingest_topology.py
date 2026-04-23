import os
import json
import requests
import random
from datetime import datetime, timezone
from dotenv import load_dotenv

# Force load the .env.local file
load_dotenv(".env.local")

TOKEN = os.getenv("TINYBIRD_TOKEN")
BASE_URL = os.getenv("TINYBIRD_API_URL", "https://api.tinybird.co")
API_URL = f"{BASE_URL}/v0/events?name=network_edges"

# ─────────────────────────────────────────────
# AFRICAN INTERNET EXCHANGE POINT NODE REGISTRY
# Source shapes: PeeringDB + World Bank (synthetic values)
# Each node = one city with its socio-economic feature vector
# ─────────────────────────────────────────────
NODES = {
    "NBO": {
        "city": "Nairobi",
        "country": "KE",
        "lat": -1.286,
        "lon": 36.817,
        "gdp_per_capita": 2080.0,       # USD, World Bank 2023
        "fiber_index": 0.74,            # Normalized [0,1]
        "ixp_count": 2,                 # KIXP + 1 private
        "mean_latency_ms": 22.0,        # Baseline intra-city
        "asns": [36866, 33771, 15399],  # Safaricom, KENIC, etc.
    },
    "LOS": {
        "city": "Lagos",
        "country": "NG",
        "lat": 6.524,
        "lon": 3.379,
        "gdp_per_capita": 2184.0,
        "fiber_index": 0.61,
        "ixp_count": 1,                 # IXPN
        "mean_latency_ms": 28.0,
        "asns": [29465, 37148, 36351],
    },
    "JNB": {
        "city": "Johannesburg",
        "country": "ZA",
        "lat": -26.204,
        "lon": 28.047,
        "gdp_per_capita": 6994.0,
        "fiber_index": 0.89,
        "ixp_count": 3,                 # JINX, NAPAfrica, etc.
        "mean_latency_ms": 12.0,
        "asns": [3741, 16637, 36937],
    },
    "CPT": {
        "city": "Cape Town",
        "country": "ZA",
        "lat": -33.925,
        "lon": 18.424,
        "gdp_per_capita": 6994.0,
        "fiber_index": 0.85,
        "ixp_count": 1,
        "mean_latency_ms": 14.0,
        "asns": [3741, 37153],
    },
    "ACC": {
        "city": "Accra",
        "country": "GH",
        "lat": 5.603,
        "lon": -0.187,
        "gdp_per_capita": 2363.0,
        "fiber_index": 0.52,
        "ixp_count": 1,                 # GIX
        "mean_latency_ms": 35.0,
        "asns": [29614, 37122],
    },
    "DAR": {
        "city": "Dar es Salaam",
        "country": "TZ",
        "lat": -6.792,
        "lon": 39.208,
        "gdp_per_capita": 1136.0,
        "fiber_index": 0.41,
        "ixp_count": 1,                 # TZIX
        "mean_latency_ms": 42.0,
        "asns": [37182, 36936],
    },
    "ADD": {
        "city": "Addis Ababa",
        "country": "ET",
        "lat": 9.145,
        "lon": 40.489,
        "gdp_per_capita": 925.0,
        "fiber_index": 0.29,
        "ixp_count": 0,                 # No IXP — key research node
        "mean_latency_ms": 68.0,
        "asns": [24757],                # Ethio Telecom monopoly
    },
    "KIN": {
        "city": "Kinshasa",
        "country": "CD",
        "lat": -4.322,
        "lon": 15.322,
        "gdp_per_capita": 577.0,        # Lowest GDP — highest loss weight
        "fiber_index": 0.18,
        "ixp_count": 0,
        "mean_latency_ms": 89.0,
        "asns": [36916, 37342],
    },
    "CMN": {
        "city": "Casablanca",
        "country": "MA",
        "lat": 33.589,
        "lon": -7.604,
        "gdp_per_capita": 3795.0,
        "fiber_index": 0.67,
        "ixp_count": 1,                 # MAG-IX
        "mean_latency_ms": 18.0,
        "asns": [6713, 36925],
    },
    "CPV": {
        "city": "Kampala",
        "country": "UG",
        "lat": 0.347,
        "lon": 32.582,
        "gdp_per_capita": 883.0,
        "fiber_index": 0.33,
        "ixp_count": 1,                 # UIXP
        "mean_latency_ms": 55.0,
        "asns": [36977, 37271],
    },
}

# ─────────────────────────────────────────────
# DETOUR CLASSIFICATION LOGIC
# Mirrors the Rust Trombone Detector logic in Python for now.
# Rust will own this in production.
# ─────────────────────────────────────────────
TRANSIT_HUBS = {
    "LON": {"lat": 51.509, "lon": -0.118},   # London — LINX
    "FRA": {"lat": 50.110, "lon": 8.682},    # Frankfurt — DE-CIX
    "AMS": {"lat": 52.370, "lon": 4.895},    # Amsterdam — AMS-IX
    "PAR": {"lat": 48.857, "lon": 2.347},    # Paris — France-IX
}

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance. Mirrors the Rust geodesic computation."""
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def classify_detour(src_id: str, dst_id: str, via_hub: str | None) -> dict:
    """
    Classifies a routing event into one of three categories:
      - DIRECT:      No detour, regional path used.
      - TROMBONE:    Traffic routed through European hub unnecessarily.
      - POLICY:      Detour exists but is BGP policy-driven (valley-free constraint).

    Returns a dict with detour_type, geodesic_km, bgp_path_km, detour_ratio.
    """
    src = NODES[src_id]
    dst = NODES[dst_id]
    geodesic_km = _haversine_km(src["lat"], src["lon"], dst["lat"], dst["lon"])

    if via_hub is None:
        return {
            "detour_type": "DIRECT",
            "geodesic_km": round(geodesic_km, 2),
            "bgp_path_km": round(geodesic_km * random.uniform(1.0, 1.4), 2),
            "detour_ratio": round(random.uniform(1.0, 1.4), 3),
            "transit_hub": "NONE",
        }

    hub = TRANSIT_HUBS[via_hub]
    bgp_path_km = (
        _haversine_km(src["lat"], src["lon"], hub["lat"], hub["lon"])
        + _haversine_km(hub["lat"], hub["lon"], dst["lat"], dst["lon"])
    )
    detour_ratio = bgp_path_km / geodesic_km if geodesic_km > 0 else 1.0

    # BGP Trombone threshold: ratio > 2.0 (tunable hyperparameter — see thesis §3.2)
    TROMBONE_THRESHOLD = 2.0
    detour_type = "TROMBONE" if detour_ratio > TROMBONE_THRESHOLD else "POLICY"

    return {
        "detour_type": detour_type,
        "geodesic_km": round(geodesic_km, 2),
        "bgp_path_km": round(bgp_path_km, 2),
        "detour_ratio": round(detour_ratio, 3),
        "transit_hub": via_hub,
    }

# ─────────────────────────────────────────────
# SYNTHETIC BGP EVENT GENERATOR
# Generates realistic network_edges events shaped for the GNN feature vector:
# x_i = [gdp_per_capita, fiber_index, ixp_count, mean_latency_ms]
# ─────────────────────────────────────────────
def generate_edge_event() -> dict:
    node_ids = list(NODES.keys())
    src_id, dst_id = random.sample(node_ids, 2)
    src, dst = NODES[src_id], NODES[dst_id]

    # ~40% of African inter-city traffic still routes via European hubs
    # This ratio is the core empirical claim we are measuring
    routes_via_europe = random.random() < 0.40
    via_hub = random.choice(list(TRANSIT_HUBS.keys())) if routes_via_europe else None

    detour = classify_detour(src_id, dst_id, via_hub)

    # Observed latency = geodesic baseline + detour penalty + jitter
    jitter_ms = random.gauss(0, 3.0)
    detour_penalty_ms = (detour["detour_ratio"] - 1.0) * src["mean_latency_ms"] * 8
    observed_latency_ms = round(
        src["mean_latency_ms"] + detour_penalty_ms + jitter_ms, 2
    )

    # Synthetic AS path: src_asn → (optional hub ASN) → dst_asn
    src_asn = random.choice(src["asns"])
    dst_asn = random.choice(dst["asns"])
    as_path = [src_asn, dst_asn] if via_hub is None else [src_asn, 1273, dst_asn]
    # ASN 1273 = Vodafone/CWC — common European transit AS for African traffic

    return {
        # ── Topology identifiers ──
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "src_city": src_id,
        "dst_city": dst_id,
        "src_asn": src_asn,
        "dst_asn": dst_asn,
        "as_path": json.dumps(as_path),

        # ── GNN Node Feature Vector (src) ──
        "src_gdp_per_capita": src["gdp_per_capita"],
        "src_fiber_index": src["fiber_index"],
        "src_ixp_count": src["ixp_count"],
        "src_mean_latency_ms": src["mean_latency_ms"],

        # ── GNN Node Feature Vector (dst) ──
        "dst_gdp_per_capita": dst["gdp_per_capita"],
        "dst_fiber_index": dst["fiber_index"],
        "dst_ixp_count": dst["ixp_count"],
        "dst_mean_latency_ms": dst["mean_latency_ms"],

        # ── Trombone Detector Output ──
        "detour_type": detour["detour_type"],
        "geodesic_km": detour["geodesic_km"],
        "bgp_path_km": detour["bgp_path_km"],
        "detour_ratio": detour["detour_ratio"],
        "transit_hub": detour["transit_hub"],

        # ── Observed Performance ──
        "observed_latency_ms": observed_latency_ms,
    }

# ─────────────────────────────────────────────
# INGESTION LOOP
# ─────────────────────────────────────────────
def ingest_batch(batch_size: int = 20) -> None:
    events = [generate_edge_event() for _ in range(batch_size)]
    ndjson_payload = "\n".join(json.dumps(e) for e in events)

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/x-ndjson",
    }

    response = requests.post(API_URL, data=ndjson_payload, headers=headers)

    if response.status_code == 202:
        trombone_count = sum(1 for e in events if e["detour_type"] == "TROMBONE")
        policy_count = sum(1 for e in events if e["detour_type"] == "POLICY")
        direct_count = sum(1 for e in events if e["detour_type"] == "DIRECT")
        print(f"✅ Ingested {batch_size} events | "
              f"🔴 TROMBONE: {trombone_count} | "
              f"🟡 POLICY: {policy_count} | "
              f"🟢 DIRECT: {direct_count}")
    else:
        print(f"❌ Ingestion failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    if not TOKEN:
        print("❌ Error: TINYBIRD_TOKEN not found in .env.local")
    else:
        print(f"📡 Connecting to: {API_URL}")
        print(f"🔑 Using Token (first 10 chars): {TOKEN[:10]}...")
        print(f"🌍 Nodes loaded: {len(NODES)} African cities")
        print(f"🔁 Generating synthetic BGP topology events...\n")

        # Run 5 batches of 20 = 100 events total
        # Scale this up once schema is validated in Tinybird
        for i in range(5):
            ingest_batch(batch_size=20)

        print(f"\n🏁 Done. Check your Tinybird 'network_edges' Data Source.")
        print(f"   Next step: validate schema → then build GraphSAGE on this shape.")