#!/usr/bin/env python3
# data/ris_live.py
#
# Connects to RIPE RIS Live WebSocket, subscribes to African ASN prefixes,
# classifies trombone detours, and streams events to Tinybird.
#
# Usage:
#   pip install websockets aiohttp python-dotenv
#   python data/ris_live.py
#
# Environment variables (in .env.local):
#   TINYBIRD_TOKEN   — your Tinybird API token
#   TINYBIRD_API_URL — defaults to https://api.tinybird.co

import os
import json
import math
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
import aiohttp
import websockets
from dotenv import load_dotenv

from asn_map import (
    ASN_TO_NODE,
    TRACKED_ASNS,
    TRANSIT_HUBS,
    as_path_contains_hub,
    asn_to_node,
)

load_dotenv(".env.local")

TOKEN       = os.getenv("TINYBIRD_TOKEN")
BASE_URL    = os.getenv("TINYBIRD_API_URL", "https://api.tinybird.co")
CLIENT_ID   = "project-kijiji-ris-client-v1"
RIS_WS_URL  = f"wss://ris-live.ripe.net/v1/ws/?client={CLIENT_ID}"

# RIS route collectors with good African BGP visibility
# rrc01 = London, rrc03 = Amsterdam, rrc04 = Geneva,
# rrc11 = New York, rrc12 = Frankfurt, rrc22 = Nairobi (best for us)
COLLECTORS = ["rrc22", "rrc01", "rrc12"]

# Approximate city coordinates for geodesic distance calc
CITY_COORDS = {
    "NBO": (-1.286,  36.817),
    "LOS": ( 6.524,   3.379),
    "JNB": (-26.195, 28.034),
    "CPT": (-33.925, 18.424),
    "ACC": ( 5.556,  -0.197),
    "DAR": (-6.792,  39.208),
    "ADD": ( 9.025,  38.747),
    "KIN": (-4.322,  15.322),
    "KMP": ( 0.347,  32.582),
    "LUS": (-15.417, 28.283),
    # Transit hub approximate coords
    "London":    (51.509, -0.118),
    "Paris":     (48.857,  2.352),
    "Frankfurt": (50.110,  8.682),
    "Amsterdam": (52.370,  4.895),
    "New York":  (40.713, -74.006),
    "Atlanta":   (33.749, -84.388),
    "Stockholm": (59.334, 18.065),
}

TROMBONE_RATIO_THRESHOLD = 2.0  # flag if actual path is 2× the direct distance
SPEED_OF_LIGHT_KM_MS     = 200  # ~200 km/ms in fibre (accounts for refraction)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ris_live")


# ── Geodesic helpers ──────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def estimate_path_km(as_path: list[int], src_node: str, dst_node: str) -> float:
    """
    Estimate the physical path length by summing geodesic segments
    through any known intermediate transit hubs in the AS path.
    Falls back to direct distance if no hub coordinates are found.
    """
    # Build waypoint list: src -> any hub coords -> dst
    waypoints = [CITY_COORDS[src_node]]

    for asn in as_path:
        hub_name = TRANSIT_HUBS.get(asn)
        if hub_name:
            # Extract city name (before the parenthesis)
            city = hub_name.split(" (")[0]
            if city in CITY_COORDS:
                waypoints.append(CITY_COORDS[city])

    waypoints.append(CITY_COORDS[dst_node])

    total = 0.0
    for i in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[i]
        lat2, lon2 = waypoints[i + 1]
        total += haversine_km(lat1, lon1, lat2, lon2)
    return total


def classify_detour(
    src_node: str,
    dst_node: str,
    as_path: list[int],
) -> Optional[dict]:
    """
    Returns a trombone event dict if the path is a detour, else None.
    Mirrors the logic in engine/src/trombone.rs.
    """
    if src_node not in CITY_COORDS or dst_node not in CITY_COORDS:
        return None

    lat1, lon1 = CITY_COORDS[src_node]
    lat2, lon2 = CITY_COORDS[dst_node]
    direct_km  = haversine_km(lat1, lon1, lat2, lon2)

    if direct_km < 100:  # same metro, skip
        return None

    hub_name  = as_path_contains_hub(as_path)
    actual_km = estimate_path_km(as_path, src_node, dst_node)
    ratio     = actual_km / direct_km if direct_km > 0 else 1.0

    if ratio < TROMBONE_RATIO_THRESHOLD:
        return None

    wasted_ms = round((actual_km - direct_km) / SPEED_OF_LIGHT_KM_MS)

    return {
        "src":        src_node,
        "dst":        dst_node,
        "via":        hub_name or "unknown",
        "direct_km":  round(direct_km),
        "actual_km":  round(actual_km),
        "ratio":      round(ratio, 2),
        "wasted_ms":  wasted_ms,
        "as_path":    as_path,
    }


# ── Tinybird ingest ───────────────────────────────────────────────────────────

async def post_to_tinybird(session: aiohttp.ClientSession, datasource: str, rows: list[dict]):
    """
    POST NDJSON rows to a Tinybird Events API datasource.
    Silently skips if TOKEN is not set (dev mode).
    """
    if not TOKEN:
        log.debug("No TINYBIRD_TOKEN — skipping ingest for %s", datasource)
        return

    ndjson = "\n".join(json.dumps(r) for r in rows)
    url    = f"{BASE_URL}/v0/events?name={datasource}"

    try:
        async with session.post(
            url,
            data=ndjson,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type":  "application/x-ndjson",
            },
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 202:
                body = await resp.text()
                log.warning("Tinybird %s → HTTP %d: %s", datasource, resp.status, body[:120])
    except Exception as e:
        log.warning("Tinybird ingest failed: %s", e)


# ── BGP message processor ─────────────────────────────────────────────────────

def extract_asns_from_path(raw_path) -> list[int]:
    """
    Parse AS path from RIS Live message.
    Handles both list-of-ints and space-separated string formats.
    Strips AS sets ({}) and prepend duplicates.
    """
    if not raw_path:
        return []
    if isinstance(raw_path, str):
        parts = raw_path.split()
    else:
        parts = [str(p) for p in raw_path]

    result = []
    seen   = set()
    for p in parts:
        # skip AS sets like {12345,67890}
        p = p.strip("{}")
        if "," in p:
            continue
        try:
            asn = int(p)
            if asn not in seen:
                result.append(asn)
                seen.add(asn)
        except ValueError:
            continue
    return result


def process_bgp_update(msg_data: dict) -> Optional[dict]:
    """
    Process a single ris_message BGP update.
    Returns a normalised event dict or None if not relevant to our topology.
    """
    msg_type = msg_data.get("type")
    if msg_type not in ("UPDATE", "ANNOUNCE"):
        return None

    body       = msg_data.get("body", {})
    path_raw   = body.get("path", [])
    as_path    = extract_asns_from_path(path_raw)
    prefix     = body.get("prefix", "")
    timestamp  = msg_data.get("timestamp", datetime.now(timezone.utc).timestamp())
    peer_asn   = int(msg_data.get("peer_asn", 0))
    host       = msg_data.get("host", "")  # which RRC collector

    if not as_path:
        return None

    # Find African ASNs in this path
    src_asn = as_path[-1]   # origin (rightmost)
    dst_asn = as_path[0]    # destination perspective (leftmost = peer)

    src_info = asn_to_node(src_asn)
    dst_info = asn_to_node(dst_asn) or asn_to_node(peer_asn)

    if not src_info:
        return None  # origin not in our topology

    src_node = src_info["node"]
    dst_node = dst_info["node"] if dst_info else None

    # Classify trombone if we have both endpoints
    trombone = None
    if dst_node and dst_node != src_node:
        trombone = classify_detour(src_node, dst_node, as_path)

    event = {
        "ts":         datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
        "src_node":   src_node,
        "src_asn":    src_asn,
        "src_op":     src_info["operator"],
        "dst_node":   dst_node or "UNK",
        "prefix":     prefix,
        "as_path":    " ".join(str(a) for a in as_path),
        "path_len":   len(as_path),
        "collector":  host,
        "is_trombone": trombone is not None,
        "trombone_via":      trombone["via"]       if trombone else None,
        "trombone_ratio":    trombone["ratio"]     if trombone else None,
        "trombone_wasted_ms": trombone["wasted_ms"] if trombone else None,
        "direct_km":  trombone["direct_km"]  if trombone else None,
        "actual_km":  trombone["actual_km"]  if trombone else None,
    }

    if trombone:
        log.info(
            "TROMBONE  %s → %s  via %s  ratio=%.2f  +%dms wasted",
            src_node, dst_node, trombone["via"], trombone["ratio"], trombone["wasted_ms"],
        )
    else:
        log.debug("UPDATE  %s → %s  prefix=%s  path_len=%d",
                  src_node, dst_node or "?", prefix, len(as_path))

    return event


# ── WebSocket connection + subscription ───────────────────────────────────────
def build_subscriptions() -> list[dict]:
    subs = []
    for collector in COLLECTORS:
        # Subscribe to all messages touching any of our tracked ASNs
        # RIS Live supports 'path' filter: match ASN anywhere in AS path
        for asn in list(TRACKED_ASNS)[:20]:  # RIS Live recommends batching
            subs.append({
                "type": "ris_subscribe",
                "data": {
                    "host":    collector,
                    "path":    str(asn),
                    "type":    "UPDATE",
                    "require": "announcements",
                },
            })
    return subs


async def run_ris_stream(session: aiohttp.ClientSession):
    """
    Main WebSocket loop. Connects, subscribes, processes messages forever.
    Reconnects automatically on disconnect.
    """
    buffer      = []
    buffer_size = 20   # batch size for Tinybird ingest
    flush_every = 30   # also flush every N seconds regardless

    async def flush():
        if buffer:
            await post_to_tinybird(session, "bgp_events", buffer.copy())
            buffer.clear()

    log.info("Connecting to RIS Live: %s", RIS_WS_URL)

    async with websockets.connect(
        RIS_WS_URL,
        ping_interval=20,
        ping_timeout=10,
        max_size=2**20,  # 1MB max message
    ) as ws:
        log.info("Connected. Sending subscriptions for %d collectors...", len(COLLECTORS))

        for sub in build_subscriptions():
            await ws.send(json.dumps(sub))
            await asyncio.sleep(0.05)  # rate-limit subscription sends

        log.info("Subscriptions sent. Waiting for BGP events...")
        last_flush = asyncio.get_event_loop().time()
        msg_count  = 0

        async for raw_msg in ws:
            try:
                envelope = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            msg_type = envelope.get("type")

            # ACK from server
            if msg_type == "ris_subscribed":
                log.info("Subscription ACK: %s", envelope.get("data", {}).get("host", "?"))
                continue

            # Error from server
            if msg_type == "ris_error":
                log.warning("RIS error: %s", envelope.get("data", ""))
                continue

            # Actual BGP message
            if msg_type == "ris_message":
                event = process_bgp_update(envelope.get("data", {}))
                if event:
                    buffer.append(event)
                    msg_count += 1

                    if len(buffer) >= buffer_size:
                        await flush()

            # Periodic flush
            now = asyncio.get_event_loop().time()
            if now - last_flush > flush_every:
                await flush()
                last_flush = now
                if msg_count > 0:
                    log.info("Stats: %d relevant BGP events processed", msg_count)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    log.info("Project Kijiji — RIS Live BGP Ingestion")
    log.info("Tracking %d African ASNs across %d cities", len(TRACKED_ASNS), 10)
    if not TOKEN:
        log.warning("TINYBIRD_TOKEN not set — events will be classified but NOT sent to Tinybird")
    log.info("Tinybird target: %s", BASE_URL)

    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            try:
                await run_ris_stream(session)
            except websockets.exceptions.ConnectionClosed as e:
                log.warning("WebSocket closed (%s) — reconnecting in 5s...", e)
                await asyncio.sleep(5)
            except Exception as e:
                log.error("Unexpected error: %s — reconnecting in 10s...", e)
                await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())