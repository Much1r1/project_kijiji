#!/usr/bin/env python3
# data/bridge.py
#
# FastAPI WebSocket bridge: relays classified BGP events from ris_live.py
# to the React dashboard in real time.
#
# Run alongside ris_live.py:
#   uvicorn data.bridge:app --host 0.0.0.0 --port 8000 --reload
#
# Dashboard connects to: ws://localhost:8000/ws/events

import asyncio
import json
import os
from collections import deque
from datetime import datetime, timezone
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Kijiji BGP Bridge", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Event bus ─────────────────────────────────────────────────────────────────
# In-memory ring buffer of recent events (survives page reloads)
MAX_HISTORY = 200
event_history: deque = deque(maxlen=MAX_HISTORY)

# Connected dashboard clients
connected_clients: Set[WebSocket] = set()

# Stats counters
stats = {
    "total_events":     0,
    "trombone_events":  0,
    "unique_nodes_seen": set(),
    "started_at":       datetime.now(timezone.utc).isoformat(),
}


async def broadcast(event: dict):
    """Send an event to all connected WebSocket clients."""
    stats["total_events"] += 1
    if event.get("is_trombone"):
        stats["trombone_events"] += 1
    stats["unique_nodes_seen"].add(event.get("src_node", ""))

    event_history.append(event)
    dead = set()

    for client in connected_clients:
        try:
            await client.send_json(event)
        except Exception:
            dead.add(client)

    connected_clients.difference_update(dead)


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "clients":         len(connected_clients),
        "total_events":    stats["total_events"],
        "trombone_events": stats["trombone_events"],
        "nodes_seen":      list(stats["unique_nodes_seen"]),
        "started_at":      stats["started_at"],
        "uptime_s":        (
            datetime.now(timezone.utc) -
            datetime.fromisoformat(stats["started_at"])
        ).seconds,
    }


@app.get("/history")
async def history(limit: int = 50, trombones_only: bool = False):
    """Return recent event history (useful for dashboard cold-start)."""
    events = list(event_history)
    if trombones_only:
        events = [e for e in events if e.get("is_trombone")]
    return {"events": events[-limit:], "total": len(events)}


@app.post("/ingest")
async def ingest(event: dict):
    """
    HTTP endpoint for ris_live.py to POST classified events.
    Can also be called from Tinybird pipes via webhook.
    """
    event["received_at"] = datetime.now(timezone.utc).isoformat()
    await broadcast(event)
    return {"status": "ok", "clients_notified": len(connected_clients)}


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    """
    Dashboard connects here to receive live BGP events.
    On connect: sends last 50 events from history (cold-start hydration).
    Then streams new events as they arrive.
    """
    await websocket.accept()
    connected_clients.add(websocket)

    # Send connection confirmation + history
    await websocket.send_json({
        "type":    "connected",
        "history": list(event_history)[-50:],
        "stats": {
            "total_events":    stats["total_events"],
            "trombone_events": stats["trombone_events"],
        },
    })

    try:
        while True:
            # Keep connection alive; actual data pushed via broadcast()
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)