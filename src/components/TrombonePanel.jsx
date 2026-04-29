import { useState, useEffect } from "react";
import { TROMBONE_EVENTS } from "../data/topology.js";

const CLASS_COLOR = {
  TROMBONE: "#ef4444",
  POLICY:   "#f59e0b",
  DIRECT:   "#00ff88",
};

function RatioBar({ ratio, max = 6 }) {
  const pct = Math.min((ratio / max) * 100, 100);
  const color = ratio > 4 ? "#ef4444" : ratio > 2.5 ? "#f59e0b" : "#00ff88";
  return (
    <div className="ratio-bar-wrap">
      <div className="ratio-bar-bg">
        <div className="ratio-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="ratio-val" style={{ color }}>{ratio.toFixed(2)}×</span>
    </div>
  );
}

export default function TrombonePanel() {
  const [events, setEvents] = useState(TROMBONE_EVENTS);
  const [tick, setTick]     = useState(0);

  // Simulate live stream — add synthetic events periodically
  useEffect(() => {
    const vias = ["London", "Frankfurt", "Paris", "Amsterdam"];
    const pairs = [
      ["KIN","NBO"], ["ADD","ACC"], ["KMP","LOS"], ["LUS","KIN"],
    ];
    const id = setInterval(() => {
      const [src, dst] = pairs[Math.floor(Math.random() * pairs.length)];
      const direct = 1500 + Math.random() * 3000;
      const ratio  = 2.0 + Math.random() * 3.5;
      setEvents(prev => [{
        src, dst,
        via:      vias[Math.floor(Math.random() * vias.length)],
        directKm: Math.round(direct),
        actualKm: Math.round(direct * ratio),
        ratio:    parseFloat(ratio.toFixed(2)),
        wastedMs: Math.round((ratio - 1) * 28),
        fresh:    true,
      }, ...prev.slice(0, 11)]);
      setTick(t => t + 1);
    }, 3200);
    return () => clearInterval(id);
  }, []);

  const totalWasted = events.reduce((s, e) => s + e.wastedMs, 0);

  return (
    <div className="panel trombone-panel">
      <div className="panel-header">
        <span className="panel-title">TROMBONE DETECTOR</span>
        <span className="live-badge">● LIVE</span>
      </div>

      <div className="trombone-stats">
        <div className="stat-chip">
          <span className="stat-label">DETOURS</span>
          <span className="stat-val red">{events.length}</span>
        </div>
        <div className="stat-chip">
          <span className="stat-label">AVG RATIO</span>
          <span className="stat-val amber">
            {(events.reduce((s,e)=>s+e.ratio,0)/events.length).toFixed(2)}×
          </span>
        </div>
        <div className="stat-chip">
          <span className="stat-label">MS WASTED</span>
          <span className="stat-val red">{totalWasted}</span>
        </div>
      </div>

      <div className="event-feed">
        {events.map((ev, i) => (
          <div key={i} className={`event-row ${ev.fresh && i === 0 ? "event-fresh" : ""}`}>
            <div className="event-route">
              <span className="node-id">{ev.src}</span>
              <span className="arrow">→</span>
              <span className="node-id dim">via {ev.via}</span>
              <span className="arrow">→</span>
              <span className="node-id">{ev.dst}</span>
            </div>
            <div className="event-meta">
              <span className="meta-km">{ev.directKm.toLocaleString()}km direct</span>
              <span className="meta-sep">/</span>
              <span className="meta-km red">{ev.actualKm.toLocaleString()}km actual</span>
            </div>
            <RatioBar ratio={ev.ratio} />
            <span className="wasted-badge">+{ev.wastedMs}ms wasted</span>
          </div>
        ))}
      </div>
    </div>
  );
}