// src/components/FragilityRank.jsx
import { motion } from "framer-motion";
import { NODES } from "../data/topology.js";

const TIER_COLOR = {
  CRITICAL: "#ef4444",
  HIGH:     "#f59e0b",
  MEDIUM:   "#3b82f6",
  LOW:      "#10b981",
};

function computeFragility() {
  const gdps      = NODES.map(n => n.gdp);
  const ixps      = NODES.map(n => n.ixp);
  const latencies = NODES.map(n => n.latency);

  const gdpMin = Math.min(...gdps),      gdpMax = Math.max(...gdps);
  const ixpMin = Math.min(...ixps),      ixpMax = Math.max(...ixps);
  const latMin = Math.min(...latencies), latMax = Math.max(...latencies);

  return NODES
    .map(n => {
      const gdpW = 1 - (n.gdp     - gdpMin) / (gdpMax - gdpMin + 1e-8);
      const ixpW = 1 - (n.ixp     - ixpMin) / (ixpMax - ixpMin + 1e-8);
      const latW =     (n.latency  - latMin) / (latMax - latMin + 1e-8);
      const score = 0.40 * gdpW + 0.35 * ixpW + 0.25 * latW;
      const tier  = score > 0.7  ? "CRITICAL"
                  : score > 0.45 ? "HIGH"
                  : score > 0.25 ? "MEDIUM"
                  : "LOW";
      return { id: n.id, city: n.label, score: parseFloat(score.toFixed(3)), tier };
    })
    .sort((a, b) => b.score - a.score);
}

const RANKED = computeFragility();

export default function FragilityRank({ onNodeSelect }) {
  return (
    <div className="panel fragility-panel">
      <div className="panel-header">
        <span className="panel-title">FRAGILITY INDEX</span>
        <span className="panel-sub">SDG 9.4 · GDP⁻¹ × IXP⁻¹ × Latency</span>
      </div>

      <div className="frag-list">
        {RANKED.map((node, i) => {
          const color = TIER_COLOR[node.tier] || "#64748b";
          const barW  = `${(node.score * 100).toFixed(0)}%`;

          return (
            <motion.div
              key={node.id}
              className="frag-row"
              onClick={() => onNodeSelect?.(node)}
              whileHover={{ x: 3 }}
              transition={{ duration: 0.15 }}
            >
              <span className="frag-rank">{i + 1}</span>
              <div className="frag-info">
                <div className="frag-city-row">
                  <span className="frag-id">{node.id}</span>
                  <span className="frag-city-name">{node.city}</span>
                  <span className="frag-tier" style={{ color }}>{node.tier}</span>
                </div>
                <div className="frag-bar-track">
                  <motion.div
                    className="frag-bar-fill"
                    style={{ backgroundColor: color }}
                    initial={{ width: 0 }}
                    animate={{ width: barW }}
                    transition={{ duration: 0.6, delay: i * 0.05 }}
                  />
                </div>
              </div>
              <span className="frag-score" style={{ color }}>{node.score}</span>
            </motion.div>
          );
        })}
      </div>

      <div className="frag-legend">
        {Object.entries(TIER_COLOR).map(([tier, color]) => (
          <div key={tier} className="legend-item">
            <span className="legend-dot" style={{ backgroundColor: color }} />
            <span style={{ fontSize: "8px", color: "var(--text-muted)" }}>{tier}</span>
          </div>
        ))}
      </div>
    </div>
  );
}