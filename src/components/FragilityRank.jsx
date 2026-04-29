// src/components/FragilityRank.jsx
import { motion } from "framer-motion"

const TIER_COLOR = {
  Critical: "#ef4444",
  High:     "#f59e0b",
  Moderate: "#3b82f6",
  Stable:   "#10b981",
}

export default function FragilityRank({ fragility, onSelect, selectedNode }) {
  if (!fragility?.length) return null

  return (
    <div className="panel fragility-panel">
      <div className="panel-header">
        <span className="panel-title">FRAGILITY INDEX</span>
        <span className="panel-sub">SDG 9.4 Priority</span>
      </div>

      <div className="frag-list">
        {fragility.map((node, i) => {
          const color    = TIER_COLOR[node.tier] || "#64748b"
          const isActive = selectedNode?.id === node.id
          const barW     = `${(node.fragility * 100).toFixed(0)}%`

          return (
            <motion.div
              key={node.id}
              className={`frag-row ${isActive ? "active" : ""}`}
              onClick={() => onSelect?.(node)}
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
              <span className="frag-score" style={{ color }}>
                {node.fragility}
              </span>
            </motion.div>
          )
        })}
      </div>

      <div className="frag-legend">
        {Object.entries(TIER_COLOR).map(([tier, color]) => (
          <div key={tier} className="legend-item">
            <span className="legend-dot" style={{ backgroundColor: color }} />
            <span>{tier}</span>
          </div>
        ))}
      </div>
    </div>
  )
}