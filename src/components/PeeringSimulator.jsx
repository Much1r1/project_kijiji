import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { PROPOSED_PEERS, NODES } from "../data/topology.js";

const IMPACT_COLOR = { Critical: "#ef4444", High: "#f59e0b", Medium: "#00ff88" };
const COST_COLOR   = { High: "#ef4444",     Medium: "#f59e0b", Low: "#00ff88" };

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="tt-label">{payload[0].payload.city}</p>
      <p className="tt-val">{payload[0].value}ms reduction</p>
    </div>
  );
}

export default function PeeringSimulator({ onProposalChange }) {
  const [selected, setSelected] = useState(null);

  const handleSelect = (idx) => {
    const next = selected === idx ? null : idx;
    setSelected(next);
    onProposalChange(next);
  };

  const proposal = selected != null ? PROPOSED_PEERS[selected] : null;

  // Build bar chart data from affected cities
  const chartData = proposal
    ? proposal.affectedCities.map(id => {
        const node = NODES.find(n => n.id === id);
        return {
          city: node?.label || id,
          reduction: Math.round(proposal.predictedLatencyReduction * (0.6 + Math.random() * 0.8)),
        };
      })
    : [];

  return (
    <div className="panel peering-panel">
      <div className="panel-header">
        <span className="panel-title">PEERING SIMULATOR</span>
        <span className="panel-sub">GNN · GraphSAGE Link Predictor</span>
      </div>

      <p className="sim-hint">SELECT A PROPOSED PEERING LINK TO SIMULATE REGIONAL LATENCY DIVIDEND</p>

      <div className="proposal-list">
        {PROPOSED_PEERS.map((p, i) => (
          <div
            key={i}
            className={`proposal-card ${selected === i ? "proposal-active" : ""}`}
            onClick={() => handleSelect(i)}
          >
            <div className="proposal-top">
              <span className="proposal-label">{p.label}</span>
              <span className="impact-badge" style={{ color: IMPACT_COLOR[p.sdgImpact] }}>
                {p.sdgImpact}
              </span>
            </div>
            <div className="proposal-metrics">
              <span className="metric">
                <span className="metric-label">Δ LATENCY</span>
                <span className="metric-val green">−{p.predictedLatencyReduction}ms</span>
              </span>
              <span className="metric">
                <span className="metric-label">DIVIDEND</span>
                <span className="metric-val">{p.regionalDividend}×</span>
              </span>
              <span className="metric">
                <span className="metric-label">COST</span>
                <span className="metric-val" style={{ color: COST_COLOR[p.cost] }}>{p.cost}</span>
              </span>
            </div>
          </div>
        ))}
      </div>

      {proposal && (
        <div className="dividend-chart">
          <p className="chart-title">REGIONAL LATENCY DIVIDEND — {proposal.label}</p>
          <ResponsiveContainer width="100%" height={120}>
            <BarChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <XAxis dataKey="city" tick={{ fill: "#6b7280", fontSize: 9, fontFamily: "monospace" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#6b7280", fontSize: 9, fontFamily: "monospace" }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="reduction" radius={[2, 2, 0, 0]}>
                {chartData.map((_, idx) => (
                  <Cell key={idx} fill={idx === 0 ? "#ef4444" : "#00ff88"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="sdg-note">
            ⚡ SDG 9.4 — Inverse-GDP loss weighting prioritises {proposal.affectedCities[0]} cluster
          </p>
        </div>
      )}
    </div>
  );
}