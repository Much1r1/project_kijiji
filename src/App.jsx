// src/App.jsx  (updated — live BGP feed via useLiveBGP hook)
import { useState } from "react"
import TopologyMap      from "./components/TopologyMap.jsx"
import TrombonePanel    from "./components/TrombonePanel.jsx"
import PeeringSimulator from "./components/PeeringSimulator.jsx"
import FragilityRank    from "./components/FragilityRank.jsx"
import { useLiveBGP }   from "./hooks/useLiveBGP.js"
import "./App.css"

export default function App() {
  const [activeProposal, setActiveProposal] = useState(null)
  const [selectedNode,   setSelectedNode]   = useState(null)

  const { events, trombones, stats, isLive, status } = useLiveBGP()

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-left">
          <span className="logo-mark">◈</span>
          <div>
            <h1 className="app-title">KIJIJI</h1>
            <p className="app-sub">African IXP Routing Intelligence Platform</p>
          </div>
        </div>
        <div className="header-right">
          <div className="header-stat">
            <span className="hs-label">MODEL</span>
            <span className="hs-val">GraphSAGE v0.1</span>
          </div>
          <div className="header-stat">
            <span className="hs-label">NODES</span>
            <span className="hs-val green">10</span>
          </div>
          <div className="header-stat">
            <span className="hs-label">EVENTS</span>
            <span className="hs-val green">{stats.total}</span>
          </div>
          <div className="header-stat">
            <span className="hs-label">DETOURS</span>
            <span className="hs-val red">{stats.tromboneCount}</span>
          </div>

          {/* Live / Synthetic status pill */}
          <div className={`live-badge ${isLive ? "live-badge--live" : "live-badge--synthetic"}`}>
            <span className="live-dot" />
            {isLive ? "LIVE · RIS" : status === "connecting" ? "CONNECTING" : "SYNTHETIC"}
          </div>

          <div className="pulse-dot" />
        </div>
      </header>

      {/* ── Main grid ── */}
      <main className="app-grid">
        {/* Left column */}
        <div className="col-left">
          <TrombonePanel trombones={trombones} />
          <FragilityRank onNodeSelect={setSelectedNode} liveEvents={events} />
        </div>

        {/* Centre — topology map */}
        <div className="col-centre">
          <TopologyMap
            activeProposal={activeProposal}
            onNodeClick={setSelectedNode}
            liveEvents={events}
          />
          {selectedNode && (
            <div className="node-detail">
              <button className="close-btn" onClick={() => setSelectedNode(null)}>✕</button>
              <span className="nd-id">{selectedNode.id}</span>
              <span className="nd-name">{selectedNode.label}</span>
              <div className="nd-stats">
                <span>Latency <strong>{selectedNode.latency}ms</strong></span>
                <span>IXP pts <strong>{selectedNode.ixp}</strong></span>
                <span>Fiber <strong>{(selectedNode.fiber * 100).toFixed(0)}%</strong></span>
                <span>GDP/cap <strong>${selectedNode.gdp.toLocaleString()}</strong></span>
              </div>
            </div>
          )}
        </div>

        {/* Right column */}
        <div className="col-right">
          <PeeringSimulator onProposalChange={setActiveProposal} />
        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer">
        <span>KIJIJI · SDG 9.4 Research Platform · DAAD 2025</span>
        <span>Data: {isLive ? "RIPE RIS Live · PeeringDB · World Bank" : "Synthetic · RIPE RIS topology"}</span>
        <span>Model: GraphSAGE · PyTorch Geometric</span>
      </footer>
    </div>
  )
}