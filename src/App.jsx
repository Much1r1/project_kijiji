import { useState } from "react";
import TopologyMap      from "./components/TopologyMap.jsx";
import TrombonePanel    from "./components/TrombonePanel.jsx";
import PeeringSimulator from "./components/PeeringSimulator.jsx";
import FragilityRank    from "./components/FragilityRank.jsx";
import "./App.css";

export default function App() {
  const [activeProposal, setActiveProposal] = useState(null);
  const [selectedNode,   setSelectedNode]   = useState(null);

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
            <span className="hs-label">DETOURS</span>
            <span className="hs-val red">LIVE</span>
          </div>
          <div className="pulse-dot" />
        </div>
      </header>

      {/* ── Main grid ── */}
      <main className="app-grid">
        {/* Left column */}
        <div className="col-left">
          <TrombonePanel />
          <FragilityRank onNodeSelect={setSelectedNode} />
        </div>

        {/* Centre — topology map */}
        <div className="col-centre">
          <TopologyMap
            activeProposal={activeProposal}
            onNodeClick={setSelectedNode}
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
        <span>Data: RIPE RIS · PeeringDB · World Bank</span>
        <span>Model: GraphSAGE · PyTorch Geometric</span>
      </footer>
    </div>
  );
}