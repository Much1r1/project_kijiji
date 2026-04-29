import { useEffect, useRef, useState, useCallback } from "react";
import * as d3 from "d3";
import { NODES, EDGES, PROPOSED_PEERS } from "../data/topology.js";

const TYPE_COLOR = {
  direct:   "#00ff88",
  policy:   "#f59e0b",
  trombone: "#ef4444",
  proposed: "#818cf8",
};

const TYPE_DASH = {
  direct:   [],
  policy:   [6, 4],
  trombone: [3, 3],
  proposed: [8, 4],
};

export default function TopologyMap({ activeProposal, onNodeClick }) {
  const canvasRef = useRef(null);
  const simRef    = useRef(null);
  const nodesRef  = useRef([]);
  const edgesRef  = useRef([]);
  const rafRef    = useRef(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  const draw = useCallback((ctx, nodes, edges, width, height, hovered) => {
    ctx.clearRect(0, 0, width, height);

    // Grid background
    ctx.strokeStyle = "rgba(0,255,136,0.04)";
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x < width; x += gridSize) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }
    for (let y = 0; y < height; y += gridSize) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }

    // Draw edges
    edges.forEach(edge => {
      const s = nodes.find(n => n.id === (edge.source.id || edge.source));
      const t = nodes.find(n => n.id === (edge.target.id || edge.target));
      if (!s || !t) return;

      const color = TYPE_COLOR[edge.type] || TYPE_COLOR.direct;
      const dash  = TYPE_DASH[edge.type]  || [];

      // Glow for trombone edges
      if (edge.type === "trombone") {
        ctx.save();
        ctx.strokeStyle = "rgba(239,68,68,0.15)";
        ctx.lineWidth = 8;
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(t.x, t.y); ctx.stroke();
        ctx.restore();
      }

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = edge.type === "trombone" ? 2 : 1.5;
      ctx.globalAlpha = edge.type === "proposed" ? 0.5 : 0.8;
      ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(t.x, t.y); ctx.stroke();
      ctx.restore();

      // Latency label on edge midpoint
      if (edge.latency) {
        const mx = (s.x + t.x) / 2;
        const my = (s.y + t.y) / 2;
        ctx.save();
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.6;
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(`${edge.latency}ms`, mx, my - 4);
        ctx.restore();
      }
    });

    // Draw nodes
    nodes.forEach(node => {
      if (node.x == null) return;
      const isHovered = hovered === node.id;
      const r = isHovered ? 10 : 7;

      // IXP indicator — ring if has IXP
      if (node.ixp > 0) {
        ctx.save();
        ctx.strokeStyle = "#00ff88";
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        ctx.arc(node.x, node.y, r + 5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // Node glow
      const grd = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, r * 2.5);
      grd.addColorStop(0, node.ixp === 0 ? "rgba(239,68,68,0.3)" : "rgba(0,255,136,0.2)");
      grd.addColorStop(1, "transparent");
      ctx.save();
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r * 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Node core
      ctx.save();
      ctx.fillStyle = node.ixp === 0 ? "#7f1d1d" : "#052e16";
      ctx.strokeStyle = node.ixp === 0 ? "#ef4444" : "#00ff88";
      ctx.lineWidth = isHovered ? 2 : 1;
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      // Label
      ctx.save();
      ctx.fillStyle = isHovered ? "#ffffff" : "#a3e6c8";
      ctx.font = `${isHovered ? "bold " : ""}10px 'JetBrains Mono', monospace`;
      ctx.textAlign = "center";
      ctx.fillText(node.id, node.x, node.y - r - 5);
      ctx.restore();
    });

    // Tooltip for hovered node
    if (hovered) {
      const node = nodes.find(n => n.id === hovered);
      if (node) {
        const pad = 10;
        const tw = 160;
        const th = 80;
        const tx = Math.min(node.x + 14, (canvasRef.current?.width || 800) - tw - pad);
        const ty = Math.min(node.y - 10, (canvasRef.current?.height || 600) - th - pad);

        ctx.save();
        ctx.fillStyle = "rgba(0,0,0,0.92)";
        ctx.strokeStyle = "#00ff88";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(tx, ty, tw, th, 4);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#00ff88";
        ctx.font = "bold 11px 'JetBrains Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText(node.label, tx + pad, ty + 18);

        ctx.fillStyle = "#6b7280";
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.fillText(`GDP/cap  $${node.gdp.toLocaleString()}`, tx + pad, ty + 34);
        ctx.fillText(`Latency  ${node.latency}ms`, tx + pad, ty + 47);
        ctx.fillText(`IXP pts  ${node.ixp}`, tx + pad, ty + 60);
        ctx.fillText(`Fiber    ${(node.fiber * 100).toFixed(0)}%`, tx + pad, ty + 73);
        ctx.restore();
      }
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const resize = () => {
      canvas.width  = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    // Clone data for D3 (it mutates objects)
    const nodes = NODES.map(n => ({ ...n }));
    const edges = EDGES.map(e => ({ ...e }));

    // Add proposed edge if active
    if (activeProposal != null && PROPOSED_PEERS[activeProposal]) {
      const p = PROPOSED_PEERS[activeProposal];
      edges.push({ source: p.source, target: p.target, type: "proposed", latency: null });
    }

    nodesRef.current = nodes;
    edgesRef.current = edges;

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(edges).id(d => d.id).distance(120).strength(0.6))
      .force("charge", d3.forceManyBody().strength(-320))
      .force("center", d3.forceCenter(canvas.width / 2, canvas.height / 2))
      .force("collision", d3.forceCollide(28))
      .on("tick", () => {
        draw(ctx, nodesRef.current, edgesRef.current, canvas.width, canvas.height, null);
      });

    simRef.current = sim;

    return () => {
      sim.stop();
      window.removeEventListener("resize", resize);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [activeProposal, draw]);

  // Mouse interaction
  const getNodeAt = useCallback((x, y) => {
    return nodesRef.current.find(n => {
      const dx = n.x - x;
      const dy = n.y - y;
      return Math.sqrt(dx * dx + dy * dy) < 14;
    });
  }, []);

  const handleMouseMove = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const node = getNodeAt(x, y);
    const id = node ? node.id : null;
    setHoveredNode(id);
    canvasRef.current.style.cursor = id ? "pointer" : "default";

    const ctx = canvasRef.current.getContext("2d");
    draw(ctx, nodesRef.current, edgesRef.current,
      canvasRef.current.width, canvasRef.current.height, id);
  }, [draw, getNodeAt]);

  const handleClick = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const node = getNodeAt(e.clientX - rect.left, e.clientY - rect.top);
    if (node && onNodeClick) onNodeClick(node);
  }, [getNodeAt, onNodeClick]);

  return (
    <div className="topology-wrap">
      <div className="panel-header">
        <span className="panel-title">NETWORK TOPOLOGY</span>
        <div className="legend">
          {Object.entries(TYPE_COLOR).map(([k, v]) => (
            <span key={k} className="legend-item">
              <span className="legend-dot" style={{ background: v }} />
              {k.toUpperCase()}
            </span>
          ))}
        </div>
      </div>
      <canvas
        ref={canvasRef}
        className="topology-canvas"
        onMouseMove={handleMouseMove}
        onClick={handleClick}
      />
    </div>
  );
}