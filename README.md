# Project Kijiji 🌍
### Distributed Network Observability & Predictive Modeling Platform

> *"How can Graph Neural Networks predict optimal peering points to mitigate routing detours in Sub-Saharan digital infrastructure?"*

A research-grade platform that maps, detects, and predicts African Internet routing inefficiencies — quantifying the latency and economic cost of traffic that unnecessarily routes through European hubs before reaching its destination.

Built as a DAAD scholarship research project targeting **SDG 9.4** (resilient infrastructure).

---

## Table of Contents
1. [The Research Problem](#the-research-problem)
2. [Architecture Overview](#architecture-overview)
3. [Module Breakdown](#module-breakdown)
4. [Tech Stack](#tech-stack)
5. [Data Sources](#data-sources)
6. [Getting Started](#getting-started)
7. [File Structure](#file-structure)
8. [Research Pillars](#research-pillars)
9. [Roadmap](#roadmap)

---

## The Research Problem

A packet sent from Nairobi to Lagos — two African cities ~4,000km apart — often routes through Frankfurt or London before arriving. This "trombone routing" pattern is a legacy of colonial-era infrastructure and the lack of direct IXP (Internet Exchange Point) peering agreements between African ASes (Autonomous Systems).

**The cost is measurable:**
- Added latency: 100–200ms per round trip
- Economic waste: transit fees paid to European carriers
- Human cost: degraded access to cloud services, real-time applications, and digital economy tools for populations with the least connectivity alternatives

This platform answers: *where should new peering agreements be established, and what is the predicted regional latency dividend if they were?*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    DASHBOARD (React)                     │
│         3D Topology Map │ Trombone Panel │ Simulator     │
└──────────────┬──────────────────────────┬───────────────┘
               │  REST/WebSocket          │  GNN Inference
┌──────────────▼──────────┐  ┌────────────▼──────────────┐
│    RUST ENGINE           │  │    PYTHON GNN CORE         │
│  Trombone detection      │  │  GraphSAGE encoder         │
│  Geodesic computation    │  │  Peering simulator         │
│  BGP path parsing        │  │  Weighted loss training    │
└──────────────┬──────────┘  └────────────┬──────────────┘
               │  Events (NDJSON)          │  Features
┌──────────────▼──────────────────────────▼──────────────┐
│                   CLICKHOUSE                            │
│        bgp_events │ city_metrics │ as_to_geo_mapping    │
└──────────────────────────────────────────────────────────┘
               ▲
               │ Ingestion
┌──────────────┴──────────┐
│   DATA SOURCES           │
│  RIPE RIS / RouteViews   │
│  PeeringDB API           │
│  World Bank API          │
│  CAIDA ITDK              │
└─────────────────────────┘
```

---

## Module Breakdown

### 1. The Trombone Detector (`data/` + `engine/`)
Compares actual BGP paths against geodesic optimal routes and classifies detours:

| Class | Meaning | Detection Logic |
|---|---|---|
| `DIRECT` | No detour | `bgp_path ≤ 1.4 × geodesic` |
| `POLICY` | Intentional routing (BGP valley-free) | `1.4 < ratio ≤ 2.0` |
| `TROMBONE` | Unnecessary European transit | `ratio > 2.0` |

The `2.0` threshold is a **tunable hyperparameter** (`TROMBONE_THRESHOLD`) calibrated against known detour cases from CAIDA studies.

### 2. The Peering Recommendation Engine (`model/`)
The core GNN. Given the current African AS topology as a graph, it predicts the **Regional Latency Dividend** of adding a new peering link between two cities — not just for those cities, but for the entire neighboring cluster.

**Model:** GraphSAGE (inductive, works on unseen ASes)
**Node feature vector:** `[gdp_per_capita, fiber_index, ixp_count, mean_latency_ms]`
**Loss function:** Inverse-GDP weighted MSE — the model penalizes errors more heavily in low-GDP, high-population nodes

**Baselines:**
- BGP Valley-Free Path Model (policy compliance baseline)
- Dijkstra Shortest Geodesic Path (geometric ceiling)

### 3. The Fragility Forecast (`model/evaluate.py`)
Correlates IXP adoption rates and World Bank connectivity indices with routing quality metrics. Replaces the original sentiment-scraping approach with structured, citable data from authoritative sources.

---

## Tech Stack

| Layer | Technology | Justification |
|---|---|---|
| ML / GNN | PyTorch Geometric | GraphSAGE, SEAL implementations |
| Data Plane | Rust + Petgraph | Systems-level path computation |
| Database | ClickHouse | High-cardinality BGP event streams |
| Ingestion (proto) | Tinybird | Rapid schema validation |
| Frontend | React + React-Force-Graph | Interactive topology visualization |
| BGP Data | RIPE RIS / RouteViews | Public BGP collector feeds |
| IXP Data | PeeringDB API | IXP membership and capacity |
| Socio-economic | World Bank API | GDP, connectivity indices |
| AS Geolocation | CAIDA ITDK | AS-to-city mapping with confidence scores |

---

## Data Sources

| Source | What we pull | Table |
|---|---|---|
| [RIPE NCC RIS](https://ris.ripe.net/) | BGP updates, AS paths | `bgp_events` |
| [RouteViews](http://www.routeviews.org/) | Alternative BGP collector | `bgp_events` |
| [PeeringDB](https://www.peeringdb.com/api/) | IXP count, capacity per city | `city_metrics` |
| [World Bank API](https://data.worldbank.org/) | GDP per capita | `city_metrics` |
| [CAIDA ITDK](https://www.caida.org/catalog/datasets/itdk/) | AS → geographic mapping | `as_to_geo_mapping` |

> **Note:** `data/ingest_topology.py` currently generates **synthetic** data that mirrors the shape of real BGP events. This is intentional for schema validation before connecting live feeds.

---

## Getting Started

### Prerequisites
```bash
python >= 3.11
rust >= 1.75
node >= 20
```

### Python environment
```bash
cd project_kijiji
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables
```bash
cp .env.example .env.local
# Fill in:
# TINYBIRD_TOKEN=...
# TINYBIRD_API_URL=https://api.europe-west2.gcp.tinybird.co
# WORLD_BANK_API_KEY=...  (optional, public endpoints don't need one)
# PEERINGDB_TOKEN=...
```

### Run the ingestion (synthetic mode)
```bash
python data/ingest_topology.py
# Expected output:
# ✅ Ingested 20 events | 🔴 TROMBONE: 8 | 🟡 POLICY: 4 | 🟢 DIRECT: 8
```

### Train the GNN
```bash
# Coming next — model/ directory
python model/train.py --epochs 100 --hidden-dim 64
```

### Build the Rust engine
```bash
cd engine
cargo build --release
./target/release/kijiji-engine --help
```

---

## File Structure

```
project_kijiji/
│
├── data/                          # Data ingestion layer
│   ├── ingest_topology.py         # Synthetic BGP event generator → Tinybird/ClickHouse
│   ├── as_to_geo_mapping.py       # ASN → city bridge (CAIDA ITDK)
│   └── schemas/
│       ├── bgp_events.sql         # ClickHouse DDL: raw BGP stream
│       ├── city_metrics.sql       # ClickHouse DDL: socio-economic features
│       └── as_to_geo_mapping.sql  # ClickHouse DDL: ASN→city bridge table
│
├── model/                         # GNN core (PyTorch Geometric)
│   ├── graph_sage.py              # GraphSAGE encoder + MLP decoder
│   ├── loss.py                    # Inverse-GDP weighted MSE loss
│   ├── train.py                   # Temporal walk-forward training loop
│   ├── evaluate.py                # Baseline comparison + fragility metrics
│   └── simulate_peering.py        # Transductive adjacency perturbation
│
├── engine/                        # Rust data plane
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── trombone.rs            # Detour classifier (3-class)
│       ├── geodesic.rs            # Great-circle distance (Haversine)
│       └── bgp_parser.rs          # BGP MRT format parser
│
├── dashboard/                     # React frontend
│   ├── src/
│   │   ├── TopologyMap.jsx        # 3D force-graph, live edge streaming
│   │   ├── TrombonePanel.jsx      # Detour classification feed
│   │   └── PeeringSimulator.jsx   # Latency dividend UI
│   └── package.json
│
├── docs/
│   ├── ARCHITECTURE.md            # This file (expanded)
│   └── THESIS_NOTES.md            # Research decisions, citations, open questions
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Research Pillars

These map directly to the three thesis modules for DAAD evaluation:

### Pillar 1 — Anomaly Detection
*Can we reliably classify African BGP routing detours by cause?*
- Metric: F1 score on 3-class classifier (TROMBONE / POLICY / DIRECT)
- Baseline: Rule-based threshold only (no ML)
- Contribution: Labeled taxonomy of African routing detours

### Pillar 2 — Predictive Peering (Core GNN)
*Can GraphSAGE predict which peering agreements deliver the highest regional latency dividend?*
- Metric: MAE on predicted vs. simulated latency delta
- Baselines: Valley-Free routing, Dijkstra geodesic
- Validation: Temporal walk-forward split (train 2022–2023, test 2024)
- Contribution: Actionable peering recommendations with quantified regional impact

### Pillar 3 — Socio-Technical Fragility
*Do IXP penetration rates and GDP correlate with routing quality in predictable ways?*
- Metric: Pearson correlation, regression R²
- Data: PeeringDB IXP counts × World Bank GDP × trombone ratio per city
- Contribution: Empirical evidence linking infrastructure investment gaps to routing quality

---

## Roadmap

- [x] Synthetic BGP event ingestion → Tinybird
- [x] Trombone detector (Python prototype)
- [x] Node feature vector design
- [ ] `model/graph_sage.py` — GraphSAGE encoder
- [ ] `model/loss.py` — Inverse-GDP weighted MSE
- [ ] `model/train.py` — Temporal walk-forward training
- [ ] `engine/trombone.rs` — Rust production detector
- [ ] Live RIPE RIS feed connection
- [ ] `dashboard/TopologyMap.jsx` — 3D topology visualization
- [ ] DAAD proposal draft (see `docs/THESIS_NOTES.md`)

---