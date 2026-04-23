"""
model/graph_sage.py
────────────────────────────────────────────────────────────────────────────
Project Kijiji — Peering Recommendation Engine
GraphSAGE-based link predictor for African IXP topology.

Research question:
    Can GraphSAGE learn the topological and socio-economic conditions under
    which new peering agreements reduce latency detours, and predict where
    new links deliver the highest Regional Latency Dividend?

Architecture:
    Encoder  — Two SAGEConv layers producing 64-dim node embeddings.
               Inductive: handles cities not seen during training (new IXPs).
    Decoder  — MLP that scores a candidate (u, v) edge as a predicted
               latency dividend in milliseconds.

Node feature vector  x_i = [gdp_per_capita, fiber_index, ixp_count, mean_latency_ms]
    All features are normalised to [0, 1] before being fed to the model.
    gdp_per_capita is INVERTED so that low-GDP nodes get higher representation
    weight, aligning with SDG 9.4.

Usage:
    See __main__ block at the bottom for a full synthetic training run.
    For production, replace build_synthetic_graph() with a ClickHouse loader.
────────────────────────────────────────────────────────────────────────────
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Feature vector indices — must stay in sync with build_synthetic_graph()
FEAT_GDP        = 0   # GDP per capita (normalised, inverted)
FEAT_FIBER      = 1   # Fiber density index [0, 1]
FEAT_IXP        = 2   # IXP count (normalised)
FEAT_LATENCY    = 3   # Mean latency ms (normalised)

IN_FEATURES     = 4   # x_i dimension
HIDDEN_DIM      = 64  # SAGEConv embedding dimension
OUT_DIM         = 32  # Final node embedding dimension


# ─────────────────────────────────────────────────────────────────────────────
# AFRICAN IXP NODE REGISTRY
# Mirrors data/ingest_topology.py NODES — single source of truth eventually
# moves to ClickHouse city_metrics table.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CityNode:
    city_id:         str
    city:            str
    country:         str
    gdp_per_capita:  float   # USD
    fiber_index:     float   # [0, 1]
    ixp_count:       int
    mean_latency_ms: float   # ms, baseline intra-region

AFRICAN_NODES: list[CityNode] = [
    CityNode("NBO", "Nairobi",        "KE", 2080.0, 0.74, 2, 22.0),
    CityNode("LOS", "Lagos",          "NG", 2184.0, 0.61, 1, 28.0),
    CityNode("JNB", "Johannesburg",   "ZA", 6994.0, 0.89, 3, 12.0),
    CityNode("CPT", "Cape Town",      "ZA", 6994.0, 0.85, 1, 14.0),
    CityNode("ACC", "Accra",          "GH", 2363.0, 0.52, 1, 35.0),
    CityNode("DAR", "Dar es Salaam",  "TZ", 1136.0, 0.41, 1, 42.0),
    CityNode("ADD", "Addis Ababa",    "ET",  925.0, 0.29, 0, 68.0),  # No IXP
    CityNode("KIN", "Kinshasa",       "CD",  577.0, 0.18, 0, 89.0),  # Lowest GDP
    CityNode("CMN", "Casablanca",     "MA", 3795.0, 0.67, 1, 18.0),
    CityNode("KLA", "Kampala",        "UG",  883.0, 0.33, 1, 55.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(values: list[float], invert: bool = False) -> list[float]:
    """Min-max normalise a list to [0, 1]. Invert for low=high-weight features."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    normed = [(v - lo) / (hi - lo) for v in values]
    return [1.0 - n for n in normed] if invert else normed


def build_node_features(nodes: list[CityNode]) -> Tensor:
    """
    Builds the node feature matrix X of shape [N, 4].

    GDP is INVERTED: a city with lower GDP gets a higher feature value,
    so the model naturally attends more to underserved nodes. This is the
    implementation of the SDG 9.4 alignment in the feature space.
    """
    gdp_norm     = _normalise([n.gdp_per_capita  for n in nodes], invert=True)
    fiber_norm   = _normalise([n.fiber_index      for n in nodes])
    ixp_norm     = _normalise([float(n.ixp_count) for n in nodes])
    latency_norm = _normalise([n.mean_latency_ms  for n in nodes])

    rows = [
        [gdp_norm[i], fiber_norm[i], ixp_norm[i], latency_norm[i]]
        for i in range(len(nodes))
    ]
    return torch.tensor(rows, dtype=torch.float)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC GRAPH BUILDER
# Produces a PyG Data object from AFRICAN_NODES.
# Replace this with a ClickHouse loader when live data is ready.
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance — mirrors engine/src/geodesic.rs"""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

# Rough city coordinates for geodesic edge weights
_COORDS: dict[str, tuple[float, float]] = {
    "NBO": (-1.286,  36.817), "LOS": ( 6.524,   3.379),
    "JNB": (-26.204, 28.047), "CPT": (-33.925,  18.424),
    "ACC": ( 5.603,  -0.187), "DAR": (-6.792,   39.208),
    "ADD": ( 9.145,  40.489), "KIN": (-4.322,   15.322),
    "CMN": (33.589,  -7.604), "KLA": ( 0.347,   32.582),
}


def build_synthetic_graph(
    nodes: list[CityNode],
    edge_prob: float = 0.45,
    seed: int = 42,
) -> Data:
    """
    Constructs a synthetic African IXP topology graph.

    Edges represent existing peering relationships (not all city pairs are
    connected — the missing edges are what the GNN learns to recommend).

    Edge weight = observed_latency / geodesic_latency (detour ratio).
    A ratio near 1.0 means efficient routing; > 2.0 flags a trombone detour.

    Args:
        nodes:      List of CityNode objects.
        edge_prob:  Probability of a directed edge existing between any pair.
        seed:       Random seed for reproducibility.

    Returns:
        PyG Data object with:
            x           — [N, 4] node feature matrix
            edge_index  — [2, E] connectivity in COO format
            edge_attr   — [E, 1] detour ratio per edge
            y           — [N] latency label (mean_latency_ms, normalised)
            node_ids    — list of city_id strings (for interpretability)
    """
    random.seed(seed)
    N = len(nodes)
    x = build_node_features(nodes)

    # Build edge list: sparse directed graph, ~45% connectivity by default
    src_list, dst_list, attr_list = [], [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if random.random() > edge_prob:
                continue

            src_id = nodes[i].city_id
            dst_id = nodes[j].city_id
            lat_i, lon_i = _COORDS[src_id]
            lat_j, lon_j = _COORDS[dst_id]

            geodesic_km    = _haversine_km(lat_i, lon_i, lat_j, lon_j)
            # Simulate ~40% of routes going via European hub (+65% path length)
            via_europe     = random.random() < 0.40
            observed_km    = geodesic_km * (random.uniform(1.55, 1.85) if via_europe
                                            else random.uniform(1.0,  1.35))
            detour_ratio   = observed_km / geodesic_km

            src_list.append(i)
            dst_list.append(j)
            attr_list.append([detour_ratio])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float)

    # Node-level label: normalised mean latency (regression target)
    latency_vals = torch.tensor(
        [n.mean_latency_ms for n in nodes], dtype=torch.float
    )
    y = (latency_vals - latency_vals.min()) / (latency_vals.max() - latency_vals.min())

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        node_ids=[n.city_id for n in nodes],
        num_nodes=N,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL: GraphSAGE ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class KijijiEncoder(nn.Module):
    """
    Two-layer GraphSAGE encoder.

    Layer 1: IN_FEATURES (4) → HIDDEN_DIM (64)
        Embeds raw city features into a latent space where proximity in
        embedding space reflects routing similarity, not just geography.

    Layer 2: HIDDEN_DIM (64) → OUT_DIM (32)
        Aggregates 1-hop neighbourhood context. A city's embedding now
        encodes not just its own features but its neighbours' connectivity.

    SAGEConv is chosen over GCNConv because it is INDUCTIVE:
        - Generalises to new nodes (new IXPs) not seen during training
        - Uses neighbour sampling rather than full adjacency at inference time
        - Critical for a dynamic topology where ASes appear and disappear

    Dropout (p=0.3) is applied between layers to regularise — the graph is
    small (10 nodes in synthetic mode) so overfitting is a real risk.
    """

    def __init__(
        self,
        in_features:  int = IN_FEATURES,
        hidden_dim:   int = HIDDEN_DIM,
        out_dim:      int = OUT_DIM,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.conv1   = SAGEConv(in_features, hidden_dim)
        self.conv2   = SAGEConv(hidden_dim,  out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1     = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x:          [N, in_features] node feature matrix
            edge_index: [2, E] edge connectivity in COO format

        Returns:
            z: [N, out_dim] node embedding matrix
        """
        # Layer 1: embed raw features into neighbourhood-aware latent space
        z = self.conv1(x, edge_index)       # [N, hidden_dim]
        z = self.bn1(z)
        z = F.relu(z)
        z = self.dropout(z)

        # Layer 2: aggregate 1-hop regional context
        z = self.conv2(z, edge_index)       # [N, out_dim]
        z = F.relu(z)

        return z


# ─────────────────────────────────────────────────────────────────────────────
# MODEL: MLP DECODER (Link Score Predictor)
# ─────────────────────────────────────────────────────────────────────────────

class PeeringDecoder(nn.Module):
    """
    MLP decoder that scores a candidate peering link (u, v).

    Input:  Concatenation of embeddings z_u and z_v → [2 * out_dim]
    Output: Scalar score representing predicted Regional Latency Dividend (ms)

    The decoder learns that the value of a peering link between u and v is
    a non-linear function of both endpoints' neighbourhood embeddings —
    not just their pairwise geodesic distance.

    Architecture: 3-layer MLP with ReLU activations and dropout.
    Output activation: Softplus — ensures the predicted dividend is positive
    (you can't gain negative latency by adding a peering link) while keeping
    gradients alive near zero.
    """

    def __init__(self, in_dim: int = OUT_DIM * 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),   # output > 0; predicted latency saving in ms
        )

    def forward(self, z_u: Tensor, z_v: Tensor) -> Tensor:
        """
        Args:
            z_u: [B, out_dim] source node embeddings
            z_v: [B, out_dim] target node embeddings

        Returns:
            score: [B, 1] predicted latency dividend
        """
        return self.net(torch.cat([z_u, z_v], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# FULL MODEL: Encoder + Decoder
# ─────────────────────────────────────────────────────────────────────────────

class KijijiGNN(nn.Module):
    """
    Full GraphSAGE link predictor for African IXP peering recommendation.

    Combines KijijiEncoder and PeeringDecoder into a single trainable module.

    Training mode:  scores positive edges (existing peering) vs negative
                    edges (no current peering — these are the candidates).
    Inference mode: given any (u, v) pair, returns predicted latency dividend.
    Simulation:     simulate_peering() modifies the adjacency matrix and
                    re-runs inference to compute regional latency delta
                    (transductive adjacency perturbation).
    """

    def __init__(self):
        super().__init__()
        self.encoder = KijijiEncoder()
        self.decoder = PeeringDecoder()

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Score all edges in edge_index given node embeddings z."""
        z_u = z[edge_index[0]]
        z_v = z[edge_index[1]]
        return self.decoder(z_u, z_v).squeeze(-1)

    def forward(self, data: Data) -> tuple[Tensor, Tensor]:
        """
        Full forward pass for training.

        Returns:
            pos_scores: [E_pos] scores for existing edges
            neg_scores: [E_neg] scores for sampled negative edges
        """
        z = self.encode(data.x, data.edge_index)

        # Score existing (positive) edges
        pos_scores = self.decode(z, data.edge_index)

        # Sample negative edges (city pairs with no current peering)
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1),
        )
        neg_scores = self.decode(z, neg_edge_index)

        return pos_scores, neg_scores

    @torch.no_grad()
    def simulate_peering(
        self,
        data: Data,
        new_src: int,
        new_dst: int,
    ) -> dict:
        """
        Transductive adjacency perturbation.

        Adds a synthetic edge (new_src → new_dst) to the graph and measures
        the change in predicted latency scores across ALL nodes — the
        Regional Latency Dividend.

        This is the core simulation mechanism for Thesis Pillar 2.

        Args:
            data:    Current graph state
            new_src: Source node index (e.g. index of "ADD" Addis Ababa)
            new_dst: Target node index (e.g. index of "NBO" Nairobi)

        Returns:
            dict with:
                baseline_scores    — per-node mean predicted latency (before)
                perturbed_scores   — per-node mean predicted latency (after)
                latency_dividend   — per-node improvement in ms
                regional_dividend  — mean improvement across all nodes
                most_improved      — node index with highest dividend
        """
        self.eval()

        # ── Baseline: embeddings on original graph ──
        z_base = self.encode(data.x, data.edge_index)
        base_scores = self._node_latency_scores(z_base, data.edge_index)

        # ── Perturbation: add the proposed peering link ──
        new_edge   = torch.tensor([[new_src], [new_dst]], dtype=torch.long)
        perturbed_edge_index = torch.cat([data.edge_index, new_edge], dim=1)
        z_pert     = self.encode(data.x, perturbed_edge_index)
        pert_scores = self._node_latency_scores(z_pert, perturbed_edge_index)

        dividend = base_scores - pert_scores   # positive = improvement
        node_ids = data.node_ids if hasattr(data, "node_ids") else list(range(data.num_nodes))

        return {
            "new_edge":           (node_ids[new_src], node_ids[new_dst]),
            "baseline_scores":    base_scores.tolist(),
            "perturbed_scores":   pert_scores.tolist(),
            "latency_dividend":   dividend.tolist(),
            "regional_dividend":  float(dividend.mean()),
            "most_improved_node": node_ids[int(dividend.argmax())],
            "most_improved_ms":   float(dividend.max()),
        }

    def _node_latency_scores(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Mean outgoing edge score per node — proxy for routing quality."""
        N = z.size(0)
        scores = torch.zeros(N)
        counts = torch.zeros(N)
        if edge_index.size(1) == 0:
            return scores
        raw = self.decode(z, edge_index)
        for i, src in enumerate(edge_index[0]):
            scores[src] += raw[i]
            counts[src] += 1
        mask = counts > 0
        scores[mask] = scores[mask] / counts[mask]
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# LOSS: model/loss.py (inlined here, will be split out next)
# ─────────────────────────────────────────────────────────────────────────────

def weighted_latency_loss(
    pos_scores:  Tensor,
    neg_scores:  Tensor,
    gdp_weights: Tensor,
) -> Tensor:
    """
    Inverse-GDP weighted binary cross-entropy loss.

    Positive edges (existing peering) should score HIGH.
    Negative edges (no current peering) should score LOW.

    The gdp_weights vector encodes the SDG 9.4 objective:
        - Low-GDP nodes → weight close to 1.0
        - High-GDP nodes → weight close to 0.0

    Errors on routes involving underserved cities are penalised more,
    so the model learns to prioritise those connections.

    Args:
        pos_scores:  [E_pos] scores for existing edges (want: high)
        neg_scores:  [E_neg] scores for sampled negative edges (want: low)
        gdp_weights: [N] per-node inverse-GDP weight in [0, 1]

    Returns:
        Scalar loss tensor.
    """
    # Binary labels: 1 for positive (existing) edges, 0 for negative
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])

    # Normalise scores to [0, 1] for BCE
    scores_norm = torch.sigmoid(scores)

    # Clamp to avoid log(0)
    eps = 1e-7
    scores_norm = scores_norm.clamp(eps, 1 - eps)

    bce = -(labels * torch.log(scores_norm) + (1 - labels) * torch.log(1 - scores_norm))

    # Apply GDP weight to positive edges only — we care more about
    # correctly predicting peering value for underserved nodes
    mean_gdp_weight = gdp_weights.mean()
    weights = torch.cat([
        torch.full_like(pos_scores, float(mean_gdp_weight)),
        torch.ones_like(neg_scores),
    ])

    return (bce * weights).mean()


def get_gdp_weights(data: Data) -> Tensor:
    """Extract the inverted GDP feature column as the weight vector."""
    return data.x[:, FEAT_GDP]   # already inverted in build_node_features()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP (will move to model/train.py)
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:     KijijiGNN,
    data:      Data,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad()
    pos_scores, neg_scores = model(data)
    gdp_weights = get_gdp_weights(data)
    loss = weighted_latency_loss(pos_scores, neg_scores, gdp_weights)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(model: KijijiGNN, data: Data) -> dict:
    """
    Evaluation metrics for link prediction quality.

    Temporal walk-forward validation logic lives in model/evaluate.py.
    This function gives per-epoch sanity metrics during training.
    """
    model.eval()
    pos_scores, neg_scores = model(data)

    # AUC proxy: fraction of pos_score > neg_score pairs
    pos_mean = float(pos_scores.mean())
    neg_mean = float(neg_scores.mean())
    separation = pos_mean - neg_mean

    gdp_weights = get_gdp_weights(data)
    loss = float(weighted_latency_loss(pos_scores, neg_scores, gdp_weights))

    return {
        "loss":       loss,
        "pos_mean":   round(pos_mean, 4),
        "neg_mean":   round(neg_mean, 4),
        "separation": round(separation, 4),   # want this > 0 and growing
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT — full synthetic training run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  Project Kijiji — GraphSAGE Peering Recommendation Engine")
    print("=" * 62)

    # ── 1. Build graph ──
    graph = build_synthetic_graph(AFRICAN_NODES, seed=42)
    print(f"\n📍 Nodes : {graph.num_nodes} African cities")
    print(f"🔗 Edges : {graph.edge_index.size(1)} existing peering links")
    print(f"📐 Features per node: {graph.x.size(1)}")
    print(f"\nNode feature matrix (x):\n")
    header = f"  {'City':5}  {'GDP↑':>6}  {'Fiber':>6}  {'IXP':>6}  {'Latency':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, city_id in enumerate(graph.node_ids):
        feats = graph.x[i]
        print(f"  {city_id:5}  {feats[0]:.3f}   {feats[1]:.3f}   {feats[2]:.3f}   {feats[3]:.3f}")

    # ── 2. Init model ──
    model    = KijijiGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model parameters: {total_params:,}")
    print(f"   Encoder: KijijiEncoder  ({IN_FEATURES}→{HIDDEN_DIM}→{OUT_DIM})")
    print(f"   Decoder: PeeringDecoder ({OUT_DIM*2}→32→16→1)")

    # ── 3. Train ──
    print(f"\n{'─'*62}")
    print(f"  Training  (100 epochs, Adam lr=1e-3)")
    print(f"{'─'*62}")
    print(f"  {'Epoch':>6}  {'Loss':>8}  {'Pos':>7}  {'Neg':>7}  {'Sep':>7}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}")

    for epoch in range(1, 101):
        loss = train_epoch(model, graph, optimizer)
        if epoch % 10 == 0:
            metrics = evaluate(model, graph)
            print(f"  {epoch:>6}  {metrics['loss']:>8.4f}  "
                  f"{metrics['pos_mean']:>7.4f}  {metrics['neg_mean']:>7.4f}  "
                  f"{metrics['separation']:>7.4f}")

    # ── 4. Simulate a peering link ──
    print(f"\n{'─'*62}")
    print(f"  Peering Simulation: What if ADD ↔ NBO were directly peered?")
    print(f"{'─'*62}")

    node_ids  = graph.node_ids
    add_idx   = node_ids.index("ADD")   # Addis Ababa — no IXP, high latency
    nbo_idx   = node_ids.index("NBO")   # Nairobi — regional hub

    result = model.simulate_peering(graph, new_src=add_idx, new_dst=nbo_idx)

    print(f"\n  Proposed link  : {result['new_edge'][0]} → {result['new_edge'][1]}")
    print(f"  Regional dividend (mean across all nodes): "
          f"{result['regional_dividend']:+.4f}")
    print(f"  Most improved node : {result['most_improved_node']}")
    print(f"  Max improvement    : {result['most_improved_ms']:+.4f}")
    print(f"\n  Per-node latency dividend:")
    print(f"  {'City':5}  {'Dividend':>10}")
    print(f"  {'─'*5}  {'─'*10}")
    for city_id, div in zip(node_ids, result["latency_dividend"]):
        bar = "█" * max(0, int(div * 40))
        print(f"  {city_id:5}  {div:>+10.4f}  {bar}")

    print(f"\n✅ Done. Next: model/train.py for temporal walk-forward training.")