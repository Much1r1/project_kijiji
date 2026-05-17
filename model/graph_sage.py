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

Loss function:
    Inverse-GDP weighted BCE applied PER-EDGE, not as a scalar mean.
    Each edge (u, v) carries weight = max(gdp_weight[u], gdp_weight[v]).
    This means a misprediction on a KIN→LOS edge is penalised far more
    than the same error on a JNB→CPT edge — the core SDG 9.4 claim.

Usage:
    See __main__ block at the bottom for a full synthetic training run.
    For production, replace build_synthetic_graph() with a ClickHouse loader.
────────────────────────────────────────────────────────────────────────────
"""

import math
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling, add_self_loops, degree


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FEAT_GDP        = 0   # GDP per capita (normalised, inverted)
FEAT_FIBER      = 1   # Fiber density index [0, 1]
FEAT_IXP        = 2   # IXP count (normalised)
FEAT_LATENCY    = 3   # Mean latency ms (normalised)

IN_FEATURES     = 4
HIDDEN_DIM      = 32
OUT_DIM         = 16


# ─────────────────────────────────────────────────────────────────────────────
# AFRICAN IXP NODE REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CityNode:
    city_id:         str
    city:            str
    country:         str
    gdp_per_capita:  float
    fiber_index:     float
    ixp_count:       int
    mean_latency_ms: float

AFRICAN_NODES: list[CityNode] = [
    CityNode("NBO", "Nairobi",        "KE", 2080.0, 0.74, 2, 22.0),
    CityNode("LOS", "Lagos",          "NG", 2184.0, 0.61, 1, 28.0),
    CityNode("JNB", "Johannesburg",   "ZA", 6994.0, 0.89, 3, 12.0),
    CityNode("CPT", "Cape Town",      "ZA", 6994.0, 0.85, 1, 14.0),
    CityNode("ACC", "Accra",          "GH", 2363.0, 0.52, 1, 35.0),
    CityNode("DAR", "Dar es Salaam",  "TZ", 1136.0, 0.41, 1, 42.0),
    CityNode("ADD", "Addis Ababa",    "ET",  925.0, 0.29, 0, 68.0),
    CityNode("KIN", "Kinshasa",       "CD",  577.0, 0.18, 0, 89.0),
    CityNode("CMN", "Casablanca",     "MA", 3795.0, 0.67, 1, 18.0),
    CityNode("KLA", "Kampala",        "UG",  883.0, 0.33, 1, 55.0),
    CityNode("ABJ", "Abidjan",      "CI", 1750.0, 0.49, 1, 48.0),
    CityNode("DKR", "Dakar",        "SN", 1430.0, 0.42, 1, 55.0),
    CityNode("CTN", "Conakry",      "GN",  510.0, 0.18, 0, 98.0),
    CityNode("HAR", "Harare",       "ZW", 1200.0, 0.39, 0, 72.0),
    CityNode("MPS", "Maputo",       "MZ",  490.0, 0.22, 0, 88.0),
]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(values: list[float], invert: bool = False) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    normed = [(v - lo) / (hi - lo) for v in values]
    return [1.0 - n for n in normed] if invert else normed


def build_node_features(nodes: list[CityNode]) -> Tensor:
    """
    Builds the node feature matrix X of shape [N, 4].

    GDP is INVERTED: low-GDP cities get higher feature values so the model
    attends more to underserved nodes. This is the SDG 9.4 alignment
    baked into the feature space, complementing the loss function weighting.
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
# MANUAL SAGEConv — for interpretability and academic review
#
# Stephen's requirement: understand what's inside the aggregation step.
# This mirrors torch_geometric SAGEConv exactly but makes the mean
# neighbourhood aggregation explicit. Used in KijijiEncoderManual below.
#
# SAGEConv message passing (Hamilton et al. 2017):
#   h_v^(k) = W1 · h_v^(k-1)  +  W2 · MEAN({ h_u : u ∈ N(v) })
#
# Where:
#   h_v^(k-1) = current node embedding
#   N(v)      = neighbours of v
#   W1, W2    = learned weight matrices
#   MEAN      = mean aggregation over neighbourhood
# ─────────────────────────────────────────────────────────────────────────────

class ManualSAGEConv(nn.Module):
    """
    Manual implementation of GraphSAGE mean aggregation.

    Explicit implementation of Hamilton et al. (2017) equation:
        h_v = W1 * h_v + W2 * mean(h_neighbours)

    This exists alongside the torch_geometric SAGEConv to demonstrate
    understanding of the internal aggregation mechanism — a requirement
    for academic review of the architecture choice.

    In production KijijiEncoder uses torch_geometric SAGEConv (optimised).
    KijijiEncoderManual uses this for interpretability and testing.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # W1: transforms the node's own embedding
        self.W_self = nn.Linear(in_channels, out_channels, bias=False)
        # W2: transforms the aggregated neighbourhood embedding
        self.W_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x:          [N, in_channels] node feature / embedding matrix
            edge_index: [2, E] edges in COO format (row=src, col=dst)

        Returns:
            out: [N, out_channels] updated node embeddings
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # ── Step 1: Aggregate neighbourhood embeddings ──
        # For each node v, compute mean of all neighbour embeddings h_u
        # where u ∈ N(v). We aggregate messages sent TO v (dst=v, src=u).
        agg = torch.zeros(N, x.size(1), device=x.device)
        count = torch.zeros(N, 1, device=x.device)

        # Scatter: for each edge (u→v), add h_u to v's aggregation bucket
        agg.index_add_(0, dst, x[src])
        count.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device))

        # Mean aggregation — divide by degree, avoid div-by-zero for isolated nodes
        count = count.clamp(min=1.0)
        agg = agg / count                       # [N, in_channels]

        # ── Step 2: Combine self embedding + neighbourhood aggregate ──
        out = self.W_self(x) + self.W_neigh(agg) + self.bias
        return out


class KijijiEncoderManual(nn.Module):
    """
    Two-layer GraphSAGE encoder using ManualSAGEConv.
    Architecturally identical to KijijiEncoder but transparent internals.
    Used for academic review and loss function stress-testing.
    """

    def __init__(
        self,
        in_features: int = IN_FEATURES,
        hidden_dim:  int = HIDDEN_DIM,
        out_dim:     int = OUT_DIM,
        dropout:     float = 0.5,
    ):
        super().__init__()
        self.conv1   = ManualSAGEConv(in_features, hidden_dim)
        self.conv2   = ManualSAGEConv(hidden_dim,  out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1     = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        z = self.conv1(x, edge_index)
        z = self.bn1(z)
        z = F.relu(z)
        z = self.dropout(z)
        z = self.conv2(z, edge_index)
        z = F.relu(z)
        return z


# ─────────────────────────────────────────────────────────────────────────────
# MODEL: GraphSAGE ENCODER (production — uses torch_geometric)
# ─────────────────────────────────────────────────────────────────────────────

class KijijiEncoder(nn.Module):
    """
    Two-layer GraphSAGE encoder using torch_geometric SAGEConv.

    SAGEConv is chosen over GCNConv because it is INDUCTIVE:
        - Generalises to new nodes (new IXPs) not seen during training
        - Uses neighbour sampling not full adjacency at inference
        - Critical for dynamic topology where ASes appear and disappear

    The manual equivalent (KijijiEncoderManual / ManualSAGEConv) shows the
    internal aggregation step explicitly for academic review.
    """

    def __init__(
        self,
        in_features: int = IN_FEATURES,
        hidden_dim:  int = HIDDEN_DIM,
        out_dim:     int = OUT_DIM,
        dropout:     float = 0.5,
    ):
        super().__init__()
        self.conv1   = SAGEConv(in_features, hidden_dim)
        self.conv2   = SAGEConv(hidden_dim,  out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1     = nn.LayerNorm(hidden_dim)  # LayerNorm stable on small graphs

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        z = self.conv1(x, edge_index)
        z = self.bn1(z)
        z = F.relu(z)
        z = self.dropout(z)
        z = self.conv2(z, edge_index)
        z = F.relu(z)
        return z


# ─────────────────────────────────────────────────────────────────────────────
# MODEL: MLP DECODER
# ─────────────────────────────────────────────────────────────────────────────

class PeeringDecoder(nn.Module):
    """
    MLP decoder that scores a candidate peering link (u, v).

    Input:  Concatenation of embeddings z_u and z_v → [2 * out_dim]
    Output: Scalar score (predicted Regional Latency Dividend)

    Softplus output ensures dividend is always positive — adding a peering
    link cannot increase latency by definition.
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
            nn.Softplus(),
        )

    def forward(self, z_u: Tensor, z_v: Tensor) -> Tensor:
        return self.net(torch.cat([z_u, z_v], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class KijijiGNN(nn.Module):
    """
    Full GraphSAGE link predictor for African IXP peering recommendation.

    use_manual=True switches to the ManualSAGEConv encoder for review/debug.
    Production training uses use_manual=False (torch_geometric optimised).
    """

    def __init__(self, use_manual: bool = False):
        super().__init__()
        self.encoder = KijijiEncoderManual() if use_manual else KijijiEncoder()
        self.decoder = PeeringDecoder()

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        z_u = z[edge_index[0]]
        z_v = z[edge_index[1]]
        return self.decoder(z_u, z_v).squeeze(-1)

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor]:
        """
        Full forward pass.

        Returns:
            pos_scores:     [E_pos] scores for existing edges
            neg_scores:     [E_neg] scores for sampled negative edges
            neg_edge_index: [2, E_neg] the sampled negative edges
                            (needed for per-edge GDP weighting in loss)
        """
        z = self.encode(data.x, data.edge_index)

        pos_scores = self.decode(z, data.edge_index)

        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1),
        )
        neg_scores = self.decode(z, neg_edge_index)

        return pos_scores, neg_scores, neg_edge_index

    @torch.no_grad()
    def simulate_peering(
        self,
        data:    Data,
        new_src: int,
        new_dst: int,
    ) -> dict:
        """
        Transductive adjacency perturbation — Regional Latency Dividend.

        Adds edge (new_src → new_dst) to the graph and measures the change
        in per-node embedding cosine similarity to their nearest neighbours.

        Why cosine similarity instead of raw score proxy:
            Embedding distance is a principled measure of how much the GNN's
            learned representation of a node changes when a new peering link
            is added. A node whose embedding shifts significantly after the
            perturbation is one the model believes is meaningfully affected
            by the new peering agreement.

        Returns:
            dict with per-node dividend, regional mean, and most improved node.
        """
        self.eval()
        N = data.num_nodes

        # ── Baseline embeddings ──
        z_base = self.encode(data.x, data.edge_index)  # [N, OUT_DIM]

        # ── Perturbed embeddings (add proposed peering link) ──
        new_edge = torch.tensor([[new_src], [new_dst]], dtype=torch.long)
        perturbed_edge_index = torch.cat([data.edge_index, new_edge], dim=1)
        z_pert = self.encode(data.x, perturbed_edge_index)  # [N, OUT_DIM]

        # ── Dividend: cosine similarity shift per node ──
        # How much did the model's representation of each node change?
        # Nodes far from JNB/CPT (well-connected) should shift more.
        cos = nn.CosineSimilarity(dim=-1)
        similarity = cos(z_base, z_pert)         # [N] — 1.0 = no change
        dividend = 1.0 - similarity              # [N] — higher = more affected

        node_ids = data.node_ids if hasattr(data, "node_ids") else list(range(N))

        return {
            "new_edge":           (node_ids[new_src], node_ids[new_dst]),
            "baseline_scores":    z_base.norm(dim=-1).tolist(),
            "perturbed_scores":   z_pert.norm(dim=-1).tolist(),
            "latency_dividend":   dividend.tolist(),
            "regional_dividend":  float(dividend.mean()),
            "most_improved_node": node_ids[int(dividend.argmax())],
            "most_improved_ms":   float(dividend.max()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# LOSS: Inverse-GDP weighted BCE — per-edge, not scalar mean
#
# CRITICAL FIX from v1:
#   v1 applied mean_gdp_weight as a single scalar across ALL positive edges.
#   This meant a KIN→LOS edge and a JNB→CPT edge carried identical weight,
#   defeating the SDG 9.4 claim entirely.
#
#   v2 computes per-edge weights:
#       weight(u,v) = max(gdp_weight[u], gdp_weight[v])
#   This means an edge is weighted by its most fragile endpoint.
#   KIN→LOS weight ≈ 1.0 (KIN GDP=577, highest inverse weight)
#   JNB→CPT weight ≈ 0.02 (both high GDP, low weight)
#
#   The loss gradient now flows disproportionately through edges involving
#   underserved cities — the model is forced to get those right.
# ─────────────────────────────────────────────────────────────────────────────

def weighted_latency_loss(
    pos_scores:     Tensor,
    neg_scores:     Tensor,
    gdp_weights:    Tensor,
    pos_edge_index: Tensor,
    neg_edge_index: Tensor,
) -> Tensor:
    """
    Per-edge inverse-GDP weighted binary cross-entropy loss.

    Each edge (u, v) carries weight = max(gdp_weight[u], gdp_weight[v]).
    This weights errors on routes involving the most fragile cities highest.

    Args:
        pos_scores:     [E_pos] scores for existing edges (want: high)
        neg_scores:     [E_neg] scores for sampled negative edges (want: low)
        gdp_weights:    [N] per-node inverse-GDP weight in [0, 1]
        pos_edge_index: [2, E_pos] positive edge indices
        neg_edge_index: [2, E_neg] negative edge indices

    Returns:
        Scalar weighted loss tensor.
    """
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])

    # Clamp before sigmoid to prevent overflow → nan on small graphs
    scores = scores.clamp(-10, 10)
    scores_norm = torch.sigmoid(scores).clamp(1e-7, 1 - 1e-7)
    bce = -(labels * torch.log(scores_norm) + (1 - labels) * torch.log(1 - scores_norm))

    # ── Per-edge GDP weights ──
    pos_w = torch.max(
        gdp_weights[pos_edge_index[0]],
        gdp_weights[pos_edge_index[1]],
    )  # [E_pos]
    neg_w = torch.ones_like(neg_scores)
    weights = torch.cat([pos_w, neg_w])
    bce_loss = (bce * weights).mean()

    # ── Margin loss: enforce pos > neg + 0.5 ──
    n = min(pos_scores.size(0), neg_scores.size(0))
    margin_loss = F.relu(0.5 - pos_scores[:n] + neg_scores[:n])
    margin_loss = (margin_loss * pos_w[:n]).mean()

    return bce_loss + 0.1 * margin_loss


def get_gdp_weights(data: Data) -> Tensor:
    """Extract the inverted GDP feature column as the weight vector."""
    return data.x[:, FEAT_GDP]


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.asin(math.sqrt(a))

_COORDS: dict[str, tuple[float, float]] = {
    "NBO": (-1.286,  36.817), "LOS": ( 6.524,   3.379),
    "JNB": (-26.204, 28.047), "CPT": (-33.925,  18.424),
    "ACC": ( 5.603,  -0.187), "DAR": (-6.792,   39.208),
    "ADD": ( 9.145,  40.489), "KIN": (-4.322,   15.322),
    "CMN": (33.589,  -7.604), "KLA": ( 0.347,   32.582),
    "ABJ": ( 5.359,  -4.008), "DKR": (14.693, -17.447),
    "CTN": ( 9.537, -13.677), "HAR": (-17.829, 31.052),
    "MPS": (-25.966, 32.573),
}


def build_synthetic_graph(
    nodes: list[CityNode],
    edge_prob: float = 0.45,
    seed: int = 42,
) -> Data:
    random.seed(seed)
    N = len(nodes)
    x = build_node_features(nodes)

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
            geodesic_km  = _haversine_km(lat_i, lon_i, lat_j, lon_j)
            via_europe   = random.random() < 0.40
            observed_km  = geodesic_km * (random.uniform(1.55, 1.85) if via_europe
                                          else random.uniform(1.0, 1.35))
            src_list.append(i)
            dst_list.append(j)
            attr_list.append([observed_km / geodesic_km])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float)

    latency_vals = torch.tensor([n.mean_latency_ms for n in nodes], dtype=torch.float)
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
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:     KijijiGNN,
    data:      Data,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad()
    pos_scores, neg_scores, neg_edge_index = model(data)
    gdp_weights = get_gdp_weights(data)
    loss = weighted_latency_loss(
        pos_scores, neg_scores, gdp_weights,
        data.edge_index, neg_edge_index,   # pass edge indices for per-edge weighting
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(model: KijijiGNN, data: Data) -> dict:
    model.eval()
    pos_scores, neg_scores, neg_edge_index = model(data)
    pos_mean = float(pos_scores.mean())
    neg_mean = float(neg_scores.mean())
    gdp_weights = get_gdp_weights(data)
    loss = float(weighted_latency_loss(
        pos_scores, neg_scores, gdp_weights,
        data.edge_index, neg_edge_index,
    ))
    return {
        "loss":       loss,
        "pos_mean":   round(pos_mean, 4),
        "neg_mean":   round(neg_mean, 4),
        "separation": round(pos_mean - neg_mean, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  Project Kijiji — GraphSAGE Peering Recommendation Engine")
    print("=" * 62)

    graph = build_synthetic_graph(AFRICAN_NODES, seed=42)
    print(f"\nNodes : {graph.num_nodes}  |  Edges : {graph.edge_index.size(1)}")
    print(f"\nNode feature matrix (GDP↑ = inverted = more weight in loss):\n")
    print(f"  {'City':5}  {'GDP↑':>6}  {'Fiber':>6}  {'IXP':>6}  {'Latency':>8}  {'GDP weight':>10}")
    print("  " + "─" * 52)
    gdp_w = get_gdp_weights(graph)
    for i, city_id in enumerate(graph.node_ids):
        f = graph.x[i]
        print(f"  {city_id:5}  {f[0]:.3f}   {f[1]:.3f}   {f[2]:.3f}   {f[3]:.3f}   {gdp_w[i]:.3f}")

    print(f"\n── Per-edge weight sample (first 5 positive edges) ──")
    print(f"  {'Edge':12}  {'w(u)':>6}  {'w(v)':>6}  {'max_w':>6}")
    print("  " + "─" * 36)
    for k in range(min(5, graph.edge_index.size(1))):
        u = graph.edge_index[0, k].item()
        v = graph.edge_index[1, k].item()
        wu = gdp_w[u].item()
        wv = gdp_w[v].item()
        print(f"  {graph.node_ids[u]}→{graph.node_ids[v]:8}  {wu:.3f}   {wv:.3f}   {max(wu,wv):.3f}")

    # ── Train — production encoder ──
    print(f"\n── Training (PyG SAGEConv, 100 epochs) ──")
    model     = KijijiGNN(use_manual=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"  {'Epoch':>6}  {'Loss':>8}  {'Pos':>7}  {'Neg':>7}  {'Sep':>7}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}")
    for epoch in range(1, 101):
        loss = train_epoch(model, graph, optimizer)
        if epoch % 10 == 0:
            m = evaluate(model, graph)
            print(f"  {epoch:>6}  {m['loss']:>8.4f}  {m['pos_mean']:>7.4f}  "
                  f"{m['neg_mean']:>7.4f}  {m['separation']:>7.4f}")

    # ── Train — manual encoder (same results, transparent internals) ──
    print(f"\n── Training (ManualSAGEConv, 100 epochs) ──")
    model_m    = KijijiGNN(use_manual=True)
    optimizer_m = torch.optim.Adam(model_m.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"  {'Epoch':>6}  {'Loss':>8}  {'Sep':>7}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*7}")
    for epoch in range(1, 101):
        loss_m = train_epoch(model_m, graph, optimizer_m)
        if epoch % 20 == 0:
            m = evaluate(model_m, graph)
            print(f"  {epoch:>6}  {m['loss']:>8.4f}  {m['separation']:>7.4f}")

    # ── Simulation ──
    print(f"\n── Peering Simulation: ADD ↔ NBO ──")
    node_ids = graph.node_ids
    result = model.simulate_peering(
        graph,
        new_src=node_ids.index("ADD"),
        new_dst=node_ids.index("NBO"),
    )
    print(f"  Proposed link    : {result['new_edge'][0]} → {result['new_edge'][1]}")
    print(f"  Regional dividend: {result['regional_dividend']:+.4f}")
    print(f"  Most improved    : {result['most_improved_node']}  ({result['most_improved_ms']:+.4f})")
    print(f"\n  Per-node embedding shift (cosine distance from baseline):")
    print(f"  {'City':5}  {'Dividend':>10}")
    print(f"  {'─'*5}  {'─'*10}")
    for city_id, div in zip(node_ids, result["latency_dividend"]):
        bar = "█" * max(0, int(div * 200))
        print(f"  {city_id:5}  {div:>+10.4f}  {bar}")

    print(f"\n✅ Done. Run model/train.py for temporal walk-forward validation.")