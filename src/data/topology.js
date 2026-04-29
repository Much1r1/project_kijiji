// African IXP Topology Data
// Nodes represent cities with socio-economic features used by the GNN

export const NODES = [
  { id: "NBO", label: "Nairobi",       lat: -1.286,  lng: 36.817,  gdp: 2100,  fiber: 0.72, ixp: 2, latency: 42  },
  { id: "LOS", label: "Lagos",         lat: 6.524,   lng: 3.379,   gdp: 2200,  fiber: 0.65, ixp: 1, latency: 58  },
  { id: "JNB", label: "Johannesburg",  lat: -26.195, lng: 28.034,  gdp: 7400,  fiber: 0.91, ixp: 3, latency: 28  },
  { id: "CPT", label: "Cape Town",     lat: -33.925, lng: 18.424,  gdp: 7100,  fiber: 0.88, ixp: 2, latency: 31  },
  { id: "ACC", label: "Accra",         lat: 5.556,   lng: -0.197,  gdp: 2300,  fiber: 0.58, ixp: 1, latency: 67  },
  { id: "DAR", label: "Dar es Salaam", lat: -6.792,  lng: 39.208,  gdp: 1100,  fiber: 0.44, ixp: 1, latency: 89  },
  { id: "ADD", label: "Addis Ababa",   lat: 9.025,   lng: 38.747,  gdp: 900,   fiber: 0.38, ixp: 0, latency: 112 },
  { id: "KIN", label: "Kinshasa",      lat: -4.322,  lng: 15.322,  gdp: 550,   fiber: 0.21, ixp: 0, latency: 134 },
  { id: "KMP", label: "Kampala",       lat: 0.347,   lng: 32.582,  gdp: 850,   fiber: 0.41, ixp: 0, latency: 98  },
  { id: "LUS", label: "Lusaka",        lat: -15.417, lng: 28.283,  gdp: 1200,  fiber: 0.47, ixp: 1, latency: 76  },
];

// Edges represent existing peering / fiber connections
export const EDGES = [
  { source: "NBO", target: "DAR",  latency: 28,  type: "direct"   },
  { source: "NBO", target: "KMP",  latency: 22,  type: "direct"   },
  { source: "NBO", target: "ADD",  latency: 45,  type: "direct"   },
  { source: "NBO", target: "LOS",  latency: 89,  type: "trombone" },
  { source: "JNB", target: "CPT",  latency: 12,  type: "direct"   },
  { source: "JNB", target: "LUS",  latency: 38,  type: "direct"   },
  { source: "JNB", target: "DAR",  latency: 55,  type: "direct"   },
  { source: "LOS", target: "ACC",  latency: 18,  type: "direct"   },
  { source: "LOS", target: "KIN",  latency: 142, type: "trombone" },
  { source: "DAR", target: "LUS",  latency: 41,  type: "direct"   },
  { source: "KIN", target: "LUS",  latency: 98,  type: "policy"   },
  { source: "ADD", target: "KMP",  latency: 67,  type: "policy"   },
];

// Proposed peering links for the simulator
export const PROPOSED_PEERS = [
  {
    source: "KIN",
    target: "LOS",
    label: "Kinshasa ↔ Lagos",
    predictedLatencyReduction: 87,
    regionalDividend: 3.2,
    affectedCities: ["KIN", "LUS", "DAR", "KMP"],
    cost: "High",
    sdgImpact: "Critical",
  },
  {
    source: "ADD",
    target: "NBO",
    label: "Addis Ababa ↔ Nairobi",
    predictedLatencyReduction: 34,
    regionalDividend: 1.8,
    affectedCities: ["ADD", "KMP", "DAR"],
    cost: "Medium",
    sdgImpact: "High",
  },
  {
    source: "ACC",
    target: "KIN",
    label: "Accra ↔ Kinshasa",
    predictedLatencyReduction: 61,
    regionalDividend: 2.4,
    affectedCities: ["ACC", "KIN", "LOS"],
    cost: "High",
    sdgImpact: "Critical",
  },
];

// Trombone detour examples (from Rust engine output)
export const TROMBONE_EVENTS = [
  { src: "KIN", dst: "ACC", via: "London",   directKm: 2046, actualKm: 11484, ratio: 5.61, wastedMs: 172 },
  { src: "ADD", dst: "LOS", via: "Frankfurt", directKm: 4891, actualKm: 12203, ratio: 2.50, wastedMs: 94  },
  { src: "KMP", dst: "KIN", via: "Paris",    directKm: 2108, actualKm: 9876,  ratio: 4.68, wastedMs: 143 },
  { src: "DAR", dst: "ACC", via: "London",   directKm: 4312, actualKm: 10987, ratio: 2.55, wastedMs: 98  },
];