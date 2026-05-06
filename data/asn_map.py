# data/asn_map.py
# Maps African ASNs to Project Kijiji node IDs
# Sources: PeeringDB, CAIDA ITDK, AFRINIC registry
#
# Format: ASN -> { node_id, city, country, operator, tier }
# tier: 1=national backbone, 2=regional ISP, 3=last-mile

ASN_TO_NODE = {
    # ── NAIROBI (NBO) ──────────────────────────────────────────────
    33771:  {"node": "NBO", "city": "Nairobi",       "country": "KE", "operator": "Safaricom",          "tier": 1},
    36914:  {"node": "NBO", "city": "Nairobi",       "country": "KE", "operator": "Liquid Telecom KE",  "tier": 1},
    37061:  {"node": "NBO", "city": "Nairobi",       "country": "KE", "operator": "KIXP/TESPOK",        "tier": 1},
    15399:  {"node": "NBO", "city": "Nairobi",       "country": "KE", "operator": "Telkom Kenya",       "tier": 1},
    37197:  {"node": "NBO", "city": "Nairobi",       "country": "KE", "operator": "Jamii Telecom",      "tier": 2},
    36944:  {"node": "NBO", "city": "Nairobi",       "country": "KE", "operator": "AccessKenya",        "tier": 2},

    # ── LAGOS (LOS) ────────────────────────────────────────────────
    37148:  {"node": "LOS", "city": "Lagos",         "country": "NG", "operator": "MTN Nigeria",        "tier": 1},
    29465:  {"node": "LOS", "city": "Lagos",         "country": "NG", "operator": "Airtel Nigeria",     "tier": 1},
    36873:  {"node": "LOS", "city": "Lagos",         "country": "NG", "operator": "Smile Comms NG",     "tier": 2},
    37282:  {"node": "LOS", "city": "Lagos",         "country": "NG", "operator": "MainOne",            "tier": 1},
    37179:  {"node": "LOS", "city": "Lagos",         "country": "NG", "operator": "Rack Centre",        "tier": 2},
    30844:  {"node": "LOS", "city": "Lagos",         "country": "NG", "operator": "Liquid Telecom NG",  "tier": 1},

    # ── JOHANNESBURG (JNB) ─────────────────────────────────────────
    2905:   {"node": "JNB", "city": "Johannesburg",  "country": "ZA", "operator": "Telkom SA",          "tier": 1},
    36874:  {"node": "JNB", "city": "Johannesburg",  "country": "ZA", "operator": "MTN SA",             "tier": 1},
    37271:  {"node": "JNB", "city": "Johannesburg",  "country": "ZA", "operator": "Liquid Telecom ZA",  "tier": 1},
    37153:  {"node": "JNB", "city": "Johannesburg",  "country": "ZA", "operator": "NAPAfrica/JINX",     "tier": 1},
    328512: {"node": "JNB", "city": "Johannesburg",  "country": "ZA", "operator": "Afrihost",           "tier": 2},
    37680:  {"node": "JNB", "city": "Johannesburg",  "country": "ZA", "operator": "Cool Ideas",         "tier": 2},

    # ── CAPE TOWN (CPT) ────────────────────────────────────────────
    36937:  {"node": "CPT", "city": "Cape Town",     "country": "ZA", "operator": "Vodacom SA",         "tier": 1},
    37100:  {"node": "CPT", "city": "Cape Town",     "country": "ZA", "operator": "WIOCC",              "tier": 1},
    37549:  {"node": "CPT", "city": "Cape Town",     "country": "ZA", "operator": "Herotel",            "tier": 2},

    # ── ACCRA (ACC) ────────────────────────────────────────────────
    29614:  {"node": "ACC", "city": "Accra",         "country": "GH", "operator": "MTN Ghana",          "tier": 1},
    37308:  {"node": "ACC", "city": "Accra",         "country": "GH", "operator": "Ghana IX",           "tier": 1},
    36916:  {"node": "ACC", "city": "Accra",         "country": "GH", "operator": "Vodafone Ghana",     "tier": 1},
    37456:  {"node": "ACC", "city": "Accra",         "country": "GH", "operator": "Busy Internet",      "tier": 2},

    # ── DAR ES SALAAM (DAR) ────────────────────────────────────────
    37182:  {"node": "DAR", "city": "Dar es Salaam", "country": "TZ", "operator": "TTCL",               "tier": 1},
    37040:  {"node": "DAR", "city": "Dar es Salaam", "country": "TZ", "operator": "Liquid Telecom TZ",  "tier": 1},
    37253:  {"node": "DAR", "city": "Dar es Salaam", "country": "TZ", "operator": "TISPA/DEIX",         "tier": 1},
    36925:  {"node": "DAR", "city": "Dar es Salaam", "country": "TZ", "operator": "Vodacom TZ",         "tier": 1},

    # ── ADDIS ABABA (ADD) ──────────────────────────────────────────
    24757:  {"node": "ADD", "city": "Addis Ababa",   "country": "ET", "operator": "Ethio Telecom",      "tier": 1},
    # NOTE: Ethiopia has near-monopoly — very few ASNs
    # Ethio Telecom is the sole licensed ISP as of 2024

    # ── KINSHASA (KIN) ─────────────────────────────────────────────
    36986:  {"node": "KIN", "city": "Kinshasa",      "country": "CD", "operator": "Vodacom DRC",        "tier": 1},
    37594:  {"node": "KIN", "city": "Kinshasa",      "country": "CD", "operator": "Liquid Telecom DRC", "tier": 1},
    37008:  {"node": "KIN", "city": "Kinshasa",      "country": "CD", "operator": "MTN DRC",            "tier": 1},
    37342:  {"node": "KIN", "city": "Kinshasa",      "country": "CD", "operator": "Orange DRC",         "tier": 2},

    # ── KAMPALA (KMP) ──────────────────────────────────────────────
    36991:  {"node": "KMP", "city": "Kampala",       "country": "UG", "operator": "MTN Uganda",         "tier": 1},
    37075:  {"node": "KMP", "city": "Kampala",       "country": "UG", "operator": "Uganda Telecom",     "tier": 1},
    37122:  {"node": "KMP", "city": "Kampala",       "country": "UG", "operator": "UIXP",               "tier": 1},
    36977:  {"node": "KMP", "city": "Kampala",       "country": "UG", "operator": "Airtel Uganda",      "tier": 1},

    # ── LUSAKA (LUS) ───────────────────────────────────────────────
    37154:  {"node": "LUS", "city": "Lusaka",        "country": "ZM", "operator": "ZAMTEL",             "tier": 1},
    37228:  {"node": "LUS", "city": "Lusaka",        "country": "ZM", "operator": "MTN Zambia",         "tier": 1},
    36924:  {"node": "LUS", "city": "Lusaka",        "country": "ZM", "operator": "Liquid Telecom ZM",  "tier": 1},
    328232: {"node": "LUS", "city": "Lusaka",        "country": "ZM", "operator": "ZICTA/ZINX",         "tier": 1},
}

# Reverse map: node_id -> list of ASNs
NODE_TO_ASNS = {}
for asn, info in ASN_TO_NODE.items():
    node = info["node"]
    NODE_TO_ASNS.setdefault(node, []).append(asn)

# Flat set of all tracked ASNs (for fast membership check)
TRACKED_ASNS = set(ASN_TO_NODE.keys())

# Known European/US transit hubs used in trombone paths
# These are the "via" cities we detect when African traffic detours
TRANSIT_HUBS = {
    1273:   "London (CWC/Vodafone)",
    174:    "Ashburn (Cogent)",
    3356:   "Dallas (Lumen/Level3)",
    6762:   "Frankfurt (Telecom Italia Sparkle)",
    5511:   "Paris (Orange)",
    3257:   "Frankfurt (GTT)",
    1299:   "Stockholm (Telia Carrier)",
    2914:   "San Jose (NTT)",
    7018:   "Atlanta (AT&T)",
    3320:   "Frankfurt (Deutsche Telekom)",
    6453:   "New York (TATA Comms)",
    6461:   "Seattle (Zayo)",
}

def asn_to_node(asn: int) -> dict | None:
    """Look up node info for an ASN. Returns None if not in our topology."""
    return ASN_TO_NODE.get(asn)

def is_transit_hub(asn: int) -> str | None:
    """Returns hub name if ASN is a known non-African transit provider."""
    return TRANSIT_HUBS.get(asn)

def as_path_contains_hub(as_path: list[int]) -> str | None:
    """
    Scan an AS path for non-African transit hubs.
    Returns the first hub name found, or None.
    Used to detect trombone routing via European/US backbone.
    """
    for asn in as_path:
        hub = is_transit_hub(asn)
        if hub:
            return hub
    return None