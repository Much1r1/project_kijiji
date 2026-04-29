//! engine/src/trombone.rs
//! ─────────────────────────────────────────────────────────────────────────
//! BGP Trombone Detour Detector
//!
//! Classifies every BGP routing event into one of three categories:
//!
//!   DIRECT       — Traffic takes a near-optimal regional path.
//!                  detour_ratio ≤ 1.4
//!
//!   POLICY       — Detour exists but is BGP valley-free policy driven.
//!                  ISPs intentionally route via a higher-tier AS.
//!                  1.4 < detour_ratio ≤ TROMBONE_THRESHOLD
//!
//!   TROMBONE     — Unnecessary detour through a European/non-regional hub.
//!                  detour_ratio > TROMBONE_THRESHOLD (default: 2.0)
//!                  This is the target of Thesis Pillar 1.
//!
//! The 2.0 threshold is a tunable hyperparameter. It is calibrated against
//! known detour cases from CAIDA trombone routing studies. See THESIS_NOTES.md
//! §3.2 for justification.
//!
//! This module is the production implementation of the Python prototype in
//! data/ingest_topology.py classify_detour(). Rust owns this on the hot path.
//! ─────────────────────────────────────────────────────────────────────────

use serde::{Deserialize, Serialize};

use crate::geodesic::{haversine_km, Coord};

// ─────────────────────────────────────────────────────────────────────────
// TUNABLE HYPERPARAMETER
// See thesis §3.2 — empirically calibrated against CAIDA trombone studies.
// Exposed as a constant so it can be overridden via CLI (see main.rs).
// ─────────────────────────────────────────────────────────────────────────

pub const TROMBONE_THRESHOLD: f64 = 2.0;
pub const POLICY_THRESHOLD:   f64 = 1.4;

// ─────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────

/// Three-class detour taxonomy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DetourClass {
    /// Near-optimal regional routing. detour_ratio ≤ 1.4.
    Direct,
    /// Intentional BGP policy routing. 1.4 < ratio ≤ 2.0.
    Policy,
    /// Unnecessary European/non-regional transit. ratio > 2.0.
    Trombone,
}

impl DetourClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            DetourClass::Direct   => "DIRECT",
            DetourClass::Policy   => "POLICY",
            DetourClass::Trombone => "TROMBONE",
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            DetourClass::Direct   => "🟢",
            DetourClass::Policy   => "🟡",
            DetourClass::Trombone => "🔴",
        }
    }
}

/// A single BGP routing event ingested from the data pipeline.
#[derive(Debug, Clone, Deserialize)]
pub struct BgpEvent {
    /// ISO timestamp of the observation.
    pub timestamp: String,

    /// Source city code (e.g. "NBO").
    pub src_city: String,

    /// Destination city code (e.g. "LOS").
    pub dst_city: String,

    /// Source AS number.
    pub src_asn: u32,

    /// Destination AS number.
    pub dst_asn: u32,

    /// Source city coordinates.
    pub src_lat: f64,
    pub src_lon: f64,

    /// Destination city coordinates.
    pub dst_lat: f64,
    pub dst_lon: f64,

    /// Transit hub used (e.g. "FRA"), if any.
    /// None means the event was routed directly.
    pub transit_hub: Option<String>,

    /// Transit hub coordinates (required if transit_hub is Some).
    pub hub_lat: Option<f64>,
    pub hub_lon: Option<f64>,

    /// Observed end-to-end latency in milliseconds.
    pub observed_latency_ms: f64,
}

/// Full classification result for one BGP event.
#[derive(Debug, Clone, Serialize)]
pub struct TromboneResult {
    pub timestamp:           String,
    pub src_city:            String,
    pub dst_city:            String,
    pub src_asn:             u32,
    pub dst_asn:             u32,
    pub geodesic_km:         f64,
    pub bgp_path_km:         f64,
    pub detour_ratio:        f64,
    pub detour_class:        String,
    pub transit_hub:         String,
    pub observed_latency_ms: f64,

    /// Estimated latency waste in ms due to the detour.
    /// = observed_latency_ms × (1 - 1/detour_ratio)
    /// Zero for DIRECT events.
    pub latency_waste_ms:    f64,
}

// ─────────────────────────────────────────────────────────────────────────
// CLASSIFIER
// ─────────────────────────────────────────────────────────────────────────

/// Classifies a BGP routing event using the trombone detection algorithm.
///
/// # Algorithm
/// 1. Compute geodesic distance between src and dst (Haversine).
/// 2. Compute BGP path distance:
///    - If transit hub present: src→hub + hub→dst
///    - Otherwise: geodesic × small_overhead_factor (1.0–1.35)
/// 3. detour_ratio = bgp_path_km / geodesic_km
/// 4. Classify by threshold.
///
/// # Arguments
/// * `event`     — The BGP event to classify.
/// * `threshold` — Trombone threshold (default: TROMBONE_THRESHOLD = 2.0).
///
/// # Returns
/// A `TromboneResult` with full classification metadata.
pub fn classify(event: &BgpEvent, threshold: f64) -> TromboneResult {
    let src  = Coord::new(event.src_lat, event.src_lon);
    let dst  = Coord::new(event.dst_lat, event.dst_lon);

    let geodesic_km = haversine_km(src, dst);

    // BGP path distance: via hub if present, else direct with small overhead
    let bgp_path_km = match (event.hub_lat, event.hub_lon) {
        (Some(hlat), Some(hlon)) => {
            let hub = Coord::new(hlat, hlon);
            haversine_km(src, hub) + haversine_km(hub, dst)
        }
        _ => {
            // No hub recorded — path is near-geodesic with small routing overhead
            geodesic_km * 1.15
        }
    };

    let detour_ratio = if geodesic_km > 0.0 {
        bgp_path_km / geodesic_km
    } else {
        1.0
    };

    let detour_class = classify_ratio(detour_ratio, threshold);

    // Latency waste: portion of observed latency attributable to the detour
    let latency_waste_ms = match detour_class {
        DetourClass::Direct => 0.0,
        _ => event.observed_latency_ms * (1.0 - 1.0 / detour_ratio),
    };

    TromboneResult {
        timestamp:           event.timestamp.clone(),
        src_city:            event.src_city.clone(),
        dst_city:            event.dst_city.clone(),
        src_asn:             event.src_asn,
        dst_asn:             event.dst_asn,
        geodesic_km:         round2(geodesic_km),
        bgp_path_km:         round2(bgp_path_km),
        detour_ratio:        round3(detour_ratio),
        detour_class:        detour_class.as_str().to_string(),
        transit_hub:         event.transit_hub.clone().unwrap_or_else(|| "NONE".into()),
        observed_latency_ms: event.observed_latency_ms,
        latency_waste_ms:    round2(latency_waste_ms),
    }
}

/// Classify a raw detour ratio into the three-class taxonomy.
pub fn classify_ratio(ratio: f64, threshold: f64) -> DetourClass {
    if ratio > threshold {
        DetourClass::Trombone
    } else if ratio > POLICY_THRESHOLD {
        DetourClass::Policy
    } else {
        DetourClass::Direct
    }
}

// ─────────────────────────────────────────────────────────────────────────
// BATCH PROCESSING
// ─────────────────────────────────────────────────────────────────────────

/// Classify a batch of BGP events and return aggregated statistics.
pub fn classify_batch(
    events:    &[BgpEvent],
    threshold: f64,
) -> (Vec<TromboneResult>, BatchStats) {
    let results: Vec<TromboneResult> = events
        .iter()
        .map(|e| classify(e, threshold))
        .collect();

    let stats = BatchStats::from_results(&results);
    (results, stats)
}

/// Aggregate statistics for a batch of classified events.
#[derive(Debug, Serialize)]
pub struct BatchStats {
    pub total:            usize,
    pub trombone_count:   usize,
    pub policy_count:     usize,
    pub direct_count:     usize,
    pub trombone_rate:    f64,
    pub mean_detour_ratio: f64,
    pub total_waste_ms:   f64,
    pub mean_waste_ms:    f64,
}

impl BatchStats {
    pub fn from_results(results: &[TromboneResult]) -> Self {
        let total = results.len();
        if total == 0 {
            return Self {
                total: 0, trombone_count: 0, policy_count: 0, direct_count: 0,
                trombone_rate: 0.0, mean_detour_ratio: 0.0,
                total_waste_ms: 0.0, mean_waste_ms: 0.0,
            };
        }

        let trombone_count = results.iter().filter(|r| r.detour_class == "TROMBONE").count();
        let policy_count   = results.iter().filter(|r| r.detour_class == "POLICY").count();
        let direct_count   = results.iter().filter(|r| r.detour_class == "DIRECT").count();
        let total_waste_ms: f64 = results.iter().map(|r| r.latency_waste_ms).sum();
        let mean_detour:    f64 = results.iter().map(|r| r.detour_ratio).sum::<f64>()
                                  / total as f64;

        Self {
            total,
            trombone_count,
            policy_count,
            direct_count,
            trombone_rate:    round3(trombone_count as f64 / total as f64),
            mean_detour_ratio: round3(mean_detour),
            total_waste_ms:   round2(total_waste_ms),
            mean_waste_ms:    round2(total_waste_ms / total as f64),
        }
    }

    pub fn print_summary(&self) {
        println!("  ┌─────────────────────────────────────────┐");
        println!("  │  Batch Classification Summary            │");
        println!("  ├─────────────────────────────────────────┤");
        println!("  │  Total events    : {:>6}                │", self.total);
        println!("  │  🔴 TROMBONE     : {:>6}  ({:>5.1}%)     │",
            self.trombone_count, self.trombone_rate * 100.0);
        println!("  │  🟡 POLICY       : {:>6}                │", self.policy_count);
        println!("  │  🟢 DIRECT       : {:>6}                │", self.direct_count);
        println!("  ├─────────────────────────────────────────┤");
        println!("  │  Mean detour ratio : {:.3}               │", self.mean_detour_ratio);
        println!("  │  Total latency waste: {:.1}ms             │", self.total_waste_ms);
        println!("  │  Mean waste/event   : {:.1}ms             │", self.mean_waste_ms);
        println!("  └─────────────────────────────────────────┘");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────

fn round2(v: f64) -> f64 { (v * 100.0).round() / 100.0 }
fn round3(v: f64) -> f64 { (v * 1000.0).round() / 1000.0 }

// ─────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(
        src_lat: f64, src_lon: f64,
        dst_lat: f64, dst_lon: f64,
        hub_lat: Option<f64>, hub_lon: Option<f64>,
        hub_name: Option<&str>,
        latency_ms: f64,
    ) -> BgpEvent {
        BgpEvent {
            timestamp: "2024-01-01T00:00:00Z".into(),
            src_city: "NBO".into(), dst_city: "LOS".into(),
            src_asn: 36866, dst_asn: 29465,
            src_lat, src_lon, dst_lat, dst_lon,
            transit_hub: hub_name.map(|s| s.to_string()),
            hub_lat, hub_lon,
            observed_latency_ms: latency_ms,
        }
    }

    #[test]
    fn direct_route_classified_direct() {
        // Nairobi → Lagos, no hub
        let event = make_event(
            -1.286, 36.817, 6.524, 3.379,
            None, None, None, 45.0,
        );
        let result = classify(&event, TROMBONE_THRESHOLD);
        assert_eq!(result.detour_class, "DIRECT");
        assert_eq!(result.latency_waste_ms, 0.0);
    }

    #[test]
    fn frankfurt_detour_classified_trombone() {
        // Nairobi → Frankfurt → Lagos
        // geodesic NBO→LOS ≈ 3961km
        // via FRA: NBO→FRA ≈ 6360km + FRA→LOS ≈ 5136km = ~11496km
        // ratio ≈ 2.9 → TROMBONE
        let event = make_event(
            -1.286, 36.817,   // Nairobi
             6.524,  3.379,   // Lagos
            Some(50.110), Some(8.682), Some("FRA"),  // Frankfurt
            180.0,
        );
        let result = classify(&event, TROMBONE_THRESHOLD);
        assert_eq!(result.detour_class, "TROMBONE",
            "ratio was {}", result.detour_ratio);
        assert!(result.detour_ratio > 2.0);
        assert!(result.latency_waste_ms > 0.0);
    }

    #[test]
    fn moderate_detour_classified_policy() {
        // DAR → ACC via JNB: routes south to Johannesburg then west to Accra.
        // Verified ratio: direct=4584km, via JNB=7127km → ratio=1.555
        // Cleanly in POLICY band (1.4 < ratio ≤ 2.0).
        let event = make_event(
            -6.792, 39.208,    // Dar es Salaam
             5.603, -0.187,    // Accra
            Some(-26.204), Some(28.047), Some("JNB"),  // Johannesburg detour
            95.0,
        );
        let result = classify(&event, TROMBONE_THRESHOLD);
        assert!(result.detour_ratio > POLICY_THRESHOLD,
            "expected > {POLICY_THRESHOLD}, got {}", result.detour_ratio);
        assert!(result.detour_ratio < TROMBONE_THRESHOLD,
            "expected < {TROMBONE_THRESHOLD}, got {}", result.detour_ratio);
        assert_eq!(result.detour_class, "POLICY");
    }

    #[test]
    fn batch_stats_correct() {
        let events = vec![
            // Direct: NBO → DAR (no hub)
            make_event(-1.286, 36.817, -6.792, 39.208, None, None, None, 30.0),
            // Trombone: NBO → LOS via FRA
            make_event(-1.286, 36.817, 6.524, 3.379,
                       Some(50.110), Some(8.682), Some("FRA"), 180.0),
        ];
        let (results, stats) = classify_batch(&events, TROMBONE_THRESHOLD);
        assert_eq!(stats.total, 2);
        assert_eq!(stats.trombone_count, 1);
        assert_eq!(stats.direct_count, 1);
        assert!(stats.total_waste_ms > 0.0);
    }

    #[test]
    fn zero_geodesic_does_not_panic() {
        // Same src and dst coordinates — edge case
        let event = make_event(
            -1.286, 36.817, -1.286, 36.817,
            None, None, None, 1.0,
        );
        let result = classify(&event, TROMBONE_THRESHOLD);
        assert_eq!(result.detour_ratio, 1.0);
    }
}
// (test fix appended below original)