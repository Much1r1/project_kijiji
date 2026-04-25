//! engine/src/main.rs
mod geodesic;
mod trombone;

use std::io::{self, BufRead};
use trombone::{classify_batch, BgpEvent, TROMBONE_THRESHOLD};

struct Config {
    threshold:  f64,
    stats_only: bool,
    demo:       bool,
}

impl Config {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        if args.iter().any(|a| a == "--help" || a == "-h") {
            eprintln!("kijiji-engine [--threshold <f64>] [--stats-only] [--demo]");
            std::process::exit(0);
        }
        let threshold = args.windows(2)
            .find(|w| w[0] == "--threshold")
            .and_then(|w| w[1].parse::<f64>().ok())
            .unwrap_or(TROMBONE_THRESHOLD);
        Config {
            threshold,
            stats_only: args.iter().any(|a| a == "--stats-only"),
            demo:       args.iter().any(|a| a == "--demo"),
        }
    }
}

fn demo_events() -> Vec<BgpEvent> {
    vec![
        BgpEvent {
            timestamp: "2024-07-01T08:00:00Z".into(),
            src_city: "NBO".into(), dst_city: "DAR".into(),
            src_asn: 36866, dst_asn: 37182,
            src_lat: -1.286, src_lon: 36.817,
            dst_lat: -6.792, dst_lon: 39.208,
            transit_hub: None, hub_lat: None, hub_lon: None,
            observed_latency_ms: 24.0,
        },
        BgpEvent {
            timestamp: "2024-07-01T08:05:00Z".into(),
            src_city: "JNB".into(), dst_city: "CPT".into(),
            src_asn: 3741, dst_asn: 3741,
            src_lat: -26.204, src_lon: 28.047,
            dst_lat: -33.925, dst_lon: 18.424,
            transit_hub: None, hub_lat: None, hub_lon: None,
            observed_latency_ms: 18.0,
        },
        BgpEvent {
            timestamp: "2024-07-01T08:10:00Z".into(),
            src_city: "NBO".into(), dst_city: "LOS".into(),
            src_asn: 36866, dst_asn: 29465,
            src_lat: -1.286, src_lon: 36.817,
            dst_lat:  6.524, dst_lon:  3.379,
            transit_hub: Some("FRA".into()),
            hub_lat: Some(50.110), hub_lon: Some(8.682),
            observed_latency_ms: 187.0,
        },
        BgpEvent {
            timestamp: "2024-07-01T08:15:00Z".into(),
            src_city: "KIN".into(), dst_city: "ACC".into(),
            src_asn: 36916, dst_asn: 29614,
            src_lat: -4.322, src_lon: 15.322,
            dst_lat:  5.603, dst_lon: -0.187,
            transit_hub: Some("LON".into()),
            hub_lat: Some(51.509), hub_lon: Some(-0.118),
            observed_latency_ms: 210.0,
        },
        BgpEvent {
            timestamp: "2024-07-01T08:20:00Z".into(),
            src_city: "ADD".into(), dst_city: "KLA".into(),
            src_asn: 24757, dst_asn: 36977,
            src_lat:  9.145, src_lon: 40.489,
            dst_lat:  0.347, dst_lon: 32.582,
            transit_hub: Some("NBO".into()),
            hub_lat: Some(-1.286), hub_lon: Some(36.817),
            observed_latency_ms: 55.0,
        },
        BgpEvent {
            timestamp: "2024-07-01T08:25:00Z".into(),
            src_city: "DAR".into(), dst_city: "LOS".into(),
            src_asn: 37182, dst_asn: 29465,
            src_lat: -6.792, src_lon: 39.208,
            dst_lat:  6.524, dst_lon:  3.379,
            transit_hub: Some("AMS".into()),
            hub_lat: Some(52.370), hub_lon: Some(4.895),
            observed_latency_ms: 195.0,
        },
        BgpEvent {
            timestamp: "2024-07-01T08:30:00Z".into(),
            src_city: "CMN".into(), dst_city: "ACC".into(),
            src_asn: 6713, dst_asn: 29614,
            src_lat: 33.589, src_lon: -7.604,
            dst_lat:  5.603, dst_lon: -0.187,
            transit_hub: None, hub_lat: None, hub_lon: None,
            observed_latency_ms: 42.0,
        },
    ]
}

fn read_events_from_stdin() -> Vec<BgpEvent> {
    let stdin = io::stdin();
    let mut events = Vec::new();
    for (i, line) in stdin.lock().lines().enumerate() {
        let line = match line { Ok(l) => l, Err(e) => { eprintln!("line {i}: {e}"); continue; } };
        let line = line.trim();
        if line.is_empty() { continue; }
        match serde_json::from_str::<BgpEvent>(line) {
            Ok(e)  => events.push(e),
            Err(e) => eprintln!("⚠️  parse error line {i}: {e}"),
        }
    }
    events
}

fn main() {
    let config = Config::from_args();

    eprintln!("🌍 Kijiji Engine — BGP Trombone Detector  (threshold: {:.2}×)", config.threshold);

    let events: Vec<BgpEvent> = if config.demo {
        eprintln!("   Mode: DEMO\n");
        demo_events()
    } else {
        eprintln!("   Mode: STDIN\n");
        read_events_from_stdin()
    };

    if events.is_empty() {
        eprintln!("❌ No events. Use --demo or pipe NDJSON via stdin.");
        std::process::exit(1);
    }

    let (results, stats) = classify_batch(&events, config.threshold);

    if !config.stats_only {
        for result in &results {
            if let Ok(json) = serde_json::to_string(result) { println!("{json}"); }
        }
    }

    // Per-event summary to stderr
    eprintln!("  {:<6}  {:<6}  {:<10}  {:>10}  {:>10}  {:>8}  {:>12}",
        "Src", "Dst", "Class", "Geodesic", "BGP Path", "Ratio", "Waste(ms)");
    eprintln!("  {}  {}  {}  {}  {}  {}  {}",
        "─".repeat(6),"─".repeat(6),"─".repeat(10),
        "─".repeat(10),"─".repeat(10),"─".repeat(8),"─".repeat(12));
    for r in &results {
        let icon = match r.detour_class.as_str() { "TROMBONE" => "🔴", "POLICY" => "🟡", _ => "🟢" };
        eprintln!("  {:<6}  {:<6}  {} {:<8}  {:>8.0}km  {:>8.0}km  {:>8.3}  {:>10.1}ms",
            r.src_city, r.dst_city, icon, r.detour_class,
            r.geodesic_km, r.bgp_path_km, r.detour_ratio, r.latency_waste_ms);
    }
    eprintln!();
    stats.print_summary();
    eprintln!("\n✅ Pipe stdout → ClickHouse or Python GNN pipeline.");
}