//! engine/src/geodesic.rs
//! ─────────────────────────────────────────────────────────────────────────
//! Great-circle distance computation (Haversine formula).
//!
//! This is the geometric ground truth that the Trombone Detector compares
//! actual BGP paths against. If a BGP path is more than 2× the geodesic
//! distance, it is flagged as a trombone detour.
//!
//! This module mirrors the Python `_haversine_km()` in graph_sage.py.
//! Rust owns this computation in production because:
//!   - It runs on every BGP event (high throughput requirement)
//!   - f64 precision matters for short inter-city paths (<500km)
//!   - No Python GIL overhead on the hot path
//! ─────────────────────────────────────────────────────────────────────────

use std::f64::consts::PI;

/// Earth's mean radius in kilometres (WGS-84).
const EARTH_RADIUS_KM: f64 = 6371.0;

/// A geographic coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coord {
    pub lat: f64,   // decimal degrees, positive = North
    pub lon: f64,   // decimal degrees, positive = East
}

impl Coord {
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    fn to_radians(self) -> (f64, f64) {
        (self.lat * PI / 180.0, self.lon * PI / 180.0)
    }
}

/// Haversine great-circle distance between two coordinates, in kilometres.
///
/// # Arguments
/// * `a` - Source coordinate
/// * `b` - Destination coordinate
///
/// # Returns
/// Distance in kilometres (f64, always ≥ 0.0).
///
/// # Example
/// ```
/// let nairobi  = Coord::new(-1.286,  36.817);
/// let lagos    = Coord::new( 6.524,   3.379);
/// let dist_km  = haversine_km(nairobi, lagos);
/// assert!((dist_km - 3961.0).abs() < 10.0);
/// ```
pub fn haversine_km(a: Coord, b: Coord) -> f64 {
    let (phi1, lambda1) = a.to_radians();
    let (phi2, lambda2) = b.to_radians();

    let d_phi    = phi2 - phi1;
    let d_lambda = lambda2 - lambda1;

    let h = (d_phi / 2.0).sin().powi(2)
        + phi1.cos() * phi2.cos() * (d_lambda / 2.0).sin().powi(2);

    2.0 * EARTH_RADIUS_KM * h.sqrt().asin()
}

/// BGP path distance estimate from a sequence of hop coordinates.
///
/// Sums the geodesic distance between each consecutive hop pair.
/// Used to estimate total BGP path length when hop coordinates are known
/// (via CAIDA ITDK AS geolocation).
///
/// # Arguments
/// * `hops` - Ordered slice of coordinates [src, hop1, hop2, ..., dst]
///
/// # Returns
/// Total path length in kilometres.
pub fn path_distance_km(hops: &[Coord]) -> f64 {
    if hops.len() < 2 {
        return 0.0;
    }
    hops.windows(2)
        .map(|pair| haversine_km(pair[0], pair[1]))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nairobi_to_lagos() {
        let nbo = Coord::new(-1.286, 36.817);
        let los = Coord::new(6.524, 3.379);
        let dist = haversine_km(nbo, los);
        // Known approximate distance ~3961km
        assert!((dist - 3961.0).abs() < 20.0, "got {dist:.1}km");
    }

    #[test]
    fn same_point_is_zero() {
        let nbo = Coord::new(-1.286, 36.817);
        assert!(haversine_km(nbo, nbo) < 1e-6);
    }

    #[test]
    fn symmetric() {
        let nbo = Coord::new(-1.286, 36.817);
        let jnb = Coord::new(-26.204, 28.047);
        let diff = (haversine_km(nbo, jnb) - haversine_km(jnb, nbo)).abs();
        assert!(diff < 1e-6, "asymmetry: {diff}");
    }

    #[test]
    fn path_distance_sums_hops() {
        let nbo = Coord::new(-1.286,  36.817);  // Nairobi
        let fra = Coord::new(50.110,   8.682);  // Frankfurt (European hub)
        let los = Coord::new( 6.524,   3.379);  // Lagos
        // Nairobi→Frankfurt→Lagos path should be much longer than direct
        let direct  = haversine_km(nbo, los);
        let via_fra = path_distance_km(&[nbo, fra, los]);
        assert!(via_fra > direct * 1.5,
            "via_fra={via_fra:.0}km, direct={direct:.0}km");
    }
}