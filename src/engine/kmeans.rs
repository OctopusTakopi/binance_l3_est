//! Mini-batch K-Means clustering for order book quantity levels.
//!
//! # Allocations
//!
//! After the first call to [`MiniBatchKMeans::fit`] the implementation is
//! effectively zero-alloc on subsequent calls:
//!
//! | Buffer | Strategy |
//! |-|-|
//! | `points`, `labels`, `batch_indices`, `sort_buf` | Vec fields — capacity retained between calls, only `clear()` is called |
//! | `counts`, `sums` (per-iteration) | `[u32; MAX_K]` / `[f64; MAX_K]` — stack, zero cost |
//! | `centroid_order`, `label_map` | `[usize; MAX_K]` — stack |
//! | `centroids` | `[Point; MAX_K]` — stack inside the struct |

use rand::Rng;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};

/// Upper bound on the number of clusters. All inline arrays are sized to this.
/// Must be ≥ num_clusters passed to [`MiniBatchKMeans::new`].
const MAX_K: usize = 16;

// ── Internal point type ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
struct Point {
    qty: f64,
}

// ── Normalisation ──────────────────────────────────────────────────────────────

fn normalize(points: &mut [Point]) {
    if points.is_empty() {
        return;
    }
    let mut min_q = f64::MAX;
    let mut max_q = f64::MIN;
    for p in points.iter() {
        min_q = min_q.min(p.qty);
        max_q = max_q.max(p.qty);
    }
    let range_q = max_q - min_q;
    if range_q > 0.0 {
        let inv = 1.0 / range_q;
        for p in points.iter_mut() {
            p.qty = (p.qty - min_q) * inv;
        }
    }
}

// ── MiniBatchKMeans ────────────────────────────────────────────────────────────

/// Mini-batch K-Means with stable, sorted cluster labels.
///
/// All buffers used during [`fit`] are owned by the struct and reused across
/// calls — no heap allocation occurs in the steady-state training loop.
pub struct MiniBatchKMeans {
    num_clusters: usize,
    batch_size: usize,
    max_iter: usize,

    /// Centroid positions, stack-allocated.
    centroids: [Point; MAX_K],
    /// Whether `centroids` has been initialised at least once.
    centroids_ready: bool,

    // ── Reusable heap buffers (capacity is retained between fit() calls) ─────
    /// Normalised point buffer.
    points: Vec<Point>,
    /// Output label buffer; returned by reference from `fit`.
    labels: Vec<usize>,
    /// Scratch buffer for mini-batch index selection.
    batch_indices: Vec<usize>,
    /// Scratch buffer for sorting points during centroid initialisation.
    sort_buf: Vec<Point>,
}

impl MiniBatchKMeans {
    pub fn new(num_clusters: usize, batch_size: usize, max_iter: usize) -> Self {
        assert!(
            num_clusters <= MAX_K,
            "num_clusters ({num_clusters}) exceeds MAX_K ({MAX_K})"
        );
        Self {
            num_clusters,
            batch_size,
            max_iter,
            centroids: [Point::default(); MAX_K],
            centroids_ready: false,
            points: Vec::new(),
            labels: Vec::new(),
            batch_indices: Vec::new(),
            sort_buf: Vec::new(),
        }
    }

    /// Fit on the order book slice and return label assignments.
    ///
    /// The returned slice is valid until the next call to `fit`.
    /// The returned slice is valid until the next call to `fit_iter`.
    pub fn fit_iter<'a, I>(&mut self, iter: I) -> &[usize]
    where
        I: Iterator<Item = (&'a Decimal, &'a VecDeque<Decimal>)>,
    {
        // ── 1. Populate point buffer (reuse capacity) ────────────────────────
        self.points.clear();
        for (_price, deq) in iter {
            for &qty in deq.iter() {
                if qty > Decimal::ZERO {
                    if let Some(q) = qty.to_f64() {
                        self.points.push(Point { qty: q });
                    }
                }
            }
        }

        if self.points.is_empty() {
            self.labels.clear();
            return &self.labels;
        }

        normalize(&mut self.points);

        // ── 2. Initialise centroids on first call or reset ───────────────────
        if !self.centroids_ready {
            self.initialize_centroids();
            self.centroids_ready = true;
        }

        // ── 3. Mini-batch update loop — zero heap allocation ─────────────────
        // Stack-allocated accumulators: MAX_K × 8 bytes = 128 bytes each.
        let mut counts = [0u32; MAX_K];
        let mut sums = [0.0f64; MAX_K];

        let k = self.num_clusters;
        let n = self.points.len();
        let batch = self.batch_size.min(n);
        let mut rng = rand::rng();

        // Reserve once if needed, then only clear() in the loop.
        if self.batch_indices.capacity() < batch {
            self.batch_indices.reserve(batch);
        }

        for _ in 0..self.max_iter {
            self.batch_indices.clear();
            for _ in 0..batch {
                self.batch_indices.push(rng.random_range(0..n));
            }

            // Reset accumulators (stack only, bounded by MAX_K).
            counts[..k].fill(0);
            sums[..k].fill(0.0);

            for &idx in &self.batch_indices {
                let qty = self.points[idx].qty;
                let c = self.closest_centroid(qty);
                sums[c] += qty;
                counts[c] += 1;
            }

            for i in 0..k {
                if counts[i] > 0 {
                    let lr = 1.0 / counts[i] as f64;
                    self.centroids[i].qty =
                        (1.0 - lr) * self.centroids[i].qty + lr * (sums[i] / counts[i] as f64);
                }
            }
        }

        // ── 4. Assign labels ─────────────────────────────────────────────────
        self.labels.clear();
        if self.labels.capacity() < n {
            self.labels.reserve(n - self.labels.len());
        }
        for p in &self.points {
            self.labels.push(self.closest_centroid(p.qty));
        }

        // ── 5. Stabilise: remap labels so cluster-0 = smallest centroid ──────
        // All temporaries on the stack (bounded by MAX_K = 16).
        let mut centroid_order = [0usize; MAX_K];
        for i in 0..k {
            centroid_order[i] = i;
        }
        centroid_order[..k].sort_unstable_by(|&a, &b| {
            self.centroids[a]
                .qty
                .partial_cmp(&self.centroids[b].qty)
                .unwrap_or(Ordering::Equal)
        });

        let mut label_map = [0usize; MAX_K];
        for (new_label, &old_label) in centroid_order[..k].iter().enumerate() {
            label_map[old_label] = new_label;
        }
        for label in self.labels.iter_mut() {
            *label = label_map[*label];
        }

        &self.labels
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Euclidean (1-D) nearest centroid.
    #[inline(always)]
    fn closest_centroid(&self, qty: f64) -> usize {
        let mut min_dist = f64::INFINITY;
        let mut min_idx = 0;
        for i in 0..self.num_clusters {
            let d = (qty - self.centroids[i].qty).abs();
            if d < min_dist {
                min_dist = d;
                min_idx = i;
            }
        }
        min_idx
    }

    /// Deterministic centroid initialisation: evenly spaced along sorted qty.
    /// Uses `sort_buf` as scratch — allocation only on the very first call.
    fn initialize_centroids(&mut self) {
        self.sort_buf.clear();
        self.sort_buf.extend_from_slice(&self.points);
        self.sort_buf
            .sort_unstable_by(|a, b| a.qty.partial_cmp(&b.qty).unwrap_or(Ordering::Equal));

        let n = self.sort_buf.len();
        let k = self.num_clusters;
        let step = if k > 1 { (n - 1) / (k - 1) } else { 0 };
        for i in 0..k {
            let idx = (i * step).min(n - 1);
            self.centroids[i] = self.sort_buf[idx];
        }
        // Pad remaining slots with the first point if n < k.
        for i in n.min(k)..k {
            self.centroids[i] = self.sort_buf[0];
        }
    }
}

// ── Public helpers ─────────────────────────────────────────────────────────────

/// Combine an order book with pre-computed K-Means labels into a
/// price → `[(qty, cluster)]` map for rendering.
pub fn build_clustered_orders<'a, I>(
    iter: I,
    labels: &[usize],
) -> Vec<(&'a Decimal, Vec<(Decimal, usize)>)>
where
    I: Iterator<Item = (&'a Decimal, &'a VecDeque<Decimal>)>,
{
    let mut out = Vec::with_capacity(200);
    let mut idx = 0;
    for (price, deq) in iter {
        let mut level_orders = Vec::with_capacity(deq.len());
        for &qty in deq.iter() {
            if qty > Decimal::ZERO {
                if idx < labels.len() {
                    level_orders.push((qty, labels[idx]));
                    idx += 1;
                }
            }
        }
        out.push((price, level_orders));
    }
    out
}

/// Convenience wrapper: creates a fresh `MiniBatchKMeans`, fits it, and
/// returns a fully-owned clustered book. Prefer calling `fit` directly on a
/// persistent `MiniBatchKMeans` if you want to avoid repeated init overhead.
#[allow(dead_code)]
pub fn cluster_order_book(
    order_book: &BTreeMap<Decimal, VecDeque<Decimal>>,
    num_classes: usize,
    batch_size: usize,
    max_iter: usize,
) -> Vec<(&Decimal, Vec<(Decimal, usize)>)> {
    let mut kmeans = MiniBatchKMeans::new(num_classes, batch_size, max_iter);
    let labels = kmeans.fit_iter(order_book.iter()).to_vec(); // one-shot, so copy is fine
    build_clustered_orders(order_book.iter(), &labels)
}
