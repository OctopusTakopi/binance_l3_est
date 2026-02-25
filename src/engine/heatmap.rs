//! Heatmap state: rolling z-score statistics and depth-time pixel buffer.

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::collections::{BTreeMap, VecDeque};

/// Manages the depth-time heatmap: rolling orderbook statistics and the
/// accumulated pixel-column buffer.
pub struct HeatmapState {
    /// Columns stored as normalised z+offset values; newest at back.
    pub data: VecDeque<Vec<f64>>,
    /// Rolling EMA of per-level quantity magnitude.
    pub rolling_mean_qty: f64,
    /// Rolling EMA of per-level quantity std deviation.
    pub rolling_std_qty: f64,
    pub warmup_samples: usize,
    /// Maximum Z-score observed during the warmup period, used for auto-scaling UI.
    pub max_z_score: f64,
    /// Maximum number of columns kept (== displayed width in pixels).
    pub width: usize,
    /// Number of price levels rendered per column (== displayed height in pixels).
    pub height: usize,
    /// Pool of pre-allocated column vectors to reuse instead of allocating new ones.
    pub snapshot_pool: Vec<Vec<f64>>,
}

impl HeatmapState {
    /// Create a new heatmap with the given pixel dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: VecDeque::new(),
            rolling_mean_qty: 0.0,
            rolling_std_qty: 1.0,
            warmup_samples: 0,
            max_z_score: 1.0,
            width,
            height,
            snapshot_pool: Vec::new(),
        }
    }

    /// Reset all accumulated state (called on symbol change or refetch).
    pub fn reset(&mut self) {
        self.data.clear();
        self.rolling_mean_qty = 0.0;
        self.rolling_std_qty = 1.0;
        self.warmup_samples = 0;
        self.max_z_score = 1.0;
        self.snapshot_pool.extend(self.data.drain(..));
    }

    /// Update rolling quantity statistics from the current book state.
    ///
    /// Should be called frequently so the EMA can settle before heatmap
    /// rendering begins.
    pub fn update_rolling_stats(
        &mut self,
        bids: &BTreeMap<Decimal, VecDeque<Decimal>>,
        asks: &BTreeMap<Decimal, VecDeque<Decimal>>,
    ) {
        if bids.is_empty() || asks.is_empty() {
            return;
        }

        let iter = bids
            .values()
            .chain(asks.values())
            .map(|dq| {
                let mut sum = 0.0;
                for &q in dq {
                    sum += q.to_f64().unwrap_or(0.0);
                }
                sum
            })
            .filter(|&q| q > 0.0);

        let mut count = 0_usize;
        let mut sum = 0.0;
        for q in iter.clone() {
            sum += q;
            count += 1;
        }

        if count == 0 {
            return;
        }

        let m = sum / count as f64;
        let mut var_sum = 0.0;
        for q in iter {
            var_sum += (q - m).powi(2);
        }
        let v = var_sum / count as f64;
        let s = v.sqrt().max(1e-9);

        let alpha = if self.warmup_samples < 30 { 0.05 } else { 0.01 };

        if self.rolling_mean_qty == 0.0 {
            self.rolling_mean_qty = m;
            self.rolling_std_qty = s;
        } else {
            self.rolling_mean_qty = self.rolling_mean_qty * (1.0 - alpha) + m * alpha;
            self.rolling_std_qty = self.rolling_std_qty * (1.0 - alpha) + s * alpha;
        }
        self.warmup_samples += 1;
    }

    /// Compute and append a new column snapshot (z-scores + offset encoded as
    /// sign convention) only if past the warmup threshold.
    pub fn append(
        &mut self,
        bids: &BTreeMap<Decimal, VecDeque<Decimal>>,
        asks: &BTreeMap<Decimal, VecDeque<Decimal>>,
    ) {
        self.update_rolling_stats(bids, asks);

        let mean = self.rolling_mean_qty;
        let std = self.rolling_std_qty.max(1e-9);

        if self.warmup_samples < 30 {
            let mut local_max_z = 0.0_f64;
            for (_, qty_deq) in asks.iter().take(self.height / 2) {
                let mut qty = 0.0;
                for &q in qty_deq {
                    qty += q.to_f64().unwrap_or(0.0);
                }
                if qty > 0.0 {
                    local_max_z = local_max_z.max((qty - mean) / std);
                }
            }
            for (_, qty_deq) in bids.iter().rev().take(self.height / 2) {
                let mut qty = 0.0;
                for &q in qty_deq {
                    qty += q.to_f64().unwrap_or(0.0);
                }
                if qty > 0.0 {
                    local_max_z = local_max_z.max((qty - mean) / std);
                }
            }
            self.max_z_score = self.max_z_score.max(local_max_z);
            return;
        }

        let mut snapshot = self
            .snapshot_pool
            .pop()
            .unwrap_or_else(|| vec![0.0_f64; self.height]);
        // Fast clear just in case it's a reused vector
        snapshot.fill(0.0_f64);

        // Top half → asks (reversed so best ask is nearest mid).
        let ask_iter = asks.iter().take(self.height / 2).rev();
        for (cell, (_, qty_deq)) in snapshot.iter_mut().take(self.height / 2).zip(ask_iter) {
            let mut qty = 0.0;
            for &q in qty_deq {
                qty += q.to_f64().unwrap_or(0.0);
            }
            if qty > 0.0 {
                *cell = (qty - mean) / std + 10.0;
            }
        }

        // Bottom half → bids (reversed so best bid is nearest mid).
        let bid_iter = bids.iter().rev().take(self.height / 2);
        for (cell, (_, qty_deq)) in snapshot.iter_mut().skip(self.height / 2).zip(bid_iter) {
            let mut qty = 0.0;
            for &q in qty_deq {
                qty += q.to_f64().unwrap_or(0.0);
            }
            if qty > 0.0 {
                *cell = -((qty - mean) / std + 10.0); // Negative → bid
            }
        }

        self.data.push_back(snapshot);
        if self.data.len() > self.width {
            if let Some(old_snapshot) = self.data.pop_front() {
                self.snapshot_pool.push(old_snapshot);
            }
        }
    }
}
