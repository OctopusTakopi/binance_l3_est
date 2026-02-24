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
    /// Number of samples since the last reset — used for warmup gating.
    pub warmup_samples: usize,
    /// Maximum number of columns kept (== displayed width in pixels).
    pub width: usize,
    /// Number of price levels rendered per column (== displayed height in pixels).
    pub height: usize,
}

impl HeatmapState {
    /// Create a new heatmap with the given pixel dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: VecDeque::new(),
            rolling_mean_qty: 0.0,
            rolling_std_qty: 1.0,
            warmup_samples: 0,
            width,
            height,
        }
    }

    /// Reset all accumulated state (called on symbol change or refetch).
    pub fn reset(&mut self) {
        self.data.clear();
        self.rolling_mean_qty = 0.0;
        self.rolling_std_qty = 1.0;
        self.warmup_samples = 0;
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

        let levels: Vec<f64> = bids
            .values()
            .chain(asks.values())
            .map(|v| v.iter().sum::<Decimal>().to_f64().unwrap_or(0.0))
            .filter(|&q| q > 0.0)
            .collect();

        if levels.is_empty() {
            return;
        }

        let m = levels.iter().sum::<f64>() / levels.len() as f64;
        let v = levels.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / levels.len() as f64;
        let s = v.sqrt().max(1e-9);

        let alpha = if self.warmup_samples < 200 {
            0.05
        } else {
            0.01
        };

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
        if self.warmup_samples < 200 {
            return;
        }

        let mean = self.rolling_mean_qty;
        let std = self.rolling_std_qty.max(1e-9);

        let mut snapshot = vec![0.0_f64; self.height];

        // Top half → asks (reversed so best ask is nearest mid).
        let ask_iter = asks.iter().take(self.height / 2).rev();
        for (cell, (_, qty_deq)) in snapshot.iter_mut().take(self.height / 2).zip(ask_iter) {
            let qty = qty_deq.iter().sum::<Decimal>().to_f64().unwrap_or(0.0);
            if qty > 0.0 {
                *cell = (qty - mean) / std + 10.0;
            }
        }

        // Bottom half → bids (reversed so best bid is nearest mid).
        let bid_iter = bids.iter().rev().take(self.height / 2);
        for (cell, (_, qty_deq)) in snapshot.iter_mut().skip(self.height / 2).zip(bid_iter) {
            let qty = qty_deq.iter().sum::<Decimal>().to_f64().unwrap_or(0.0);
            if qty > 0.0 {
                *cell = -((qty - mean) / std + 10.0); // Negative → bid
            }
        }

        self.data.push_back(snapshot);
        if self.data.len() > self.width {
            self.data.pop_front();
        }
    }
}
