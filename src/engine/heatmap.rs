//! Heatmap state: rolling z-score statistics and depth-time pixel buffer.

use crate::engine::order_book::OrderBook;
use std::collections::VecDeque;

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

    /// Incremental ID to track when data has changed.
    pub last_update_id: u64,
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
            last_update_id: 0,
        }
    }

    /// Reset all accumulated state (called on symbol change or refetch).
    pub fn reset(&mut self) {
        self.snapshot_pool.extend(self.data.drain(..));
        self.rolling_mean_qty = 0.0;
        self.rolling_std_qty = 1.0;
        self.warmup_samples = 0;
        self.max_z_score = 1.0;
        self.last_update_id = self.last_update_id.wrapping_add(1);
    }

    /// Update rolling quantity statistics from the current book state.
    pub fn update_rolling_stats(&mut self, book: &OrderBook) {
        let mut count = 0_usize;
        let mut sum = 0.0;
        
        let levels_per_side = self.height / 2;
        
        // Use a single pass for mean and variance if possible, or just be efficient with iterators
        let bids = book.iter_bids().take(levels_per_side);
        let asks = book.iter_asks().take(levels_per_side);

        let mut qtys = Vec::with_capacity(self.height); // Small stack-ish alloc
        for (_, qty) in bids.chain(asks) {
            if qty > 0.0 {
                sum += qty;
                count += 1;
                qtys.push(qty);
            }
        }

        if count == 0 {
            return;
        }

        let m = sum / count as f64;
        let mut var_sum = 0.0;
        for q in &qtys {
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

    /// Compute and append a new column snapshot.
    pub fn append(&mut self, book: &OrderBook) {
        // update_rolling_stats is called by FeatureEngine before append, 
        // so we don't necessarily need to call it here again if we trust the caller.
        // But for safety:
        // self.update_rolling_stats(book); 

        let mean = self.rolling_mean_qty;
        let std = self.rolling_std_qty.max(1e-9);
        let levels_per_side = self.height / 2;

        if self.warmup_samples < 30 {
            let mut local_max_z = 0.0_f64;
            for (_, qty) in book.iter_asks().take(levels_per_side) {
                if qty > 0.0 {
                    local_max_z = local_max_z.max((qty - mean) / std);
                }
            }
            for (_, qty) in book.iter_bids().take(levels_per_side) {
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
        
        if snapshot.len() != self.height {
            snapshot.resize(self.height, 0.0);
        }
        snapshot.fill(0.0_f64);

        // Optimization: avoid collect() and intermediate vectors.
        // Top half -> asks (reversed so best ask is nearest mid)
        // Best ask is at index levels_per_side - 1, worst at 0.
        let mut i = levels_per_side;
        for (_, qty) in book.iter_asks().take(levels_per_side) {
            i -= 1;
            if qty > 0.0 {
                snapshot[i] = (qty - mean) / std + 10.0;
            }
        }

        // Bottom half -> bids. Best bid at levels_per_side, worst at height-1.
        let mut i = levels_per_side;
        for (_, qty) in book.iter_bids().take(levels_per_side) {
            if qty > 0.0 {
                snapshot[i] = -((qty - mean) / std + 10.0);
            }
            i += 1;
            if i >= self.height { break; }
        }

        self.data.push_back(snapshot);
        self.last_update_id = self.last_update_id.wrapping_add(1);
        if self.data.len() > self.width {
            if let Some(old_snapshot) = self.data.pop_front() {
                self.snapshot_pool.push(old_snapshot);
            }
        }
    }
}
