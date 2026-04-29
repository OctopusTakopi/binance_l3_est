use crate::engine::{
    heatmap::HeatmapState,
    metrics::{MetricsState, Side},
    order_book::{DepthUpdateStatus, OrderBook},
    twap::TwapDetector,
};
use crate::types::{DepthUpdate, Trade};

/// Unified engine that manages all "book-observing" features.
/// This fulfills the "spread" architecture: features update in tandem with the book.
pub struct FeatureEngine {
    pub metrics: MetricsState,
    pub heatmap: HeatmapState,
    pub twap: TwapDetector,

    /// SOFP: Rolling trade-size distribution.
    pub rolling_trade_mean: f64,
    pub rolling_trade_std: f64,

    /// Global warmup counter: number of depth-update batches seen.
    pub warmup_samples: usize,

    /// Internal counter for batching heatmap updates.
    pub update_counter: u32,
}

impl FeatureEngine {
    pub fn new(heatmap_w: usize, heatmap_h: usize) -> Self {
        Self {
            metrics: MetricsState::default(),
            heatmap: HeatmapState::new(heatmap_w, heatmap_h),
            twap: TwapDetector::new(),
            rolling_trade_mean: 0.0,
            rolling_trade_std: 1.0,
            warmup_samples: 0,
            update_counter: 0,
        }
    }

    /// Reset all feature state (called on symbol change).
    pub fn reset(&mut self) {
        self.metrics.reset();
        self.heatmap.reset();
        self.twap.reset();
        self.rolling_trade_mean = 0.0;
        self.rolling_trade_std = 1.0;
        self.warmup_samples = 0;
        self.update_counter = 0;
    }

    /// Reset only book-dependent state (called on snapshots).
    pub fn reset_book_only(&mut self) {
        // We preserve rolling statistics and metrics continuity even on refetch.
        // Preserve heatmap state across routine refetches.
        self.update_counter = 0;
    }

    /// Handle a trade: update rolling stats, CTR metrics, and TWAP.
    pub fn on_trade(&mut self, trade: &Trade, book: &mut OrderBook) {
        let qty = trade.quantity;
        if qty > 0.0 {
            const ALPHA: f64 = 0.1;
            if self.rolling_trade_mean == 0.0 {
                self.rolling_trade_mean = qty;
                self.rolling_trade_std = qty * 0.1;
            } else {
                let diff = qty - self.rolling_trade_mean;
                self.rolling_trade_mean += ALPHA * diff;
                self.rolling_trade_std =
                    (1.0 - ALPHA) * self.rolling_trade_std + ALPHA * diff.abs();
            }
        }

        // Metrics: track fills at the book price.
        let trade_ticks = book.price_to_ticks(trade.price);
        let best_bid = book.book.best_bid().unwrap_or_default();
        let best_ask = book.book.best_ask().unwrap_or_default();

        if trade.is_buyer_maker {
            let is_tob = trade_ticks == best_bid;
            // For now, let's just use a simple check for in_top20
            let in_top20 = book.iter_bids().take(20).any(|(p, _)| p == trade_ticks);
            self.metrics.on_fill(Side::Bid, is_tob, in_top20, qty);
        } else {
            let is_tob = trade_ticks == best_ask;
            let in_top20 = book.iter_asks().take(20).any(|(p, _)| p == trade_ticks);
            self.metrics.on_fill(Side::Ask, is_tob, in_top20, qty);
        }

        self.twap.on_trade(trade);

        // MTQR: store trade in the attribution buffer.
        book.record_trade(trade);
    }

    /// Handle a depth update: feed metrics via closure and update heatmap.
    pub fn on_depth_update(
        &mut self,
        update: DepthUpdate,
        book: &mut OrderBook,
    ) -> Result<(), crate::engine::order_book::OrderBookError> {
        let mean = self.rolling_trade_mean;

        let status = book.process_update(update, mean)?;
        if status == DepthUpdateStatus::IgnoredStale {
            return Ok(());
        }

        for lc in &book.last_changes {
            let side = if lc.is_bid { Side::Bid } else { Side::Ask };
            if lc.inflow > 0.0 {
                self.metrics
                    .on_inflow(side, lc.is_tob, lc.in_top20, lc.inflow);
            } else if lc.cancel > 0.0 {
                self.metrics
                    .on_cancel(side, lc.is_tob, lc.in_top20, lc.cancel);
            }
        }

        // Batch heatmap updates (every 50 messages ≈ 500ms on Binance).
        self.update_counter += 1;
        if self.update_counter >= 50 {
            self.update_counter = 0;
            self.heatmap.update_rolling_stats(book);
            if self.heatmap.warmup_samples >= 30 {
                self.heatmap.append(book);
            }
            self.warmup_samples = self.warmup_samples.saturating_add(1);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::order_book::OrderBook;
    use crate::types::{DepthUpdate, Trade};

    #[test]
    fn test_feature_engine_trade_rolling_stats() {
        let mut fe = FeatureEngine::new(10, 10);
        let mut book = OrderBook::default();

        // Initial trade
        let trade1 = Trade {
            event_type: "aggTrade".to_string(),
            event_time: 1000,
            symbol: "BTCUSDT".into(),
            trade_id: 1,
            price: 50000.0,
            quantity: 2.0,
            order_type: "LIMIT".to_string(),
            transaction_time: 1000,
            is_buyer_maker: false,
        };

        fe.on_trade(&trade1, &mut book);
        assert_eq!(fe.rolling_trade_mean, 2.0);

        // Second trade
        let trade2 = Trade {
            event_type: "aggTrade".to_string(),
            event_time: 1100,
            symbol: "BTCUSDT".into(),
            trade_id: 2,
            price: 50000.0,
            quantity: 4.0,
            order_type: "LIMIT".to_string(),
            transaction_time: 1100,
            is_buyer_maker: false,
        };

        fe.on_trade(&trade2, &mut book);
        // alpha = 0.1. mean = 2.0 + 0.1 * (4.0 - 2.0) = 2.2
        assert!((fe.rolling_trade_mean - 2.2).abs() < 1e-6);
    }

    #[test]
    fn test_feature_engine_depth_update_flow() {
        let mut fe = FeatureEngine::new(10, 10);
        let mut book = OrderBook::default();
        book.is_synced = true;
        book.last_applied_u = 100;

        let update = DepthUpdate {
            event_type: "depthUpdate".to_string(),
            event_time: 2000,
            transaction_time: 2000,
            symbol: "BTCUSDT".into(),
            capital_u: 101,
            small_u: 101,
            pu: Some(100),
            b: vec![[50000.0, 1.5]],
            a: vec![],
        };

        let res = fe.on_depth_update(update, &mut book);
        assert!(res.is_ok());

        // Check if metrics were updated via the closure
        // Side::Bid, is_tob=true (since book was empty), in_top20=true
        assert_eq!(fe.metrics.inflows_bid_top1, 1.5);
    }

    #[test]
    fn test_reset_book_only_preserves_heatmap_state() {
        let mut fe = FeatureEngine::new(10, 10);
        fe.heatmap.warmup_samples = 42;
        fe.heatmap.rolling_mean_qty = 7.0;
        fe.heatmap.data.push_back(vec![1.0, 2.0]);
        fe.update_counter = 9;

        fe.reset_book_only();

        assert_eq!(fe.heatmap.warmup_samples, 42);
        assert_eq!(fe.heatmap.rolling_mean_qty, 7.0);
        assert_eq!(fe.heatmap.data.len(), 1);
        assert_eq!(fe.update_counter, 0);
    }
}
