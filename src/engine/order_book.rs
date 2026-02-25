//! Core order book: storage, state machine sync, and L3-estimation logic.

use crate::types::{BookTicker, DepthUpdate, OrderBookSnapshot, Trade};
use egui_plot::PlotPoint;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::collections::{BTreeMap, HashMap, VecDeque};

/// The core L3-estimated order book.
///
/// Bids and asks are stored as price-keyed `BTreeMap`s where each level holds a
/// `VecDeque<Decimal>` representing the individual estimated order queue.
pub struct OrderBook {
    pub bids: BTreeMap<Decimal, VecDeque<Decimal>>,
    pub asks: BTreeMap<Decimal, VecDeque<Decimal>>,
    pub last_applied_u: u64,
    pub is_synced: bool,
    /// Short-term trade buffer used by MTQR (Marker-Triggered Queue Refining).
    /// Keyed by price; values are `(qty, transaction_time_ms)`.
    pub trade_buffer: HashMap<Decimal, VecDeque<(Decimal, u64)>>,
    /// Pre-allocated cache for VWAP liquidity curve visualizations
    pub cached_buy_points: Vec<PlotPoint>,
    pub cached_sell_points: Vec<PlotPoint>,
}

impl Default for OrderBook {
    fn default() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_applied_u: 0,
            is_synced: false,
            trade_buffer: HashMap::new(),
            cached_buy_points: Vec::new(),
            cached_sell_points: Vec::new(),
        }
    }
}

impl OrderBook {
    // ── Snapshot ───────────────────────────────────────────────────────────────

    /// Replace book state with a full REST snapshot and reset sync state.
    pub fn apply_snapshot(&mut self, snap: OrderBookSnapshot) {
        self.bids.clear();
        self.asks.clear();
        for bid in &snap.bids {
            let price = bid[0];
            let qty = bid[1];
            if qty > Decimal::ZERO {
                self.bids.insert(price, VecDeque::from(vec![qty]));
            }
        }
        for ask in &snap.asks {
            let price = ask[0];
            let qty = ask[1];
            if qty > Decimal::ZERO {
                self.asks.insert(price, VecDeque::from(vec![qty]));
            }
        }
        self.last_applied_u = snap.last_update_id;
        self.is_synced = false;
    }

    /// Clear all book-specific state (called on symbol change).
    pub fn reset(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.last_applied_u = 0;
        self.is_synced = false;
        self.trade_buffer.clear();
    }

    // ── Depth Update Processing ────────────────────────────────────────────────

    /// Attempt to apply a depth update, respecting the sequence-number sync
    /// protocol.
    ///
    /// Returns `Some(results)` with all [`LevelChangeResult`]s so the caller can
    /// update metrics, or `None` if a refetch was triggered.
    pub fn process_update(
        &mut self,
        update: DepthUpdate,
        rolling_trade_mean: f64,
        refetch: &mut bool,
        results_buf: &mut Vec<LevelChangeResult>,
    ) {
        if update.small_u < self.last_applied_u {
            return;
        }

        if self.is_synced {
            if (update.pu as u64) != self.last_applied_u {
                log::warn!(
                    "Message gap detected! pu={}, last={}",
                    update.pu,
                    self.last_applied_u
                );
                *refetch = true;
                return;
            }
            self.apply_update(&update, rolling_trade_mean, results_buf);
        } else if update.capital_u <= self.last_applied_u && self.last_applied_u <= update.small_u {
            self.apply_update(&update, rolling_trade_mean, results_buf);
            self.is_synced = true;
        } else {
            log::warn!(
                "Initial gap! U={}, u={}, last={}",
                update.capital_u,
                update.small_u,
                self.last_applied_u
            );
            *refetch = true;
        }
    }

    /// Directly apply bids/asks from a depth update to the book.
    /// Returns all [`LevelChangeResult`]s for the caller (metrics dispatch).
    fn apply_update(
        &mut self,
        update: &DepthUpdate,
        rolling_trade_mean: f64,
        results_buf: &mut Vec<LevelChangeResult>,
    ) {
        let cap_needed = results_buf.len() + update.b.len() + update.a.len();
        if results_buf.capacity() < cap_needed {
            results_buf.reserve(cap_needed - results_buf.len());
        }

        for bid in &update.b {
            let price = bid[0];
            let qty = bid[1];
            if qty == Decimal::ZERO {
                self.bids.remove(&price);
            } else {
                results_buf.push(self.process_level_change(
                    &price,
                    qty,
                    true,
                    update.transaction_time,
                    rolling_trade_mean,
                ));
            }
        }
        for ask in &update.a {
            let price = ask[0];
            let qty = ask[1];
            if qty == Decimal::ZERO {
                self.asks.remove(&price);
            } else {
                results_buf.push(self.process_level_change(
                    &price,
                    qty,
                    false,
                    update.transaction_time,
                    rolling_trade_mean,
                ));
            }
        }
        self.last_applied_u = update.small_u;
    }

    // ── BookTicker Anchor ──────────────────────────────────────────────────────

    /// Use a best-bid/ask tick snapshot to prune crossed quotes and reconcile
    /// top-of-book quantities.
    pub fn apply_ticker_anchor(&mut self, ticker: BookTicker) {
        // Safety pruning: resolve any crossed book from stale depth updates.
        // Keep bids <= best_bid_price
        let mut too_high = self.bids.split_off(&ticker.best_bid_price);
        if let Some(best_bid_val) = too_high.remove(&ticker.best_bid_price) {
            self.bids.insert(ticker.best_bid_price, best_bid_val);
        }

        // Keep asks >= best_ask_price
        self.asks = self.asks.split_off(&ticker.best_ask_price);

        // Sync best bid.
        Self::reconcile_level(
            &mut self.bids,
            ticker.best_bid_price,
            ticker.best_bid_qty,
            true,
        );
        // Sync best ask.
        Self::reconcile_level(
            &mut self.asks,
            ticker.best_ask_price,
            ticker.best_ask_qty,
            false,
        );
    }

    fn reconcile_level(
        side: &mut BTreeMap<Decimal, VecDeque<Decimal>>,
        ticker_price: Decimal,
        ticker_qty: Decimal,
        is_bid: bool,
    ) {
        let best_price = if is_bid {
            side.keys().next_back().copied()
        } else {
            side.keys().next().copied()
        };

        let Some(best) = best_price else { return };

        if best == ticker_price {
            if let Some(queue) = side.get_mut(&best) {
                let est_total: Decimal = queue.iter().sum();
                if ticker_qty < est_total {
                    let mut to_remove = est_total - ticker_qty;
                    while to_remove > Decimal::ZERO && !queue.is_empty() {
                        if queue[0] <= to_remove {
                            to_remove -= queue[0];
                            queue.pop_front();
                        } else {
                            queue[0] -= to_remove;
                            to_remove = Decimal::ZERO;
                        }
                    }
                } else if ticker_qty > est_total {
                    queue.push_back(ticker_qty - est_total);
                }
            }
        } else {
            let new_best = if is_bid {
                ticker_price > best
            } else {
                ticker_price < best
            };
            if new_best {
                side.insert(ticker_price, VecDeque::from([ticker_qty]));
            }
        }
    }

    // ── Level Change / SOFP / MTQR ────────────────────────────────────────────

    /// Process a level-quantity change with SOFP fragmentation and MTQR fill
    /// attribution.
    ///
    /// Returns `(inflow_diff, cancel_diff, is_tob, in_top20)` for metrics
    /// tracking.
    pub fn process_level_change(
        &mut self,
        price: &Decimal,
        new_total_qty: Decimal,
        is_bid: bool,
        ts: u64,
        rolling_trade_mean: f64,
    ) -> LevelChangeResult {
        // 1. Determine position *before* mutable borrow.
        let is_tob = if is_bid {
            self.bids.keys().next_back() == Some(price)
        } else {
            self.asks.keys().next() == Some(price)
        };
        let in_top20 = if is_bid {
            self.bids.keys().rev().take(20).any(|p| p == price)
        } else {
            self.asks.keys().take(20).any(|p| p == price)
        };

        // 2. Mutable work.
        let side: &mut BTreeMap<Decimal, VecDeque<Decimal>> = if is_bid {
            &mut self.bids
        } else {
            &mut self.asks
        };

        if let Some(queue) = side.get_mut(price) {
            let old_total: Decimal = queue.iter().sum();

            if new_total_qty > old_total {
                // INFLOW: Liquidity added.
                let diff = new_total_qty - old_total;

                if !is_tob && diff > dec!(0.1) {
                    let avg_trade = rolling_trade_mean.max(0.001);
                    let diff_f = diff.to_f64().unwrap_or(0.0);
                    if diff_f > avg_trade * 2.0 && diff_f < avg_trade * 20.0 {
                        let num_fragments = (diff_f / avg_trade).min(5.0).max(2.0) as usize;
                        let fragment_val = (diff / Decimal::from(num_fragments)).round_dp(8);
                        let mut remaining = diff;
                        for i in 0..num_fragments {
                            if i == num_fragments - 1 {
                                queue.push_back(remaining);
                            } else {
                                queue.push_back(fragment_val);
                                remaining -= fragment_val;
                            }
                        }
                    } else {
                        queue.push_back(diff);
                    }
                } else {
                    queue.push_back(diff);
                }

                return LevelChangeResult {
                    inflow: diff.to_f64().unwrap_or(0.0),
                    cancel: 0.0,
                    is_tob,
                    in_top20,
                    is_bid,
                };
            } else if new_total_qty < old_total {
                // OUTFLOW: Liquidity reduced.
                let mut remaining_to_remove = old_total - new_total_qty;

                // MTQR: consume matched trades first (FIFO).
                if let Some(trade_deq) = self.trade_buffer.get_mut(price) {
                    while let Some(&trade) = trade_deq.front() {
                        if ts >= trade.1 && ts - trade.1 < 1000 {
                            let trade_qty = trade.0;
                            if !queue.is_empty() && queue[0] < trade_qty {
                                queue[0] = trade_qty;
                            }
                            let consumed = remaining_to_remove.min(trade_qty);
                            let mut inner_remaining = consumed;
                            while inner_remaining > Decimal::ZERO && !queue.is_empty() {
                                if queue[0] <= inner_remaining {
                                    inner_remaining -= queue[0];
                                    queue.pop_front();
                                } else {
                                    queue[0] -= inner_remaining;
                                    inner_remaining = Decimal::ZERO;
                                }
                            }
                            remaining_to_remove -= consumed;
                            trade_deq.pop_front();
                            if remaining_to_remove == Decimal::ZERO {
                                break;
                            }
                        } else if trade.1 > ts {
                            break;
                        } else {
                            trade_deq.pop_front();
                        }
                    }
                }

                if remaining_to_remove > Decimal::ZERO {
                    // Cancel / modification (LIFO with priority reset).
                    let cancel = remaining_to_remove.to_f64().unwrap_or(0.0);

                    if let Some(pos) = queue.iter().rposition(|&x| x == remaining_to_remove) {
                        queue.remove(pos);
                    } else {
                        while remaining_to_remove > Decimal::ZERO && !queue.is_empty() {
                            let last_idx = queue.len() - 1;
                            if queue[last_idx] <= remaining_to_remove {
                                remaining_to_remove -= queue[last_idx];
                                queue.pop_back();
                            } else {
                                queue[last_idx] -= remaining_to_remove;
                                remaining_to_remove = Decimal::ZERO;
                            }
                        }
                    }

                    return LevelChangeResult {
                        inflow: 0.0,
                        cancel,
                        is_tob,
                        in_top20,
                        is_bid,
                    };
                }
            }
        } else {
            side.insert(*price, VecDeque::from(vec![new_total_qty]));
        }

        LevelChangeResult {
            inflow: 0.0,
            cancel: 0.0,
            is_tob,
            in_top20,
            is_bid,
        }
    }

    // ── Trade Buffer ───────────────────────────────────────────────────────────

    /// Record a trade for MTQR attribution and expire entries older than 10 s.
    pub fn record_trade(&mut self, trade: &Trade) {
        self.trade_buffer
            .entry(trade.price)
            .or_default()
            .push_back((trade.quantity, trade.transaction_time));
        let now = trade.transaction_time;
        for deq in self.trade_buffer.values_mut() {
            while let Some(&front) = deq.front() {
                if now > front.1 + 10_000 {
                    deq.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    // ── Analytics Helpers ──────────────────────────────────────────────────────

    /// Compute the VWAP buy and sell market-impact curves for the Sweep window.
    pub fn get_liquidity_curves(&mut self) -> (&[PlotPoint], &[PlotPoint]) {
        let best_bid = self
            .bids
            .keys()
            .next_back()
            .and_then(|p| p.to_f64())
            .unwrap_or(0.0);
        let best_ask = self
            .asks
            .keys()
            .next()
            .and_then(|p| p.to_f64())
            .unwrap_or(0.0);
        let mid = (best_bid + best_ask) / 2.0;

        self.cached_buy_points.clear();
        self.cached_sell_points.clear();

        if mid == 0.0 {
            return (&self.cached_buy_points, &self.cached_sell_points);
        }

        self.cached_buy_points.push(PlotPoint::new(0.0, 0.0));
        let mut cum_qty = 0.0;
        let mut weighted_price = 0.0;
        for (price, qty_deq) in self.asks.iter() {
            let p_f64 = price.to_f64().unwrap_or(0.0);
            let mut level_qty = 0.0;
            for &q in qty_deq {
                level_qty += q.to_f64().unwrap_or(0.0);
            }
            weighted_price += level_qty * p_f64;
            cum_qty += level_qty;
            if cum_qty > 0.0 {
                let vwap = weighted_price / cum_qty;
                let delta_p = ((vwap - mid) / mid * 10000.0).abs();
                let usd = cum_qty * mid;
                self.cached_buy_points.push(PlotPoint::new(usd, delta_p));
            }
        }

        self.cached_sell_points.push(PlotPoint::new(0.0, 0.0));
        let mut cum_qty = 0.0;
        let mut weighted_price = 0.0;
        for (price, qty_deq) in self.bids.iter().rev() {
            let p_f64 = price.to_f64().unwrap_or(0.0);
            let mut level_qty = 0.0;
            for &q in qty_deq {
                level_qty += q.to_f64().unwrap_or(0.0);
            }
            weighted_price += level_qty * p_f64;
            cum_qty += level_qty;
            if cum_qty > 0.0 {
                let vwap = weighted_price / cum_qty;
                let delta_p = ((mid - vwap) / mid * 10000.0).abs();
                let usd = cum_qty * mid;
                self.cached_sell_points.push(PlotPoint::new(usd, delta_p));
            }
        }

        (&self.cached_buy_points, &self.cached_sell_points)
    }

    /// Compute the scalar buy and sell price impact (in bps) for a given USD
    /// trade size.
    pub fn calculate_liquidity_impact(&self, execute_usd: f64) -> (f64, f64) {
        if self.bids.is_empty() || self.asks.is_empty() {
            return (0.0, 0.0);
        }
        let best_bid = self
            .bids
            .keys()
            .next_back()
            .and_then(|p| p.to_f64())
            .unwrap_or(0.0);
        let best_ask = self
            .asks
            .keys()
            .next()
            .and_then(|p| p.to_f64())
            .unwrap_or(0.0);
        let mid = (best_bid + best_ask) / 2.0;
        if mid == 0.0 {
            return (0.0, 0.0);
        }
        let quantity = execute_usd / mid;

        // Buy (sweep asks)
        let buy_vwap = sweep_vwap(self.asks.iter(), quantity, best_ask);
        let buy_impact = ((buy_vwap - mid) / mid * 10000.0).abs();

        // Sell (sweep bids, reversed)
        let sell_vwap = sweep_vwap(self.bids.iter().rev(), quantity, best_bid);
        let sell_impact = ((mid - sell_vwap) / mid * 10000.0).abs();

        (buy_impact, sell_impact)
    }
}

/// Result returned by [`OrderBook::process_level_change`].
pub struct LevelChangeResult {
    /// Quantity of liquidity added at this level (0 if outflow).
    pub inflow: f64,
    /// Quantity attributed to cancellations at this level (0 if inflow).
    pub cancel: f64,
    pub is_tob: bool,
    pub in_top20: bool,
    pub is_bid: bool,
}

// ── Private helper ──────────────────────────────────────────────────────────

fn sweep_vwap<'a>(
    iter: impl Iterator<Item = (&'a Decimal, &'a VecDeque<Decimal>)>,
    quantity: f64,
    fallback: f64,
) -> f64 {
    let mut cum_qty = 0.0;
    let mut weighted_price = 0.0;
    let mut last_price = fallback;
    for (price, qty_deq) in iter {
        last_price = price.to_f64().unwrap_or(0.0);
        let mut level_qty = 0.0;
        for &q in qty_deq {
            level_qty += q.to_f64().unwrap_or(0.0);
        }
        let remaining = quantity - cum_qty;
        if level_qty >= remaining {
            weighted_price += remaining * last_price;
            cum_qty += remaining;
            break;
        } else {
            weighted_price += level_qty * last_price;
            cum_qty += level_qty;
        }
    }
    if cum_qty < quantity {
        weighted_price += (quantity - cum_qty) * last_price;
        cum_qty = quantity;
    }
    if cum_qty > 0.0 {
        weighted_price / cum_qty
    } else {
        fallback
    }
}
