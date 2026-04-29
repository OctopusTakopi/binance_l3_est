//! Core order book: storage, state machine sync, and L3-estimation logic.

use crate::engine::fast_order_book::FastOrderBook;
use crate::types::{BookTicker, DepthUpdate, OrderBookSnapshot, Trade};
use egui_plot::PlotPoint;
use std::collections::{HashMap, VecDeque};

const QTY_EPSILON: f64 = 1e-12;
const TRADE_BUFFER_WINDOW_MS: u64 = 10_000;

#[derive(Clone, Copy)]
struct BufferedTrade {
    seq: u64,
    quantity: f64,
    transaction_time: u64,
}

#[derive(Clone, Copy)]
struct TradeExpiryRef {
    seq: u64,
    price_ticks: i64,
    transaction_time: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum OrderBookError {
    SequenceGap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthUpdateStatus {
    Applied,
    IgnoredStale,
}

pub struct OrderBook {
    pub book: FastOrderBook,
    pub last_applied_u: u64,
    pub is_synced: bool,
    trade_buffer: HashMap<i64, VecDeque<BufferedTrade>>,
    trade_expiry_queue: VecDeque<TradeExpiryRef>,
    pub last_changes: Vec<LevelChangeResult>,
    pub cached_buy_points: Vec<PlotPoint>,
    pub cached_sell_points: Vec<PlotPoint>,

    // Liquidity Caching
    last_liquidity_id: u64,
    cached_buy_impact: Option<f64>,
    cached_sell_impact: Option<f64>,
    last_impact_usd: f64,

    tick_size: f64,
    best_bid_ticks: Option<i64>,
    best_bid_qty: Option<f64>,
    best_ask_ticks: Option<i64>,
    best_ask_qty: Option<f64>,
    best_bid_from_ticker: bool,
    best_ask_from_ticker: bool,
    last_top_update_id: Option<u64>,
}

impl Default for OrderBook {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl OrderBook {
    pub fn new(tick_size: f64) -> Self {
        Self {
            book: FastOrderBook::new(tick_size, 0.0),
            last_applied_u: 0,
            is_synced: false,
            trade_buffer: HashMap::with_capacity(100),
            trade_expiry_queue: VecDeque::with_capacity(100),
            last_changes: Vec::with_capacity(40),
            cached_buy_points: Vec::new(),
            cached_sell_points: Vec::new(),
            last_liquidity_id: 0,
            cached_buy_impact: None,
            cached_sell_impact: None,
            last_impact_usd: 0.0,
            tick_size: tick_size.max(QTY_EPSILON),
            best_bid_ticks: None,
            best_bid_qty: None,
            best_ask_ticks: None,
            best_ask_qty: None,
            best_bid_from_ticker: false,
            best_ask_from_ticker: false,
            last_top_update_id: None,
        }
    }

    pub fn set_tick_size(&mut self, tick_size: f64) {
        if tick_size > QTY_EPSILON {
            self.tick_size = tick_size;
            self.book.tick_size = tick_size;
        }
    }

    pub fn price_to_ticks(&self, price: f64) -> i64 {
        self.book.price_to_ticks(price)
    }

    pub fn ticks_to_price(&self, ticks: i64) -> f64 {
        self.book.ticks_to_price(ticks)
    }

    pub fn apply_snapshot(&mut self, snap: OrderBookSnapshot) {
        let mid = if !snap.bids.is_empty() && !snap.asks.is_empty() {
            (snap.bids[0][0] + snap.asks[0][0]) / 2.0
        } else if !snap.bids.is_empty() {
            snap.bids[0][0]
        } else if !snap.asks.is_empty() {
            snap.asks[0][0]
        } else {
            0.0
        };

        if mid > 0.0 {
            let mid_ticks = self.price_to_ticks(mid);
            let new_base = mid_ticks - (crate::engine::fast_order_book::WINDOW_SIZE as i64 / 2);
            // Fully reset on snapshot to ensure no stale levels remain
            self.book.reset_with_base(new_base);
        }

        self.clear_top_of_book_cache();
        self.trade_buffer.clear();
        self.trade_expiry_queue.clear();
        self.last_changes.clear();
        self.cached_buy_points.clear();
        self.cached_sell_points.clear();
        self.last_liquidity_id = 0;
        self.cached_buy_impact = None;
        self.cached_sell_impact = None;
        self.last_impact_usd = 0.0;

        for bid in &snap.bids {
            let price_ticks = self.price_to_ticks(bid[0]);
            let qty = bid[1];
            if qty > QTY_EPSILON {
                self.book.process_update(price_ticks, qty, true);
            }
        }
        for ask in &snap.asks {
            let price_ticks = self.price_to_ticks(ask[0]);
            let qty = ask[1];
            if qty > QTY_EPSILON {
                self.book.process_update(price_ticks, qty, false);
            }
        }

        self.last_applied_u = snap.last_update_id;
        self.last_top_update_id = Some(snap.last_update_id);
        self.is_synced = false;
        self.seed_top_of_book_cache_from_book();
    }

    pub fn reset(&mut self) {
        self.book.slide_window(
            self.book.base_price_ticks + (crate::engine::fast_order_book::WINDOW_SIZE as i64 / 2),
            true,
        );
        self.last_applied_u = 0;
        self.is_synced = false;
        self.trade_buffer.clear();
        self.trade_expiry_queue.clear();
        self.last_changes.clear();
        self.cached_buy_points.clear();
        self.cached_sell_points.clear();
        self.last_liquidity_id = 0;
        self.cached_buy_impact = None;
        self.cached_sell_impact = None;
        self.last_impact_usd = 0.0;
        self.last_top_update_id = None;
        self.clear_top_of_book_cache();
    }

    pub fn process_update(
        &mut self,
        update: DepthUpdate,
        rolling_trade_mean: f64,
    ) -> Result<DepthUpdateStatus, OrderBookError> {
        if self.last_applied_u == 0 {
            self.reset_after_gap();
            return Err(OrderBookError::SequenceGap);
        }

        if update.small_u <= self.last_applied_u {
            self.last_changes.clear();
            return Ok(DepthUpdateStatus::IgnoredStale);
        }

        if !self.is_synced {
            let next_update_id = self.last_applied_u.saturating_add(1);
            if update.capital_u > next_update_id || update.small_u < next_update_id {
                log::warn!(
                    "Sequence gap on snapshot bridge: U={} u={} expected {}",
                    update.capital_u,
                    update.small_u,
                    next_update_id
                );
                self.reset_after_gap();
                return Err(OrderBookError::SequenceGap);
            }
        } else if let Some(prev_u) = update.pu {
            if prev_u != self.last_applied_u {
                log::warn!(
                    "Sequence gap: pu={} != last={}",
                    prev_u,
                    self.last_applied_u
                );
                self.reset_after_gap();
                return Err(OrderBookError::SequenceGap);
            }
        } else {
            let next_update_id = self.last_applied_u.saturating_add(1);
            if update.capital_u > next_update_id || update.small_u < next_update_id {
                log::warn!(
                    "Sequence gap on diff continuity: U={} u={} expected {}",
                    update.capital_u,
                    update.small_u,
                    next_update_id
                );
                self.reset_after_gap();
                return Err(OrderBookError::SequenceGap);
            }
        }

        self.last_changes.clear();
        self.apply_update(&update, rolling_trade_mean);
        self.last_applied_u = update.small_u;
        self.last_top_update_id = Some(self.last_top_update_id.unwrap_or(0).max(update.small_u));
        self.is_synced = true;

        // Trigger lazy re-centering if mid-price moved too far
        self.check_window_health();

        Ok(DepthUpdateStatus::Applied)
    }

    fn check_window_health(&mut self) {
        if let Some(mid) = self.mid_price() {
            let mid_ticks = self.price_to_ticks(mid);
            self.book.slide_window(mid_ticks, false);
        }
    }

    fn apply_update(&mut self, update: &DepthUpdate, rolling_trade_mean: f64) {
        for bid in &update.b {
            let price_ticks = self.price_to_ticks(bid[0]);
            let res = self.process_level_change(
                price_ticks,
                bid[1],
                true,
                update.transaction_time,
                rolling_trade_mean,
            );
            self.last_changes.push(res);

            let cached_best_bid = self.best_bid_ticks;
            if self.should_refresh_top_of_book_from_diff(true, price_ticks, cached_best_bid) {
                self.refresh_top_of_book_cache_from_diff(true, price_ticks);
            }
        }
        for ask in &update.a {
            let price_ticks = self.price_to_ticks(ask[0]);
            let res = self.process_level_change(
                price_ticks,
                ask[1],
                false,
                update.transaction_time,
                rolling_trade_mean,
            );
            self.last_changes.push(res);

            let cached_best_ask = self.best_ask_ticks;
            if self.should_refresh_top_of_book_from_diff(false, price_ticks, cached_best_ask) {
                self.refresh_top_of_book_cache_from_diff(false, price_ticks);
            }
        }
    }

    pub fn apply_ticker_anchor(&mut self, ticker: BookTicker) {
        if self.last_applied_u == 0 {
            return;
        }
        if ticker.update_id <= self.last_top_update_id.unwrap_or(self.last_applied_u) {
            return;
        }

        let best_bid_ticks = self.price_to_ticks(ticker.best_bid_price);
        let best_ask_ticks = self.price_to_ticks(ticker.best_ask_price);

        // Re-center the window if the ticker prices fall outside it; otherwise the
        // overlay would be silently dropped by reconcile_level.
        if self.book.get_idx(best_bid_ticks).is_none()
            || self.book.get_idx(best_ask_ticks).is_none()
        {
            let mid_ticks = (best_bid_ticks + best_ask_ticks) / 2;
            self.book.slide_window(mid_ticks, true);
        }

        // Clear bids above best_bid_ticks and asks below best_ask_ticks
        // In BTreeMap this was split_off. In FastOrderBook, we can just clear the bitset range.
        self.clear_bids_above(best_bid_ticks);
        self.clear_asks_below(best_ask_ticks);

        self.reconcile_level(best_bid_ticks, ticker.best_bid_qty, true);
        self.reconcile_level(best_ask_ticks, ticker.best_ask_qty, false);

        self.best_bid_ticks = Some(best_bid_ticks);
        self.best_bid_qty = Some(ticker.best_bid_qty);
        self.best_bid_from_ticker = true;
        self.best_ask_ticks = Some(best_ask_ticks);
        self.best_ask_qty = Some(ticker.best_ask_qty);
        self.best_ask_from_ticker = true;
        self.last_top_update_id = Some(ticker.update_id);
    }

    fn clear_bids_above(&mut self, price_ticks: i64) {
        let rel = price_ticks + 1 - self.book.base_price_ticks;
        let start_idx = if rel < 0 {
            0
        } else if rel >= crate::engine::fast_order_book::WINDOW_SIZE as i64 {
            return; // No bids are above
        } else {
            rel as usize
        };

        // Use find_next to skip empty ranges instead of linear scan
        let mut curr = self.book.bids_bitset.find_next(start_idx);
        while let Some(i) = curr {
            self.book.total_qtys[i] = 0;
            self.book.bids_bitset.clear(i);
            self.book.asks_bitset.clear(i);
            self.book.clear_queue(i);
            curr = self.book.bids_bitset.find_next(i + 1);
        }
    }

    fn clear_asks_below(&mut self, price_ticks: i64) {
        let rel = price_ticks - self.book.base_price_ticks;
        let end_idx = if rel < 0 {
            return; // No asks are below
        } else if rel >= crate::engine::fast_order_book::WINDOW_SIZE as i64 {
            crate::engine::fast_order_book::WINDOW_SIZE
        } else {
            rel as usize
        };

        // Use find_first and find_next until we reach end_idx
        let mut curr = self.book.asks_bitset.find_first();
        while let Some(i) = curr {
            if i >= end_idx {
                break;
            }
            self.book.total_qtys[i] = 0;
            self.book.asks_bitset.clear(i);
            self.book.bids_bitset.clear(i);
            self.book.clear_queue(i);
            curr = self.book.asks_bitset.find_next(i + 1);
        }
    }

    fn reconcile_level(&mut self, price_ticks: i64, ticker_qty: f64, is_bid: bool) {
        let idx = match self.book.get_idx(price_ticks) {
            Some(i) => i,
            None => return,
        };

        // If the level is currently on the opposite side, wipe it before applying ticker.
        // After ensure_side, est_total may be reset to 0, falling into the fresh-level path.
        self.book.ensure_side(idx, is_bid);

        let ticker_qty_u64 = self.book.f64_to_u64(ticker_qty);
        let est_total = self.book.total_qtys[idx];

        if est_total > 0 {
            if ticker_qty_u64 < est_total {
                let mut to_remove = est_total - ticker_qty_u64;
                while to_remove > 0 && !self.book.queues[idx].is_empty(&mut self.book.pool) {
                    let first = self.book.queues[idx].first_size(&mut self.book.pool);
                    if first <= to_remove {
                        self.book.queues[idx].pop_front(&mut self.book.pool);
                        to_remove -= first;
                    } else {
                        self.book.queues[idx]
                            .mut_first_size(&mut self.book.pool, |s| *s -= to_remove);
                        to_remove = 0;
                    }
                }
                self.book.total_qtys[idx] = ticker_qty_u64;
                if is_bid {
                    self.book.bids_bitset.set(idx);
                } else {
                    self.book.asks_bitset.set(idx);
                }
            } else if ticker_qty_u64 > est_total {
                self.book.queues[idx].push_back(&mut self.book.pool, ticker_qty_u64 - est_total);
                self.book.total_qtys[idx] = ticker_qty_u64;
                if is_bid {
                    self.book.bids_bitset.set(idx);
                } else {
                    self.book.asks_bitset.set(idx);
                }
            }
        } else {
            self.book.process_update(price_ticks, ticker_qty, is_bid);
        }
    }

    /// Whether `price_ticks` is in the top-N levels of the given side, treating the
    /// query as a hypothetical insert: it returns true if `price_ticks` already
    /// exists in the top-N OR if it would land in top-N when inserted (i.e. fewer
    /// than N existing levels strictly outrank it).
    pub fn is_in_top_n(&self, price_ticks: i64, n: usize, is_bid: bool) -> bool {
        if n == 0 {
            return false;
        }
        if is_bid {
            // Bids rank by descending price; walk down from best_bid.
            let mut curr = self.book.best_bid();
            for _ in 0..n {
                match curr {
                    None => return true, // fewer than n bids exist
                    Some(ticks) if ticks == price_ticks => return true,
                    Some(ticks) if ticks < price_ticks => return true, // walked past it
                    Some(ticks) => {
                        let idx = self.book.get_idx(ticks).unwrap();
                        if idx == 0 {
                            return true;
                        }
                        curr = self
                            .book
                            .bids_bitset
                            .find_prev(idx - 1)
                            .map(|i| self.book.base_price_ticks + i as i64);
                    }
                }
            }
            false
        } else {
            // Asks rank by ascending price; walk up from best_ask.
            let mut curr = self.book.best_ask();
            for _ in 0..n {
                match curr {
                    None => return true,
                    Some(ticks) if ticks == price_ticks => return true,
                    Some(ticks) if ticks > price_ticks => return true,
                    Some(ticks) => {
                        let idx = self.book.get_idx(ticks).unwrap();
                        curr = self
                            .book
                            .asks_bitset
                            .find_next(idx + 1)
                            .map(|i| self.book.base_price_ticks + i as i64);
                    }
                }
            }
            false
        }
    }

    pub fn process_level_change(
        &mut self,
        price_ticks: i64,
        new_total_qty: f64,
        is_bid: bool,
        ts: u64,
        rolling_trade_mean: f64,
    ) -> LevelChangeResult {
        let is_tob = if is_bid {
            self.best_bid_ticks.is_none_or(|best| price_ticks >= best)
        } else {
            self.best_ask_ticks.is_none_or(|best| price_ticks <= best)
        };

        let in_top20 = self.is_in_top_n(price_ticks, 20, is_bid);

        let idx = match self.book.get_idx(price_ticks) {
            Some(i) => i,
            None => {
                // Bootstrap the window the first time we see a price; the book has no
                // anchor yet so center on this update. Subsequent out-of-window prices
                // are dropped (lazy re-centering handles intra-window drift).
                if !self.book.initialized {
                    self.book.slide_window(price_ticks, true);
                    match self.book.get_idx(price_ticks) {
                        Some(i) => i,
                        None => {
                            return LevelChangeResult {
                                inflow: 0.0,
                                cancel: 0.0,
                                is_tob,
                                in_top20,
                                is_bid,
                            };
                        }
                    }
                } else {
                    return LevelChangeResult {
                        inflow: 0.0,
                        cancel: 0.0,
                        is_tob,
                        in_top20,
                        is_bid,
                    };
                }
            }
        };

        self.book.ensure_side(idx, is_bid);

        let new_total_u64 = self.book.f64_to_u64(new_total_qty);
        let old_total = self.book.total_qtys[idx];

        if old_total > 0 {
            if new_total_u64 > old_total {
                let diff = new_total_u64 - old_total;

                if !is_tob && diff > self.book.f64_to_u64(0.1) {
                    let avg_trade = rolling_trade_mean.max(0.001);
                    let avg_trade_u64 = self.book.f64_to_u64(avg_trade);
                    if diff > avg_trade_u64 * 2 && diff < avg_trade_u64 * 20 {
                        let num_fragments =
                            (diff as f64 / avg_trade_u64 as f64).clamp(2.0, 5.0) as usize;
                        let fragment_val = diff / num_fragments as u64;
                        let mut remaining = diff;
                        for i in 0..num_fragments {
                            if i == num_fragments - 1 {
                                self.book.queues[idx].push_back(&mut self.book.pool, remaining);
                            } else {
                                self.book.queues[idx].push_back(&mut self.book.pool, fragment_val);
                                remaining -= fragment_val;
                            }
                        }
                    } else {
                        self.book.queues[idx].push_back(&mut self.book.pool, diff);
                    }
                } else {
                    self.book.queues[idx].push_back(&mut self.book.pool, diff);
                }

                self.book.total_qtys[idx] = new_total_u64;
                if is_bid {
                    self.book.bids_bitset.set(idx);
                } else {
                    self.book.asks_bitset.set(idx);
                }

                return LevelChangeResult {
                    inflow: self.book.u64_to_f64(diff),
                    cancel: 0.0,
                    is_tob,
                    in_top20,
                    is_bid,
                };
            } else if new_total_u64 < old_total {
                let mut remaining_to_remove = old_total - new_total_u64;
                let mut remove_trade_bucket = false;

                if let Some(trade_deq) = self.trade_buffer.get_mut(&price_ticks) {
                    while let Some(trade) = trade_deq.front_mut() {
                        if ts >= trade.transaction_time && ts - trade.transaction_time < 1000 {
                            let trade_qty_u64 = self.book.f64_to_u64(trade.quantity);
                            let consumed = trade_qty_u64.min(remaining_to_remove);

                            let mut inner_remaining = consumed;
                            while inner_remaining > 0
                                && !self.book.queues[idx].is_empty(&mut self.book.pool)
                            {
                                let first = self.book.queues[idx].first_size(&mut self.book.pool);
                                if first <= inner_remaining {
                                    inner_remaining -= first;
                                    self.book.queues[idx].pop_front(&mut self.book.pool);
                                } else {
                                    self.book.queues[idx]
                                        .mut_first_size(&mut self.book.pool, |s| {
                                            *s -= inner_remaining
                                        });
                                    inner_remaining = 0;
                                }
                            }

                            remaining_to_remove -= consumed;
                            trade.quantity -= self.book.u64_to_f64(consumed);
                            self.book.total_qtys[idx] -= consumed;

                            if trade.quantity <= QTY_EPSILON {
                                trade.quantity = 0.0;
                                trade_deq.pop_front();
                            }

                            if remaining_to_remove == 0 {
                                break;
                            }
                        } else if trade.transaction_time > ts {
                            break;
                        } else {
                            trade_deq.pop_front();
                        }
                    }
                    remove_trade_bucket = trade_deq.is_empty();
                }
                if remove_trade_bucket {
                    self.trade_buffer.remove(&price_ticks);
                }

                if remaining_to_remove == 0 {
                    if new_total_u64 == 0 {
                        if is_bid {
                            self.book.bids_bitset.clear(idx);
                        } else {
                            self.book.asks_bitset.clear(idx);
                        }
                        self.book.clear_queue(idx);
                    } else {
                        self.book.total_qtys[idx] = new_total_u64;
                        if is_bid {
                            self.book.bids_bitset.set(idx);
                        } else {
                            self.book.asks_bitset.set(idx);
                        }
                    }

                    return LevelChangeResult {
                        inflow: 0.0,
                        cancel: 0.0,
                        is_tob,
                        in_top20,
                        is_bid,
                    };
                }

                if remaining_to_remove > 0 {
                    let cancel_u64 = remaining_to_remove;

                    // remove_order_by_size_simd can be used here if we want FIFO/exact match
                    // But the original logic used pop_back if no exact match found.
                    let mut exact_found = false;
                    unsafe {
                        if self.book.queues[idx]
                            .remove_order_by_size_simd(&mut self.book.pool, remaining_to_remove)
                        {
                            exact_found = true;
                        }
                    }

                    if !exact_found {
                        while remaining_to_remove > 0
                            && !self.book.queues[idx].is_empty(&mut self.book.pool)
                        {
                            if let Some(back_qty) =
                                self.book.queues[idx].pop_back(&mut self.book.pool)
                            {
                                if back_qty <= remaining_to_remove {
                                    remaining_to_remove -= back_qty;
                                } else {
                                    self.book.queues[idx].push_back(
                                        &mut self.book.pool,
                                        back_qty - remaining_to_remove,
                                    );
                                    remaining_to_remove = 0;
                                }
                            } else {
                                break;
                            }
                        }
                    }

                    if new_total_u64 == 0 {
                        if is_bid {
                            self.book.bids_bitset.clear(idx);
                        } else {
                            self.book.asks_bitset.clear(idx);
                        }
                        self.book.clear_queue(idx);
                        self.book.total_qtys[idx] = 0;
                    } else {
                        self.book.total_qtys[idx] = new_total_u64;
                        if is_bid {
                            self.book.bids_bitset.set(idx);
                        } else {
                            self.book.asks_bitset.set(idx);
                        }
                    }

                    return LevelChangeResult {
                        inflow: 0.0,
                        cancel: self.book.u64_to_f64(cancel_u64),
                        is_tob,
                        in_top20,
                        is_bid,
                    };
                }
            }
        } else if new_total_u64 > 0 {
            self.book.process_update(price_ticks, new_total_qty, is_bid);
            return LevelChangeResult {
                inflow: new_total_qty,
                cancel: 0.0,
                is_tob,
                in_top20,
                is_bid,
            };
        }

        LevelChangeResult {
            inflow: 0.0,
            cancel: 0.0,
            is_tob,
            in_top20,
            is_bid,
        }
    }

    pub fn record_trade(&mut self, trade: &Trade) {
        let price_ticks = self.price_to_ticks(trade.price);
        let seq = trade.trade_id;
        self.trade_buffer
            .entry(price_ticks)
            .or_default()
            .push_back(BufferedTrade {
                seq,
                quantity: trade.quantity,
                transaction_time: trade.transaction_time,
            });
        self.trade_expiry_queue.push_back(TradeExpiryRef {
            seq,
            price_ticks,
            transaction_time: trade.transaction_time,
        });
        self.expire_stale_trades(trade.transaction_time);
    }

    pub fn get_liquidity_curves(&mut self) -> (&[PlotPoint], &[PlotPoint]) {
        if self.last_liquidity_id == self.last_applied_u && !self.cached_buy_points.is_empty() {
            return (&self.cached_buy_points, &self.cached_sell_points);
        }

        self.cached_buy_points.clear();
        self.cached_sell_points.clear();

        let Some(mid) = self.mid_price() else {
            return (&self.cached_buy_points, &self.cached_sell_points);
        };
        if mid <= QTY_EPSILON {
            return (&self.cached_buy_points, &self.cached_sell_points);
        }

        self.cached_buy_points.push(PlotPoint::new(0.0, 0.0));
        let mut cum_qty = 0.0;
        let mut weighted_price = 0.0;

        // Iterate asks (ascending)
        let mut curr_idx = self.book.asks_bitset.find_first();
        let mut count = 0;
        while let Some(idx) = curr_idx {
            let price = self.ticks_to_price(self.book.base_price_ticks + idx as i64);
            let qty = self.book.u64_to_f64(self.book.total_qtys[idx]);
            weighted_price += qty * price;
            cum_qty += qty;
            if cum_qty > QTY_EPSILON {
                let vwap = weighted_price / cum_qty;
                let delta_p = ((vwap - mid) / mid * 10_000.0).abs();
                self.cached_buy_points
                    .push(PlotPoint::new(cum_qty * mid, delta_p));
            }
            count += 1;
            if count >= 100 {
                break;
            }

            curr_idx = self.book.asks_bitset.find_next(idx + 1);
        }

        self.cached_sell_points.push(PlotPoint::new(0.0, 0.0));
        let mut cum_qty = 0.0;
        let mut weighted_price = 0.0;

        // Iterate bids (descending)
        let mut curr_idx = self.book.bids_bitset.find_last();
        let mut count = 0;
        while let Some(idx) = curr_idx {
            let price = self.ticks_to_price(self.book.base_price_ticks + idx as i64);
            let qty = self.book.u64_to_f64(self.book.total_qtys[idx]);
            weighted_price += qty * price;
            cum_qty += qty;
            if cum_qty > QTY_EPSILON {
                let vwap = weighted_price / cum_qty;
                let delta_p = ((mid - vwap) / mid * 10_000.0).abs();
                self.cached_sell_points
                    .push(PlotPoint::new(cum_qty * mid, delta_p));
            }
            count += 1;
            if count >= 100 {
                break;
            }

            if idx == 0 {
                break;
            }
            curr_idx = self.book.bids_bitset.find_prev(idx - 1);
        }

        self.last_liquidity_id = self.last_applied_u;
        (&self.cached_buy_points, &self.cached_sell_points)
    }

    pub fn calculate_liquidity_impact(&mut self, execute_usd: f64) -> Option<(f64, f64)> {
        if self.last_liquidity_id == self.last_applied_u
            && (execute_usd - self.last_impact_usd).abs() < 1.0
            && let (Some(b), Some(s)) = (self.cached_buy_impact, self.cached_sell_impact)
        {
            return Some((b, s));
        }

        let mid = self.mid_price()?;
        if mid <= QTY_EPSILON {
            return None;
        }

        let quantity = execute_usd / mid;

        // Buy impact (sweep asks)
        let mut remaining = quantity;
        let mut notional = 0.0;
        let mut filled = 0.0;
        let mut curr_idx = self.book.asks_bitset.find_first();
        while let Some(idx) = curr_idx {
            if remaining <= QTY_EPSILON {
                break;
            }
            let price = self.ticks_to_price(self.book.base_price_ticks + idx as i64);
            let qty = self.book.u64_to_f64(self.book.total_qtys[idx]);
            let take = qty.min(remaining);
            notional += take * price;
            filled += take;
            remaining -= take;
            curr_idx = self.book.asks_bitset.find_next(idx + 1);
        }
        let buy_impact = if filled > 0.0 && remaining <= QTY_EPSILON {
            let vwap = notional / filled;
            Some(((vwap - mid) / mid * 10_000.0).abs())
        } else {
            None
        };

        // Sell impact (sweep bids)
        let mut remaining = quantity;
        let mut notional = 0.0;
        let mut filled = 0.0;
        let mut curr_idx = self.book.bids_bitset.find_last();
        while let Some(idx) = curr_idx {
            if remaining <= QTY_EPSILON {
                break;
            }
            let price = self.ticks_to_price(self.book.base_price_ticks + idx as i64);
            let qty = self.book.u64_to_f64(self.book.total_qtys[idx]);
            let take = qty.min(remaining);
            notional += take * price;
            filled += take;
            remaining -= take;
            if idx == 0 {
                break;
            }
            curr_idx = self.book.bids_bitset.find_prev(idx - 1);
        }
        let sell_impact = if filled > 0.0 && remaining <= QTY_EPSILON {
            let vwap = notional / filled;
            Some(((mid - vwap) / mid * 10_000.0).abs())
        } else {
            None
        };

        self.cached_buy_impact = buy_impact;
        self.cached_sell_impact = sell_impact;
        self.last_impact_usd = execute_usd;
        self.last_liquidity_id = self.last_applied_u;

        if let (Some(b), Some(s)) = (buy_impact, sell_impact) {
            Some((b, s))
        } else {
            None
        }
    }

    pub fn best_bid_ask(&self) -> Option<(f64, f64)> {
        Some((
            self.ticks_to_price(self.book.best_bid()?),
            self.ticks_to_price(self.book.best_ask()?),
        ))
    }

    pub fn mid_price(&self) -> Option<f64> {
        let (best_bid, best_ask) = self.best_bid_ask()?;
        Some((best_bid + best_ask) / 2.0)
    }

    pub fn iter_asks(&self) -> impl Iterator<Item = (i64, f64)> + '_ {
        let mut curr_idx = self.book.asks_bitset.find_first();
        std::iter::from_fn(move || {
            let idx = curr_idx?;
            let price_ticks = self.book.base_price_ticks + idx as i64;
            let qty = self.book.u64_to_f64(self.book.total_qtys[idx]);
            curr_idx = self.book.asks_bitset.find_next(idx + 1);
            Some((price_ticks, qty))
        })
    }

    pub fn iter_bids(&self) -> impl Iterator<Item = (i64, f64)> + '_ {
        let mut curr_idx = self.book.bids_bitset.find_last();
        std::iter::from_fn(move || {
            let idx = curr_idx?;
            let price_ticks = self.book.base_price_ticks + idx as i64;
            let qty = self.book.u64_to_f64(self.book.total_qtys[idx]);
            if idx == 0 {
                curr_idx = None;
            } else {
                curr_idx = self.book.bids_bitset.find_prev(idx - 1);
            }
            Some((price_ticks, qty))
        })
    }

    pub fn iter_asks_with_orders(
        &self,
    ) -> impl Iterator<Item = (i64, impl Iterator<Item = f64> + '_)> + '_ {
        let mut curr_idx = self.book.asks_bitset.find_first();
        std::iter::from_fn(move || {
            let idx = curr_idx?;
            let price_ticks = self.book.base_price_ticks + idx as i64;
            let q_ptr = self.book.queues[idx];
            let pool = &self.book.pool;
            let order_iter = q_ptr.iter(pool).map(|q| self.book.u64_to_f64(q));

            curr_idx = self.book.asks_bitset.find_next(idx + 1);
            Some((price_ticks, order_iter))
        })
    }

    pub fn iter_bids_with_orders(
        &self,
    ) -> impl Iterator<Item = (i64, impl Iterator<Item = f64> + '_)> + '_ {
        let mut curr_idx = self.book.bids_bitset.find_last();
        std::iter::from_fn(move || {
            let idx = curr_idx?;
            let price_ticks = self.book.base_price_ticks + idx as i64;
            let q_ptr = self.book.queues[idx];
            let pool = &self.book.pool;
            let order_iter = q_ptr.iter(pool).map(|q| self.book.u64_to_f64(q));

            if idx == 0 {
                curr_idx = None;
            } else {
                curr_idx = self.book.bids_bitset.find_prev(idx - 1);
            }
            Some((price_ticks, order_iter))
        })
    }

    fn seed_top_of_book_cache_from_book(&mut self) {
        if let Some(price_ticks) = self.book.best_bid() {
            let idx = self.book.get_idx(price_ticks).unwrap();
            self.best_bid_ticks = Some(price_ticks);
            self.best_bid_qty = Some(self.book.u64_to_f64(self.book.total_qtys[idx]));
        } else {
            self.best_bid_ticks = None;
            self.best_bid_qty = None;
        }

        if let Some(price_ticks) = self.book.best_ask() {
            let idx = self.book.get_idx(price_ticks).unwrap();
            self.best_ask_ticks = Some(price_ticks);
            self.best_ask_qty = Some(self.book.u64_to_f64(self.book.total_qtys[idx]));
        } else {
            self.best_ask_ticks = None;
            self.best_ask_qty = None;
        }
    }

    fn clear_top_of_book_cache(&mut self) {
        self.best_bid_ticks = None;
        self.best_bid_qty = None;
        self.best_ask_ticks = None;
        self.best_ask_qty = None;
        self.best_bid_from_ticker = false;
        self.best_ask_from_ticker = false;
    }

    fn refresh_top_of_book_cache_from_diff(&mut self, is_bid: bool, price_ticks: i64) {
        let cached_best_ticks = if is_bid {
            self.best_bid_ticks
        } else {
            self.best_ask_ticks
        };

        if cached_best_ticks.is_none() {
            self.promote_top_of_book_from_side(is_bid);
            return;
        }

        self.update_top_of_book_cache_from_side(is_bid, price_ticks);
    }

    fn should_refresh_top_of_book_from_diff(
        &self,
        is_bid: bool,
        price_ticks: i64,
        cached_best_ticks: Option<i64>,
    ) -> bool {
        match cached_best_ticks {
            None => true,
            Some(best_ticks) if price_ticks == best_ticks => true,
            Some(best_ticks) if is_bid => price_ticks > best_ticks,
            Some(best_ticks) => price_ticks < best_ticks,
        }
    }

    fn update_top_of_book_cache_from_side(&mut self, is_bid: bool, price_ticks: i64) {
        let idx = self.book.get_idx(price_ticks);
        let qty = idx.and_then(|i| {
            let q = self.book.total_qtys[i];
            if q > 0 {
                Some(self.book.u64_to_f64(q))
            } else {
                None
            }
        });

        let cached_best_ticks = if is_bid {
            self.best_bid_ticks
        } else {
            self.best_ask_ticks
        };
        let best_from_ticker = if is_bid {
            self.best_bid_from_ticker
        } else {
            self.best_ask_from_ticker
        };

        if qty.is_none()
            && best_from_ticker
            && cached_best_ticks.is_some_and(|best_ticks| price_ticks != best_ticks)
        {
            return;
        }

        if is_bid {
            if let Some(qty) = qty {
                self.best_bid_ticks = Some(price_ticks);
                self.best_bid_qty = Some(qty);
                self.best_bid_from_ticker = false;
            } else {
                self.promote_top_of_book_from_side(true);
            }
        } else if let Some(qty) = qty {
            self.best_ask_ticks = Some(price_ticks);
            self.best_ask_qty = Some(qty);
            self.best_ask_from_ticker = false;
        } else {
            self.promote_top_of_book_from_side(false);
        }
    }

    fn promote_top_of_book_from_side(&mut self, is_bid: bool) {
        if is_bid {
            if let Some(price_ticks) = self.book.best_bid() {
                let idx = self.book.get_idx(price_ticks).unwrap();
                self.best_bid_ticks = Some(price_ticks);
                self.best_bid_qty = Some(self.book.u64_to_f64(self.book.total_qtys[idx]));
                self.best_bid_from_ticker = false;
            } else {
                self.best_bid_ticks = None;
                self.best_bid_qty = None;
                self.best_bid_from_ticker = false;
            }
        } else if let Some(price_ticks) = self.book.best_ask() {
            let idx = self.book.get_idx(price_ticks).unwrap();
            self.best_ask_ticks = Some(price_ticks);
            self.best_ask_qty = Some(self.book.u64_to_f64(self.book.total_qtys[idx]));
            self.best_ask_from_ticker = false;
        } else {
            self.best_ask_ticks = None;
            self.best_ask_qty = None;
            self.best_ask_from_ticker = false;
        }
    }

    fn reset_after_gap(&mut self) {
        self.book.slide_window(
            self.book.base_price_ticks + (crate::engine::fast_order_book::WINDOW_SIZE as i64 / 2),
            true,
        );
        self.trade_buffer.clear();
        self.trade_expiry_queue.clear();
        self.last_changes.clear();
        self.cached_buy_points.clear();
        self.cached_sell_points.clear();
        self.last_applied_u = 0;
        self.last_top_update_id = None;
        self.is_synced = false;
        self.clear_top_of_book_cache();
    }

    fn expire_stale_trades(&mut self, current_time: u64) {
        while self
            .trade_expiry_queue
            .front()
            .is_some_and(|entry| current_time > entry.transaction_time + TRADE_BUFFER_WINDOW_MS)
        {
            let expired = self
                .trade_expiry_queue
                .pop_front()
                .expect("front expiry entry must exist");
            let mut remove_bucket = false;
            if let Some(trades) = self.trade_buffer.get_mut(&expired.price_ticks) {
                if trades
                    .front()
                    .is_some_and(|buffered_trade| buffered_trade.seq == expired.seq)
                {
                    trades.pop_front();
                }
                remove_bucket = trades.is_empty();
            }
            if remove_bucket {
                self.trade_buffer.remove(&expired.price_ticks);
            }
        }
    }
}

pub struct LevelChangeResult {
    pub inflow: f64,
    pub cancel: f64,
    pub is_tob: bool,
    pub in_top20: bool,
    pub is_bid: bool,
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::utils::SymbolStr;

    fn approx_eq(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-9, "{left} != {right}");
    }

    fn trade(trade_id: u64, price: f64, quantity: f64, transaction_time: u64) -> Trade {
        Trade {
            event_type: "trade".to_string(),
            event_time: transaction_time,
            symbol: SymbolStr::from("BTCUSDT"),
            trade_id,
            price,
            quantity,
            order_type: "MARKET".to_string(),
            transaction_time,
            is_buyer_maker: true,
        }
    }

    fn buffered_trade(
        seq: u64,
        _price: f64,
        quantity: f64,
        transaction_time: u64,
    ) -> BufferedTrade {
        BufferedTrade {
            seq,
            quantity,
            transaction_time,
        }
    }

    #[test]
    fn test_tick_round_trip() {
        let ob = OrderBook::new(0.1);
        let price = 50000.3;
        let ticks = ob.price_to_ticks(price);
        assert_eq!(ticks, 500003);
        approx_eq(ob.ticks_to_price(ticks), price);
    }

    #[test]
    fn test_process_update_inflow() {
        let mut ob = OrderBook::default();
        ob.last_applied_u = 100;
        ob.is_synced = true;
        // Set a base price so the window is initialized
        ob.book.slide_window(ob.price_to_ticks(50000.0), true);

        let update = DepthUpdate {
            event_type: "depthUpdate".to_string(),
            event_time: 1000,
            transaction_time: 1000,
            symbol: SymbolStr::from("BTCUSDT"),
            capital_u: 101,
            small_u: 101,
            pu: Some(100),
            b: vec![[50000.0, 1.5]],
            a: vec![],
        };

        let res = ob.process_update(update, 0.0);
        assert_eq!(res, Ok(DepthUpdateStatus::Applied));
        assert_eq!(ob.last_changes.len(), 1);
        approx_eq(ob.last_changes[0].inflow, 1.5);
        assert!(ob.last_changes[0].is_bid);
    }

    #[test]
    fn test_process_update_gap() {
        let mut ob = OrderBook::default();
        ob.last_applied_u = 100;
        ob.is_synced = true;

        let update = DepthUpdate {
            event_type: "depthUpdate".to_string(),
            event_time: 1000,
            transaction_time: 1000,
            symbol: SymbolStr::from("BTCUSDT"),
            capital_u: 105,
            small_u: 105,
            pu: Some(104),
            b: vec![[50000.0, 1.5]],
            a: vec![],
        };

        let res = ob.process_update(update, 0.0);
        assert_eq!(res, Err(OrderBookError::SequenceGap));
    }

    #[test]
    fn test_first_update_must_bridge_snapshot() {
        let mut ob = OrderBook::default();
        ob.apply_snapshot(OrderBookSnapshot {
            last_update_id: 100,
            bids: vec![[50000.0, 1.0]],
            asks: vec![[50001.0, 2.0]],
        });

        let update = DepthUpdate {
            event_type: "depthUpdate".to_string(),
            event_time: 1000,
            transaction_time: 1000,
            symbol: SymbolStr::from("BTCUSDT"),
            capital_u: 102,
            small_u: 102,
            pu: Some(100),
            b: vec![[50000.0, 1.5]],
            a: vec![],
        };

        let res = ob.process_update(update, 0.0);
        assert_eq!(res, Err(OrderBookError::SequenceGap));
    }

    #[test]
    fn test_stale_update_is_ignored() {
        let mut ob = OrderBook::default();
        ob.last_applied_u = 105;
        ob.is_synced = true;

        let update = DepthUpdate {
            event_type: "depthUpdate".to_string(),
            event_time: 1000,
            transaction_time: 1000,
            symbol: SymbolStr::from("BTCUSDT"),
            capital_u: 104,
            small_u: 105,
            pu: Some(104),
            b: vec![[50000.0, 1.5]],
            a: vec![],
        };

        let res = ob.process_update(update, 0.0);
        assert_eq!(res, Ok(DepthUpdateStatus::IgnoredStale));
    }

    #[test]
    fn test_ticker_anchor_ignored_before_snapshot() {
        let mut ob = OrderBook::default();
        ob.apply_ticker_anchor(BookTicker {
            update_id: 10,
            symbol: SymbolStr::from("BTCUSDT"),
            best_bid_price: 50000.0,
            best_bid_qty: 1.0,
            best_ask_price: 50001.0,
            best_ask_qty: 2.0,
            transaction_time: 1_000,
            event_time: 1_000,
        });

        assert!(ob.book.best_bid().is_none());
        assert!(ob.book.best_ask().is_none());
        assert!(ob.best_bid_ask().is_none());
    }

    #[test]
    fn test_ticker_overlay_survives_unrelated_diff_removal() {
        let mut ob = OrderBook::new(1.0);
        ob.apply_snapshot(OrderBookSnapshot {
            last_update_id: 10,
            bids: vec![[100.0, 1.0]],
            asks: vec![[103.0, 1.0]],
        });
        ob.apply_ticker_anchor(BookTicker {
            update_id: 11,
            symbol: SymbolStr::from("BTCUSDT"),
            best_bid_price: 101.0,
            best_bid_qty: 2.0,
            best_ask_price: 104.0,
            best_ask_qty: 3.0,
            transaction_time: 1_000,
            event_time: 1_000,
        });

        let res = ob.process_update(
            DepthUpdate {
                event_type: "depthUpdate".to_string(),
                event_time: 1_001,
                transaction_time: 1_001,
                symbol: SymbolStr::from("BTCUSDT"),
                capital_u: 11,
                small_u: 11,
                pu: None,
                b: vec![],
                a: vec![[103.0, 0.0]],
            },
            0.0,
        );

        assert_eq!(res, Ok(DepthUpdateStatus::Applied));
        assert_eq!(ob.best_bid_ask(), Some((101.0, 104.0)));
    }

    #[test]
    fn test_exact_diff_invalidation_clears_ticker_overlay() {
        let mut ob = OrderBook::new(1.0);
        ob.apply_snapshot(OrderBookSnapshot {
            last_update_id: 10,
            bids: vec![[100.0, 1.0]],
            asks: vec![[103.0, 1.0]],
        });
        ob.apply_ticker_anchor(BookTicker {
            update_id: 11,
            symbol: SymbolStr::from("BTCUSDT"),
            best_bid_price: 101.0,
            best_bid_qty: 2.0,
            best_ask_price: 104.0,
            best_ask_qty: 3.0,
            transaction_time: 1_000,
            event_time: 1_000,
        });
        let _ = ob.process_update(
            DepthUpdate {
                event_type: "depthUpdate".to_string(),
                event_time: 1_001,
                transaction_time: 1_001,
                symbol: SymbolStr::from("BTCUSDT"),
                capital_u: 11,
                small_u: 11,
                pu: None,
                b: vec![],
                a: vec![[103.0, 0.0]],
            },
            0.0,
        );

        let res = ob.process_update(
            DepthUpdate {
                event_type: "depthUpdate".to_string(),
                event_time: 1_002,
                transaction_time: 1_002,
                symbol: SymbolStr::from("BTCUSDT"),
                capital_u: 12,
                small_u: 12,
                pu: Some(11),
                b: vec![],
                a: vec![[104.0, 0.0]],
            },
            0.0,
        );

        assert_eq!(res, Ok(DepthUpdateStatus::Applied));
        assert!(ob.best_bid_ask().is_none());
    }

    #[test]
    fn matched_trade_fully_consuming_level_removes_level_and_preserves_residual_trade() {
        let mut ob = OrderBook::new(1.0);
        let price_ticks = 100;
        // Initialize window
        ob.book.slide_window(100, true);

        let idx = ob.book.get_idx(price_ticks).unwrap();
        ob.book.total_qtys[idx] = ob.book.f64_to_u64(0.5);
        ob.book.bids_bitset.set(idx);
        ob.book.queues[idx] = crate::engine::simd_queue::QueuePtrs::new(&mut ob.book.pool);
        let q0_u64 = ob.book.f64_to_u64(0.2);
        let q1_u64 = ob.book.f64_to_u64(0.3);
        ob.book.queues[idx].push_back(&mut ob.book.pool, q0_u64);
        ob.book.queues[idx].push_back(&mut ob.book.pool, q1_u64);

        ob.trade_buffer.insert(
            price_ticks,
            VecDeque::from([buffered_trade(1, 100.0, 1.0, 100)]),
        );

        let result = ob.process_level_change(price_ticks, 0.0, true, 150, 0.0);

        assert_eq!(result.inflow, 0.0);
        assert_eq!(result.cancel, 0.0);
        assert_eq!(ob.book.total_qtys[idx], 0);

        let trade_quantity = ob
            .trade_buffer
            .get(&price_ticks)
            .and_then(|trades| trades.front().map(|trade| trade.quantity))
            .expect("residual trade should remain buffered");
        approx_eq(trade_quantity, 0.5);
    }

    #[test]
    fn record_trade_expires_only_stale_front_entries() {
        let mut ob = OrderBook::new(1.0);

        ob.record_trade(&trade(1, 100.0, 1.0, 100));
        ob.record_trade(&trade(2, 101.0, 1.0, 150));
        ob.record_trade(&trade(3, 102.0, 1.0, 10_101));

        assert!(!ob.trade_buffer.contains_key(&100));
        assert_eq!(
            ob.trade_buffer
                .get(&101)
                .and_then(|trades| trades.front().map(|trade| trade.seq)),
            Some(2)
        );
        assert_eq!(
            ob.trade_buffer
                .get(&102)
                .and_then(|trades| trades.front().map(|trade| trade.seq)),
            Some(3)
        );
        assert_eq!(ob.trade_expiry_queue.len(), 2);
    }
}
