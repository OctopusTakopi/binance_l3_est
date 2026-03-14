//! Core order book: storage, state machine sync, and L3-estimation logic.

use crate::types::{BookTicker, DepthUpdate, OrderBookSnapshot, Trade};
use egui_plot::PlotPoint;
use std::collections::{BTreeMap, HashMap, VecDeque};

type OrderQueue = VecDeque<f64>;
type BookSide = BTreeMap<i64, (f64, OrderQueue)>;

const QTY_EPSILON: f64 = 1e-12;

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
    pub bids: BookSide,
    pub asks: BookSide,
    pub last_applied_u: u64,
    pub is_synced: bool,
    pub trade_buffer: HashMap<i64, VecDeque<Trade>>,
    pub last_changes: Vec<LevelChangeResult>,
    pub cached_buy_points: Vec<PlotPoint>,
    pub cached_sell_points: Vec<PlotPoint>,
    tick_size: f64,
    best_bid_ticks: Option<i64>,
    best_bid_qty: Option<f64>,
    best_ask_ticks: Option<i64>,
    best_ask_qty: Option<f64>,
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
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_applied_u: 0,
            is_synced: false,
            trade_buffer: HashMap::with_capacity(100),
            last_changes: Vec::with_capacity(40),
            cached_buy_points: Vec::new(),
            cached_sell_points: Vec::new(),
            tick_size: tick_size.max(QTY_EPSILON),
            best_bid_ticks: None,
            best_bid_qty: None,
            best_ask_ticks: None,
            best_ask_qty: None,
            last_top_update_id: None,
        }
    }

    pub fn set_tick_size(&mut self, tick_size: f64) {
        if tick_size > QTY_EPSILON {
            self.tick_size = tick_size;
        }
    }

    pub fn price_to_ticks(&self, price: f64) -> i64 {
        (price / self.tick_size).round() as i64
    }

    pub fn ticks_to_price(&self, ticks: i64) -> f64 {
        ticks as f64 * self.tick_size
    }

    pub fn apply_snapshot(&mut self, snap: OrderBookSnapshot) {
        self.bids.clear();
        self.asks.clear();
        self.clear_top_of_book_cache();
        self.trade_buffer.clear();
        self.last_changes.clear();
        self.cached_buy_points.clear();
        self.cached_sell_points.clear();

        for bid in &snap.bids {
            let price_ticks = self.price_to_ticks(bid[0]);
            let qty = bid[1];
            if qty > QTY_EPSILON {
                self.bids.insert(price_ticks, (qty, VecDeque::from([qty])));
            }
        }
        for ask in &snap.asks {
            let price_ticks = self.price_to_ticks(ask[0]);
            let qty = ask[1];
            if qty > QTY_EPSILON {
                self.asks.insert(price_ticks, (qty, VecDeque::from([qty])));
            }
        }

        self.last_applied_u = snap.last_update_id;
        self.last_top_update_id = Some(snap.last_update_id);
        self.is_synced = false;
        self.seed_top_of_book_cache_from_book();
    }

    pub fn reset(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.last_applied_u = 0;
        self.is_synced = false;
        self.trade_buffer.clear();
        self.last_changes.clear();
        self.cached_buy_points.clear();
        self.cached_sell_points.clear();
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

        Ok(DepthUpdateStatus::Applied)
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

        let mut too_high = self.bids.split_off(&best_bid_ticks);
        if let Some(best_bid_val) = too_high.remove(&best_bid_ticks) {
            self.bids.insert(best_bid_ticks, best_bid_val);
        }

        self.asks = self.asks.split_off(&best_ask_ticks);

        Self::reconcile_level(&mut self.bids, best_bid_ticks, ticker.best_bid_qty, true);
        Self::reconcile_level(&mut self.asks, best_ask_ticks, ticker.best_ask_qty, false);

        self.best_bid_ticks = Some(best_bid_ticks);
        self.best_bid_qty = Some(ticker.best_bid_qty);
        self.best_ask_ticks = Some(best_ask_ticks);
        self.best_ask_qty = Some(ticker.best_ask_qty);
        self.last_top_update_id = Some(ticker.update_id);
    }

    fn reconcile_level(side: &mut BookSide, price_ticks: i64, ticker_qty: f64, is_bid: bool) {
        if side.is_empty() {
            return;
        }

        let best_ticks = if is_bid {
            side.keys().next_back().copied()
        } else {
            side.keys().next().copied()
        };

        let Some(best) = best_ticks else { return };

        if best == price_ticks {
            if let Some((est_total, queue)) = side.get_mut(&best) {
                if ticker_qty < *est_total {
                    let mut to_remove = *est_total - ticker_qty;
                    while to_remove > QTY_EPSILON && !queue.is_empty() {
                        if queue[0] <= to_remove + QTY_EPSILON {
                            to_remove -= queue[0];
                            queue.pop_front();
                        } else {
                            queue[0] -= to_remove;
                            to_remove = 0.0;
                        }
                    }
                    *est_total = ticker_qty;
                } else if ticker_qty > *est_total {
                    queue.push_back(ticker_qty - *est_total);
                    *est_total = ticker_qty;
                }
            }
        } else {
            let new_best = if is_bid {
                price_ticks > best
            } else {
                price_ticks < best
            };
            if new_best {
                side.insert(price_ticks, (ticker_qty, VecDeque::from([ticker_qty])));
            }
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
            self.best_bid_ticks.map_or(true, |best| price_ticks >= best)
        } else {
            self.best_ask_ticks.map_or(true, |best| price_ticks <= best)
        };
        let in_top20 = if is_bid {
            self.bids.keys().rev().take(20).any(|&p| p == price_ticks) || self.bids.len() < 20
        } else {
            self.asks.keys().take(20).any(|&p| p == price_ticks) || self.asks.len() < 20
        };

        let side = if is_bid {
            &mut self.bids
        } else {
            &mut self.asks
        };

        if let Some((total_qty, queue)) = side.get_mut(&price_ticks) {
            let old_total = *total_qty;

            if new_total_qty > old_total + QTY_EPSILON {
                let diff = new_total_qty - old_total;

                if !is_tob && diff > 0.1 {
                    let avg_trade = rolling_trade_mean.max(0.001);
                    if diff > avg_trade * 2.0 && diff < avg_trade * 20.0 {
                        let num_fragments = (diff / avg_trade).clamp(2.0, 5.0) as usize;
                        let fragment_val = diff / num_fragments as f64;
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

                *total_qty = new_total_qty;

                return LevelChangeResult {
                    inflow: diff,
                    cancel: 0.0,
                    is_tob,
                    in_top20,
                    is_bid,
                };
            } else if new_total_qty < old_total - QTY_EPSILON {
                let mut remaining_to_remove = old_total - new_total_qty;

                if let Some(trade_deq) = self.trade_buffer.get_mut(&price_ticks) {
                    while let Some(trade) = trade_deq.front_mut() {
                        if ts >= trade.transaction_time && ts - trade.transaction_time < 1000 {
                            let trade_qty = trade.quantity;
                            if !queue.is_empty() && queue[0] < trade_qty {
                                queue[0] = trade_qty;
                            }
                            let consumed = trade.quantity.min(remaining_to_remove);

                            let mut inner_remaining = consumed;
                            while inner_remaining > QTY_EPSILON && !queue.is_empty() {
                                if queue[0] <= inner_remaining + QTY_EPSILON {
                                    inner_remaining -= queue[0];
                                    queue.pop_front();
                                } else {
                                    queue[0] -= inner_remaining;
                                    inner_remaining = 0.0;
                                }
                            }

                            remaining_to_remove -= consumed;
                            trade.quantity -= consumed;
                            *total_qty -= consumed;

                            if trade.quantity <= QTY_EPSILON {
                                trade.quantity = 0.0;
                                trade_deq.pop_front();
                            }

                            if remaining_to_remove <= QTY_EPSILON {
                                break;
                            }
                        } else if trade.transaction_time > ts {
                            break;
                        } else {
                            trade_deq.pop_front();
                        }
                    }
                }

                if remaining_to_remove > QTY_EPSILON {
                    let cancel = remaining_to_remove;

                    if let Some(pos) = queue
                        .iter()
                        .rposition(|&x| (x - remaining_to_remove).abs() <= QTY_EPSILON)
                    {
                        queue.remove(pos);
                    } else {
                        while remaining_to_remove > QTY_EPSILON && !queue.is_empty() {
                            let last_idx = queue.len() - 1;
                            if queue[last_idx] <= remaining_to_remove + QTY_EPSILON {
                                remaining_to_remove -= queue[last_idx];
                                queue.pop_back();
                            } else {
                                queue[last_idx] -= remaining_to_remove;
                                remaining_to_remove = 0.0;
                            }
                        }
                    }

                    if new_total_qty <= QTY_EPSILON {
                        side.remove(&price_ticks);
                    } else {
                        *total_qty = new_total_qty;
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
        } else if new_total_qty > QTY_EPSILON {
            side.insert(
                price_ticks,
                (new_total_qty, VecDeque::from([new_total_qty])),
            );
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
        self.trade_buffer
            .entry(price_ticks)
            .or_default()
            .push_back(trade.clone());
        let now = trade.transaction_time;
        for deq in self.trade_buffer.values_mut() {
            while let Some(front) = deq.front() {
                if now > front.transaction_time + 10_000 {
                    deq.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    pub fn get_liquidity_curves(&mut self) -> (&[PlotPoint], &[PlotPoint]) {
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
        for (price_ticks, (level_total, _)) in &self.asks {
            let price = self.ticks_to_price(*price_ticks);
            weighted_price += level_total * price;
            cum_qty += *level_total;
            if cum_qty > QTY_EPSILON {
                let vwap = weighted_price / cum_qty;
                let delta_p = ((vwap - mid) / mid * 10_000.0).abs();
                self.cached_buy_points
                    .push(PlotPoint::new(cum_qty * mid, delta_p));
            }
        }

        self.cached_sell_points.push(PlotPoint::new(0.0, 0.0));
        let mut cum_qty = 0.0;
        let mut weighted_price = 0.0;
        for (price_ticks, (level_total, _)) in self.bids.iter().rev() {
            let price = self.ticks_to_price(*price_ticks);
            weighted_price += level_total * price;
            cum_qty += *level_total;
            if cum_qty > QTY_EPSILON {
                let vwap = weighted_price / cum_qty;
                let delta_p = ((mid - vwap) / mid * 10_000.0).abs();
                self.cached_sell_points
                    .push(PlotPoint::new(cum_qty * mid, delta_p));
            }
        }

        (&self.cached_buy_points, &self.cached_sell_points)
    }

    pub fn calculate_liquidity_impact(&self, execute_usd: f64) -> Option<(f64, f64)> {
        let mid = self.mid_price()?;
        if mid <= QTY_EPSILON {
            return None;
        }

        let quantity = execute_usd / mid;
        let buy_vwap = sweep_vwap(self.asks.iter(), quantity, self.tick_size)?;
        let sell_vwap = sweep_vwap(self.bids.iter().rev(), quantity, self.tick_size)?;

        Some((
            ((buy_vwap - mid) / mid * 10_000.0).abs(),
            ((mid - sell_vwap) / mid * 10_000.0).abs(),
        ))
    }

    pub fn best_bid_ask(&self) -> Option<(f64, f64)> {
        Some((
            self.ticks_to_price(self.best_bid_ticks?),
            self.ticks_to_price(self.best_ask_ticks?),
        ))
    }

    pub fn mid_price(&self) -> Option<f64> {
        let (best_bid, best_ask) = self.best_bid_ask()?;
        Some((best_bid + best_ask) / 2.0)
    }

    fn seed_top_of_book_cache_from_book(&mut self) {
        if let Some((price_ticks, (qty, _))) = self.bids.last_key_value() {
            self.best_bid_ticks = Some(*price_ticks);
            self.best_bid_qty = Some(*qty);
        } else {
            self.best_bid_ticks = None;
            self.best_bid_qty = None;
        }

        if let Some((price_ticks, (qty, _))) = self.asks.first_key_value() {
            self.best_ask_ticks = Some(*price_ticks);
            self.best_ask_qty = Some(*qty);
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
        let side = if is_bid { &self.bids } else { &self.asks };
        let qty = side.get(&price_ticks).map(|(qty, _)| *qty);

        if is_bid {
            if let Some(qty) = qty {
                self.best_bid_ticks = Some(price_ticks);
                self.best_bid_qty = Some(qty);
            } else {
                self.promote_top_of_book_from_side(true);
            }
        } else if let Some(qty) = qty {
            self.best_ask_ticks = Some(price_ticks);
            self.best_ask_qty = Some(qty);
        } else {
            self.promote_top_of_book_from_side(false);
        }
    }

    fn promote_top_of_book_from_side(&mut self, is_bid: bool) {
        if is_bid {
            if let Some((price_ticks, (qty, _))) = self.bids.last_key_value() {
                self.best_bid_ticks = Some(*price_ticks);
                self.best_bid_qty = Some(*qty);
            } else {
                self.best_bid_ticks = None;
                self.best_bid_qty = None;
            }
        } else if let Some((price_ticks, (qty, _))) = self.asks.first_key_value() {
            self.best_ask_ticks = Some(*price_ticks);
            self.best_ask_qty = Some(*qty);
        } else {
            self.best_ask_ticks = None;
            self.best_ask_qty = None;
        }
    }

    fn reset_after_gap(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.trade_buffer.clear();
        self.last_changes.clear();
        self.cached_buy_points.clear();
        self.cached_sell_points.clear();
        self.last_applied_u = 0;
        self.last_top_update_id = None;
        self.is_synced = false;
        self.clear_top_of_book_cache();
    }
}

pub struct LevelChangeResult {
    pub inflow: f64,
    pub cancel: f64,
    pub is_tob: bool,
    pub in_top20: bool,
    pub is_bid: bool,
}

fn sweep_vwap<'a>(
    iter: impl Iterator<Item = (&'a i64, &'a (f64, OrderQueue))>,
    quantity: f64,
    tick_size: f64,
) -> Option<f64> {
    let mut remaining = quantity;
    let mut notional = 0.0;
    let mut filled = 0.0;

    for (price_ticks, (level_total, _)) in iter {
        if remaining <= QTY_EPSILON {
            break;
        }

        let price = *price_ticks as f64 * tick_size;
        let take_qty = level_total.min(remaining);
        notional += take_qty * price;
        filled += take_qty;
        remaining -= take_qty;
    }

    if filled <= QTY_EPSILON || remaining > QTY_EPSILON {
        None
    } else {
        Some(notional / filled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::SymbolStr;

    fn approx_eq(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-9, "{left} != {right}");
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

        assert!(ob.bids.is_empty());
        assert!(ob.asks.is_empty());
        assert!(ob.best_bid_ask().is_none());
    }
}
