use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::{BTreeMap, HashMap, VecDeque};

use binance_l3_est::engine::order_book::OrderBook;
use binance_l3_est::types::{DepthUpdate, OrderBookSnapshot, Trade};
use binance_l3_est::utils::SymbolStr;
use egui_plot::PlotPoint;

// --- OLD ORDER BOOK IMPLEMENTATION ---

type OrderQueue = VecDeque<f64>;
type BookSide = BTreeMap<i64, (f64, OrderQueue)>;

const QTY_EPSILON: f64 = 1e-12;

#[derive(Debug, PartialEq, Eq)]
pub enum OldOrderBookError { SequenceGap }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OldDepthUpdateStatus { Applied, IgnoredStale }

pub struct OldOrderBook {
    pub bids: BookSide,
    pub asks: BookSide,
    pub last_applied_u: u64,
    pub is_synced: bool,
    pub trade_buffer: HashMap<i64, VecDeque<Trade>>,
    pub last_changes: Vec<OldLevelChangeResult>,
    pub cached_buy_points: Vec<PlotPoint>,
    pub cached_sell_points: Vec<PlotPoint>,
    tick_size: f64,
}

pub struct OldLevelChangeResult {
    pub inflow: f64,
    pub cancel: f64,
    pub is_tob: bool,
    pub in_top20: bool,
    pub is_bid: bool,
}

impl OldOrderBook {
    pub fn new(tick_size: f64) -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_applied_u: 0,
            is_synced: false,
            trade_buffer: HashMap::new(),
            last_changes: Vec::new(),
            cached_buy_points: Vec::new(),
            cached_sell_points: Vec::new(),
            tick_size,
        }
    }

    pub fn price_to_ticks(&self, price: f64) -> i64 {
        (price / self.tick_size).round() as i64
    }

    pub fn apply_snapshot(&mut self, snap: OrderBookSnapshot) {
        self.bids.clear();
        self.asks.clear();
        for bid in &snap.bids {
            self.bids.insert(self.price_to_ticks(bid[0]), (bid[1], VecDeque::from([bid[1]])));
        }
        for ask in &snap.asks {
            self.asks.insert(self.price_to_ticks(ask[0]), (ask[1], VecDeque::from([ask[1]])));
        }
        self.last_applied_u = snap.last_update_id;
    }

    pub fn process_update(&mut self, update: DepthUpdate, _mean: f64) -> Result<OldDepthUpdateStatus, OldOrderBookError> {
        for bid in &update.b {
            self.process_level_change(self.price_to_ticks(bid[0]), bid[1], true);
        }
        for ask in &update.a {
            self.process_level_change(self.price_to_ticks(ask[0]), ask[1], false);
        }
        Ok(OldDepthUpdateStatus::Applied)
    }

    fn process_level_change(&mut self, price_ticks: i64, new_total_qty: f64, is_bid: bool) {
        let side = if is_bid { &mut self.bids } else { &mut self.asks };
        if let Some((total_qty, queue)) = side.get_mut(&price_ticks) {
            if new_total_qty > *total_qty + QTY_EPSILON {
                queue.push_back(new_total_qty - *total_qty);
                *total_qty = new_total_qty;
            } else if new_total_qty < *total_qty - QTY_EPSILON {
                let mut to_rem = *total_qty - new_total_qty;
                while to_rem > QTY_EPSILON && !queue.is_empty() {
                    let last = queue.len() - 1;
                    if queue[last] <= to_rem + QTY_EPSILON { to_rem -= queue[last]; queue.pop_back(); }
                    else { queue[last] -= to_rem; to_rem = 0.0; }
                }
                if new_total_qty <= QTY_EPSILON { side.remove(&price_ticks); }
                else { *total_qty = new_total_qty; }
            }
        } else if new_total_qty > QTY_EPSILON {
            side.insert(price_ticks, (new_total_qty, VecDeque::from([new_total_qty])));
        }
    }
}

// --- BENCHMARK ---

fn generate_synthetic_data(num_updates: usize, tick_size: f64) -> (OrderBookSnapshot, Vec<DepthUpdate>) {
    let mut rng = StdRng::seed_from_u64(42);
    let mut bids = Vec::new();
    let mut asks = Vec::new();
    let mid = 50000.0;
    for i in 1..=500 {
        bids.push([mid - (i as f64) * tick_size, rng.random_range(0.1..5.0)]);
        asks.push([mid + (i as f64) * tick_size, rng.random_range(0.1..5.0)]);
    }
    let snapshot = OrderBookSnapshot { last_update_id: 1000, bids, asks };
    let mut updates = Vec::new();
    for i in 0..num_updates {
        let mut b = Vec::new();
        let mut a = Vec::new();
        for _ in 0..2 {
            b.push([mid - (rng.random_range(1..=500) as f64) * tick_size, rng.random_range(0.0..6.0)]);
            a.push([mid + (rng.random_range(1..=500) as f64) * tick_size, rng.random_range(0.0..6.0)]);
        }
        updates.push(DepthUpdate {
            event_type: "depthUpdate".to_string(), event_time: 0, transaction_time: 0,
            symbol: SymbolStr::from("BTCUSDT"), capital_u: 0, small_u: 1001 + i as u64,
            pu: None, b, a,
        });
    }
    (snapshot, updates)
}

fn bench_comparison(c: &mut Criterion) {
    let tick_size = 0.1;
    let (snapshot, updates) = generate_synthetic_data(1000, tick_size);
    let mut new_ob = OrderBook::new(tick_size);
    new_ob.apply_snapshot(snapshot.clone());
    let mut old_ob = OldOrderBook::new(tick_size);
    old_ob.apply_snapshot(snapshot);

    let mut g = c.benchmark_group("OrderBook Comparison");
    g.bench_function("New (RingBuffer + Bitset)", |b| {
        let mut i = 0;
        b.iter(|| {
            let update = &updates[i % updates.len()];
            let _ = new_ob.process_update(black_box(update.clone()), 1.0);
            i += 1;
        })
    });
    g.bench_function("Old (BTreeMap)", |b| {
        let mut i = 0;
        b.iter(|| {
            let update = &updates[i % updates.len()];
            let _ = old_ob.process_update(black_box(update.clone()), 1.0);
            i += 1;
        })
    });
    g.finish();
}

criterion_group!(benches, bench_comparison);
criterion_main!(benches);
