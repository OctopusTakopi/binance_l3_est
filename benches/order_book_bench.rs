use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Duration;

use binance_l3_est::engine::order_book::OrderBook;
use binance_l3_est::types::{DepthUpdate, OrderBookSnapshot};
use binance_l3_est::utils::SymbolStr;

fn generate_synthetic_data(
    num_updates: usize,
    tick_size: f64,
) -> (OrderBookSnapshot, Vec<DepthUpdate>) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut bids = Vec::new();
    let mut asks = Vec::new();

    let mid_price = 50000.0;

    for i in 1..=500 {
        let price = mid_price - (i as f64) * tick_size;
        let qty = rng.random_range(0.1..5.0);
        bids.push([price, qty]);
    }

    for i in 1..=500 {
        let price = mid_price + (i as f64) * tick_size;
        let qty = rng.random_range(0.1..5.0);
        asks.push([price, qty]);
    }

    let snapshot = OrderBookSnapshot {
        last_update_id: 1000,
        bids,
        asks,
    };

    let mut updates = Vec::with_capacity(num_updates);
    let mut current_u = 1000;

    for _ in 0..num_updates {
        let pu = current_u;
        current_u += 1;

        let num_bids = rng.random_range(1..=5);
        let num_asks = rng.random_range(1..=5);

        let mut b = Vec::new();
        let mut a = Vec::new();

        for _ in 0..num_bids {
            let offset = rng.random_range(1..=500);
            let price = mid_price - (offset as f64) * tick_size;
            let qty = rng.random_range(0.0..6.0); // 0.0 means cancel
            b.push([price, qty]);
        }

        for _ in 0..num_asks {
            let offset = rng.random_range(1..=500);
            let price = mid_price + (offset as f64) * tick_size;
            let qty = rng.random_range(0.0..6.0); // 0.0 means cancel
            a.push([price, qty]);
        }

        updates.push(DepthUpdate {
            event_type: "depthUpdate".to_string(),
            event_time: 1000000 + current_u,
            transaction_time: 1000000 + current_u,
            symbol: SymbolStr::from("BTCUSDT"),
            capital_u: current_u,
            small_u: current_u,
            pu: Some(pu),
            b,
            a,
        });
    }

    (snapshot, updates)
}

fn bench_order_book(c: &mut Criterion) {
    let tick_size = 0.1;
    let (snapshot, updates) = generate_synthetic_data(10_000, tick_size);

    c.bench_function("order_book_process_10k_updates", |b| {
        b.iter(|| {
            let mut ob = OrderBook::new(tick_size);
            ob.apply_snapshot(snapshot.clone());

            // Process all updates
            for update in &updates {
                let _ = ob.process_update(black_box(update.clone()), 1.0);
            }
        })
    });
}

criterion_group!(benches, bench_order_book);
criterion_main!(benches);
