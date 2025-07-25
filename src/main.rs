mod kmeans;

use eframe::egui;
use egui::{Align2, Color32, TextureOptions, Vec2};
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoint, PlotPoints, Text};
use futures_util::{SinkExt, StreamExt};
use once_cell::sync::Lazy;
use reqwest::blocking;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::Deserialize;
use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::sync::mpsc::{self as std_mpsc, Receiver as StdReceiver, Sender as StdSender};
use std::thread;
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message as WsMessage};

#[derive(Deserialize)]
struct ExchangeInfo {
    symbols: Vec<SymbolInfo>,
}

#[derive(Deserialize)]
struct SymbolInfo {
    symbol: String,
    filters: Vec<Filter>,
}

#[derive(Deserialize)]
struct Filter {
    #[serde(rename = "filterType")]
    filter_type: String,
    #[serde(rename = "tickSize")]
    tick_size: Option<String>,
    #[serde(rename = "stepSize")]
    step_size: Option<String>,
}

#[derive(Deserialize)]
struct OrderBookSnapshot {
    #[serde(rename = "lastUpdateId")]
    last_update_id: u64,
    bids: Vec<Vec<Decimal>>,
    asks: Vec<Vec<Decimal>>,
}

#[allow(dead_code)]
#[derive(Deserialize, Clone)]
struct DepthUpdate {
    e: String,
    #[serde(rename = "E")]
    event_time: u64,
    #[serde(rename = "T")]
    transaction_time: u64,
    s: String,
    #[serde(rename = "U")]
    capital_u: u64,
    #[serde(rename = "u")]
    small_u: u64,
    pu: i64,
    b: Vec<Vec<Decimal>>,
    a: Vec<Vec<Decimal>>,
}

enum AppMessage {
    Snapshot(OrderBookSnapshot),
    Update(DepthUpdate),
}

enum Control {
    Refetch,
    ChangeSymbol(String),
}

static BID_COLORS: Lazy<Vec<Color32>> = Lazy::new(|| {
    vec![
        Color32::from_rgb(222, 235, 247), // Light Blue
        Color32::from_rgb(204, 227, 245), // Lighter Blue
        Color32::from_rgb(158, 202, 225), // Blue
        Color32::from_rgb(129, 189, 231), // Light Medium Blue
        Color32::from_rgb(107, 174, 214), // Medium Blue
        Color32::from_rgb(78, 157, 202),  // Medium Deep Blue
        Color32::from_rgb(49, 130, 189),  // Deep Blue
        Color32::from_rgb(33, 113, 181),  // Darker Deep Blue
        Color32::from_rgb(16, 96, 168),   // Dark Blue
        Color32::from_rgb(8, 81, 156),    // Darkest Blue
    ]
});

static ASK_COLORS: Lazy<Vec<Color32>> = Lazy::new(|| {
    vec![
        Color32::from_rgb(254, 230, 206), // Light Orange
        Color32::from_rgb(253, 216, 186), // Lighter Orange
        Color32::from_rgb(253, 174, 107), // Orange
        Color32::from_rgb(253, 159, 88),  // Light Deep Orange
        Color32::from_rgb(253, 141, 60),  // Deep Orange
        Color32::from_rgb(245, 126, 47),  // Medium Red-Orange
        Color32::from_rgb(230, 85, 13),   // Red-Orange
        Color32::from_rgb(204, 75, 12),   // Darker Red-Orange
        Color32::from_rgb(179, 65, 10),   // Dark Red
        Color32::from_rgb(166, 54, 3),    // Darkest Red
    ]
});

fn main() -> eframe::Result {
    // Fetch the symbol from command-line arguments or default to DOGEUSDT
    let args: Vec<String> = env::args().collect();
    let symbol: String = if args.len() > 1 {
        args[1].to_ascii_lowercase()
    } else {
        "dogeusdt".to_string()
    };

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Order Book Visualizer",
        options,
        Box::new(move |cc| Ok(Box::new(MyApp::new(cc, symbol)))),
    )
}

struct MyApp {
    symbol: String,
    edited_symbol: String,
    bids: BTreeMap<Decimal, VecDeque<Decimal>>,
    asks: BTreeMap<Decimal, VecDeque<Decimal>>,
    last_applied_u: u64,
    is_synced: bool,
    rx: StdReceiver<AppMessage>,
    update_buffer: VecDeque<DepthUpdate>,
    control_tx: Sender<Control>,
    kmeans_mode: bool,
    price_prec: usize,
    qty_prec: usize,
    brighter_step: usize,
    batch_size: usize,
    max_iter: usize,
    heatmap_data: VecDeque<Vec<f64>>,
    heatmap_width: usize,
    heatmap_height: usize,
    update_counter: u32,
    show_heatmap: bool,
    show_liquidity_cost: bool,
    execute_usd: f64,
    max_liquidity_usd: f64,
    rolling_max_qty: f64,
}

impl MyApp {
    fn new(cc: &eframe::CreationContext<'_>, symbol: String) -> Self {
        let (tx, rx) = std_mpsc::channel();
        let (control_tx, control_rx) = mpsc::channel(1);
        let ctx = cc.egui_ctx.clone();
        let s = symbol.clone();
        thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                Self::fetch_and_stream_loop(&tx, &ctx, control_rx, s).await;
            });
        });

        let mut price_prec = 2;
        let mut qty_prec = 2;
        Self::fetch_precision(&symbol.to_uppercase(), &mut price_prec, &mut qty_prec);

        Self {
            symbol: symbol.clone(),
            edited_symbol: symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_applied_u: 0,
            is_synced: false,
            rx,
            update_buffer: VecDeque::new(),
            control_tx,
            kmeans_mode: false,
            price_prec,
            qty_prec,
            brighter_step: 5,
            batch_size: 1024,
            max_iter: 1024,
            heatmap_data: VecDeque::new(),
            heatmap_width: 480,
            heatmap_height: 320,
            update_counter: 0,
            show_heatmap: true,
            show_liquidity_cost: true,
            execute_usd: 0.0,
            max_liquidity_usd: 100000.0,
            rolling_max_qty: 1.0,
        }
    }

    fn fetch_precision(symbol: &str, price_prec: &mut usize, qty_prec: &mut usize) {
        let url = "https://fapi.binance.com/fapi/v1/exchangeInfo".to_string();
        if let Ok(resp) = blocking::get(&url)
            && let Ok(info) = resp.json::<ExchangeInfo>()
            && let Some(sym_info) = info.symbols.into_iter().find(|s| s.symbol == *symbol)
        {
            for filter in sym_info.filters {
                if filter.filter_type == "PRICE_FILTER" {
                    if let Some(ts) = filter.tick_size {
                        let tick_size = ts.parse::<f64>().unwrap_or(1.0);
                        if tick_size > 0.0 {
                            *price_prec = (-tick_size.log10()).ceil() as usize;
                        }
                    }
                } else if filter.filter_type == "LOT_SIZE"
                    && let Some(ss) = filter.step_size
                {
                    let step_size = ss.parse::<f64>().unwrap_or(1.0);
                    if step_size > 0.0 {
                        *qty_prec = (-step_size.log10()).ceil() as usize;
                    }
                }
            }
        }
    }

    async fn fetch_and_stream_loop(
        tx: &StdSender<AppMessage>,
        ctx: &egui::Context,
        mut control_rx: Receiver<Control>,
        mut symbol: String,
    ) {
        loop {
            let ws_url_str = format!("wss://fstream.binance.com/ws/{symbol}@depth@0ms");
            let (mut ws_stream, response) = match connect_async(ws_url_str).await {
                Ok(pair) => pair,
                Err(e) => {
                    println!("WebSocket connection error: {e:?}");
                    return;
                }
            };
            println!("WebSocket connected: {response:?}");

            let tx_clone = tx.clone();
            let ctx_clone = ctx.clone();
            let ws_handle = tokio::spawn(async move {
                while let Some(result) = ws_stream.next().await {
                    match result {
                        Ok(message) => match message {
                            WsMessage::Text(text) => {
                                match serde_json::from_str::<DepthUpdate>(&text) {
                                    Ok(update) => {
                                        tx_clone.send(AppMessage::Update(update)).unwrap();
                                        ctx_clone.request_repaint();
                                    }
                                    Err(e) => println!("Update JSON error: {e:?}"),
                                }
                            }
                            WsMessage::Ping(payload) => {
                                if let Err(e) = ws_stream.send(WsMessage::Pong(payload)).await {
                                    println!("Pong send error: {e:?}");
                                    break;
                                }
                            }
                            WsMessage::Pong(_) => {}
                            WsMessage::Close(_) => {
                                println!("Connection closed by server.");
                                break;
                            }
                            _ => {}
                        },
                        Err(e) => {
                            println!("WebSocket error: {e:?}");
                            break;
                        }
                    }
                }
            });

            let client = reqwest::Client::new();
            let snap_url = format!(
                "https://fapi.binance.com/fapi/v1/depth?symbol={}&limit=1000",
                symbol.to_uppercase()
            );
            match client.get(snap_url).send().await {
                Ok(resp) => match resp.json::<OrderBookSnapshot>().await {
                    Ok(snap) => {
                        println!("Snapshot fetched successfully.");
                        tx.send(AppMessage::Snapshot(snap)).unwrap();
                    }
                    Err(e) => println!("Snapshot JSON error: {e:?}"),
                },
                Err(e) => println!("Snapshot request error: {e:?}"),
            }

            if let Some(ctrl) = control_rx.recv().await {
                ws_handle.abort();
                match ctrl {
                    Control::Refetch => {
                        println!("Refetch triggered, restarting connection.");
                    }
                    Control::ChangeSymbol(new_symbol) => {
                        symbol = new_symbol;
                        println!("Changing symbol to {symbol}, restarting connection.");
                    }
                }
            } else {
                break;
            }
        }
    }

    fn process_update(&mut self, update: DepthUpdate) {
        if update.small_u < self.last_applied_u {
            return;
        }

        if self.is_synced {
            if (update.pu as u64) != self.last_applied_u {
                println!(
                    "Warning: Message gap detected! pu: {}, last: {}",
                    update.pu, self.last_applied_u
                );
                self.update_buffer.clear();
                let _ = self.control_tx.try_send(Control::Refetch);
                return;
            }
            self.apply_update(&update);
            self.last_applied_u = update.small_u;
        } else if update.capital_u <= self.last_applied_u && self.last_applied_u <= update.small_u {
            self.apply_update(&update);
            self.last_applied_u = update.small_u;
            self.is_synced = true;
        } else {
            println!(
                "Initial gap detected! U: {}, u: {}, last: {}",
                update.capital_u, update.small_u, self.last_applied_u
            );
            self.update_buffer.clear();
            let _ = self.control_tx.try_send(Control::Refetch);
        }
        self.update_counter += 1;
        if self.update_counter.is_multiple_of(10) {
            // Append every 10 updates to reduce frequency
            self.append_to_heatmap();
            self.update_counter = 0;
        }
    }

    fn append_to_heatmap(&mut self) {
        if self.bids.is_empty() || self.asks.is_empty() {
            return;
        }

        let mut snapshot = vec![0.0; self.heatmap_height];

        // Get max level sum for normalization
        let current_max = self
            .bids
            .values()
            .chain(self.asks.values())
            .map(|v| v.iter().sum::<Decimal>().to_f64().unwrap_or(0.0))
            .fold(0.0, f64::max);

        // Smooth update (EMA)
        let a = 0.3;
        self.rolling_max_qty = self.rolling_max_qty * (1.0 - a) + current_max * a;
        let max_qty = self.rolling_max_qty.max(1e-6) / 3.0;

        // Fill asks (top half)
        let ask_iter = self.asks.iter().take(self.heatmap_height / 2).rev();
        for (snapshot_cell, (_, qty_deq)) in snapshot
            .iter_mut()
            .take(self.heatmap_height / 2)
            .zip(ask_iter)
        {
            let sum = qty_deq.iter().sum::<Decimal>().to_f64().unwrap_or(0.0) / max_qty;
            *snapshot_cell = sum.clamp(0.0, 1.0);
        }

        // Fill bids (bottom half)
        let bid_iter = self.bids.iter().rev().take(self.heatmap_height / 2);
        for (snapshot_cell, (_, qty_deq)) in snapshot
            .iter_mut()
            .skip(self.heatmap_height / 2)
            .zip(bid_iter)
        {
            let sum = qty_deq.iter().sum::<Decimal>().to_f64().unwrap_or(0.0) / max_qty;
            *snapshot_cell = -sum.clamp(0.0, 1.0);
        }

        self.heatmap_data.push_back(snapshot);
        if self.heatmap_data.len() > self.heatmap_width {
            self.heatmap_data.pop_front();
        }
    }

    fn calculate_liquidity_impact(&self) -> (f64, f64) {
        if self.bids.is_empty() || self.asks.is_empty() {
            return (0.0, 0.0);
        }

        let best_bid = self
            .bids
            .keys()
            .next_back()
            .cloned()
            .unwrap_or(Decimal::ZERO)
            .to_f64()
            .unwrap_or(0.0);
        let best_ask = self
            .asks
            .keys()
            .next()
            .cloned()
            .unwrap_or(Decimal::ZERO)
            .to_f64()
            .unwrap_or(0.0);
        let mid = (best_bid + best_ask) / 2.0;
        if mid == 0.0 {
            return (0.0, 0.0);
        }

        let quantity = self.execute_usd / mid;

        // For buy impact (sweep asks)
        let mut cum_qty = 0.0;
        let mut weighted_price = 0.0;
        let mut last_price = best_ask;
        for (price, qty_deq) in self.asks.iter() {
            last_price = price.to_f64().unwrap_or(0.0);
            let level_qty = qty_deq.iter().sum::<Decimal>().to_f64().unwrap_or(0.0);
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
        let buy_vwap = if cum_qty > 0.0 {
            weighted_price / cum_qty
        } else {
            best_ask
        };
        let buy_impact = ((buy_vwap - mid) / mid * 10000.0).abs();

        // For sell impact (sweep bids)
        let mut cum_qty = 0.0;
        let mut weighted_price = 0.0;
        let mut last_price = best_bid;
        for (price, qty_deq) in self.bids.iter().rev() {
            last_price = price.to_f64().unwrap_or(0.0);
            let level_qty = qty_deq.iter().sum::<Decimal>().to_f64().unwrap_or(0.0);
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
        let sell_vwap = if cum_qty > 0.0 {
            weighted_price / cum_qty
        } else {
            best_bid
        };
        let sell_impact = ((mid - sell_vwap) / mid * 10000.0).abs();

        (buy_impact, sell_impact)
    }

    fn get_liquidity_curves(&self) -> (Vec<PlotPoint>, Vec<PlotPoint>) {
        let best_bid = self
            .bids
            .keys()
            .next_back()
            .cloned()
            .unwrap_or(Decimal::ZERO)
            .to_f64()
            .unwrap_or(0.0);
        let best_ask = self
            .asks
            .keys()
            .next()
            .cloned()
            .unwrap_or(Decimal::ZERO)
            .to_f64()
            .unwrap_or(0.0);
        let mid = (best_bid + best_ask) / 2.0;
        if mid == 0.0 {
            return (vec![], vec![]);
        }

        // Buy curve (sweep asks)
        let mut buy_points = vec![PlotPoint::new(0.0, 0.0)];
        let mut cum_qty = 0.0;
        let mut weighted_price = 0.0;
        for (price, qty_deq) in self.asks.iter() {
            let level_qty = qty_deq.iter().sum::<Decimal>().to_f64().unwrap_or(0.0);
            weighted_price += level_qty * price.to_f64().unwrap_or(0.0);
            cum_qty += level_qty;
            let vwap = weighted_price / cum_qty;
            let delta_p = ((vwap - mid) / mid * 10000.0).abs();
            let usd = cum_qty * mid;
            buy_points.push(PlotPoint::new(usd, delta_p));
        }

        // Sell curve (sweep bids)
        let mut sell_points = vec![PlotPoint::new(0.0, 0.0)];
        let mut cum_qty = 0.0;
        let mut weighted_price = 0.0;
        for (price, qty_deq) in self.bids.iter().rev() {
            let level_qty = qty_deq.iter().sum::<Decimal>().to_f64().unwrap_or(0.0);
            weighted_price += level_qty * price.to_f64().unwrap_or(0.0);
            cum_qty += level_qty;
            let vwap = weighted_price / cum_qty;
            let delta_p = ((mid - vwap) / mid * 10000.0).abs();
            let usd = cum_qty * mid;
            sell_points.push(PlotPoint::new(usd, delta_p));
        }

        (buy_points, sell_points)
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                AppMessage::Snapshot(snap) => {
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

                    while let Some(update) = self.update_buffer.pop_front() {
                        self.process_update(update);
                    }
                }
                AppMessage::Update(update) => {
                    if self.last_applied_u == 0 {
                        self.update_buffer.push_back(update);
                    } else {
                        self.process_update(update);
                    }
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(format!(
                "{} Perpetual Order Book",
                self.symbol.to_uppercase()
            ));
            if ui.button("Toggle K-Means Mode").clicked() {
                self.kmeans_mode = !self.kmeans_mode;
            }
            if ui.button("Toggle Sweep / Liquidity-Cost").clicked() {
                self.show_liquidity_cost = !self.show_liquidity_cost;
            }
            if ui.button("Toggle Depth-Time Heatmap").clicked() {
                self.show_heatmap = !self.show_heatmap;
            }

            ui.horizontal(|ui| {
                ui.label("Symbol:");
                ui.text_edit_singleline(&mut self.edited_symbol);
                if ui.button("Change Symbol").clicked() && self.edited_symbol != self.symbol {
                    Self::fetch_precision(
                        &self.edited_symbol.to_uppercase(),
                        &mut self.price_prec,
                        &mut self.qty_prec,
                    );
                    let _ = self
                        .control_tx
                        .try_send(Control::ChangeSymbol(self.edited_symbol.clone()));
                    self.symbol = self.edited_symbol.clone();
                    self.bids.clear();
                    self.asks.clear();
                    self.last_applied_u = 0;
                    self.is_synced = false;
                    self.heatmap_data.clear();
                }
            });

            if self.kmeans_mode {
                ui.horizontal(|ui| {
                    ui.label("K-means Batch Size:");
                    ui.add(egui::Slider::new(&mut self.batch_size, 32..=2048));
                });
                ui.horizontal(|ui| {
                    ui.label("K-means Max Iter:");
                    ui.add(egui::Slider::new(&mut self.max_iter, 64..=2048));
                });
            } else {
                ui.horizontal(|ui| {
                    ui.label("Age mode brighter step %:");
                    ui.add(egui::Slider::new(&mut self.brighter_step, 1..=10));
                });
            }

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    egui::Grid::new("order_book_grid")
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("Asks");
                            ui.label("Price");
                            ui.label("Quantity");
                            ui.end_row();

                            for (price, qty) in self.asks.iter().take(20).rev() {
                                ui.label("");
                                ui.label(format!(
                                    "{:.1$}",
                                    price.to_f64().unwrap_or(0.0),
                                    self.price_prec
                                ));
                                ui.label(format!(
                                    "{:.1$}",
                                    qty.iter().sum::<Decimal>().to_f64().unwrap_or(0.0),
                                    self.qty_prec
                                ));
                                ui.end_row();
                            }

                            ui.label("Bids");
                            ui.label("Price");
                            ui.label("Quantity");
                            ui.end_row();

                            for (price, qty) in self.bids.iter().rev().take(20) {
                                ui.label("");
                                ui.label(format!(
                                    "{:.1$}",
                                    price.to_f64().unwrap_or(0.0),
                                    self.price_prec
                                ));
                                ui.label(format!(
                                    "{:.1$}",
                                    qty.iter().sum::<Decimal>().to_f64().unwrap_or(0.0),
                                    self.qty_prec
                                ));
                                ui.end_row();
                            }
                        });
                });

                ui.vertical(|ui| {
                    let bid_levels: Vec<(&Decimal, Decimal)> = self
                        .bids
                        .iter()
                        .rev()
                        .take(200)
                        .map(|(key, deque)| {
                            let sum = deque.iter().cloned().sum::<Decimal>(); // Sum the VecDeque<Decimal>
                            (key, sum)
                        })
                        .collect();
                    let ask_levels: Vec<(&Decimal, Decimal)> = self
                        .asks
                        .iter()
                        .take(200)
                        .map(|(key, deque)| {
                            let sum = deque.iter().cloned().sum::<Decimal>(); // Sum the VecDeque<Decimal>
                            (key, sum)
                        })
                        .collect();
                    let mut max_qty: f64 = 0.0;
                    for (_, qty) in &bid_levels {
                        max_qty = max_qty.max(qty.to_f64().unwrap_or(0.0));
                    }
                    for (_, qty) in &ask_levels {
                        max_qty = max_qty.max(qty.to_f64().unwrap_or(0.0));
                    }

                    let step = 1.0;
                    let mut bars: Vec<Bar> = Vec::new();

                    let max_bid_order: Decimal = self
                        .bids
                        .values()
                        .rev()
                        .take(200)
                        .flat_map(|dq| dq.iter())
                        .cloned()
                        .max()
                        .unwrap_or(Decimal::ZERO);
                    let max_ask_order: Decimal = self
                        .asks
                        .values()
                        .take(200)
                        .flat_map(|dq| dq.iter())
                        .cloned()
                        .max()
                        .unwrap_or(Decimal::ZERO);
                    let second_max_bid_order = {
                        let mut orders: Vec<_> = self
                            .bids
                            .values()
                            .rev()
                            .take(200)
                            .flat_map(|dq| dq.iter())
                            .cloned()
                            .collect();
                        orders.sort_by(|a, b| b.cmp(a)); // Sort in descending order
                        orders.get(1).cloned().unwrap_or(Decimal::ZERO)
                    };
                    let second_max_ask_order = {
                        let mut orders: Vec<_> = self
                            .asks
                            .values()
                            .take(200)
                            .flat_map(|dq| dq.iter())
                            .cloned()
                            .collect();
                        orders.sort_by(|a, b| b.cmp(a)); // Sort in descending order
                        orders.get(1).cloned().unwrap_or(Decimal::ZERO)
                    };

                    if !self.kmeans_mode {
                        for (i, (_, qty_deq)) in self.asks.iter().take(200).enumerate() {
                            let x = (i as f64 + 0.5) * step + 0.5;
                            let mut offset = 0.0;

                            for (j, &qty) in qty_deq.iter().enumerate() {
                                if qty <= dec!(0.0) {
                                    continue;
                                }
                                let color = if qty == max_ask_order {
                                    Color32::GOLD
                                } else if qty == second_max_ask_order {
                                    Color32::from_rgb(184, 134, 11)
                                } else {
                                    self.get_order_color(
                                        j,
                                        Color32::DARK_RED,
                                        self.brighter_step as f32 / 100.0,
                                    )
                                };
                                let bar = Bar::new(x, qty.to_f64().unwrap_or(0.0))
                                    .fill(color)
                                    .base_offset(offset)
                                    .width(step * 0.9);
                                bars.push(bar);
                                offset += qty.to_f64().unwrap_or(0.0);
                            }
                        }

                        // Color Mapping for Bids
                        for (i, (_, qty_deq)) in self.bids.iter().rev().take(200).enumerate() {
                            let x = -(i as f64 + 0.5) * step - 0.5;
                            let mut offset = 0.0;

                            for (j, &qty) in qty_deq.iter().enumerate() {
                                if qty <= dec!(0.0) {
                                    continue;
                                }
                                let color = if qty == max_bid_order {
                                    Color32::GOLD
                                } else if qty == second_max_bid_order {
                                    Color32::from_rgb(184, 134, 11)
                                } else {
                                    self.get_order_color(
                                        j,
                                        Color32::DARK_GREEN,
                                        self.brighter_step as f32 / 100.0,
                                    )
                                };
                                let bar = Bar::new(x, qty.to_f64().unwrap_or(0.0))
                                    .fill(color)
                                    .base_offset(offset)
                                    .width(step * 0.9);
                                bars.push(bar);
                                offset += qty.to_f64().unwrap_or(0.0);
                            }
                        }
                    } else {
                        let asks_for_cluster: BTreeMap<Decimal, VecDeque<Decimal>> = self
                            .asks
                            .iter()
                            .take(200)
                            .map(|(&k, v)| (k, v.clone()))
                            .collect();
                        let mut kmeans_asks =
                            kmeans::MiniBatchKMeans::new(10, self.batch_size, self.max_iter);
                        let labels_asks = kmeans_asks.fit(&asks_for_cluster);
                        let clustered_asks =
                            kmeans::build_clustered_orders(&asks_for_cluster, &labels_asks);

                        let bids_for_cluster: BTreeMap<Decimal, VecDeque<Decimal>> = self
                            .bids
                            .iter()
                            .rev()
                            .take(200)
                            .map(|(&k, v)| (k, v.clone()))
                            .collect();
                        let mut kmeans_bids =
                            kmeans::MiniBatchKMeans::new(10, self.batch_size, self.max_iter);
                        let labels_bids = kmeans_bids.fit(&bids_for_cluster);
                        let clustered_bids =
                            kmeans::build_clustered_orders(&bids_for_cluster, &labels_bids);

                        // Asks in K-Means mode
                        for (i, (_, qty_deq)) in clustered_asks.iter().enumerate() {
                            let x = (i as f64 + 0.5) * step + 0.5;
                            let mut offset = 0.0;

                            for &(qty, cluster) in qty_deq.iter() {
                                if qty <= dec!(0.0) {
                                    continue;
                                }
                                let color = if qty == max_ask_order {
                                    Color32::GOLD
                                } else {
                                    ASK_COLORS
                                        .get(cluster % ASK_COLORS.len())
                                        .cloned()
                                        .unwrap_or(Color32::GRAY)
                                };
                                let bar = Bar::new(x, qty.to_f64().unwrap_or(0.0))
                                    .fill(color)
                                    .base_offset(offset)
                                    .width(step * 0.9);
                                bars.push(bar);
                                offset += qty.to_f64().unwrap_or(0.0);
                            }
                        }

                        // Bids in K-Means mode
                        for (i, (_, qty_deq)) in clustered_bids.iter().rev().enumerate() {
                            let x = -(i as f64 + 0.5) * step - 0.5;
                            let mut offset = 0.0;

                            for &(qty, cluster) in qty_deq.iter() {
                                if qty <= dec!(0.0) {
                                    continue;
                                }
                                let color = if qty == max_bid_order {
                                    Color32::GOLD
                                } else {
                                    BID_COLORS
                                        .get(cluster % BID_COLORS.len())
                                        .cloned()
                                        .unwrap_or(Color32::GRAY)
                                };
                                let bar = Bar::new(x, qty.to_f64().unwrap_or(0.0))
                                    .fill(color)
                                    .base_offset(offset)
                                    .width(step * 0.9);
                                bars.push(bar);
                                offset += qty.to_f64().unwrap_or(0.0);
                            }
                        }
                    }

                    Plot::new("orderbook_chart")
                        .allow_drag(false)
                        .allow_scroll(false)
                        .allow_zoom(false)
                        .show_axes([true, true])
                        .show(ui, |plot_ui| {
                            plot_ui.bar_chart(BarChart::new("ob", bars));

                            for (i, (price, _)) in bid_levels.iter().enumerate() {
                                if i.is_multiple_of(20) {
                                    // Show label every 20th level
                                    let x = -(i as f64 + 0.5) * step - 0.5;
                                    plot_ui.text(
                                        Text::new(
                                            "bid",
                                            PlotPoint::new(x, -max_qty * 0.05),
                                            format!(
                                                "{:.1$}",
                                                price.to_f64().unwrap_or(0.0),
                                                self.price_prec
                                            ),
                                        )
                                        .anchor(Align2::CENTER_BOTTOM),
                                    );
                                }
                            }

                            for (i, (price, _)) in ask_levels.iter().enumerate() {
                                if i.is_multiple_of(20) {
                                    // Show label every 20th level
                                    if i == 0 {
                                        continue;
                                    }
                                    let x = (i as f64 + 0.5) * step + 0.5;
                                    plot_ui.text(
                                        Text::new(
                                            "ask",
                                            PlotPoint::new(x, -max_qty * 0.05),
                                            format!(
                                                "{:.1$}",
                                                price.to_f64().unwrap_or(0.0),
                                                self.price_prec
                                            ),
                                        )
                                        .anchor(Align2::CENTER_BOTTOM),
                                    );
                                }
                            }
                        });
                });
            });

            if self.show_heatmap {
                egui::Window::new("Depth-Time Heatmap")
                    .open(&mut self.show_heatmap)
                    .show(ctx, |ui| {
                        if !self.heatmap_data.is_empty() {
                            let width = self.heatmap_data.len();
                            let height = self.heatmap_height;
                            let mut pixels = vec![Color32::BLACK; width * height];

                            for (col, snapshot) in self.heatmap_data.iter().enumerate() {
                                for (row, &value) in snapshot.iter().enumerate() {
                                    let normalized_value = value.abs() * 10.0;
                                    let intensity = (normalized_value * 255.0) as u8;
                                    let color = if value >= 0.0 {
                                        // Ask: red intensity
                                        let intensity = (value * 255.0).clamp(0.0, 255.0) as u8;
                                        Color32::from_rgb(intensity, 0, 0)
                                    } else {
                                        // Bid: blue intensity
                                        let intensity = (-value * 255.0).clamp(0.0, 255.0) as u8;
                                        Color32::from_rgb(0, 0, intensity)
                                    };
                                    pixels[row * width + col] = color;
                                }
                            }

                            let color_image = egui::ColorImage {
                                size: [width, height],
                                source_size: Default::default(),
                                pixels,
                            };

                            let texture = ui.ctx().load_texture(
                                "heatmap",
                                color_image,
                                TextureOptions::LINEAR,
                            );

                            ui.image(&texture);
                        } else {
                            ui.label("No heatmap data yet.");
                        }
                    });
            }

            if self.show_liquidity_cost {
                egui::Window::new("Sweep / Liquidity-Cost")
                    .default_size(Vec2::new(480.0, 320.0))
                    .resizable(false)
                    .show(ctx, |ui| {
                        let (buy_points, sell_points) = self.get_liquidity_curves();
                        let buy_plot_points: PlotPoints =
                            PlotPoints::from_iter(buy_points.clone().iter().map(|p| [p.x, p.y]));
                        let buy_line = Line::new("a", buy_plot_points).color(Color32::BLUE);
                        let sell_plot_points: PlotPoints =
                            PlotPoints::from_iter(sell_points.clone().iter().map(|p| [p.x, p.y]));
                        let sell_line = Line::new("b", sell_plot_points).color(Color32::RED);

                        // Calculate min/max for x and y axes from both curves
                        let all_points = buy_points.iter().chain(sell_points.iter());
                        let x_min = all_points
                            .clone()
                            .map(|p| p.x)
                            .fold(f64::INFINITY, f64::min);
                        let x_max = all_points
                            .clone()
                            .map(|p| p.x)
                            .fold(f64::NEG_INFINITY, f64::max);
                        let y_min = all_points
                            .clone()
                            .map(|p| p.y)
                            .fold(f64::INFINITY, f64::min);
                        let y_max = all_points
                            .clone()
                            .map(|p| p.y)
                            .fold(f64::NEG_INFINITY, f64::max);

                        // Optional: Add padding to avoid clipping (adjust 0.05 for more/less density)
                        let x_range = x_max - x_min;
                        let y_range = y_max - y_min;
                        let padded_x_min = x_min - 0.05 * x_range;
                        let padded_x_max = x_max + 0.05 * x_range;
                        let padded_y_min = y_min - 0.05 * y_range;
                        let padded_y_max = y_max + 0.05 * y_range;

                        ui.allocate_ui(egui::Vec2::new(480.0, 320.0), |ui| {
                            Plot::new("liquidity_cost_plot")
                                .show_axes([true, true])
                                .default_x_bounds(padded_x_min, padded_x_max) // Set x-axis bounds
                                .default_y_bounds(padded_y_min, padded_y_max) // Set y-axis bounds (makes it denser)
                                .show(ui, |plot_ui| {
                                    plot_ui.line(buy_line);
                                    plot_ui.line(sell_line);
                                });
                        });

                        let (buy_impact, sell_impact) = self.calculate_liquidity_impact();
                        ui.label(format!(
                            "Buy impact = {buy_impact:.2} bps | Sell impact = {sell_impact:.2} bps"
                        ));

                        ui.horizontal(|ui| {
                            ui.label("Execute USD X:");
                            ui.add(egui::Slider::new(
                                &mut self.execute_usd,
                                0.0..=self.max_liquidity_usd,
                            ));
                        });
                    });
            }
        });
    }
}

impl MyApp {
    // Function to calculate color based on the order index
    fn get_order_color(&self, index: usize, base_color: Color32, step: f32) -> Color32 {
        // Brighten the color by 5% for each order index
        let brightening_factor = 1.0 + step * index as f32; // 5% brighter per order
        let r = (base_color.r() as f32 * brightening_factor).min(255.0) as u8;
        let g = (base_color.g() as f32 * brightening_factor).min(255.0) as u8;
        let b = (base_color.b() as f32 * brightening_factor).min(255.0) as u8;

        Color32::from_rgb(r, g, b)
    }
}

impl MyApp {
    fn apply_update(&mut self, update: &DepthUpdate) {
        for bid in &update.b {
            let price = bid[0];
            let qty = bid[1];
            if qty == Decimal::ZERO {
                self.bids.remove(&price);
            } else {
                let price = bid[0];
                let qty = bid[1];
                if qty > Decimal::ZERO {
                    if let Some(old_qty) = self.bids.get_mut(&price) {
                        let old_sum = old_qty.iter().sum::<Decimal>();
                        if old_sum > qty {
                            let change = old_sum - qty;
                            if let Some(pos) = old_qty.iter().rposition(|&x| x == change) {
                                old_qty.remove(pos); // Removes the last occurrence of the value
                            } else {
                                let largest_order = *old_qty.iter().max().unwrap();
                                let largest_pos =
                                    old_qty.iter().position(|&x| x == largest_order).unwrap();
                                old_qty.remove(largest_pos);
                                old_qty.push_back(largest_order - change);
                            }
                        } else if old_sum < qty {
                            if old_sum < qty {
                                let change = qty - old_sum;
                                old_qty.push_back(change);
                            }
                        } else {
                            continue;
                        }
                    } else {
                        self.bids.insert(price, VecDeque::from(vec![qty]));
                    }
                }
            }
        }
        for ask in &update.a {
            let price = ask[0];
            let qty = ask[1];
            if qty == Decimal::ZERO {
                self.asks.remove(&price);
            } else if let Some(old_qty) = self.asks.get_mut(&price) {
                let old_sum = old_qty.iter().sum::<Decimal>();
                if old_sum > qty {
                    let change = old_sum - qty;
                    if let Some(pos) = old_qty.iter().rposition(|&x| x == change) {
                        old_qty.remove(pos); // Removes the last occurrence of the value
                    } else {
                        let largest_order = *old_qty.iter().max().unwrap();
                        let largest_pos = old_qty.iter().position(|&x| x == largest_order).unwrap();
                        old_qty.remove(largest_pos);
                        old_qty.push_back(largest_order - change);
                    }
                } else if old_sum < qty {
                    if old_sum < qty {
                        let change = qty - old_sum;
                        old_qty.push_back(change);
                    }
                } else {
                    continue;
                }
            } else {
                self.asks.insert(price, VecDeque::from(vec![qty]));
            }
        }
    }
}
