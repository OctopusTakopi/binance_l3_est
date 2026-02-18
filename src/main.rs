mod kmeans;

use eframe::egui;
use egui::{Align2, Color32, TextureOptions, Vec2, Vec2b};
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoint, PlotPoints, Text};
use futures_util::{SinkExt, StreamExt};
use once_cell::sync::Lazy;
use reqwest::blocking;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
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
pub struct Trade {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "t")]
    pub trade_id: u64,
    #[serde(rename = "p")]
    pub price: Decimal,
    #[serde(rename = "q")]
    pub quantity: Decimal,
    #[serde(rename = "X")]
    pub order_type: String,
    #[serde(rename = "T")]
    pub transaction_time: u64,
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
}

#[allow(dead_code)]
#[derive(Deserialize, Clone)]
pub struct DepthUpdate {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "T")]
    pub transaction_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "U")]
    pub capital_u: u64,
    #[serde(rename = "u")]
    pub small_u: u64,
    #[serde(rename = "pu")]
    pub pu: i64,
    #[serde(rename = "b")]
    pub b: Vec<Vec<Decimal>>,
    #[serde(rename = "a")]
    pub a: Vec<Vec<Decimal>>,
}

#[allow(dead_code)]
#[derive(Deserialize, Clone)]
pub struct BookTicker {
    #[serde(rename = "u")]
    pub update_id: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b")]
    pub best_bid_price: Decimal,
    #[serde(rename = "B")]
    pub best_bid_qty: Decimal,
    #[serde(rename = "a")]
    pub best_ask_price: Decimal,
    #[serde(rename = "A")]
    pub best_ask_qty: Decimal,
    #[serde(rename = "T")]
    pub transaction_time: u64,
    #[serde(rename = "E")]
    pub event_time: u64,
}

enum AppMessage {
    Snapshot(OrderBookSnapshot),
    Update(DepthUpdate),
    Trade(Trade),
    Ticker(BookTicker),
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
    show_metrics: bool,
    execute_usd: f64,
    max_liquidity_usd: f64,
    rolling_mean_qty: f64,
    rolling_std_qty: f64,
    warmup_samples: usize,
    // SOFP: Rolling Trade Stats
    rolling_trade_mean: f64,
    rolling_trade_std: f64,
    // CTR: Metrics
    fills_bid_top1: f64,
    cancels_bid_top1: f64,
    fills_ask_top1: f64,
    cancels_ask_top1: f64,
    fills_bid_top20: f64,
    cancels_bid_top20: f64,
    fills_ask_top20: f64,
    cancels_ask_top20: f64,
    // OTR: Metrics
    inflows_bid_top1: f64,
    inflows_ask_top1: f64,
    inflows_bid_top20: f64,
    inflows_ask_top20: f64,

    ctr_history_bid_top1: VecDeque<PlotPoint>,
    ctr_history_ask_top1: VecDeque<PlotPoint>,
    ctr_history_both_top1: VecDeque<PlotPoint>,

    ctr_history_bid_top20: VecDeque<PlotPoint>,
    ctr_history_ask_top20: VecDeque<PlotPoint>,
    ctr_history_both_top20: VecDeque<PlotPoint>,

    otr_history_bid_top1: VecDeque<PlotPoint>,
    otr_history_ask_top1: VecDeque<PlotPoint>,
    otr_history_both_top1: VecDeque<PlotPoint>,

    otr_history_bid_top20: VecDeque<PlotPoint>,
    otr_history_ask_top20: VecDeque<PlotPoint>,
    otr_history_both_top20: VecDeque<PlotPoint>,

    metrics_timer: u64,
    trade_buffer: std::collections::HashMap<Decimal, VecDeque<(Decimal, u64)>>,
    heatmap_contrast: f64,
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
            show_metrics: true,
            execute_usd: 100000.0,
            max_liquidity_usd: 100000.0,
            rolling_mean_qty: 0.0,
            rolling_std_qty: 1.0,
            warmup_samples: 0,
            rolling_trade_mean: 0.0,
            rolling_trade_std: 1.0,
            fills_bid_top1: 0.0,
            cancels_bid_top1: 0.0,
            fills_ask_top1: 0.0,
            cancels_ask_top1: 0.0,
            fills_bid_top20: 0.0,
            cancels_bid_top20: 0.0,
            fills_ask_top20: 0.0,
            cancels_ask_top20: 0.0,
            inflows_bid_top1: 0.0,
            inflows_ask_top1: 0.0,
            inflows_bid_top20: 0.0,
            inflows_ask_top20: 0.0,
            ctr_history_bid_top1: VecDeque::with_capacity(2000),
            ctr_history_ask_top1: VecDeque::with_capacity(2000),
            ctr_history_both_top1: VecDeque::with_capacity(2000),
            ctr_history_bid_top20: VecDeque::with_capacity(2000),
            ctr_history_ask_top20: VecDeque::with_capacity(2000),
            ctr_history_both_top20: VecDeque::with_capacity(2000),
            otr_history_bid_top1: VecDeque::with_capacity(2000),
            otr_history_ask_top1: VecDeque::with_capacity(2000),
            otr_history_both_top1: VecDeque::with_capacity(2000),
            otr_history_bid_top20: VecDeque::with_capacity(2000),
            otr_history_ask_top20: VecDeque::with_capacity(2000),
            otr_history_both_top20: VecDeque::with_capacity(2000),
            metrics_timer: 0,
            trade_buffer: std::collections::HashMap::new(),
            heatmap_contrast: 4.0,
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
            // Use combined streams: depth and aggTrade
            let ws_url_str = format!(
                "wss://fstream.binance.com/stream?streams={symbol}@depth@0ms/{symbol}@trade/{symbol}@bookTicker"
            );
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
                #[derive(Deserialize)]
                struct CombinedStream {
                    stream: String,
                    data: serde_json::Value,
                }

                while let Some(result) = ws_stream.next().await {
                    match result {
                        Ok(message) => match message {
                            WsMessage::Text(text) => {
                                if let Ok(combined) = serde_json::from_str::<CombinedStream>(&text)
                                {
                                    if combined.stream.ends_with("@depth@0ms") {
                                        if let Ok(update) =
                                            serde_json::from_value::<DepthUpdate>(combined.data)
                                        {
                                            let _ = tx_clone.send(AppMessage::Update(update));
                                            ctx_clone.request_repaint();
                                        }
                                    } else if combined.stream.ends_with("@trade") {
                                        if let Ok(trade) =
                                            serde_json::from_value::<Trade>(combined.data)
                                        {
                                            // Filter out Binance placeholder messages (X: "NA")
                                            if trade.order_type != "NA"
                                                && trade.price > Decimal::ZERO
                                            {
                                                let _ = tx_clone.send(AppMessage::Trade(trade));
                                                ctx_clone.request_repaint();
                                            }
                                        }
                                    } else if combined.stream.ends_with("@bookTicker") {
                                        if let Ok(ticker) =
                                            serde_json::from_value::<BookTicker>(combined.data)
                                        {
                                            let _ = tx_clone.send(AppMessage::Ticker(ticker));
                                            ctx_clone.request_repaint();
                                        }
                                    }
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
            if self.warmup_samples < 200 && self.update_counter % 10 == 0 {
                self.update_rolling_stats();
            }

            self.append_to_heatmap();
            self.update_counter = 0;
        }
    }

    fn update_rolling_stats(&mut self) {
        if self.bids.is_empty() || self.asks.is_empty() {
            return;
        }

        // Collect all level quantities currently in the book
        let levels: Vec<f64> = self
            .bids
            .values()
            .chain(self.asks.values())
            .map(|v| v.iter().sum::<Decimal>().to_f64().unwrap_or(0.0))
            .filter(|&q| q > 0.0)
            .collect();

        if !levels.is_empty() {
            let m = levels.iter().sum::<f64>() / levels.len() as f64;
            let v = levels.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / levels.len() as f64;
            let s = v.sqrt().max(1e-9);

            // Update rolling stats (very slow EMA for high stability)
            // Faster alpha during warmup (0.05) vs stable (0.01)
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
    }

    fn append_to_heatmap(&mut self) {
        // ALWAYS update stats first
        self.update_rolling_stats();

        // ONLY draw to heatmap if we have passed the warmup threshold
        // This ensures the first frame is already correctly scaled
        if self.warmup_samples < 200 {
            return;
        }

        let mut snapshot = vec![0.0; self.heatmap_height];
        let mean = self.rolling_mean_qty;
        let std = self.rolling_std_qty.max(1e-9);

        // Fill asks (top half)
        let ask_iter = self.asks.iter().take(self.heatmap_height / 2).rev();
        for (snapshot_cell, (_, qty_deq)) in snapshot
            .iter_mut()
            .take(self.heatmap_height / 2)
            .zip(ask_iter)
        {
            let qty = qty_deq.iter().sum::<Decimal>().to_f64().unwrap_or(0.0);
            if qty > 0.0 {
                // Z-Score Standardization with Offset
                let z = (qty - mean) / std;
                *snapshot_cell = z + 10.0;
            }
        }

        // Fill bids (bottom half)
        let bid_iter = self.bids.iter().rev().take(self.heatmap_height / 2);
        for (snapshot_cell, (_, qty_deq)) in snapshot
            .iter_mut()
            .skip(self.heatmap_height / 2)
            .zip(bid_iter)
        {
            let qty = qty_deq.iter().sum::<Decimal>().to_f64().unwrap_or(0.0);
            if qty > 0.0 {
                let z = (qty - mean) / std;
                *snapshot_cell = -(z + 10.0); // Negative to distinguish bids
            }
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
        // SOFP & CTR: Periodically sample metrics (every 100ms approximation)
        self.metrics_timer += 1;
        if self.metrics_timer % 10 == 0 {
            let now = ctx.input(|i| i.time);

            // Top 1 Split CTR
            let ctr_bid_1 = if self.fills_bid_top1 > 0.0 {
                self.cancels_bid_top1 / self.fills_bid_top1
            } else {
                0.0
            };
            let ctr_ask_1 = if self.fills_ask_top1 > 0.0 {
                self.cancels_ask_top1 / self.fills_ask_top1
            } else {
                0.0
            };
            let total_fills_1 = self.fills_bid_top1 + self.fills_ask_top1;
            let total_cancels_1 = self.cancels_bid_top1 + self.cancels_ask_top1;
            let ctr_both_1 = if total_fills_1 > 0.0 {
                total_cancels_1 / total_fills_1
            } else {
                0.0
            };

            self.ctr_history_bid_top1
                .push_back(PlotPoint::new(now, ctr_bid_1));
            self.ctr_history_ask_top1
                .push_back(PlotPoint::new(now, ctr_ask_1));
            self.ctr_history_both_top1
                .push_back(PlotPoint::new(now, ctr_both_1));

            // Top 20 Split CTR
            let ctr_bid_20 = if self.fills_bid_top20 > 0.0 {
                self.cancels_bid_top20 / self.fills_bid_top20
            } else {
                0.0
            };
            let ctr_ask_20 = if self.fills_ask_top20 > 0.0 {
                self.cancels_ask_top20 / self.fills_ask_top20
            } else {
                0.0
            };
            let total_fills_20 = self.fills_bid_top20 + self.fills_ask_top20;
            let total_cancels_20 = self.cancels_bid_top20 + self.cancels_ask_top20;
            let ctr_both_20 = if total_fills_20 > 0.0 {
                total_cancels_20 / total_fills_20
            } else {
                0.0
            };

            self.ctr_history_bid_top20
                .push_back(PlotPoint::new(now, ctr_bid_20));
            self.ctr_history_ask_top20
                .push_back(PlotPoint::new(now, ctr_ask_20));
            self.ctr_history_both_top20
                .push_back(PlotPoint::new(now, ctr_both_20));

            // Prune history (200s limit)
            let limit = now - 200.0;
            while self
                .ctr_history_bid_top1
                .front()
                .map_or(false, |p| p.x < limit)
            {
                self.ctr_history_bid_top1.pop_front();
            }
            while self
                .ctr_history_ask_top1
                .front()
                .map_or(false, |p| p.x < limit)
            {
                self.ctr_history_ask_top1.pop_front();
            }
            while self
                .ctr_history_both_top1
                .front()
                .map_or(false, |p| p.x < limit)
            {
                self.ctr_history_both_top1.pop_front();
            }

            while self
                .ctr_history_bid_top20
                .front()
                .map_or(false, |p| p.x < limit)
            {
                self.ctr_history_bid_top20.pop_front();
            }
            while self
                .ctr_history_ask_top20
                .front()
                .map_or(false, |p| p.x < limit)
            {
                self.ctr_history_ask_top20.pop_front();
            }
            while self
                .ctr_history_both_top20
                .front()
                .map_or(false, |p| p.x < limit)
            {
                self.ctr_history_both_top20.pop_front();
            }

            // OTR History Pruning
            while self
                .otr_history_bid_top1
                .front()
                .map_or(false, |p| p.x < limit)
            {
                self.otr_history_bid_top1.pop_front();
            }
            while self
                .otr_history_ask_top1
                .front()
                .map_or(false, |p| p.x < limit)
            {
                self.otr_history_ask_top1.pop_front();
            }
            while self
                .otr_history_both_top1
                .front()
                .map_or(false, |p| p.x < limit)
            {
                self.otr_history_both_top1.pop_front();
            }
            while self
                .otr_history_bid_top20
                .front()
                .map_or(false, |p| p.x < limit)
            {
                self.otr_history_bid_top20.pop_front();
            }
            while self
                .otr_history_ask_top20
                .front()
                .map_or(false, |p| p.x < limit)
            {
                self.otr_history_ask_top20.pop_front();
            }
            while self
                .otr_history_both_top20
                .front()
                .map_or(false, |p| p.x < limit)
            {
                self.otr_history_both_top20.pop_front();
            }

            // OTR Calculation (Top-1)
            let otr_bid_1 = if self.fills_bid_top1 > 0.0 {
                self.inflows_bid_top1 / self.fills_bid_top1
            } else {
                0.0
            };
            let otr_ask_1 = if self.fills_ask_top1 > 0.0 {
                self.inflows_ask_top1 / self.fills_ask_top1
            } else {
                0.0
            };
            let otr_both_1 = if (self.fills_bid_top1 + self.fills_ask_top1) > 0.0 {
                (self.inflows_bid_top1 + self.inflows_ask_top1)
                    / (self.fills_bid_top1 + self.fills_ask_top1)
            } else {
                0.0
            };

            self.otr_history_bid_top1
                .push_back(PlotPoint::new(now, otr_bid_1));
            self.otr_history_ask_top1
                .push_back(PlotPoint::new(now, otr_ask_1));
            self.otr_history_both_top1
                .push_back(PlotPoint::new(now, otr_both_1));

            // OTR Calculation (Top-20)
            let otr_bid_20 = if self.fills_bid_top20 > 0.0 {
                self.inflows_bid_top20 / self.fills_bid_top20
            } else {
                0.0
            };
            let otr_ask_20 = if self.fills_ask_top20 > 0.0 {
                self.inflows_ask_top20 / self.fills_ask_top20
            } else {
                0.0
            };
            let otr_both_20 = if (self.fills_bid_top20 + self.fills_ask_top20) > 0.0 {
                (self.inflows_bid_top20 + self.inflows_ask_top20)
                    / (self.fills_bid_top20 + self.fills_ask_top20)
            } else {
                0.0
            };

            self.otr_history_bid_top20
                .push_back(PlotPoint::new(now, otr_bid_20));
            self.otr_history_ask_top20
                .push_back(PlotPoint::new(now, otr_ask_20));
            self.otr_history_both_top20
                .push_back(PlotPoint::new(now, otr_both_20));
        }

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
                AppMessage::Trade(trade) => {
                    // SOFP: Update rolling trade size distribution (very fast alpha for regime detection)
                    let qty = trade.quantity.to_f64().unwrap_or(0.0);
                    if qty > 0.0 {
                        let alpha = 0.1; // Fast adaptation
                        if self.rolling_trade_mean == 0.0 {
                            self.rolling_trade_mean = qty;
                            self.rolling_trade_std = qty * 0.1;
                        } else {
                            let diff = qty - self.rolling_trade_mean;
                            self.rolling_trade_mean += alpha * diff;
                            self.rolling_trade_std =
                                (1.0 - alpha) * self.rolling_trade_std + alpha * diff.abs();
                        }
                    }

                    // CTR: Track fills
                    let price = trade.price;

                    let best_bid = self
                        .bids
                        .keys()
                        .next_back()
                        .cloned()
                        .unwrap_or(Decimal::ZERO);
                    let best_ask = self.asks.keys().next().cloned().unwrap_or(Decimal::ZERO);

                    if trade.is_buyer_maker {
                        // Sell Trade (Hits Bid)
                        if price == best_bid {
                            self.fills_bid_top1 += qty;
                        }
                        if self.bids.keys().rev().take(20).any(|&p| p == price) {
                            self.fills_bid_top20 += qty;
                        }
                    } else {
                        // Buy Trade (Lifts Ask)
                        if price == best_ask {
                            self.fills_ask_top1 += qty;
                        }
                        if self.asks.keys().take(20).any(|&p| p == price) {
                            self.fills_ask_top20 += qty;
                        }
                    }

                    self.trade_buffer
                        .entry(trade.price)
                        .or_default()
                        .push_back((trade.quantity, trade.transaction_time));

                    // Cleanup old trades (older than 10s)
                    let now = trade.transaction_time;
                    for deq in self.trade_buffer.values_mut() {
                        while let Some(front) = deq.front() {
                            if now > front.1 + 10_000 {
                                deq.pop_front();
                            } else {
                                break;
                            }
                        }
                    }
                }
                AppMessage::Ticker(ticker) => {
                    self.apply_ticker_anchor(ticker);
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
            if ui.button("Toggle Microstructure Metrics").clicked() {
                self.show_metrics = !self.show_metrics;
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
                    self.rolling_mean_qty = 0.0;
                    self.rolling_std_qty = 1.0;
                    self.warmup_samples = 0;
                    self.rolling_trade_mean = 0.0;
                    self.rolling_trade_std = 1.0;
                    self.fills_bid_top1 = 0.0;
                    self.cancels_bid_top1 = 0.0;
                    self.fills_ask_top1 = 0.0;
                    self.cancels_ask_top1 = 0.0;
                    self.fills_bid_top20 = 0.0;
                    self.cancels_bid_top20 = 0.0;
                    self.fills_ask_top20 = 0.0;
                    self.cancels_ask_top20 = 0.0;
                    self.inflows_bid_top1 = 0.0;
                    self.inflows_ask_top1 = 0.0;
                    self.inflows_bid_top20 = 0.0;
                    self.inflows_ask_top20 = 0.0;

                    self.ctr_history_bid_top1.clear();
                    self.ctr_history_ask_top1.clear();
                    self.ctr_history_both_top1.clear();
                    self.ctr_history_bid_top20.clear();
                    self.ctr_history_ask_top20.clear();
                    self.ctr_history_both_top20.clear();
                    self.otr_history_bid_top1.clear();
                    self.otr_history_ask_top1.clear();
                    self.otr_history_both_top1.clear();
                    self.otr_history_bid_top20.clear();
                    self.otr_history_ask_top20.clear();
                    self.otr_history_both_top20.clear();
                    self.update_counter = 0;
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
                    ui.label("Heatmap Contrast (Z-Range):");
                    ui.add(egui::Slider::new(&mut self.heatmap_contrast, 1.0..=10.0));
                    if ui.button("Reset Stats").clicked() {
                        self.rolling_mean_qty = 0.0;
                        self.rolling_std_qty = 1.0;
                        self.warmup_samples = 0;
                        self.heatmap_data.clear();
                    }
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

            // if self.show_heatmap {
            egui::Window::new("Microstructure Metrics")
                .open(&mut self.show_metrics)
                .show(ctx, |ui| {
                    ui.label("Cancellation-to-Trade Ratio (CTR)");
                    ui.label("CTR > 1.0: Spoofing/Layering > Fills");

                    ui.allocate_ui(egui::Vec2::new(480.0, 160.0), |ui| {
                        Plot::new("top1_plot")
                            .link_axis("ctr_group", Vec2b::new(true, false))
                            .show_axes([true, true])
                            .y_axis_label("Top-1 Ratio")
                            .legend(egui_plot::Legend::default())
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(
                                        "Bid",
                                        PlotPoints::from_iter(
                                            self.ctr_history_bid_top1.iter().map(|p| [p.x, p.y]),
                                        ),
                                    )
                                    .color(Color32::GREEN),
                                );
                                plot_ui.line(
                                    Line::new(
                                        "Ask",
                                        PlotPoints::from_iter(
                                            self.ctr_history_ask_top1.iter().map(|p| [p.x, p.y]),
                                        ),
                                    )
                                    .color(Color32::RED),
                                );
                                plot_ui.line(
                                    Line::new(
                                        "Both",
                                        PlotPoints::from_iter(
                                            self.ctr_history_both_top1.iter().map(|p| [p.x, p.y]),
                                        ),
                                    )
                                    .color(Color32::WHITE),
                                );
                            });
                    });

                    ui.add_space(4.0);

                    ui.allocate_ui(egui::Vec2::new(480.0, 160.0), |ui| {
                        Plot::new("top20_plot")
                            .link_axis("ctr_group", Vec2b::new(true, false))
                            .show_axes([true, true])
                            .y_axis_label("Top-20 Ratio")
                            .legend(egui_plot::Legend::default())
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(
                                        "Bid",
                                        PlotPoints::from_iter(
                                            self.ctr_history_bid_top20.iter().map(|p| [p.x, p.y]),
                                        ),
                                    )
                                    .color(Color32::GREEN),
                                );
                                plot_ui.line(
                                    Line::new(
                                        "Ask",
                                        PlotPoints::from_iter(
                                            self.ctr_history_ask_top20.iter().map(|p| [p.x, p.y]),
                                        ),
                                    )
                                    .color(Color32::RED),
                                );
                                plot_ui.line(
                                    Line::new(
                                        "Both",
                                        PlotPoints::from_iter(
                                            self.ctr_history_both_top20.iter().map(|p| [p.x, p.y]),
                                        ),
                                    )
                                    .color(Color32::WHITE),
                                );
                            });
                    });

                    ui.add_space(8.0);
                    ui.separator();
                    ui.label("Order-to-Trade Ratio (OTR)");
                    ui.label("OTR > 1.0: Liquidity Adding > Taking (Reloading)");

                    ui.allocate_ui(egui::Vec2::new(480.0, 160.0), |ui| {
                        Plot::new("otr_top1_plot")
                            .link_axis("otr_group", Vec2b::new(true, false))
                            .show_axes([true, true])
                            .y_axis_label("Top-1 OTR")
                            .legend(egui_plot::Legend::default())
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(
                                        "Bid",
                                        PlotPoints::from_iter(
                                            self.otr_history_bid_top1.iter().map(|p| [p.x, p.y]),
                                        ),
                                    )
                                    .color(Color32::GREEN),
                                );
                                plot_ui.line(
                                    Line::new(
                                        "Ask",
                                        PlotPoints::from_iter(
                                            self.otr_history_ask_top1.iter().map(|p| [p.x, p.y]),
                                        ),
                                    )
                                    .color(Color32::RED),
                                );
                                plot_ui.line(
                                    Line::new(
                                        "Both",
                                        PlotPoints::from_iter(
                                            self.otr_history_both_top1.iter().map(|p| [p.x, p.y]),
                                        ),
                                    )
                                    .color(Color32::WHITE),
                                );
                            });
                    });

                    ui.add_space(4.0);

                    ui.allocate_ui(egui::Vec2::new(480.0, 160.0), |ui| {
                        Plot::new("otr_top20_plot")
                            .link_axis("otr_group", Vec2b::new(true, false))
                            .show_axes([true, true])
                            .y_axis_label("Top-20 OTR")
                            .legend(egui_plot::Legend::default())
                            .show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(
                                        "Bid",
                                        PlotPoints::from_iter(
                                            self.otr_history_bid_top20.iter().map(|p| [p.x, p.y]),
                                        ),
                                    )
                                    .color(Color32::GREEN),
                                );
                                plot_ui.line(
                                    Line::new(
                                        "Ask",
                                        PlotPoints::from_iter(
                                            self.otr_history_ask_top20.iter().map(|p| [p.x, p.y]),
                                        ),
                                    )
                                    .color(Color32::RED),
                                );
                                plot_ui.line(
                                    Line::new(
                                        "Both",
                                        PlotPoints::from_iter(
                                            self.otr_history_both_top20.iter().map(|p| [p.x, p.y]),
                                        ),
                                    )
                                    .color(Color32::WHITE),
                                );
                            });
                    });
                });

            egui::Window::new("Depth-Time Heatmap")
                .open(&mut self.show_heatmap)
                .show(ctx, |ui| {
                    if !self.heatmap_data.is_empty() {
                        let width = self.heatmap_data.len();
                        let height = self.heatmap_height;
                        let mut pixels = vec![Color32::BLACK; width * height];

                        for (col, snapshot) in self.heatmap_data.iter().enumerate() {
                            for (row, &value) in snapshot.iter().enumerate() {
                                let abs_val = value.abs();

                                // 1. Recover Z-score
                                let z = if value != 0.0 { abs_val - 10.0 } else { -999.0 };

                                // 2. Strict Black Background for empty/low liquidity
                                if z < -1.0 {
                                    // Threshold: 1 std deviation below mean is BLACK
                                    pixels[row * width + col] = Color32::BLACK;
                                    continue;
                                }

                                // 3. Mapping Z [-1, 3] -> [0, 1]
                                // Range = heatmap_contrast (default 4.0)
                                let t = ((z + 1.0) / self.heatmap_contrast).clamp(0.0, 1.0);

                                // 4. Gamma Correction (Square) to push mid-tones down (more contrast)
                                let t_sq = t * t;
                                let intensity = (t_sq * 255.0) as u8;

                                let color = if value > 0.0 {
                                    // ASKS: Fire (Red -> Orange -> Yellow)
                                    // R: Primary
                                    // G: Secondary (starts at t>0.5)
                                    let g_boost = ((t_sq - 0.5).max(0.0) * 2.0 * 255.0) as u8;
                                    Color32::from_rgb(intensity, g_boost, 0)
                                } else {
                                    // BIDS: Ice (Blue -> Cyan -> White-ish)
                                    // B: Primary
                                    // G: Secondary (starts at t>0.3)
                                    let g_boost = ((t_sq - 0.3).max(0.0) * 1.5 * 255.0)
                                        .clamp(0.0, 255.0)
                                        as u8;
                                    Color32::from_rgb(0, g_boost, intensity)
                                };

                                pixels[row * width + col] = color;
                            }
                        }

                        let color_image = egui::ColorImage {
                            size: [width, height],
                            source_size: Default::default(),
                            pixels,
                        };

                        let texture =
                            ui.ctx()
                                .load_texture("heatmap", color_image, TextureOptions::LINEAR);

                        ui.image(&texture);
                    } else {
                        ui.label("No heatmap data yet.");
                    }
                });
            // }

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

                        // Sweep / Liquidity-Cost Visualization
                        ui.allocate_ui(egui::Vec2::new(480.0, 320.0), |ui| {
                            Plot::new("liquidity_cost_plot")
                                .show_axes([true, true])
                                .x_axis_label("Sweep Size (USD)")
                                .y_axis_label("Price Impact (bps)")
                                .boxed_zoom_pointer_button(egui::PointerButton::Secondary)
                                .auto_bounds(egui::Vec2b::new(true, true)) // Allow egui to manage initial fit
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
        // Process Bids
        for bid in &update.b {
            let price = bid[0];
            let qty = bid[1];
            if qty == Decimal::ZERO {
                self.bids.remove(&price);
            } else {
                self.process_level_change(&price, qty, true, update.transaction_time);
            }
        }
        // Process Asks
        for ask in &update.a {
            let price = ask[0];
            let qty = ask[1];
            if qty == Decimal::ZERO {
                self.asks.remove(&price);
            } else {
                self.process_level_change(&price, qty, false, update.transaction_time);
            }
        }
        self.last_applied_u = update.small_u;
    }

    fn apply_ticker_anchor(&mut self, ticker: BookTicker) {
        // 0. Safety Pruning: Ensure no crossed book remains from stale depth updates
        // Remove Bids > Ticker.BestBid
        while let Some(&high_bid) = self.bids.keys().next_back() {
            if high_bid > ticker.best_bid_price {
                self.bids.pop_last();
            } else {
                break;
            }
        }
        // Remove Asks < Ticker.BestAsk
        while let Some(&low_ask) = self.asks.keys().next() {
            if low_ask < ticker.best_ask_price {
                self.asks.pop_first();
            } else {
                break;
            }
        }

        // 1. Sync Bids (Level 1)
        if let Some(&best_bid_price) = self.bids.keys().next_back() {
            if best_bid_price == ticker.best_bid_price {
                if let Some(queue) = self.bids.get_mut(&best_bid_price) {
                    let est_total: Decimal = queue.iter().sum();
                    let ticker_qty = ticker.best_bid_qty;

                    if ticker_qty < est_total {
                        // PRUNE: oldest orders first (FIFO)
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
                        // FILL: Inferred inflow
                        queue.push_back(ticker_qty - est_total);
                    }
                }
            } else if ticker.best_bid_price > best_bid_price {
                // New high bid price detected by ticker before depth update
                self.bids.insert(
                    ticker.best_bid_price,
                    VecDeque::from(vec![ticker.best_bid_qty]),
                );
            }
        }

        // 2. Sync Asks (Level 1)
        if let Some(&best_ask_price) = self.asks.keys().next() {
            if best_ask_price == ticker.best_ask_price {
                if let Some(queue) = self.asks.get_mut(&best_ask_price) {
                    let est_total: Decimal = queue.iter().sum();
                    let ticker_qty = ticker.best_ask_qty;

                    if ticker_qty < est_total {
                        // PRUNE: oldest orders first (FIFO)
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
                        // FILL: Inferred inflow
                        queue.push_back(ticker_qty - est_total);
                    }
                }
            } else if ticker.best_ask_price < best_ask_price {
                // New low ask price detected by ticker before depth update
                self.asks.insert(
                    ticker.best_ask_price,
                    VecDeque::from(vec![ticker.best_ask_qty]),
                );
            }
        }
    }

    fn process_level_change(
        &mut self,
        price: &Decimal,
        new_total_qty: Decimal,
        is_bid: bool,
        ts: u64,
    ) {
        // 1. Determine status first to avoid borrow conflicts later
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

        // 2. Now perform mutable work
        let side = if is_bid {
            &mut self.bids
        } else {
            &mut self.asks
        };

        if let Some(queue) = side.get_mut(price) {
            let old_total: Decimal = queue.iter().sum();

            if new_total_qty > old_total {
                // INFLOW: Liquidity added
                let diff = new_total_qty - old_total;

                // OTR Tracking
                let diff_f = diff.to_f64().unwrap_or(0.0);
                if is_bid {
                    if is_tob {
                        self.inflows_bid_top1 += diff_f;
                        self.inflows_bid_top20 += diff_f;
                    } else if in_top20 {
                        self.inflows_bid_top20 += diff_f;
                    }
                } else {
                    if is_tob {
                        self.inflows_ask_top1 += diff_f;
                        self.inflows_ask_top20 += diff_f;
                    } else if in_top20 {
                        self.inflows_ask_top20 += diff_f;
                    }
                }

                if !is_tob && diff > dec!(0.1) {
                    // Statistical Order Flow Profiling (SOFP)
                    // Instead of static 60/40, we use the rolling trade distribution
                    let avg_trade = self.rolling_trade_mean.max(0.001);
                    let diff_f = diff.to_f64().unwrap_or(0.0);

                    // FRAGMENTATION HEURISTIC:
                    // Only fragment if the addition is significantly larger than typical trades,
                    // BUT NOT so large that it's likely a single 'Whale' order (bypass if > 20x avg).
                    if diff_f > avg_trade * 2.0 && diff_f < avg_trade * 20.0 {
                        // Fragment based on market regime (avg trade size)
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
            } else if new_total_qty < old_total {
                // OUTFLOW: Liquidity reduced
                let mut remaining_to_remove = old_total - new_total_qty;

                // 3. MTQR (Marker-Triggered Queue Refining)
                if let Some(trade_deq) = self.trade_buffer.get_mut(price) {
                    while let Some(trade) = trade_deq.front() {
                        if ts >= trade.1 && ts - trade.1 < 1000 {
                            let trade_qty = trade.0;

                            // GROUND TRUTH SNAP:
                            // If an atomic trade hit the front of the queue,
                            // we know that the maker's size was AT LEAST the trade size.
                            if !queue.is_empty() && queue[0] < trade_qty {
                                // Our estimation was too small - snap to reality
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
                    // CTR: This remaining amount is a cancellation
                    let rem_f = remaining_to_remove.to_f64().unwrap_or(0.0);

                    if is_bid {
                        if is_tob {
                            self.cancels_bid_top1 += rem_f;
                            self.cancels_bid_top20 += rem_f;
                        } else if in_top20 {
                            self.cancels_bid_top20 += rem_f;
                        }
                    } else {
                        if is_tob {
                            self.cancels_ask_top1 += rem_f;
                            self.cancels_ask_top20 += rem_f;
                        } else if in_top20 {
                            self.cancels_ask_top20 += rem_f;
                        }
                    }

                    // 4. Cancellation/Modification (LIFO with Priority Reset)
                    // Updated to handle multi-order removals (robust LIFO)
                    if let Some(pos) = queue.iter().rposition(|&x| x == remaining_to_remove) {
                        queue.remove(pos);
                    } else {
                        // Iteratively remove from back (LIFO) until remaining_to_remove is exhausted
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
                }
            }
        } else {
            side.insert(*price, VecDeque::from(vec![new_total_qty]));
        }
    }
}
