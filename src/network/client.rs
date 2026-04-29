//! Background WebSocket + REST client for Binance spot and futures streams.

use crate::network::sbe::{SpotSbeEvent, parse_spot_sbe_frame};
use crate::network::{AppMessage, Control, MarketType};
use crate::types::{BookTicker, DepthUpdate, ExchangeInfo, OrderBookSnapshot, Trade};
use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use reqwest::blocking;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::sync::mpsc::Sender as StdSender;
use std::time::Duration;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::time::sleep;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, protocol::Message as WsMessage},
};

const EXCHANGE_INFO_TIMEOUT_SECS: u64 = 10;

#[derive(Clone, Copy, Debug)]
pub struct SymbolSpec {
    pub tick_size: f64,
    pub price_prec: usize,
    pub qty_prec: usize,
}

impl Default for SymbolSpec {
    fn default() -> Self {
        Self {
            tick_size: 0.01,
            price_prec: 2,
            qty_prec: 2,
        }
    }
}

#[derive(Clone, Copy)]
struct MarketEndpoints {
    exchange_info_url: &'static str,
    snapshot_base_url: &'static str,
    websocket_base_url: &'static str,
    depth_stream: &'static str,
    ticker_stream: &'static str,
    use_sbe: bool,
}

impl MarketType {
    fn endpoints(self) -> MarketEndpoints {
        match self {
            Self::Spot => MarketEndpoints {
                exchange_info_url: "https://api.binance.com/api/v3/exchangeInfo",
                snapshot_base_url: "https://api.binance.com/api/v3/depth",
                websocket_base_url: "wss://stream-sbe.binance.com/stream",
                depth_stream: "@depth",
                ticker_stream: "@bestBidAsk",
                use_sbe: true,
            },
            Self::Futures => MarketEndpoints {
                exchange_info_url: "https://fapi.binance.com/fapi/v1/exchangeInfo",
                snapshot_base_url: "https://fapi.binance.com/fapi/v1/depth",
                websocket_base_url: "wss://fstream.binance.com/stream",
                depth_stream: "@depth@0ms",
                ticker_stream: "@bookTicker",
                use_sbe: false,
            },
        }
    }
}

#[derive(Deserialize)]
struct CombinedStream {
    stream: String,
    data: Box<serde_json::value::RawValue>,
}

fn symbol_cache_key(market: MarketType, symbol: &str) -> String {
    format!("{}:{}", market.as_str(), symbol.to_uppercase())
}

fn parse_symbol_spec(info: &crate::types::SymbolInfo) -> Result<SymbolSpec> {
    let mut tick_size = None;
    let mut price_prec = None;
    let mut qty_prec = None;

    for filter in &info.filters {
        if filter.filter_type == "PRICE_FILTER" {
            if let Some(ts) = &filter.tick_size {
                let parsed = ts.parse::<f64>()?;
                if parsed > 0.0 {
                    tick_size = Some(parsed);
                    price_prec = Some(decimal_places(ts));
                }
            }
        } else if filter.filter_type == "LOT_SIZE" {
            if let Some(ss) = &filter.step_size {
                let parsed = ss.parse::<f64>()?;
                if parsed > 0.0 {
                    qty_prec = Some(decimal_places(ss));
                }
            }
        }
    }

    let tick_size = tick_size.ok_or_else(|| anyhow::anyhow!("missing PRICE_FILTER tick size"))?;
    Ok(SymbolSpec {
        tick_size,
        price_prec: price_prec.unwrap_or(2),
        qty_prec: qty_prec.unwrap_or(2),
    })
}

pub fn fetch_exchange_specs(market: MarketType) -> Result<HashMap<String, SymbolSpec>> {
    let endpoints = market.endpoints();
    let http = blocking::Client::builder()
        .timeout(Duration::from_secs(EXCHANGE_INFO_TIMEOUT_SECS))
        .build()?;

    let info = http
        .get(endpoints.exchange_info_url)
        .send()?
        .error_for_status()?
        .json::<ExchangeInfo>()?;

    let mut cache = HashMap::with_capacity(info.symbols.len());
    for sym_info in info.symbols {
        match parse_symbol_spec(&sym_info) {
            Ok(spec) => {
                cache.insert(symbol_cache_key(market, sym_info.symbol.as_str()), spec);
            }
            Err(e) => {
                log::warn!(
                    "fetch_exchange_specs: skipping {} on {}: {e}",
                    sym_info.symbol,
                    market.as_str()
                );
            }
        }
    }

    Ok(cache)
}

pub fn lookup_symbol_spec(
    spec_cache: &HashMap<String, SymbolSpec>,
    symbol: &str,
    market: MarketType,
) -> Option<SymbolSpec> {
    spec_cache.get(&symbol_cache_key(market, symbol)).copied()
}

fn refresh_market_specs(
    spec_cache: &mut HashMap<String, SymbolSpec>,
    market: MarketType,
) -> Result<()> {
    let prefix = format!("{}:", market.as_str());
    spec_cache.retain(|key, _| !key.starts_with(&prefix));
    spec_cache.extend(fetch_exchange_specs(market)?);
    Ok(())
}

pub async fn run_streaming_loop(
    tx: &StdSender<AppMessage>,
    ctx: &egui::Context,
    mut control_rx: UnboundedReceiver<Control>,
    mut symbol: String,
    mut market: MarketType,
    mut spot_api_key: Option<String>,
) {
    let mut spec_cache = HashMap::new();
    for preload_market in [MarketType::Spot, MarketType::Futures] {
        match fetch_exchange_specs(preload_market) {
            Ok(market_specs) => spec_cache.extend(market_specs),
            Err(e) => log::warn!(
                "exchange info preload failed for {}: {e}",
                preload_market.as_str()
            ),
        }
    }

    loop {
        let endpoints = market.endpoints();
        let spec = if let Some(spec) = lookup_symbol_spec(&spec_cache, &symbol, market) {
            spec
        } else {
            match refresh_market_specs(&mut spec_cache, market) {
                Ok(()) => {}
                Err(e) => {
                    let message =
                        format!("Exchange info refresh failed for {}: {e}", market.as_str());
                    log::error!("{message}");
                    let _ = tx.send(AppMessage::NetworkWarning {
                        symbol: symbol.clone(),
                        market,
                        message,
                    });
                    ctx.request_repaint();

                    tokio::select! {
                        ctrl = control_rx.recv() => {
                            match ctrl {
                                Some(ctrl) => {
                                    if !apply_control(ctrl, &mut symbol, &mut market, &mut spot_api_key) {
                                        break;
                                    }
                                }
                                None => break,
                            }
                        }
                        _ = sleep(Duration::from_millis(250)) => {}
                    }
                    continue;
                }
            }

            let Some(spec) = lookup_symbol_spec(&spec_cache, &symbol, market) else {
                let message = format!(
                    "Symbol spec not found in exchange info cache for {} on {}.",
                    symbol.to_uppercase(),
                    market.as_str()
                );
                log::error!("{message}");
                let _ = tx.send(AppMessage::NetworkWarning {
                    symbol: symbol.clone(),
                    market,
                    message,
                });
                ctx.request_repaint();

                tokio::select! {
                    ctrl = control_rx.recv() => {
                        match ctrl {
                            Some(ctrl) => {
                                if !apply_control(ctrl, &mut symbol, &mut market, &mut spot_api_key) {
                                    break;
                                }
                            }
                            None => break,
                        }
                    }
                    _ = sleep(Duration::from_millis(250)) => {}
                }
                continue;
            };

            spec
        };

        let _ = tx.send(AppMessage::SymbolSpec {
            symbol: symbol.clone(),
            market,
            spec,
        });
        ctx.request_repaint();

        let Some(active_api_key) = resolve_spot_api_key(endpoints.use_sbe, &spot_api_key) else {
            let _ = tx.send(AppMessage::SpotApiKeyRequired {
                symbol: symbol.clone(),
                market,
                message: "Spot SBE requires an API key.".to_string(),
            });
            ctx.request_repaint();

            match control_rx.recv().await {
                Some(ctrl) => {
                    if !apply_control(ctrl, &mut symbol, &mut market, &mut spot_api_key) {
                        break;
                    }
                }
                None => break,
            }
            continue;
        };

        let ws_handle = if endpoints.use_sbe {
            match connect_spot_sbe(tx, ctx, &symbol, endpoints, &active_api_key).await {
                Ok(handle) => handle,
                Err(e) => {
                    let message = format!("Spot connection error: {e}");
                    log::error!("{message}");
                    let _ = tx.send(AppMessage::NetworkWarning {
                        symbol: symbol.clone(),
                        market,
                        message,
                    });
                    ctx.request_repaint();

                    tokio::select! {
                        ctrl = control_rx.recv() => {
                            match ctrl {
                                Some(ctrl) => {
                                    if !apply_control(ctrl, &mut symbol, &mut market, &mut spot_api_key) {
                                        break;
                                    }
                                }
                                None => break,
                            }
                        }
                        _ = sleep(Duration::from_secs(2)) => {}
                    }
                    continue;
                }
            }
        } else {
            match connect_futures_json(tx, ctx, &symbol, endpoints).await {
                Ok(handle) => handle,
                Err(e) => {
                    let message = format!("Futures connection error: {e}");
                    log::error!("{message}");
                    let _ = tx.send(AppMessage::NetworkWarning {
                        symbol: symbol.clone(),
                        market,
                        message,
                    });
                    ctx.request_repaint();

                    tokio::select! {
                        ctrl = control_rx.recv() => {
                            match ctrl {
                                Some(ctrl) => {
                                    if !apply_control(ctrl, &mut symbol, &mut market, &mut spot_api_key) {
                                        break;
                                    }
                                }
                                None => break,
                            }
                        }
                        _ = sleep(Duration::from_secs(2)) => {}
                    }
                    continue;
                }
            }
        };

        let snap_url = format!(
            "{}?symbol={}&limit=1000",
            endpoints.snapshot_base_url,
            symbol.to_uppercase()
        );
        match reqwest::Client::new().get(&snap_url).send().await {
            Ok(resp) => match resp.json::<OrderBookSnapshot>().await {
                Ok(snap) => {
                    log::info!("Snapshot fetched successfully.");
                    let _ = tx.send(AppMessage::NetworkWarning {
                        symbol: symbol.clone(),
                        market,
                        message: String::new(),
                    });
                    let _ = tx.send(AppMessage::Snapshot {
                        symbol: symbol.clone(),
                        market,
                        snapshot: snap,
                    });
                }
                Err(e) => log::error!("Snapshot JSON error: {e}"),
            },
            Err(e) => log::error!("Snapshot request error: {e}"),
        }

        let mut ws_handle = ws_handle;
        tokio::select! {
            ctrl = control_rx.recv() => {
                match ctrl {
                    Some(ctrl) => {
                        ws_handle.abort();
                        if !apply_control(ctrl, &mut symbol, &mut market, &mut spot_api_key) {
                            break;
                        }
                    }
                    None => break,
                }
            }
            join_result = &mut ws_handle => {
                let message = match join_result {
                    Ok(()) => format!("{} stream disconnected.", market.as_str()),
                    Err(e) if e.is_cancelled() => String::new(),
                    Err(e) => format!("{} stream task failed: {e}", market.as_str()),
                };

                if !message.is_empty() {
                    log::warn!("{message}");
                    let _ = tx.send(AppMessage::NetworkWarning {
                        symbol: symbol.clone(),
                        market,
                        message,
                    });
                    ctx.request_repaint();
                }

                tokio::select! {
                    ctrl = control_rx.recv() => {
                        match ctrl {
                            Some(ctrl) => {
                                if !apply_control(ctrl, &mut symbol, &mut market, &mut spot_api_key) {
                                    break;
                                }
                            }
                            None => break,
                        }
                    }
                    _ = sleep(Duration::from_secs(2)) => {}
                }
                continue;
            }
        }
    }
}

fn resolve_spot_api_key(use_sbe: bool, api_key: &Option<String>) -> Option<String> {
    if !use_sbe {
        return Some(String::new());
    }

    api_key
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            env::var("BINANCE_SPOT_SBE_API_KEY")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
}

fn apply_control(
    ctrl: Control,
    symbol: &mut String,
    market: &mut MarketType,
    spot_api_key: &mut Option<String>,
) -> bool {
    match ctrl {
        Control::Refetch => {
            log::info!("Refetch triggered, restarting connection.");
            true
        }
        Control::ChangeSymbol {
            symbol: new_symbol,
            market: new_market,
            spot_api_key: new_spot_api_key,
        } => {
            *symbol = new_symbol;
            *market = new_market;
            if let Some(key) = new_spot_api_key {
                *spot_api_key = Some(key.trim().to_string());
            }
            log::info!(
                "Changing to {} on {}, restarting connection.",
                symbol,
                market.as_str()
            );
            true
        }
    }
}

async fn connect_futures_json(
    tx: &StdSender<AppMessage>,
    ctx: &egui::Context,
    symbol: &str,
    endpoints: MarketEndpoints,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let ws_url = format!(
        "{}?streams={}{}{}/{}@trade/{}{}",
        endpoints.websocket_base_url,
        symbol,
        endpoints.depth_stream,
        "",
        symbol,
        symbol,
        endpoints.ticker_stream
    );
    let (mut ws_stream, response) = connect_async(&ws_url).await?;
    log::info!("WebSocket connected: {response:?}");

    let tx_clone = tx.clone();
    let ctx_clone = ctx.clone();
    let depth_suffix = endpoints.depth_stream.to_string();
    let ticker_suffix = endpoints.ticker_stream.to_string();
    let market = MarketType::Futures;
    let mut last_repaint = std::time::Instant::now();
    let repaint_interval = std::time::Duration::from_millis(13); // ~75 FPS

    Ok(tokio::spawn(async move {
        while let Some(result) = ws_stream.next().await {
            match result {
                Ok(WsMessage::Text(text)) => {
                    let Ok(combined) = serde_json::from_str::<CombinedStream>(&text) else {
                        continue;
                    };

                    let mut needs_repaint = false;
                    
                    if combined.stream.ends_with(&depth_suffix) {
                        if let Ok(update) = serde_json::from_str::<DepthUpdate>(combined.data.get()) {
                            let _ = tx_clone.send(AppMessage::Update { market, update });
                            needs_repaint = true;
                        }
                    } else if combined.stream.ends_with("@trade") {
                        if let Ok(trade) = serde_json::from_str::<Trade>(combined.data.get()) {
                            if trade.order_type != "NA" && trade.price > 0.0 {
                                let _ = tx_clone.send(AppMessage::Trade { market, trade });
                                needs_repaint = true;
                            }
                        }
                    } else if combined.stream.ends_with(&ticker_suffix) {
                        if let Ok(ticker) = serde_json::from_str::<BookTicker>(combined.data.get()) {
                            let _ = tx_clone.send(AppMessage::Ticker { market, ticker });
                            needs_repaint = true;
                        }
                    }

                    if needs_repaint {
                        let elapsed = last_repaint.elapsed();
                        if elapsed >= repaint_interval {
                            ctx_clone.request_repaint();
                            last_repaint = std::time::Instant::now();
                        } else {
                            ctx_clone.request_repaint_after(repaint_interval - elapsed);
                        }
                    }
                }
                Ok(WsMessage::Ping(payload)) => {
                    if let Err(e) = ws_stream.send(WsMessage::Pong(payload)).await {
                        log::warn!("Pong send error: {e}");
                        break;
                    }
                }
                Ok(WsMessage::Pong(_)) => {}
                Ok(WsMessage::Close(_)) => break,
                Err(e) => {
                    log::warn!("WebSocket error: {e}");
                    break;
                }
                _ => {}
            }
        }
    }))
}

async fn connect_spot_sbe(
    tx: &StdSender<AppMessage>,
    ctx: &egui::Context,
    symbol: &str,
    endpoints: MarketEndpoints,
    api_key: &str,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let ws_url = format!(
        "{}?streams={}@trade/{}{}/{}{}",
        endpoints.websocket_base_url,
        symbol,
        symbol,
        endpoints.ticker_stream,
        symbol,
        endpoints.depth_stream
    );

    let mut request = ws_url.into_client_request()?;
    request
        .headers_mut()
        .insert("X-MBX-APIKEY", api_key.parse()?);

    let (mut ws_stream, response) = connect_async(request).await?;
    log::info!("Spot SBE WebSocket connected: {response:?}");

    let tx_clone = tx.clone();
    let ctx_clone = ctx.clone();
    let market = MarketType::Spot;
    let mut last_repaint = std::time::Instant::now();
    let repaint_interval = std::time::Duration::from_millis(13); // ~75 FPS

    Ok(tokio::spawn(async move {
        while let Some(result) = ws_stream.next().await {
            match result {
                Ok(WsMessage::Binary(payload)) => match parse_spot_sbe_frame(&payload) {
                    Some(SpotSbeEvent::Trades(trades)) => {
                        for trade in trades {
                            let _ = tx_clone.send(AppMessage::Trade { market, trade });
                        }
                        let elapsed = last_repaint.elapsed();
                        if elapsed >= repaint_interval {
                            ctx_clone.request_repaint();
                            last_repaint = std::time::Instant::now();
                        } else {
                            ctx_clone.request_repaint_after(repaint_interval - elapsed);
                        }
                    }
                    Some(SpotSbeEvent::BestBidAsk(ticker)) => {
                        let _ = tx_clone.send(AppMessage::Ticker { market, ticker });
                        let elapsed = last_repaint.elapsed();
                        if elapsed >= repaint_interval {
                            ctx_clone.request_repaint();
                            last_repaint = std::time::Instant::now();
                        } else {
                            ctx_clone.request_repaint_after(repaint_interval - elapsed);
                        }
                    }
                    Some(SpotSbeEvent::DepthDiff(update)) => {
                        let _ = tx_clone.send(AppMessage::Update { market, update });
                        let elapsed = last_repaint.elapsed();
                        if elapsed >= repaint_interval {
                            ctx_clone.request_repaint();
                            last_repaint = std::time::Instant::now();
                        } else {
                            ctx_clone.request_repaint_after(repaint_interval - elapsed);
                        }
                    }
                    None => {}
                },
                Ok(WsMessage::Ping(payload)) => {
                    if let Err(e) = ws_stream.send(WsMessage::Pong(payload)).await {
                        log::warn!("Pong send error: {e}");
                        break;
                    }
                }
                Ok(WsMessage::Text(_)) => {}
                Ok(WsMessage::Pong(_)) => {}
                Ok(WsMessage::Close(_)) => break,
                Err(e) => {
                    log::warn!("WebSocket error: {e}");
                    break;
                }
                _ => {}
            }
        }
    }))
}

fn decimal_places(value: &str) -> usize {
    let trimmed = value.trim_end_matches('0');
    trimmed
        .split_once('.')
        .map(|(_, frac)| frac.len())
        .unwrap_or(0)
}

#[allow(unused_imports)]
use crate::error;
