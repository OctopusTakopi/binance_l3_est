//! Background WebSocket + REST client for Binance futures streams.

use crate::network::{AppMessage, Control};
use crate::types::{BookTicker, DepthUpdate, ExchangeInfo, OrderBookSnapshot, Trade};
use futures_util::{SinkExt, StreamExt};
use reqwest::blocking;
use serde::Deserialize;
use std::sync::mpsc::Sender as StdSender;
use tokio::sync::mpsc::Receiver;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message as WsMessage};

// ── Precision Helper ───────────────────────────────────────────────────────────

/// Fetch the tick size (price precision) and step size (qty precision) for
/// a futures symbol from the Binance exchange-info endpoint.
///
/// Returns `(price_prec, qty_prec)` as decimal places.  Defaults to `(2, 2)`
/// on any failure.
pub fn fetch_precision(symbol: &str) -> (usize, usize) {
    let mut price_prec: usize = 2;
    let mut qty_prec: usize = 2;

    let url = "https://fapi.binance.com/fapi/v1/exchangeInfo";
    let info: ExchangeInfo = match blocking::get(url).and_then(|r| r.json::<ExchangeInfo>()) {
        Ok(info) => info,
        Err(e) => {
            log::warn!("fetch_precision: {e}");
            return (price_prec, qty_prec);
        }
    };

    let Some(sym_info) = info.symbols.into_iter().find(|s| s.symbol == symbol) else {
        log::warn!("fetch_precision: symbol {symbol} not found");
        return (price_prec, qty_prec);
    };

    for filter in sym_info.filters {
        if filter.filter_type == "PRICE_FILTER" {
            if let Some(ts) = filter.tick_size {
                if let Ok(tick_size) = ts.parse::<f64>() {
                    if tick_size > 0.0 {
                        price_prec = (-tick_size.log10()).ceil() as usize;
                    }
                }
            }
        } else if filter.filter_type == "LOT_SIZE" {
            if let Some(ss) = filter.step_size {
                if let Ok(step_size) = ss.parse::<f64>() {
                    if step_size > 0.0 {
                        qty_prec = (-step_size.log10()).ceil() as usize;
                    }
                }
            }
        }
    }

    (price_prec, qty_prec)
}

// ── Streaming Loop ─────────────────────────────────────────────────────────────

/// Combined WebSocket stream message envelope (Binance combined streams format).
#[derive(Deserialize)]
struct CombinedStream {
    stream: String,
    data: serde_json::Value,
}

/// Long-running async loop: connects to Binance combined WebSocket streams,
/// fetches the REST snapshot, and forwards [`AppMessage`]s to the UI thread.
///
/// Exits cleanly when the `control_rx` channel is closed (UI shut down).
pub async fn run_streaming_loop(
    tx: &StdSender<AppMessage>,
    ctx: &egui::Context,
    mut control_rx: Receiver<Control>,
    mut symbol: String,
) {
    loop {
        let ws_url = format!(
            "wss://fstream.binance.com/stream?streams={symbol}@depth@0ms/{symbol}@trade/{symbol}@bookTicker"
        );

        let (mut ws_stream, response) = match connect_async(&ws_url).await {
            Ok(pair) => pair,
            Err(e) => {
                log::error!("WebSocket connection error: {e}");
                return;
            }
        };
        log::info!("WebSocket connected: {response:?}");

        // Spawn a task that drains the WebSocket and forwards messages.
        let tx_clone = tx.clone();
        let ctx_clone = ctx.clone();
        let ws_handle = tokio::spawn(async move {
            while let Some(result) = ws_stream.next().await {
                match result {
                    Ok(WsMessage::Text(text)) => {
                        let Ok(combined) = serde_json::from_str::<CombinedStream>(&text) else {
                            continue;
                        };

                        if combined.stream.ends_with("@depth@0ms") {
                            if let Ok(update) = serde_json::from_value::<DepthUpdate>(combined.data)
                            {
                                let _ = tx_clone.send(AppMessage::Update(update));
                                ctx_clone.request_repaint();
                            }
                        } else if combined.stream.ends_with("@trade") {
                            if let Ok(trade) = serde_json::from_value::<Trade>(combined.data) {
                                // Filter out Binance placeholder messages (X: "NA")
                                if trade.order_type != "NA"
                                    && trade.price > rust_decimal::Decimal::ZERO
                                {
                                    let _ = tx_clone.send(AppMessage::Trade(trade));
                                    ctx_clone.request_repaint();
                                }
                            }
                        } else if combined.stream.ends_with("@bookTicker") {
                            if let Ok(ticker) = serde_json::from_value::<BookTicker>(combined.data)
                            {
                                let _ = tx_clone.send(AppMessage::Ticker(ticker));
                                ctx_clone.request_repaint();
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
                    Ok(WsMessage::Close(_)) => {
                        log::info!("Connection closed by server.");
                        break;
                    }
                    Err(e) => {
                        log::warn!("WebSocket error: {e}");
                        break;
                    }
                    _ => {}
                }
            }
        });

        // Fetch the REST snapshot concurrently with the live stream.
        let snap_url = format!(
            "https://fapi.binance.com/fapi/v1/depth?symbol={}&limit=1000",
            symbol.to_uppercase()
        );
        match reqwest::Client::new().get(&snap_url).send().await {
            Ok(resp) => match resp.json::<OrderBookSnapshot>().await {
                Ok(snap) => {
                    log::info!("Snapshot fetched successfully.");
                    let _ = tx.send(AppMessage::Snapshot(snap));
                }
                Err(e) => log::error!("Snapshot JSON error: {e}"),
            },
            Err(e) => log::error!("Snapshot request error: {e}"),
        }

        // Wait for a control command (Refetch or ChangeSymbol) before looping.
        match control_rx.recv().await {
            Some(ctrl) => {
                ws_handle.abort();
                match ctrl {
                    Control::Refetch => log::info!("Refetch triggered, restarting connection."),
                    Control::ChangeSymbol(new_symbol) => {
                        symbol = new_symbol;
                        log::info!("Changing symbol to {symbol}, restarting connection.");
                    }
                }
            }
            None => break, // UI shut down — exit cleanly
        }
    }
}

// Suppress unused-import warning — AppError is used by callers of this module.
#[allow(unused_imports)]
use crate::error;
