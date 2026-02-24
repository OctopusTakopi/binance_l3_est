//! Shared data-model types for Binance WebSocket streams and REST responses.
//!
//! # Deserialization strategy
//!
//! Binance sends prices and quantities as **JSON strings** (e.g. `"p":"0.00100"`).
//! Two approaches are used depending on downstream requirements:
//!
//! * **`Decimal`** — retained where the engine stores the value as a `BTreeMap`
//!   key or performs exact decimal arithmetic (price levels, level quantities).
//! * **`f64` via `from_str_to_f64`** — used for fields that are only read as
//!   scalars and never used as map keys (e.g. best-bid/ask quantities in
//!   `BookTicker`). Avoids the allocation overhead of `rust_decimal`.
//! * **`SymbolStr`** — replaces `String` for all symbol name fields; symbols
//!   like "BTCUSDT" (≤ 16 bytes) are stored entirely on the stack.

use crate::utils::SymbolStr;
use rust_decimal::Decimal;
use serde::Deserialize;

// ── REST: Exchange Info ────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ExchangeInfo {
    pub symbols: Vec<SymbolInfo>,
}

#[derive(Deserialize)]
pub struct SymbolInfo {
    /// Stack-allocated symbol name (e.g. "BTCUSDT").
    pub symbol: SymbolStr,
    pub filters: Vec<Filter>,
}

#[derive(Deserialize)]
pub struct Filter {
    #[serde(rename = "filterType")]
    pub filter_type: String,
    #[serde(rename = "tickSize")]
    pub tick_size: Option<String>,
    #[serde(rename = "stepSize")]
    pub step_size: Option<String>,
}

// ── REST: Order Book Snapshot ──────────────────────────────────────────────────

/// Full depth snapshot from `GET /fapi/v1/depth`.
///
/// Bids/asks are `Vec<Vec<Decimal>>` because each level is immediately inserted
/// as a `BTreeMap<Decimal, VecDeque<Decimal>>` key; `Decimal` preserves exact
/// ordering without float-key hazards.
#[derive(Deserialize)]
pub struct OrderBookSnapshot {
    #[serde(rename = "lastUpdateId")]
    pub last_update_id: u64,
    pub bids: Vec<Vec<Decimal>>,
    pub asks: Vec<Vec<Decimal>>,
}

// ── WebSocket: Aggregate Trade ─────────────────────────────────────────────────

/// A single trade event from the `@trade` stream.
///
/// `price` and `quantity` are kept as `Decimal` because:
/// - `price` is used as a `HashMap<Decimal, …>` key in the trade buffer (MTQR).
/// - `quantity` participates in exact `Decimal` arithmetic inside the engine.
#[allow(dead_code)]
#[derive(Deserialize, Clone)]
pub struct Trade {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: SymbolStr,
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

// ── WebSocket: Depth Update ────────────────────────────────────────────────────

/// An incremental depth update event from the `@depth@0ms` stream.
///
/// `b`/`a` level arrays stay as `Vec<Vec<Decimal>>` so prices can be used
/// directly as `BTreeMap` keys without conversion.
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
    pub symbol: SymbolStr,
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

// ── WebSocket: Book Ticker ─────────────────────────────────────────────────────

/// Best-bid/ask snapshot from the `@bookTicker` stream.
///
/// All price and quantity fields are `Decimal`: prices are compared to
/// `BTreeMap<Decimal, …>` keys, and quantities participate in exact
/// queue-arithmetic inside the engine.
#[allow(dead_code)]
#[derive(Deserialize, Clone)]
pub struct BookTicker {
    #[serde(rename = "u")]
    pub update_id: u64,
    #[serde(rename = "s")]
    pub symbol: SymbolStr,
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
