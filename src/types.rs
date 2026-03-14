//! Shared data-model types for Binance WebSocket streams and REST responses.

use crate::utils::{SymbolStr, from_str_to_f64, from_str_vec2};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ExchangeInfo {
    pub symbols: Vec<SymbolInfo>,
}

#[derive(Debug, Deserialize)]
pub struct SymbolInfo {
    pub symbol: SymbolStr,
    pub filters: Vec<Filter>,
}

#[derive(Debug, Deserialize)]
pub struct Filter {
    #[serde(rename = "filterType")]
    pub filter_type: String,
    #[serde(rename = "tickSize")]
    pub tick_size: Option<String>,
    #[serde(rename = "stepSize")]
    pub step_size: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OrderBookSnapshot {
    #[serde(rename = "lastUpdateId")]
    pub last_update_id: u64,
    #[serde(deserialize_with = "from_str_vec2")]
    pub bids: Vec<[f64; 2]>,
    #[serde(deserialize_with = "from_str_vec2")]
    pub asks: Vec<[f64; 2]>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct Trade {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: SymbolStr,
    #[serde(rename = "t")]
    pub trade_id: u64,
    #[serde(rename = "p", deserialize_with = "from_str_to_f64")]
    pub price: f64,
    #[serde(rename = "q", deserialize_with = "from_str_to_f64")]
    pub quantity: f64,
    #[serde(rename = "X")]
    pub order_type: String,
    #[serde(rename = "T")]
    pub transaction_time: u64,
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct DepthUpdate {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "T")]
    #[serde(default)]
    pub transaction_time: u64,
    #[serde(rename = "s")]
    pub symbol: SymbolStr,
    #[serde(rename = "U")]
    pub capital_u: u64,
    #[serde(rename = "u")]
    pub small_u: u64,
    #[serde(rename = "pu", default)]
    pub pu: Option<u64>,
    #[serde(rename = "b", deserialize_with = "from_str_vec2")]
    pub b: Vec<[f64; 2]>,
    #[serde(rename = "a", deserialize_with = "from_str_vec2")]
    pub a: Vec<[f64; 2]>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct BookTicker {
    #[serde(rename = "u")]
    pub update_id: u64,
    #[serde(rename = "s")]
    pub symbol: SymbolStr,
    #[serde(rename = "b", deserialize_with = "from_str_to_f64")]
    pub best_bid_price: f64,
    #[serde(rename = "B", deserialize_with = "from_str_to_f64")]
    pub best_bid_qty: f64,
    #[serde(rename = "a", deserialize_with = "from_str_to_f64")]
    pub best_ask_price: f64,
    #[serde(rename = "A", deserialize_with = "from_str_to_f64")]
    pub best_ask_qty: f64,
    #[serde(rename = "T")]
    pub transaction_time: u64,
    #[serde(rename = "E")]
    pub event_time: u64,
}
