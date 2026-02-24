//! Network layer: inter-thread message types for the WebSocket client.

pub mod client;

use crate::types::{BookTicker, DepthUpdate, OrderBookSnapshot, Trade};

/// Messages sent from the background network task to the UI thread.
pub enum AppMessage {
    Snapshot(OrderBookSnapshot),
    Update(DepthUpdate),
    Trade(Trade),
    Ticker(BookTicker),
}

/// Control commands sent from the UI thread to the background network task.
pub enum Control {
    Refetch,
    ChangeSymbol(String),
}
