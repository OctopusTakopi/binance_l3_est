//! Network layer: inter-thread message types for the WebSocket client.

pub mod client;
pub mod sbe;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarketType {
    Spot,
    Futures,
}

impl MarketType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Spot => "spot",
            Self::Futures => "futures",
        }
    }
}

use crate::types::{BookTicker, DepthUpdate, OrderBookSnapshot, Trade};

/// Messages sent from the background network task to the UI thread.
pub enum AppMessage {
    SymbolSpec {
        symbol: String,
        market: MarketType,
        spec: client::SymbolSpec,
    },
    Snapshot {
        symbol: String,
        market: MarketType,
        snapshot: OrderBookSnapshot,
    },
    Update {
        market: MarketType,
        update: DepthUpdate,
    },
    Trade {
        market: MarketType,
        trade: Trade,
    },
    Ticker {
        market: MarketType,
        ticker: BookTicker,
    },
    SpotApiKeyRequired {
        symbol: String,
        market: MarketType,
        message: String,
    },
    NetworkWarning {
        symbol: String,
        market: MarketType,
        message: String,
    },
}

/// Control commands sent from the UI thread to the background network task.
pub enum Control {
    Refetch,
    ChangeSymbol {
        symbol: String,
        market: MarketType,
        spot_api_key: Option<String>,
    },
}
