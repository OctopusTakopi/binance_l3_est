//! Application-level error types.
#![allow(dead_code)]

use thiserror::Error;

/// Errors that can occur within the application.
#[derive(Debug, Error)]
pub enum AppError {
    #[error("WebSocket error: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),

    #[error("HTTP request error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON deserialisation error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Channel send error: receiver dropped")]
    ChannelClosed,
}

/// Convenience alias for `Result<T, AppError>`.
pub type Result<T> = std::result::Result<T, AppError>;
