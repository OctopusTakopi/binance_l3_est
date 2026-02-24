//! The `AppWindow` trait and the shared `AppState` view passed to each window.
//!
//! To add a new analytics window:
//! 1. Create a new file in `ui/windows/`.
//! 2. Implement `AppWindow` for your struct.
//! 3. Push `Box::new(MyWindow::default())` into `App::windows` in `App::new()`.

use crate::engine::{
    heatmap::HeatmapState, metrics::MetricsState, order_book::OrderBook, twap::TwapDetector,
};

/// Read-only view of engine state shared with every window's `show` call.
///
/// UI windows receive this by `&mut` so they can mutate UI-only fields (e.g.
/// sliders) on the engine structs where necessary.
pub struct AppState<'a> {
    pub order_book: &'a mut OrderBook,
    pub metrics: &'a mut MetricsState,
    pub heatmap: &'a mut HeatmapState,
    pub twap: &'a mut TwapDetector,
    /// Rolling mean of trade size — available to windows for SOFP visualisation.
    #[allow(dead_code)]
    pub rolling_trade_mean: f64,
    pub warmup_samples: usize,
    pub price_prec: usize,
    pub qty_prec: usize,
    pub execute_usd: &'a mut f64,
    pub max_liquidity_usd: f64,
}

/// Trait implemented by every analytics window/panel.
///
/// Each window owns its own open/closed flag and any window-specific UI state
/// (sliders, toggles, etc.).  The orchestrator (`App`) simply iterates over all
/// registered windows and calls `show` on each frame.
pub trait AppWindow {
    /// Display name shown on the toggle button and as the egui window title.
    fn name(&self) -> &str;

    /// Whether this window is currently visible.
    #[allow(dead_code)]
    fn is_open(&self) -> bool;

    /// Toggle the window's open/closed state.
    fn toggle(&mut self);

    /// Draw the window contents.  Called every frame by `App::update`.
    fn show(&mut self, ctx: &egui::Context, state: &mut AppState<'_>);
}
