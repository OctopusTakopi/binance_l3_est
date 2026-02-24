//! Entry point for the Binance L3 Order Book Estimator.

mod engine;
mod error;
mod network;
mod types;
mod ui;
mod utils;

fn main() -> eframe::Result {
    env_logger::init();
    let symbol = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "dogeusdt".to_string());

    eframe::run_native(
        "Order Book Visualizer",
        eframe::NativeOptions::default(),
        Box::new(move |cc| Ok(Box::new(ui::app::App::new(cc, symbol)))),
    )
}
