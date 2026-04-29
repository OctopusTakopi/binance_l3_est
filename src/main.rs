//! Entry point for the Binance L3 Order Book Estimator.

mod engine;
mod error;
mod network;
mod types;
mod ui;
mod utils;

fn main() -> eframe::Result {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let symbol = args.next().unwrap_or_else(|| "dogeusdt".to_string());
    let market = match args.next().as_deref() {
        Some("spot") => network::MarketType::Spot,
        Some("future") | Some("futures") | Some("perp") | None => network::MarketType::Futures,
        Some(other) => {
            log::warn!("Unknown market '{other}', defaulting to futures.");
            network::MarketType::Futures
        }
    };

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0]),
        renderer: eframe::Renderer::Wgpu,
        hardware_acceleration: eframe::HardwareAcceleration::Required,
        ..Default::default()
    };

    eframe::run_native(
        "Order Book Visualizer",
        native_options,
        Box::new(move |cc| Ok(Box::new(ui::app::App::new(cc, symbol, market)))),
    )
}
