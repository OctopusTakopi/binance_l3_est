//! Sweep / Liquidity-Cost window: VWAP market-impact curves.

use crate::ui::window::{AppState, AppWindow};
use eframe::egui::{self, Color32, Vec2};
use egui_plot::{Line, Plot, PlotPoints};

pub struct LiquidityView {
    open: bool,
    last_refresh: std::time::Instant,
    cached_buy_points: Vec<[f64; 2]>,
    cached_sell_points: Vec<[f64; 2]>,
    last_update_id: u64,
}

impl Default for LiquidityView {
    fn default() -> Self {
        Self { 
            open: true,
            last_refresh: std::time::Instant::now() - std::time::Duration::from_secs(1),
            cached_buy_points: Vec::new(),
            cached_sell_points: Vec::new(),
            last_update_id: 0,
        }
    }
}

impl AppWindow for LiquidityView {
    fn name(&self) -> &str {
        "Sweep / Liquidity-Cost"
    }
    fn is_open(&self) -> bool {
        self.open
    }
    fn toggle(&mut self) {
        self.open = !self.open;
    }

    fn show(&mut self, ctx: &egui::Context, state: &mut AppState<'_>) {
        let mut open = self.open;
        egui::Window::new(self.name())
            .default_size(Vec2::new(480.0, 320.0))
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                let now = std::time::Instant::now();
                let force_refresh = now.duration_since(self.last_refresh).as_millis() > 200;

                if force_refresh || state.order_book.last_applied_u != self.last_update_id {
                    let (buy_p, sell_p) = state.order_book.get_liquidity_curves();
                    self.cached_buy_points = buy_p.iter().map(|p| [p.x, p.y]).collect();
                    self.cached_sell_points = sell_p.iter().map(|p| [p.x, p.y]).collect();
                    self.last_update_id = state.order_book.last_applied_u;
                    self.last_refresh = now;
                }

                let buy_line = Line::new("Buy", PlotPoints::new(self.cached_buy_points.clone())).color(Color32::BLUE);
                let sell_line = Line::new("Sell", PlotPoints::new(self.cached_sell_points.clone())).color(Color32::RED);

                ui.allocate_ui(egui::Vec2::new(480.0, 320.0), |ui| {
                    Plot::new("liquidity_cost_plot")
                        .show_axes([true, true])
                        .x_axis_label("Sweep Size (USD)")
                        .y_axis_label("Price Impact (bps)")
                        .boxed_zoom_pointer_button(egui::PointerButton::Secondary)
                        .auto_bounds(egui::Vec2b::new(true, true))
                        .show(ui, |plot_ui| {
                            plot_ui.line(buy_line);
                            plot_ui.line(sell_line);
                        });
                });

                if let Some((buy_impact, sell_impact)) = state
                    .order_book
                    .calculate_liquidity_impact(*state.execute_usd)
                {
                    ui.label(format!(
                        "Buy impact = {buy_impact:.2} bps | Sell impact = {sell_impact:.2} bps"
                    ));
                } else {
                    ui.label("Buy impact = N/A | Sell impact = N/A");
                }

                ui.horizontal(|ui| {
                    ui.label("Execute USD X:");
                    ui.add(egui::Slider::new(
                        state.execute_usd,
                        0.0..=state.max_liquidity_usd,
                    ));
                });
            });
        self.open = open;
    }
}
