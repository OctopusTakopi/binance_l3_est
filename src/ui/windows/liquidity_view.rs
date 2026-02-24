//! Sweep / Liquidity-Cost window: VWAP market-impact curves.

use crate::ui::window::{AppState, AppWindow};
use eframe::egui::{self, Color32, Vec2};
use egui_plot::{Line, Plot, PlotPoints};

pub struct LiquidityView {
    open: bool,
}

impl Default for LiquidityView {
    fn default() -> Self {
        Self { open: true }
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
        egui::Window::new(self.name())
            .default_size(Vec2::new(480.0, 320.0))
            .resizable(false)
            .open(&mut self.open)
            .show(ctx, |ui| {
                let (buy_points, sell_points) = state.order_book.get_liquidity_curves();

                let buy_line = Line::new(
                    "Buy",
                    PlotPoints::from_iter(buy_points.iter().map(|p| [p.x, p.y])),
                )
                .color(Color32::BLUE);

                let sell_line = Line::new(
                    "Sell",
                    PlotPoints::from_iter(sell_points.iter().map(|p| [p.x, p.y])),
                )
                .color(Color32::RED);

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

                let (buy_impact, sell_impact) = state
                    .order_book
                    .calculate_liquidity_impact(*state.execute_usd);
                ui.label(format!(
                    "Buy impact = {buy_impact:.2} bps | Sell impact = {sell_impact:.2} bps"
                ));

                ui.horizontal(|ui| {
                    ui.label("Execute USD X:");
                    ui.add(egui::Slider::new(
                        state.execute_usd,
                        0.0..=state.max_liquidity_usd,
                    ));
                });
            });
    }
}
