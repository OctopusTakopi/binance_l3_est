//! Microstructure Metrics window: CTR and OTR time-series plots.

use crate::ui::window::{AppState, AppWindow};
use eframe::egui::{self, Color32, Vec2b};
use egui_plot::{Line, Plot, PlotPoints};

pub struct MetricsView {
    open: bool,
}

impl Default for MetricsView {
    fn default() -> Self {
        Self { open: true }
    }
}

impl AppWindow for MetricsView {
    fn name(&self) -> &str {
        "Microstructure Metrics"
    }
    fn is_open(&self) -> bool {
        self.open
    }
    fn toggle(&mut self) {
        self.open = !self.open;
    }

    fn show(&mut self, ctx: &egui::Context, state: &mut AppState<'_>) {
        egui::Window::new(self.name())
            .open(&mut self.open)
            .show(ctx, |ui| {
                ui.label("Cancellation-to-Trade Ratio (CTR)");
                ui.label("CTR > 1.0: Spoofing/Layering > Fills");

                if state.warmup_samples < 30 {
                    ui.vertical_centered(|ui| {
                        ui.add_space(50.0);
                        ui.heading("Warming up metrics...");
                        ui.label(format!(
                            "Collecting data: {:.0}%",
                            (state.warmup_samples as f64 / 30.0) * 100.0
                        ));
                        ui.add_space(50.0);
                    });
                    return;
                }

                let m = &state.metrics;

                // CTR Top-1
                render_ratio_plot(
                    ui,
                    "top1_plot",
                    "Top-1 Ratio",
                    egui::Id::new("ctr_top1"),
                    &m.ctr_history_bid_top1,
                    &m.ctr_history_ask_top1,
                    &m.ctr_history_both_top1,
                );
                ui.add_space(4.0);
                // CTR Top-20
                render_ratio_plot(
                    ui,
                    "top20_plot",
                    "Top-20 Ratio",
                    egui::Id::new("ctr_top20"),
                    &m.ctr_history_bid_top20,
                    &m.ctr_history_ask_top20,
                    &m.ctr_history_both_top20,
                );

                ui.add_space(8.0);
                ui.separator();
                ui.label("Order-to-Trade Ratio (OTR)");
                ui.label("OTR > 1.0: Liquidity Adding > Taking (Reloading)");

                // OTR Top-1
                render_ratio_plot(
                    ui,
                    "otr_top1_plot",
                    "Top-1 OTR",
                    egui::Id::new("otr_top1"),
                    &m.otr_history_bid_top1,
                    &m.otr_history_ask_top1,
                    &m.otr_history_both_top1,
                );
                ui.add_space(4.0);
                // OTR Top-20
                render_ratio_plot(
                    ui,
                    "otr_top20_plot",
                    "Top-20 OTR",
                    egui::Id::new("otr_top20"),
                    &m.otr_history_bid_top20,
                    &m.otr_history_ask_top20,
                    &m.otr_history_both_top20,
                );
            });
    }
}

fn render_ratio_plot(
    ui: &mut egui::Ui,
    plot_id: &str,
    y_label: &str,
    link_group: egui::Id,
    bid_hist: &std::collections::VecDeque<egui_plot::PlotPoint>,
    ask_hist: &std::collections::VecDeque<egui_plot::PlotPoint>,
    both_hist: &std::collections::VecDeque<egui_plot::PlotPoint>,
) {
    ui.allocate_ui(egui::Vec2::new(480.0, 160.0), |ui| {
        Plot::new(plot_id)
            .link_axis(link_group, Vec2b::new(true, false))
            .show_axes([true, true])
            .y_axis_label(y_label)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        "Bid",
                        PlotPoints::from_iter(bid_hist.iter().map(|p| [p.x, p.y])),
                    )
                    .color(Color32::GREEN),
                );
                plot_ui.line(
                    Line::new(
                        "Ask",
                        PlotPoints::from_iter(ask_hist.iter().map(|p| [p.x, p.y])),
                    )
                    .color(Color32::RED),
                );
                plot_ui.line(
                    Line::new(
                        "Both",
                        PlotPoints::from_iter(both_hist.iter().map(|p| [p.x, p.y])),
                    )
                    .color(Color32::WHITE),
                );
            });
    });
}
