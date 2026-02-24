//! TWAP Detector window: FFT power spectral density plots for buy and sell
//! sides.

use crate::ui::window::{AppState, AppWindow};
use eframe::egui::{self, Align2, Color32, Vec2};
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoint, PlotPoints, Text};

pub struct TwapView {
    open: bool,
}

impl Default for TwapView {
    fn default() -> Self {
        Self { open: false }
    }
}

impl AppWindow for TwapView {
    fn name(&self) -> &str {
        "TWAP Detector"
    }
    fn is_open(&self) -> bool {
        self.open
    }
    fn toggle(&mut self) {
        self.open = !self.open;
    }

    fn show(&mut self, ctx: &egui::Context, state: &mut AppState<'_>) {
        let mut twap_open = self.open;
        egui::Window::new(self.name())
            .open(&mut twap_open)
            .default_size(Vec2::new(600.0, 600.0))
            .show(ctx, |ui| {
                let twap = &mut *state.twap;

                // Config row
                ui.horizontal(|ui| {
                    ui.label("Bin (ms):");
                    if ui
                        .add(egui::Slider::new(&mut twap.bin_ms, 100..=5000))
                        .changed()
                    {
                        twap.reset_bins();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Window (bins):");
                    ui.add(egui::Slider::new(&mut twap.window_bins, 64..=4096));
                    ui.label("Sigma:");
                    ui.add(egui::Slider::new(&mut twap.threshold_sigma, 1.0..=6.0));
                });

                ui.separator();

                let n_buy = twap.bins_buy.len();
                let n_sell = twap.bins_sell.len();
                let min_bins = 64_usize;
                let n_bins = n_buy.min(n_sell);

                if n_bins < min_bins {
                    ui.vertical_centered(|ui| {
                        ui.add_space(20.0);
                        ui.heading("Collecting data...");
                        ui.label(format!(
                            "Buy: {} / {}   Sell: {} / {}   ({:.0}%)",
                            n_buy,
                            min_bins,
                            n_sell,
                            min_bins,
                            (n_bins as f64 / min_bins as f64 * 100.0).min(100.0)
                        ));
                        ui.add_space(20.0);
                    });
                    return;
                }

                if !twap.psd_buy.is_empty() {
                    render_psd(
                        ui,
                        "twap_psd_buy",
                        "Taker Buy TWAP (Lifts Ask)",
                        Color32::from_rgb(60, 140, 255),
                        &twap.psd_buy,
                        &twap.psd_vol_buy,
                        &twap.peaks_buy,
                        n_bins,
                    );
                }

                ui.separator();

                if !twap.psd_sell.is_empty() {
                    render_psd(
                        ui,
                        "twap_psd_sell",
                        "Taker Sell TWAP (Hits Bid)",
                        Color32::from_rgb(220, 80, 80),
                        &twap.psd_sell,
                        &twap.psd_vol_sell,
                        &twap.peaks_sell,
                        n_bins,
                    );
                }

                ui.separator();
                let bin_sec = twap.bin_ms as f64 / 1000.0;
                ui.label(format!(
                    "Bins: {}  |  Bin: {}ms  |  Resolution: {:.4} Hz",
                    n_bins,
                    twap.bin_ms,
                    1.0 / (n_bins as f64 * bin_sec)
                ));
            });
        self.open = twap_open;
    }
}

/// Render a single PSD bar chart with peak markers and detected-period labels.
#[allow(clippy::too_many_arguments)]
fn render_psd(
    ui: &mut egui::Ui,
    plot_id: &str,
    label: &str,
    color_normal: Color32,
    psd: &[[f64; 2]],
    psd_vol: &[[f64; 2]],
    peaks: &[(f64, f64)],
    n_bins: usize,
) {
    let max_power = psd.iter().map(|p| p[1]).fold(0.0_f64, f64::max);
    let bar_width = if psd.len() > 1 {
        (psd[1][0] - psd[0][0]) * 0.8
    } else {
        0.001
    };

    let bars: Vec<Bar> = psd
        .iter()
        .map(|p| {
            let is_peak = peaks.iter().any(|(fp, _)| (p[0] - fp).abs() < bar_width);
            let color = if is_peak {
                Color32::from_rgb(255, 200, 0)
            } else {
                color_normal
            };
            Bar::new(p[0], p[1]).fill(color).width(bar_width)
        })
        .collect();

    ui.label(label);
    ui.allocate_ui(egui::Vec2::new(570.0, 200.0), |ui| {
        Plot::new(plot_id)
            .show_axes([true, true])
            .x_axis_label("Frequency (Hz)")
            .y_axis_label("Power")
            .allow_drag(false)
            .allow_scroll(false)
            .allow_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.bar_chart(BarChart::new(plot_id, bars));
                for (freq, _) in peaks {
                    let period = 1.0 / freq;
                    let lbl = format!("T={period:.1}s");
                    plot_ui.line(
                        Line::new(
                            lbl.clone(),
                            PlotPoints::from_iter([[*freq, 0.0], [*freq, max_power * 1.05]]),
                        )
                        .color(Color32::from_rgb(255, 80, 80))
                        .width(2.0),
                    );
                    plot_ui.text(
                        Text::new(lbl.clone(), PlotPoint::new(*freq, max_power * 1.05), lbl)
                            .anchor(Align2::CENTER_BOTTOM),
                    );
                }
            });
    });

    if peaks.is_empty() {
        ui.label("  None above threshold.");
    } else {
        for (freq, count_power) in peaks {
            let period = 1.0 / freq;
            let vol_power = psd_vol
                .iter()
                .min_by(|a, b| {
                    (a[0] - freq)
                        .abs()
                        .partial_cmp(&(b[0] - freq).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|p| p[1])
                .unwrap_or(0.0);
            let n_slice = n_bins as f64 * (2.0 * count_power).sqrt();
            let usd_slice = n_bins as f64 * (2.0 * vol_power).sqrt();
            let slices_per_hour = 3600.0 / period;
            let usd_k = usd_slice * slices_per_hour / 1000.0;
            ui.label(format!(
                "  T={period:.2}s (f={freq:.4}Hz) | ~{n_slice:.0} trades/slice | ~{usd_slice:.0} USD/slice | ~{usd_k:.0}K/hr"
            ));
        }
    }
}
