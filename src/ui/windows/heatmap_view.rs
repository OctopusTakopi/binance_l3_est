//! Heatmap window: depth-time visualisation.

use crate::ui::window::{AppState, AppWindow};
use eframe::egui::{self, Color32, TextureOptions};

pub struct HeatmapView {
    open: bool,
    pub contrast: f64,
}

impl Default for HeatmapView {
    fn default() -> Self {
        Self {
            open: true,
            contrast: 4.0,
        }
    }
}

impl AppWindow for HeatmapView {
    fn name(&self) -> &str {
        "Depth-Time Heatmap"
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
                ui.horizontal(|ui| {
                    ui.label("Contrast (Z-Range):");
                    ui.add(egui::Slider::new(&mut self.contrast, 1.0..=10.0));
                    if ui.button("Reset Stats").clicked() {
                        state.heatmap.reset();
                    }
                });
                ui.add_space(8.0);

                let hm = &state.heatmap;
                if hm.warmup_samples < 200 {
                    ui.vertical_centered(|ui| {
                        ui.add_space(50.0);
                        ui.heading("Warming up heatmap...");
                        ui.label(format!(
                            "Collecting data: {:.0}%",
                            (hm.warmup_samples as f64 / 200.0) * 100.0
                        ));
                        ui.add_space(50.0);
                    });
                    return;
                }

                if hm.data.is_empty() {
                    ui.label("No heatmap data yet.");
                    return;
                }

                let width = hm.data.len();
                let height = hm.height;
                let contrast = self.contrast;
                let mut pixels = vec![Color32::BLACK; width * height];

                for (col, snapshot) in hm.data.iter().enumerate() {
                    for (row, &value) in snapshot.iter().enumerate() {
                        let abs_val = value.abs();
                        let z = if value != 0.0 { abs_val - 10.0 } else { -999.0 };
                        if z < -1.0 {
                            continue; // BLACK background
                        }
                        let t = ((z + 1.0) / contrast).clamp(0.0, 1.0);
                        let t_sq = t * t;
                        let intensity = (t_sq * 255.0) as u8;
                        let color = if value > 0.0 {
                            let g_boost = ((t_sq - 0.5).max(0.0) * 2.0 * 255.0) as u8;
                            Color32::from_rgb(intensity, g_boost, 0)
                        } else {
                            let g_boost =
                                ((t_sq - 0.3).max(0.0) * 1.5 * 255.0).clamp(0.0, 255.0) as u8;
                            Color32::from_rgb(0, g_boost, intensity)
                        };
                        pixels[row * width + col] = color;
                    }
                }

                let color_image = egui::ColorImage {
                    size: [width, height],
                    source_size: Default::default(),
                    pixels,
                };
                let texture = ui
                    .ctx()
                    .load_texture("heatmap", color_image, TextureOptions::LINEAR);
                ui.image(&texture);
            });
    }
}
