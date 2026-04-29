//! Order book view window: bar chart (normal + k-means) and level table.

use crate::engine::kmeans::{MiniBatchKMeans, build_clustered_orders};
use crate::ui::colors::{ASK_COLORS, BID_COLORS};
use crate::ui::window::{AppState, AppWindow};
use eframe::egui::{self, Align2, Color32};
use egui_plot::{Bar, BarChart, Plot, PlotPoint, Text};

pub struct OrderBookView {
    open: bool,
    pub kmeans_mode: bool,
    pub batch_size: usize,
    pub max_iter: usize,
    pub brighter_step: usize,
    km_ask: MiniBatchKMeans,
    km_bid: MiniBatchKMeans,

    // Caching for performance
    last_update_id: u64,
    cached_bars: Vec<Bar>,
    cached_max_bid_order: f64,
    cached_max_ask_order: f64,
    cached_bid_levels: Vec<(i64, f64)>,
    cached_ask_levels: Vec<(i64, f64)>,
    cached_max_qty: f64,
}

impl Default for OrderBookView {
    fn default() -> Self {
        Self {
            open: true,
            kmeans_mode: false,
            batch_size: 256,
            max_iter: 256,
            brighter_step: 5,
            km_ask: MiniBatchKMeans::new(10, 1024, 1024),
            km_bid: MiniBatchKMeans::new(10, 1024, 1024),
            last_update_id: 0,
            cached_bars: Vec::new(),
            cached_max_bid_order: 0.0,
            cached_max_ask_order: 0.0,
            cached_bid_levels: Vec::new(),
            cached_ask_levels: Vec::new(),
            cached_max_qty: 0.0,
        }
    }
}

impl AppWindow for OrderBookView {
    fn name(&self) -> &str {
        "Order Book"
    }
    fn is_open(&self) -> bool {
        self.open
    }
    fn toggle(&mut self) {
        self.open = !self.open;
    }

    fn show(&mut self, _ctx: &egui::Context, _state: &mut AppState<'_>) {}
}

impl OrderBookView {
    pub fn render_inline(&mut self, ui: &mut egui::Ui, state: &mut AppState<'_>) {
        let ob = &state.order_book;

        ui.horizontal(|ui| {
            ui.label("Order Age Contrast:");
            ui.add(egui::Slider::new(&mut self.brighter_step, 0..=20));
            ui.separator();
            ui.checkbox(&mut self.kmeans_mode, "K-Means Clusters");
        });
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                egui::Grid::new("order_book_grid")
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("Asks");
                        ui.label("Price");
                        ui.label("Quantity");
                        ui.end_row();

                        // Take top 20 asks from cached list
                        for (price_ticks, level_total) in self.cached_ask_levels.iter().take(20).rev() {
                            ui.label("");
                            let price = ob.ticks_to_price(*price_ticks);
                            ui.label(format!("{:.1$}", price, state.price_prec));
                            ui.label(format!("{:.1$}", *level_total, state.qty_prec));
                            ui.end_row();
                        }

                        ui.label("Bids");
                        ui.label("Price");
                        ui.label("Quantity");
                        ui.end_row();

                        // Take top 20 bids from cached list
                        for (price_ticks, level_total) in self.cached_bid_levels.iter().take(20) {
                            ui.label("");
                            let price = ob.ticks_to_price(*price_ticks);
                            ui.label(format!("{:.1$}", price, state.price_prec));
                            ui.label(format!("{:.1$}", *level_total, state.qty_prec));
                            ui.end_row();
                        }
                    });
            });

            ui.vertical(|ui| {
                let step = 1.0_f64;
                if self.last_update_id != ob.last_applied_u {
                    self.cached_bid_levels = ob.iter_bids().take(200).collect();
                    self.cached_ask_levels = ob.iter_asks().take(200).collect();

                    self.cached_max_qty = self.cached_bid_levels
                        .iter()
                        .chain(self.cached_ask_levels.iter())
                        .map(|(_, q)| *q)
                        .fold(0.0_f64, f64::max);

                    let mut max_bid_order = 0.0_f64;
                    for (_, order_iter) in ob.iter_bids_with_orders().take(200) {
                        for q in order_iter { max_bid_order = max_bid_order.max(q); }
                    }
                    let mut max_ask_order = 0.0_f64;
                    for (_, order_iter) in ob.iter_asks_with_orders().take(200) {
                        for q in order_iter { max_ask_order = max_ask_order.max(q); }
                    }

                    self.cached_bars = if !self.kmeans_mode {
                        build_normal_bars(
                            ob,
                            max_ask_order,
                            max_bid_order,
                            step,
                            self.brighter_step,
                        )
                    } else {
                        build_kmeans_bars(
                            ob,
                            max_ask_order,
                            max_bid_order,
                            step,
                            &mut self.km_ask,
                            &mut self.km_bid,
                        )
                    };
                    self.cached_max_bid_order = max_bid_order;
                    self.cached_max_ask_order = max_ask_order;
                    self.last_update_id = ob.last_applied_u;
                }

                Plot::new("orderbook_chart")
                    .allow_drag(false)
                    .allow_scroll(false)
                    .allow_zoom(false)
                    .show_axes([true, true])
                    .show(ui, |plot_ui| {
                        plot_ui.bar_chart(BarChart::new("ob", self.cached_bars.clone()));

                        for (i, (price_ticks, _)) in self.cached_bid_levels.iter().enumerate() {
                            if i % 20 == 0 {
                                let x = -(i as f64 + 0.5) * step - 0.5;
                                plot_ui.text(
                                    Text::new(
                                        "bid",
                                        PlotPoint::new(x, -self.cached_max_qty * 0.05),
                                        format!(
                                            "{:.1$}",
                                            ob.ticks_to_price(*price_ticks),
                                            state.price_prec
                                        ),
                                    )
                                    .anchor(Align2::CENTER_BOTTOM),
                                );
                            }
                        }
                        for (i, (price_ticks, _)) in self.cached_ask_levels.iter().enumerate() {
                            if i % 20 == 0 && i != 0 {
                                let x = (i as f64 + 0.5) * step + 0.5;
                                plot_ui.text(
                                    Text::new(
                                        "ask",
                                        PlotPoint::new(x, -self.cached_max_qty * 0.05),
                                        format!(
                                            "{:.1$}",
                                            ob.ticks_to_price(*price_ticks),
                                            state.price_prec
                                        ),
                                    )
                                    .anchor(Align2::CENTER_BOTTOM),
                                );
                            }
                        }
                    });
            });
        });
    }

    pub fn order_color(index: usize, base: Color32, step: f32) -> Color32 {
        let f = 1.0 + step * index as f32;
        Color32::from_rgb(
            (base.r() as f32 * f).min(255.0) as u8,
            (base.g() as f32 * f).min(255.0) as u8,
            (base.b() as f32 * f).min(255.0) as u8,
        )
    }
}

fn build_normal_bars(
    ob: &crate::engine::order_book::OrderBook,
    max_ask: f64,
    max_bid: f64,
    step: f64,
    brighter_step: usize,
) -> Vec<Bar> {
    let mut bars = Vec::new();
    for (i, (_price, order_iter)) in ob.iter_asks_with_orders().take(200).enumerate() {
        let x = (i as f64 + 0.5) * step + 0.5;
        let mut offset = 0.0;
        for (j, qty) in order_iter.enumerate() {
            if qty <= 0.0 { continue; }
            let color = if qty == max_ask && qty > 0.0 {
                Color32::GOLD
            } else {
                OrderBookView::order_color(j, Color32::DARK_RED, brighter_step as f32 / 100.0)
            };
            bars.push(Bar::new(x, qty).fill(color).base_offset(offset).width(step * 0.9));
            offset += qty;
        }
    }
    for (i, (_price, order_iter)) in ob.iter_bids_with_orders().take(200).enumerate() {
        let x = -(i as f64 + 0.5) * step - 0.5;
        let mut offset = 0.0;
        for (j, qty) in order_iter.enumerate() {
            if qty <= 0.0 { continue; }
            let color = if qty == max_bid && qty > 0.0 {
                Color32::GOLD
            } else {
                OrderBookView::order_color(j, Color32::DARK_GREEN, brighter_step as f32 / 100.0)
            };
            bars.push(Bar::new(x, qty).fill(color).base_offset(offset).width(step * 0.9));
            offset += qty;
        }
    }
    bars
}

fn build_kmeans_bars(
    ob: &crate::engine::order_book::OrderBook,
    max_ask: f64,
    max_bid: f64,
    step: f64,
    km_ask: &mut MiniBatchKMeans,
    km_bid: &mut MiniBatchKMeans,
) -> Vec<Bar> {
    let labels_a = km_ask.fit_iter(ob.iter_asks_with_orders().take(200));
    let clustered_a = build_clustered_orders(ob.iter_asks_with_orders().take(200), labels_a);

    let labels_b = km_bid.fit_iter(ob.iter_bids_with_orders().take(200));
    let clustered_b = build_clustered_orders(ob.iter_bids_with_orders().take(200), labels_b);

    let mut bars = Vec::new();
    for (i, (_price, level_orders)) in clustered_a.into_iter().enumerate() {
        let x = (i as f64 + 0.5) * step + 0.5;
        let mut offset = 0.0;
        for (qty, cluster) in level_orders {
            let color = if qty == max_ask && qty > 0.0 {
                Color32::GOLD
            } else {
                ASK_COLORS.get(cluster % ASK_COLORS.len()).cloned().unwrap_or(Color32::GRAY)
            };
            bars.push(Bar::new(x, qty).fill(color).base_offset(offset).width(step * 0.9));
            offset += qty;
        }
    }
    for (i, (_price, level_orders)) in clustered_b.into_iter().enumerate() {
        let x = -(i as f64 + 0.5) * step - 0.5;
        let mut offset = 0.0;
        for (qty, cluster) in level_orders {
            let color = if qty == max_bid && qty > 0.0 {
                Color32::GOLD
            } else {
                BID_COLORS.get(cluster % BID_COLORS.len()).cloned().unwrap_or(Color32::GRAY)
            };
            bars.push(Bar::new(x, qty).fill(color).base_offset(offset).width(step * 0.9));
            offset += qty;
        }
    }
    bars
}
