//! Order book view window: bar chart (normal + k-means) and level table.

use crate::engine::kmeans::{MiniBatchKMeans, build_clustered_orders};
use crate::ui::colors::{ASK_COLORS, BID_COLORS};
use crate::ui::window::{AppState, AppWindow};
use eframe::egui::{self, Align2, Color32};
use egui_plot::{Bar, BarChart, Plot, PlotPoint, Text};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::collections::BTreeMap;
use std::collections::VecDeque;

pub struct OrderBookView {
    open: bool,
    pub kmeans_mode: bool,
    pub batch_size: usize,
    pub max_iter: usize,
    pub brighter_step: usize,
    /// Persistent KMeans for ask side — centroids warm-start across frames.
    km_ask: MiniBatchKMeans,
    /// Persistent KMeans for bid side — centroids warm-start across frames.
    km_bid: MiniBatchKMeans,
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

    /// The order book view is rendered inline in the central panel (not a
    /// floating window), so this method is a no-op.  The actual rendering is
    /// driven directly from `App::update` using `render_inline`.
    fn show(&mut self, _ctx: &egui::Context, _state: &mut AppState<'_>) {}
}

impl OrderBookView {
    /// Render the order book table + bar chart into `ui` (called from the
    /// central panel, not a floating egui::Window).
    pub fn render_inline(&mut self, ui: &mut egui::Ui, state: &mut AppState<'_>) {
        let ob = &state.order_book;

        // ── Controls ───────────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label("Order Age Contrast:");
            ui.add(egui::Slider::new(&mut self.brighter_step, 0..=20));
            ui.separator();
            ui.checkbox(&mut self.kmeans_mode, "K-Means Clusters");
        });
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            // ── Level Table ────────────────────────────────────────────────
            ui.vertical(|ui| {
                egui::Grid::new("order_book_grid")
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("Asks");
                        ui.label("Price");
                        ui.label("Quantity");
                        ui.end_row();

                        for (price, qty) in ob.asks.iter().take(20).rev() {
                            ui.label("");
                            ui.label(format!(
                                "{:.1$}",
                                price.to_f64().unwrap_or(0.0),
                                state.price_prec
                            ));
                            ui.label(format!(
                                "{:.1$}",
                                qty.iter().sum::<Decimal>().to_f64().unwrap_or(0.0),
                                state.qty_prec
                            ));
                            ui.end_row();
                        }

                        ui.label("Bids");
                        ui.label("Price");
                        ui.label("Quantity");
                        ui.end_row();

                        for (price, qty) in ob.bids.iter().rev().take(20) {
                            ui.label("");
                            ui.label(format!(
                                "{:.1$}",
                                price.to_f64().unwrap_or(0.0),
                                state.price_prec
                            ));
                            ui.label(format!(
                                "{:.1$}",
                                qty.iter().sum::<Decimal>().to_f64().unwrap_or(0.0),
                                state.qty_prec
                            ));
                            ui.end_row();
                        }
                    });
            });

            // ── Bar Chart ──────────────────────────────────────────────────
            ui.vertical(|ui| {
                let bid_levels: Vec<(&Decimal, Decimal)> = ob
                    .bids
                    .iter()
                    .rev()
                    .take(200)
                    .map(|(k, dq)| (k, dq.iter().cloned().sum::<Decimal>()))
                    .collect();
                let ask_levels: Vec<(&Decimal, Decimal)> = ob
                    .asks
                    .iter()
                    .take(200)
                    .map(|(k, dq)| (k, dq.iter().cloned().sum::<Decimal>()))
                    .collect();

                let max_qty = bid_levels
                    .iter()
                    .chain(ask_levels.iter())
                    .map(|(_, q)| q.to_f64().unwrap_or(0.0))
                    .fold(0.0_f64, f64::max);

                let max_bid_order = ob
                    .bids
                    .values()
                    .rev()
                    .take(200)
                    .flat_map(|dq| dq.iter())
                    .cloned()
                    .max()
                    .unwrap_or(Decimal::ZERO);
                let max_ask_order = ob
                    .asks
                    .values()
                    .take(200)
                    .flat_map(|dq| dq.iter())
                    .cloned()
                    .max()
                    .unwrap_or(Decimal::ZERO);

                let second_max_bid_order = nth_largest(ob.bids.values().rev().take(200), 1);
                let second_max_ask_order = nth_largest(ob.asks.values().take(200), 1);

                let step = 1.0_f64;
                let bars = if !self.kmeans_mode {
                    build_normal_bars(
                        &ob.asks,
                        &ob.bids,
                        max_ask_order,
                        second_max_ask_order,
                        max_bid_order,
                        second_max_bid_order,
                        step,
                        self.brighter_step,
                    )
                } else {
                    // Pass persistent KMeans instances so centroids warm-start.
                    build_kmeans_bars(
                        &ob.asks,
                        &ob.bids,
                        max_ask_order,
                        max_bid_order,
                        step,
                        &mut self.km_ask,
                        &mut self.km_bid,
                    )
                };

                Plot::new("orderbook_chart")
                    .allow_drag(false)
                    .allow_scroll(false)
                    .allow_zoom(false)
                    .show_axes([true, true])
                    .show(ui, |plot_ui| {
                        plot_ui.bar_chart(BarChart::new("ob", bars));

                        for (i, (price, _)) in bid_levels.iter().enumerate() {
                            if i % 20 == 0 {
                                let x = -(i as f64 + 0.5) * step - 0.5;
                                plot_ui.text(
                                    Text::new(
                                        "bid",
                                        PlotPoint::new(x, -max_qty * 0.05),
                                        format!(
                                            "{:.1$}",
                                            price.to_f64().unwrap_or(0.0),
                                            state.price_prec
                                        ),
                                    )
                                    .anchor(Align2::CENTER_BOTTOM),
                                );
                            }
                        }
                        for (i, (price, _)) in ask_levels.iter().enumerate() {
                            if i % 20 == 0 && i != 0 {
                                let x = (i as f64 + 0.5) * step + 0.5;
                                plot_ui.text(
                                    Text::new(
                                        "ask",
                                        PlotPoint::new(x, -max_qty * 0.05),
                                        format!(
                                            "{:.1$}",
                                            price.to_f64().unwrap_or(0.0),
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

    /// Compute order-age colour (brighter for newer orders).
    pub fn order_color(index: usize, base: Color32, step: f32) -> Color32 {
        let f = 1.0 + step * index as f32;
        Color32::from_rgb(
            (base.r() as f32 * f).min(255.0) as u8,
            (base.g() as f32 * f).min(255.0) as u8,
            (base.b() as f32 * f).min(255.0) as u8,
        )
    }
}

// ── Bar-building helpers ───────────────────────────────────────────────────────

fn build_normal_bars(
    asks: &BTreeMap<Decimal, VecDeque<Decimal>>,
    bids: &BTreeMap<Decimal, VecDeque<Decimal>>,
    max_ask: Decimal,
    second_max_ask: Decimal,
    max_bid: Decimal,
    second_max_bid: Decimal,
    step: f64,
    brighter_step: usize,
) -> Vec<Bar> {
    let mut bars = Vec::new();
    for (i, (_, qty_deq)) in asks.iter().take(200).enumerate() {
        let x = (i as f64 + 0.5) * step + 0.5;
        let mut offset = 0.0;
        for (j, &qty) in qty_deq.iter().enumerate() {
            if qty <= dec!(0.0) {
                continue;
            }
            let color = if qty == max_ask {
                Color32::GOLD
            } else if qty == second_max_ask {
                Color32::from_rgb(184, 134, 11)
            } else {
                OrderBookView::order_color(j, Color32::DARK_RED, brighter_step as f32 / 100.0)
            };
            bars.push(
                Bar::new(x, qty.to_f64().unwrap_or(0.0))
                    .fill(color)
                    .base_offset(offset)
                    .width(step * 0.9),
            );
            offset += qty.to_f64().unwrap_or(0.0);
        }
    }
    for (i, (_, qty_deq)) in bids.iter().rev().take(200).enumerate() {
        let x = -(i as f64 + 0.5) * step - 0.5;
        let mut offset = 0.0;
        for (j, &qty) in qty_deq.iter().enumerate() {
            if qty <= dec!(0.0) {
                continue;
            }
            let color = if qty == max_bid {
                Color32::GOLD
            } else if qty == second_max_bid {
                Color32::from_rgb(184, 134, 11)
            } else {
                OrderBookView::order_color(j, Color32::DARK_GREEN, brighter_step as f32 / 100.0)
            };
            bars.push(
                Bar::new(x, qty.to_f64().unwrap_or(0.0))
                    .fill(color)
                    .base_offset(offset)
                    .width(step * 0.9),
            );
            offset += qty.to_f64().unwrap_or(0.0);
        }
    }
    bars
}

fn build_kmeans_bars(
    asks: &BTreeMap<Decimal, VecDeque<Decimal>>,
    bids: &BTreeMap<Decimal, VecDeque<Decimal>>,
    max_ask: Decimal,
    max_bid: Decimal,
    step: f64,
    km_ask: &mut MiniBatchKMeans,
    km_bid: &mut MiniBatchKMeans,
) -> Vec<Bar> {
    // Fit using persistent instances — centroids warm-start across frames.
    // We can pass the iterator directly, avoiding the BTreeMap clone/allocation.
    let labels_a: Vec<usize> = km_ask.fit_iter(asks.iter().take(200)).to_vec();
    let clustered_a = build_clustered_orders(asks.iter().take(200), &labels_a);

    let labels_b: Vec<usize> = km_bid.fit_iter(bids.iter().rev().take(200)).to_vec();
    let clustered_b = build_clustered_orders(bids.iter().rev().take(200), &labels_b);

    let mut bars = Vec::new();
    for (i, (_, qty_deq)) in clustered_a.iter().enumerate() {
        let x = (i as f64 + 0.5) * step + 0.5;
        let mut offset = 0.0;
        for &(qty, cluster) in qty_deq.iter() {
            if qty <= dec!(0.0) {
                continue;
            }
            let color = if qty == max_ask {
                Color32::GOLD
            } else {
                ASK_COLORS
                    .get(cluster % ASK_COLORS.len())
                    .cloned()
                    .unwrap_or(Color32::GRAY)
            };
            bars.push(
                Bar::new(x, qty.to_f64().unwrap_or(0.0))
                    .fill(color)
                    .base_offset(offset)
                    .width(step * 0.9),
            );
            offset += qty.to_f64().unwrap_or(0.0);
        }
    }
    for (i, (_, qty_deq)) in clustered_b.iter().enumerate() {
        let x = -(i as f64 + 0.5) * step - 0.5;
        let mut offset = 0.0;
        for &(qty, cluster) in qty_deq.iter() {
            if qty <= dec!(0.0) {
                continue;
            }
            let color = if qty == max_bid {
                Color32::GOLD
            } else {
                BID_COLORS
                    .get(cluster % BID_COLORS.len())
                    .cloned()
                    .unwrap_or(Color32::GRAY)
            };
            bars.push(
                Bar::new(x, qty.to_f64().unwrap_or(0.0))
                    .fill(color)
                    .base_offset(offset)
                    .width(step * 0.9),
            );
            offset += qty.to_f64().unwrap_or(0.0);
        }
    }
    bars
}

/// Return the Nth-largest order quantity across all levels.
fn nth_largest<'a>(iter: impl Iterator<Item = &'a VecDeque<Decimal>>, n: usize) -> Decimal {
    let mut orders: Vec<Decimal> = iter.flat_map(|dq| dq.iter()).cloned().collect();
    orders.sort_by(|a, b| b.cmp(a));
    orders.get(n).cloned().unwrap_or(Decimal::ZERO)
}
