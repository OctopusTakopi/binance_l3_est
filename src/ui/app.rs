//! Application orchestrator: owns all engine state and drives the window system.

use std::collections::VecDeque;
use std::sync::mpsc::{self as std_mpsc, Receiver as StdReceiver};
use std::thread;

use crate::engine::{
    heatmap::HeatmapState,
    metrics::{MetricsState, Side},
    order_book::OrderBook,
    twap::TwapDetector,
};
use crate::network::{AppMessage, Control, client};
use crate::types::Trade;
use crate::ui::window::{AppState, AppWindow};
use crate::ui::windows::{
    heatmap_view::HeatmapView, liquidity_view::LiquidityView, metrics_view::MetricsView,
    order_book_view::OrderBookView, twap_view::TwapView,
};
use eframe::egui;
use rust_decimal::prelude::*;
use tokio::sync::mpsc::{self as tokio_mpsc, Sender as TokioSender};

// ── App struct ─────────────────────────────────────────────────────────────────

/// The top-level application, implementing [`eframe::App`].
///
/// All domain logic lives in the engine structs; `App` only:
/// 1. Drains the incoming message channel and dispatches to engines.
/// 2. Calls `metrics.sample/prune` periodically.
/// 3. Renders the central panel (symbol bar + order book) and delegates
///    every analytics window to the registered `windows` vec.
pub struct App {
    symbol: String,
    edited_symbol: String,
    rx: StdReceiver<AppMessage>,
    control_tx: TokioSender<Control>,

    // ── Engine state ───────────────────────────────────────────────────────
    order_book: OrderBook,
    metrics: MetricsState,
    heatmap: HeatmapState,
    twap: TwapDetector,

    // ── SOFP rolling trade distribution ───────────────────────────────────
    rolling_trade_mean: f64,
    rolling_trade_std: f64,

    // ── Counters / timers ──────────────────────────────────────────────────
    update_counter: u32,
    metrics_timer: u64,
    /// Global warmup counter: number of depth-update batches seen since last
    /// reset. Metrics and heatmap rendering are gated on this reaching 200.
    warmup_samples: usize,

    // ── Per-window state (UI-only, not engine) ─────────────────────────────
    execute_usd: f64,
    max_liquidity_usd: f64,
    price_prec: usize,
    qty_prec: usize,

    // ── Update buffer for pre-snapshot depth events ────────────────────────
    update_buffer: VecDeque<crate::types::DepthUpdate>,
    level_change_buf: Vec<crate::engine::order_book::LevelChangeResult>,

    // ── Window registry ────────────────────────────────────────────────────
    ob_view: OrderBookView,
    windows: Vec<Box<dyn AppWindow>>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, symbol: String) -> Self {
        let (tx, rx) = std_mpsc::channel();
        let (control_tx, control_rx) = tokio_mpsc::channel(1);
        let ctx = cc.egui_ctx.clone();
        let s = symbol.clone();

        // Spawn background Tokio runtime + WebSocket loop onto a dedicated OS thread.
        thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build Tokio runtime")
                .block_on(client::run_streaming_loop(&tx, &ctx, control_rx, s));
        });

        let (price_prec, qty_prec) = client::fetch_precision(&symbol.to_uppercase());

        // Register all analytics windows. Adding a new window = one line here.
        let windows: Vec<Box<dyn AppWindow>> = vec![
            Box::new(HeatmapView::default()),
            Box::new(MetricsView::default()),
            Box::new(LiquidityView::default()),
            Box::new(TwapView::default()),
        ];

        Self {
            symbol: symbol.clone(),
            edited_symbol: symbol,
            rx,
            control_tx,
            order_book: OrderBook::default(),
            metrics: MetricsState::default(),
            heatmap: HeatmapState::new(480, 320),
            twap: TwapDetector::new(),
            rolling_trade_mean: 0.0,
            rolling_trade_std: 1.0,
            update_counter: 0,
            metrics_timer: 0,
            warmup_samples: 0,
            execute_usd: 100_000.0,
            max_liquidity_usd: 100_000.0,
            price_prec,
            qty_prec,
            update_buffer: VecDeque::new(),
            level_change_buf: Vec::with_capacity(4096),
            ob_view: OrderBookView::default(),
            windows,
        }
    }

    // ── Reset helpers ────────────────────────────────────────────────────────────

    /// Full reset — called only on **symbol change**.
    /// Wipes metrics, heatmap, TWAP, and restarts warmup from zero.
    fn reset_all(&mut self) {
        self.order_book.reset();
        self.metrics.reset();
        self.heatmap.reset();
        self.twap.reset();
        self.rolling_trade_mean = 0.0;
        self.rolling_trade_std = 1.0;
        self.update_counter = 0;
        self.warmup_samples = 0;
        self.update_buffer.clear();
    }

    /// Book-only reset — called on **snapshot / refetch** events.
    ///
    /// Preserves:
    /// - `update_buffer` — pre-snapshot events needed for the Binance sync handshake
    ///   (`capital_U ≤ snapshot_id ≤ small_u`). Clearing it breaks the initial sync.
    /// - `warmup_samples` / `update_counter` — analytics continuity.
    /// - All metrics, heatmap, and TWAP state.
    fn reset_book_only(&mut self) {
        self.order_book.reset();
        // Do NOT touch update_buffer — it is drained by the Snapshot handler.
    }

    // ── Trade dispatch ─────────────────────────────────────────────────────────

    fn on_trade(&mut self, trade: Trade) {
        // SOFP: update rolling trade-size distribution.
        let qty = trade.quantity.to_f64().unwrap_or(0.0);
        if qty > 0.0 {
            const ALPHA: f64 = 0.1;
            if self.rolling_trade_mean == 0.0 {
                self.rolling_trade_mean = qty;
                self.rolling_trade_std = qty * 0.1;
            } else {
                let diff = qty - self.rolling_trade_mean;
                self.rolling_trade_mean += ALPHA * diff;
                self.rolling_trade_std =
                    (1.0 - ALPHA) * self.rolling_trade_std + ALPHA * diff.abs();
            }
        }

        // CTR: track fills at the book price.
        {
            let best_bid = self
                .order_book
                .bids
                .keys()
                .next_back()
                .cloned()
                .unwrap_or_default();
            let best_ask = self
                .order_book
                .asks
                .keys()
                .next()
                .cloned()
                .unwrap_or_default();

            if trade.is_buyer_maker {
                // Taker-sell hits bid.
                let is_tob = trade.price == best_bid;
                let in_top20 = self
                    .order_book
                    .bids
                    .keys()
                    .rev()
                    .take(20)
                    .any(|&p| p == trade.price);
                self.metrics.on_fill(Side::Bid, is_tob, in_top20, qty);
            } else {
                // Taker-buy lifts ask.
                let is_tob = trade.price == best_ask;
                let in_top20 = self
                    .order_book
                    .asks
                    .keys()
                    .take(20)
                    .any(|&p| p == trade.price);
                self.metrics.on_fill(Side::Ask, is_tob, in_top20, qty);
            }
        }

        // TWAP: bin the trade by its side.
        self.twap.on_trade(&trade);

        // MTQR: store trade in the attribution buffer.
        self.order_book.record_trade(&trade);
    }

    // ── Depth update dispatch ──────────────────────────────────────────────────

    fn on_depth_update(&mut self, update: crate::types::DepthUpdate) {
        let mut refetch = false;
        self.level_change_buf.clear();
        self.order_book.process_update(
            update,
            self.rolling_trade_mean,
            &mut refetch,
            &mut self.level_change_buf,
        );

        if refetch {
            self.order_book.is_synced = false;
            self.update_buffer.clear();
            let _ = self.control_tx.try_send(Control::Refetch);
            return;
        }

        // ── Dispatch LevelChangeResult → MetricsState ──────────────────────
        // This is the critical wiring that was missing: on_inflow and on_cancel
        // fire for every price level that changed this update cycle.
        for lc in &self.level_change_buf {
            let side = if lc.is_bid { Side::Bid } else { Side::Ask };
            if lc.inflow > 0.0 {
                self.metrics
                    .on_inflow(side, lc.is_tob, lc.in_top20, lc.inflow);
            } else if lc.cancel > 0.0 {
                self.metrics
                    .on_cancel(side, lc.is_tob, lc.in_top20, lc.cancel);
            }
        }

        // ── Heatmap & warmup ───────────────────────────────────────────────
        self.update_counter += 1;
        if self.update_counter >= 10 {
            self.update_counter = 0;
            // Always update rolling stats (drives warmup_samples in heatmap).
            self.heatmap
                .update_rolling_stats(&self.order_book.bids, &self.order_book.asks);
            // Only append pixel column once stats are warm.
            if self.heatmap.warmup_samples >= 30 {
                self.heatmap
                    .append(&self.order_book.bids, &self.order_book.asks);
            }
            // Increment global warmup counter (caps at usize::MAX).
            self.warmup_samples = self.warmup_samples.saturating_add(1);
        }
    }
}

// ── eframe::App ────────────────────────────────────────────────────────────────

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── 1. Periodic metrics sampling ──────────────────────────────────────
        self.metrics_timer += 1;
        // Sample once warmup finished (30 depth-update batches ≈ 300 raw updates)
        if self.metrics_timer % 10 == 0 && self.warmup_samples >= 30 {
            let now = ctx.input(|i| i.time);
            self.metrics.sample(now);
            self.metrics.prune(now - 200.0);
        }

        // ── 2. Drain incoming messages ────────────────────────────────────────
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                AppMessage::Snapshot(snap) => {
                    // Book-only reset — analytics state (metrics, heatmap, TWAP)
                    // is preserved so a routine refetch does not flush the plots.
                    self.reset_book_only();
                    self.order_book.apply_snapshot(snap);
                    while let Some(update) = self.update_buffer.pop_front() {
                        self.on_depth_update(update);
                    }
                }
                AppMessage::Update(update) => {
                    if self.order_book.last_applied_u == 0 {
                        self.update_buffer.push_back(update);
                    } else {
                        self.on_depth_update(update);
                    }
                }
                AppMessage::Trade(trade) => self.on_trade(trade),
                AppMessage::Ticker(ticker) => self.order_book.apply_ticker_anchor(ticker),
            }
        }

        // ── 3. Central panel ──────────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(format!(
                "{} Perpetual Order Book",
                self.symbol.to_uppercase()
            ));

            // Toggle buttons for all windows.
            ui.horizontal_wrapped(|ui| {
                for w in &mut self.windows {
                    if ui.button(format!("Toggle {}", w.name())).clicked() {
                        w.toggle();
                    }
                }
                // Order book view has its own toggle (always shown inline).
                if ui.button("Toggle K-Means Mode").clicked() {
                    self.ob_view.kmeans_mode = !self.ob_view.kmeans_mode;
                }
            });

            // Symbol change bar.
            ui.horizontal(|ui| {
                ui.label("Symbol:");
                ui.text_edit_singleline(&mut self.edited_symbol);
                if ui.button("Change Symbol").clicked() && self.edited_symbol != self.symbol {
                    let (pp, qp) = client::fetch_precision(&self.edited_symbol.to_uppercase());
                    self.price_prec = pp;
                    self.qty_prec = qp;
                    let _ = self
                        .control_tx
                        .try_send(Control::ChangeSymbol(self.edited_symbol.clone()));
                    self.symbol = self.edited_symbol.clone();
                    self.reset_all();
                }
            });

            // K-means iteration sliders.
            if self.ob_view.kmeans_mode {
                ui.horizontal(|ui| {
                    ui.label("K-means Batch Size:");
                    ui.add(egui::Slider::new(&mut self.ob_view.batch_size, 32..=2048));
                });
                ui.horizontal(|ui| {
                    ui.label("K-means Max Iter:");
                    ui.add(egui::Slider::new(&mut self.ob_view.max_iter, 64..=2048));
                });
            }

            // Order book inline view.
            let warmup_samples = self.warmup_samples;
            let mut state = AppState {
                order_book: &mut self.order_book,
                metrics: &mut self.metrics,
                heatmap: &mut self.heatmap,
                twap: &mut self.twap,
                rolling_trade_mean: self.rolling_trade_mean,
                warmup_samples,
                price_prec: self.price_prec,
                qty_prec: self.qty_prec,
                execute_usd: &mut self.execute_usd,
                max_liquidity_usd: self.max_liquidity_usd,
            };
            self.ob_view.render_inline(ui, &mut state);
        });

        // ── 4. Floating analytics windows ─────────────────────────────────────
        let warmup_samples = self.warmup_samples;
        let mut state = AppState {
            order_book: &mut self.order_book,
            metrics: &mut self.metrics,
            heatmap: &mut self.heatmap,
            twap: &mut self.twap,
            rolling_trade_mean: self.rolling_trade_mean,
            warmup_samples,
            price_prec: self.price_prec,
            qty_prec: self.qty_prec,
            execute_usd: &mut self.execute_usd,
            max_liquidity_usd: self.max_liquidity_usd,
        };
        for w in &mut self.windows {
            w.show(ctx, &mut state);
        }
    }
}
