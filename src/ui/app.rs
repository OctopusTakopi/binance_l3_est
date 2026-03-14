//! Application orchestrator: owns all engine state and drives the window system.

use std::collections::VecDeque;
use std::env;
use std::sync::mpsc::{self as std_mpsc, Receiver as StdReceiver};
use std::thread;

use crate::engine::{feature_engine::FeatureEngine, order_book::OrderBook};
use crate::network::{AppMessage, Control, MarketType, client};
use crate::types::Trade;
use crate::ui::window::{AppState, AppWindow};
use crate::ui::windows::{
    heatmap_view::HeatmapView, liquidity_view::LiquidityView, metrics_view::MetricsView,
    order_book_view::OrderBookView, twap_view::TwapView,
};
use eframe::egui;
use tokio::sync::mpsc::{self as tokio_mpsc, UnboundedSender as TokioSender};

const MAX_APP_MESSAGES_PER_FRAME: usize = 2_000;

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
    market: MarketType,
    edited_market: MarketType,
    spot_api_key: Option<String>,
    edited_spot_api_key: String,
    show_spot_key_window: bool,
    spot_key_message: Option<String>,
    network_warning: Option<String>,
    show_error_popup: bool,
    error_popup_message: Option<String>,
    pending_switch_after_key: bool,
    rx: StdReceiver<AppMessage>,
    control_tx: TokioSender<Control>,

    // ── Engine state ───────────────────────────────────────────────────────
    order_book: OrderBook,
    feature_engine: FeatureEngine,

    // ── Counters / timers ──────────────────────────────────────────────────
    metrics_timer: u64,

    // ── Per-window state (UI-only, not engine) ─────────────────────────────
    execute_usd: f64,
    max_liquidity_usd: f64,
    price_prec: usize,
    qty_prec: usize,
    symbol_spec_ready: bool,

    // ── Update buffer for pre-snapshot depth events ────────────────────────
    update_buffer: VecDeque<crate::types::DepthUpdate>,

    // ── Window registry ────────────────────────────────────────────────────
    ob_view: OrderBookView,
    windows: Vec<Box<dyn AppWindow>>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, symbol: String, market: MarketType) -> Self {
        let (tx, rx) = std_mpsc::channel();
        let (control_tx, control_rx) = tokio_mpsc::unbounded_channel();
        let ctx = cc.egui_ctx.clone();
        let s = symbol.clone();
        let m = market;
        let initial_spot_api_key = env::var("BINANCE_SPOT_SBE_API_KEY")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let has_initial_spot_api_key = initial_spot_api_key.is_some();
        let thread_spot_api_key = initial_spot_api_key.clone();

        // Spawn background Tokio runtime + WebSocket loop onto a dedicated OS thread.
        thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build Tokio runtime")
                .block_on(client::run_streaming_loop(
                    &tx,
                    &ctx,
                    control_rx,
                    s,
                    m,
                    thread_spot_api_key,
                ));
        });

        let startup_spec = match client::fetch_exchange_specs(market) {
            Ok(cache) => client::lookup_symbol_spec(&cache, &symbol, market)
                .map(Ok)
                .unwrap_or_else(|| {
                    Err(format!(
                        "Symbol spec not found in exchange info for {} on {}.",
                        symbol.to_uppercase(),
                        market.as_str()
                    ))
                }),
            Err(e) => Err(format!(
                "Failed to load exchange info for {}: {e}",
                market.as_str()
            )),
        };
        let (spec, symbol_spec_ready, startup_error) = match startup_spec {
            Ok(spec) => (spec, true, None),
            Err(message) => (client::SymbolSpec::default(), false, Some(message)),
        };

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
            market,
            edited_market: market,
            spot_api_key: initial_spot_api_key.clone(),
            edited_spot_api_key: initial_spot_api_key.unwrap_or_default(),
            show_spot_key_window: market == MarketType::Spot && !has_initial_spot_api_key,
            spot_key_message: (market == MarketType::Spot && !has_initial_spot_api_key)
                .then(|| "Spot SBE requires an API key.".to_string()),
            network_warning: startup_error.clone(),
            show_error_popup: startup_error.is_some(),
            error_popup_message: startup_error,
            pending_switch_after_key: false,
            rx,
            control_tx,
            order_book: OrderBook::new(spec.tick_size),
            feature_engine: FeatureEngine::new(480, 320),
            metrics_timer: 0,
            execute_usd: 100_000.0,
            max_liquidity_usd: 100_000.0,
            price_prec: spec.price_prec,
            qty_prec: spec.qty_prec,
            symbol_spec_ready,
            update_buffer: VecDeque::new(),
            ob_view: OrderBookView::default(),
            windows,
        }
    }

    // ── Reset helpers ────────────────────────────────────────────────────────────

    /// Full reset — called only on **symbol change**.
    /// Wipes metrics, heatmap, TWAP, and restarts warmup from zero.
    fn reset_all(&mut self) {
        self.order_book.reset();
        self.feature_engine.reset();
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
        self.feature_engine.reset_book_only();
        // Do NOT touch update_buffer — it is drained by the Snapshot handler.
    }

    // ── Trade dispatch ─────────────────────────────────────────────────────────

    fn on_trade(&mut self, trade: Trade) {
        self.feature_engine.on_trade(&trade, &mut self.order_book);
    }

    // ── Depth update dispatch ──────────────────────────────────────────────────

    fn on_depth_update(&mut self, update: crate::types::DepthUpdate) {
        use crate::engine::order_book::OrderBookError;

        let res = self
            .feature_engine
            .on_depth_update(update, &mut self.order_book);

        if let Err(OrderBookError::SequenceGap) = res {
            self.order_book.is_synced = false;
            self.update_buffer.clear();
            let _ = self.control_tx.send(Control::Refetch);
            return;
        }
    }

    fn symbol_matches_current(&self, symbol: &str) -> bool {
        symbol.eq_ignore_ascii_case(&self.symbol)
    }

    fn message_matches_current(&self, symbol: &str, market: MarketType) -> bool {
        market == self.market && self.symbol_matches_current(symbol)
    }

    fn has_spot_api_key(&self) -> bool {
        self.spot_api_key
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
    }

    fn save_spot_api_key(&mut self) -> bool {
        let key = self.edited_spot_api_key.trim().to_string();
        if key.is_empty() {
            self.spot_key_message = Some("Spot API key cannot be empty.".to_string());
            return false;
        }

        self.spot_api_key = Some(key);
        self.spot_key_message = None;
        true
    }

    fn apply_market_change(&mut self) {
        let send_result = self.control_tx.send(Control::ChangeSymbol {
            symbol: self.edited_symbol.clone(),
            market: self.edited_market,
            spot_api_key: (self.edited_market == MarketType::Spot)
                .then(|| self.spot_api_key.clone())
                .flatten(),
        });
        if send_result.is_err() {
            self.network_warning = Some("Background network task is unavailable.".to_string());
            self.error_popup_message = self.network_warning.clone();
            self.show_error_popup = true;
            return;
        }
        self.symbol = self.edited_symbol.clone();
        self.market = self.edited_market;
        self.network_warning = None;
        self.show_error_popup = false;
        self.error_popup_message = None;
        self.symbol_spec_ready = false;
        self.reset_all();
    }

    fn maybe_apply_market_change(&mut self) {
        if self.edited_market == MarketType::Spot && !self.has_spot_api_key() {
            self.pending_switch_after_key = true;
            self.show_spot_key_window = true;
            self.spot_key_message =
                Some("Enter a spot API key to enable the SBE stream.".to_string());
            return;
        }

        if !self.has_spot_api_key() && !self.edited_spot_api_key.trim().is_empty() {
            let _ = self.save_spot_api_key();
        }

        self.apply_market_change();
    }

    fn show_spot_api_key_window(&mut self, ctx: &egui::Context) {
        if !self.show_spot_key_window {
            return;
        }

        let mut open = self.show_spot_key_window;
        let mut save_clicked = false;
        let mut cancel_clicked = false;

        egui::Window::new("Spot API Key")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label("Spot market uses Binance SBE and needs an API key.");
                if let Some(message) = &self.spot_key_message {
                    if !message.is_empty() {
                        ui.label(message);
                    }
                }
                ui.add(
                    egui::TextEdit::singleline(&mut self.edited_spot_api_key)
                        .password(true)
                        .desired_width(320.0),
                );
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        save_clicked = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel_clicked = true;
                    }
                });
            });

        if save_clicked && self.save_spot_api_key() {
            self.pending_switch_after_key = false;
            open = false;
        }

        if cancel_clicked {
            self.pending_switch_after_key = false;
            open = false;
        }

        self.show_spot_key_window = open;
    }

    fn show_error_popup(&mut self, ctx: &egui::Context) {
        if !self.show_error_popup {
            return;
        }

        let mut open = self.show_error_popup;
        let mut close_clicked = false;

        egui::Window::new("Connection Error")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                if let Some(message) = &self.error_popup_message {
                    ui.label(message);
                }
                if ui.button("Close").clicked() {
                    close_clicked = true;
                }
            });

        if close_clicked {
            open = false;
        }

        self.show_error_popup = open;
    }
}

// ── eframe::App ────────────────────────────────────────────────────────────────

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── 1. Periodic metrics sampling ──────────────────────────────────────
        self.metrics_timer += 1;
        // Sample once warmup finished (30 depth-update batches ≈ 300 raw updates)
        if self.metrics_timer % 10 == 0 && self.feature_engine.warmup_samples >= 30 {
            let now = ctx.input(|i| i.time);
            self.feature_engine.metrics.sample(now);
            self.feature_engine.metrics.prune(now - 200.0);
        }

        // ── 2. Drain incoming messages ────────────────────────────────────────
        let mut processed_messages = 0usize;
        while processed_messages < MAX_APP_MESSAGES_PER_FRAME {
            let Ok(msg) = self.rx.try_recv() else {
                break;
            };
            processed_messages += 1;

            match msg {
                AppMessage::SymbolSpec {
                    symbol,
                    market,
                    spec,
                } => {
                    if !self.message_matches_current(&symbol, market) {
                        continue;
                    }
                    self.price_prec = spec.price_prec;
                    self.qty_prec = spec.qty_prec;
                    self.order_book.set_tick_size(spec.tick_size);
                    self.symbol_spec_ready = true;
                    if self.network_warning.as_deref().is_some_and(|message| {
                        message.starts_with("Exchange info")
                            || message.starts_with("Failed to load exchange info")
                            || message.starts_with("Symbol spec")
                    }) {
                        self.network_warning = None;
                        self.show_error_popup = false;
                        self.error_popup_message = None;
                    }
                }
                AppMessage::Snapshot {
                    symbol,
                    market,
                    snapshot,
                } => {
                    if !self.message_matches_current(&symbol, market) {
                        continue;
                    }
                    if !self.symbol_spec_ready {
                        continue;
                    }
                    // Book-only reset — analytics state (metrics, heatmap, TWAP)
                    // is preserved so a routine refetch does not flush the plots.
                    self.show_error_popup = false;
                    self.error_popup_message = None;
                    self.reset_book_only();
                    self.order_book.apply_snapshot(snapshot);
                    while let Some(update) = self.update_buffer.pop_front() {
                        self.on_depth_update(update);
                    }
                }
                AppMessage::Update { market, update } => {
                    if market != self.market || !self.symbol_matches_current(update.symbol.as_str())
                    {
                        continue;
                    }
                    if !self.symbol_spec_ready {
                        continue;
                    }
                    if self.order_book.last_applied_u == 0 {
                        self.update_buffer.push_back(update);
                    } else {
                        self.on_depth_update(update);
                    }
                }
                AppMessage::Trade { market, trade } => {
                    if market != self.market || !self.symbol_matches_current(trade.symbol.as_str())
                    {
                        continue;
                    }
                    if !self.symbol_spec_ready {
                        continue;
                    }
                    self.on_trade(trade)
                }
                AppMessage::Ticker { market, ticker } => {
                    if market != self.market || !self.symbol_matches_current(ticker.symbol.as_str())
                    {
                        continue;
                    }
                    if !self.symbol_spec_ready {
                        continue;
                    }
                    self.order_book.apply_ticker_anchor(ticker)
                }
                AppMessage::SpotApiKeyRequired {
                    symbol,
                    market,
                    message,
                } => {
                    if !self.message_matches_current(&symbol, market) {
                        continue;
                    }
                    self.spot_key_message = Some(message);
                    self.show_spot_key_window = true;
                }
                AppMessage::NetworkWarning {
                    symbol,
                    market,
                    message,
                } => {
                    if !self.message_matches_current(&symbol, market) {
                        continue;
                    }
                    self.network_warning = if message.trim().is_empty() {
                        None
                    } else {
                        Some(message)
                    };
                    if let Some(message) = &self.network_warning {
                        if message.starts_with("Spot connection error:")
                            || message.starts_with("Exchange info")
                            || message.starts_with("Failed to load exchange info")
                            || message.starts_with("Symbol spec")
                        {
                            self.error_popup_message = Some(message.clone());
                            self.show_error_popup = true;
                        }
                    } else {
                        self.error_popup_message = None;
                        self.show_error_popup = false;
                    }
                }
            }
        }

        if processed_messages == MAX_APP_MESSAGES_PER_FRAME {
            // Spot SBE can outpace egui if we drain indefinitely here.
            // Yield to rendering and continue draining on the next frame.
            ctx.request_repaint();
        }

        // ── 3. Central panel ──────────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(format!(
                "{} {} Order Book",
                self.symbol.to_uppercase(),
                match self.market {
                    MarketType::Spot => "Spot",
                    MarketType::Futures => "Futures",
                }
            ));
            if let Some(message) = &self.network_warning {
                ui.label(message);
            }

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
                egui::ComboBox::from_label("Market")
                    .selected_text(match self.edited_market {
                        MarketType::Spot => "Spot",
                        MarketType::Futures => "Futures",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.edited_market,
                            MarketType::Futures,
                            "Futures",
                        );
                        ui.selectable_value(&mut self.edited_market, MarketType::Spot, "Spot");
                    });

                let should_apply = self.edited_symbol != self.symbol
                    || self.edited_market != self.market
                    || self.market == MarketType::Spot
                    || self.edited_market == MarketType::Spot;

                if ui.button("Apply").clicked() && should_apply {
                    self.maybe_apply_market_change();
                }
                if ui.button("Spot API Key").clicked() {
                    self.pending_switch_after_key = false;
                    self.show_spot_key_window = true;
                    self.spot_key_message = None;
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
            let warmup_samples = self.feature_engine.warmup_samples;
            let mut state = AppState {
                order_book: &mut self.order_book,
                metrics: &mut self.feature_engine.metrics,
                heatmap: &mut self.feature_engine.heatmap,
                twap: &mut self.feature_engine.twap,
                rolling_trade_mean: self.feature_engine.rolling_trade_mean,
                warmup_samples,
                price_prec: self.price_prec,
                qty_prec: self.qty_prec,
                execute_usd: &mut self.execute_usd,
                max_liquidity_usd: self.max_liquidity_usd,
            };
            self.ob_view.render_inline(ui, &mut state);
        });

        // ── 4. Floating analytics windows ─────────────────────────────────────
        let warmup_samples = self.feature_engine.warmup_samples;
        let mut state = AppState {
            order_book: &mut self.order_book,
            metrics: &mut self.feature_engine.metrics,
            heatmap: &mut self.feature_engine.heatmap,
            twap: &mut self.feature_engine.twap,
            rolling_trade_mean: self.feature_engine.rolling_trade_mean,
            warmup_samples,
            price_prec: self.price_prec,
            qty_prec: self.qty_prec,
            execute_usd: &mut self.execute_usd,
            max_liquidity_usd: self.max_liquidity_usd,
        };
        for w in &mut self.windows {
            w.show(ctx, &mut state);
        }

        self.show_spot_api_key_window(ctx);
        self.show_error_popup(ctx);
    }
}
