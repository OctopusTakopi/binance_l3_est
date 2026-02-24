//! Metrics state: rolling CTR (Cancellation-to-Trade Ratio) and OTR
//! (Order-to-Trade Ratio) accumulators and time-series history.

use egui_plot::PlotPoint;
use std::collections::VecDeque;

// ── Side / Tier selector types ─────────────────────────────────────────────────

/// Which book side a metric event belongs to.
#[derive(Clone, Copy)]
pub enum Side {
    Bid,
    Ask,
}

/// Which level tier (top-1 or top-20).
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum Tier {
    Top1,
    Top20,
}

// ── MetricsState ────────────────────────────────────────────────────────────────

/// Accumulators for CTR and OTR, plus their time-series history for plotting.
///
/// All `fills_*`, `cancels_*`, and `inflows_*` counters are running sums since
/// the last connection; they are never reset between samples — only the history
/// `VecDeque`s are pruned.
pub struct MetricsState {
    // ── CTR accumulators ──────────────────────────────────────────────────
    pub fills_bid_top1: f64,
    pub cancels_bid_top1: f64,
    pub fills_ask_top1: f64,
    pub cancels_ask_top1: f64,
    pub fills_bid_top20: f64,
    pub cancels_bid_top20: f64,
    pub fills_ask_top20: f64,
    pub cancels_ask_top20: f64,

    // ── OTR accumulators ──────────────────────────────────────────────────
    pub inflows_bid_top1: f64,
    pub inflows_ask_top1: f64,
    pub inflows_bid_top20: f64,
    pub inflows_ask_top20: f64,

    // ── CTR history ───────────────────────────────────────────────────────
    pub ctr_history_bid_top1: VecDeque<PlotPoint>,
    pub ctr_history_ask_top1: VecDeque<PlotPoint>,
    pub ctr_history_both_top1: VecDeque<PlotPoint>,
    pub ctr_history_bid_top20: VecDeque<PlotPoint>,
    pub ctr_history_ask_top20: VecDeque<PlotPoint>,
    pub ctr_history_both_top20: VecDeque<PlotPoint>,

    // ── OTR history ───────────────────────────────────────────────────────
    pub otr_history_bid_top1: VecDeque<PlotPoint>,
    pub otr_history_ask_top1: VecDeque<PlotPoint>,
    pub otr_history_both_top1: VecDeque<PlotPoint>,
    pub otr_history_bid_top20: VecDeque<PlotPoint>,
    pub otr_history_ask_top20: VecDeque<PlotPoint>,
    pub otr_history_both_top20: VecDeque<PlotPoint>,
}

impl Default for MetricsState {
    fn default() -> Self {
        const CAP: usize = 2000;
        Self {
            fills_bid_top1: 0.0,
            cancels_bid_top1: 0.0,
            fills_ask_top1: 0.0,
            cancels_ask_top1: 0.0,
            fills_bid_top20: 0.0,
            cancels_bid_top20: 0.0,
            fills_ask_top20: 0.0,
            cancels_ask_top20: 0.0,
            inflows_bid_top1: 0.0,
            inflows_ask_top1: 0.0,
            inflows_bid_top20: 0.0,
            inflows_ask_top20: 0.0,
            ctr_history_bid_top1: VecDeque::with_capacity(CAP),
            ctr_history_ask_top1: VecDeque::with_capacity(CAP),
            ctr_history_both_top1: VecDeque::with_capacity(CAP),
            ctr_history_bid_top20: VecDeque::with_capacity(CAP),
            ctr_history_ask_top20: VecDeque::with_capacity(CAP),
            ctr_history_both_top20: VecDeque::with_capacity(CAP),
            otr_history_bid_top1: VecDeque::with_capacity(CAP),
            otr_history_ask_top1: VecDeque::with_capacity(CAP),
            otr_history_both_top1: VecDeque::with_capacity(CAP),
            otr_history_bid_top20: VecDeque::with_capacity(CAP),
            otr_history_ask_top20: VecDeque::with_capacity(CAP),
            otr_history_both_top20: VecDeque::with_capacity(CAP),
        }
    }
}

impl MetricsState {
    /// Reset all accumulators and clear all history (called on new snapshot).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    // ── Event ingestion ────────────────────────────────────────────────────────

    /// Record a trade fill event.
    pub fn on_fill(&mut self, side: Side, is_tob: bool, in_top20: bool, qty: f64) {
        match side {
            Side::Bid => {
                if is_tob {
                    self.fills_bid_top1 += qty;
                    self.fills_bid_top20 += qty;
                } else if in_top20 {
                    self.fills_bid_top20 += qty;
                }
            }
            Side::Ask => {
                if is_tob {
                    self.fills_ask_top1 += qty;
                    self.fills_ask_top20 += qty;
                } else if in_top20 {
                    self.fills_ask_top20 += qty;
                }
            }
        }
    }

    /// Record a cancellation event (outflow not explained by trades).
    pub fn on_cancel(&mut self, side: Side, is_tob: bool, in_top20: bool, qty: f64) {
        match side {
            Side::Bid => {
                if is_tob {
                    self.cancels_bid_top1 += qty;
                    self.cancels_bid_top20 += qty;
                } else if in_top20 {
                    self.cancels_bid_top20 += qty;
                }
            }
            Side::Ask => {
                if is_tob {
                    self.cancels_ask_top1 += qty;
                    self.cancels_ask_top20 += qty;
                } else if in_top20 {
                    self.cancels_ask_top20 += qty;
                }
            }
        }
    }

    /// Record a liquidity inflow event (OTR numerator).
    pub fn on_inflow(&mut self, side: Side, is_tob: bool, in_top20: bool, qty: f64) {
        match side {
            Side::Bid => {
                if is_tob {
                    self.inflows_bid_top1 += qty;
                    self.inflows_bid_top20 += qty;
                } else if in_top20 {
                    self.inflows_bid_top20 += qty;
                }
            }
            Side::Ask => {
                if is_tob {
                    self.inflows_ask_top1 += qty;
                    self.inflows_ask_top20 += qty;
                } else if in_top20 {
                    self.inflows_ask_top20 += qty;
                }
            }
        }
    }

    // ── Sampling ───────────────────────────────────────────────────────────────

    /// Compute current CTR and OTR ratios and push them to the history queues.
    pub fn sample(&mut self, now: f64) {
        // CTR Top-1
        let ctr_bid_1 = safe_div(self.cancels_bid_top1, self.fills_bid_top1);
        let ctr_ask_1 = safe_div(self.cancels_ask_top1, self.fills_ask_top1);
        let ctr_both_1 = safe_div(
            self.cancels_bid_top1 + self.cancels_ask_top1,
            self.fills_bid_top1 + self.fills_ask_top1,
        );
        self.ctr_history_bid_top1
            .push_back(PlotPoint::new(now, ctr_bid_1));
        self.ctr_history_ask_top1
            .push_back(PlotPoint::new(now, ctr_ask_1));
        self.ctr_history_both_top1
            .push_back(PlotPoint::new(now, ctr_both_1));

        // CTR Top-20
        let ctr_bid_20 = safe_div(self.cancels_bid_top20, self.fills_bid_top20);
        let ctr_ask_20 = safe_div(self.cancels_ask_top20, self.fills_ask_top20);
        let ctr_both_20 = safe_div(
            self.cancels_bid_top20 + self.cancels_ask_top20,
            self.fills_bid_top20 + self.fills_ask_top20,
        );
        self.ctr_history_bid_top20
            .push_back(PlotPoint::new(now, ctr_bid_20));
        self.ctr_history_ask_top20
            .push_back(PlotPoint::new(now, ctr_ask_20));
        self.ctr_history_both_top20
            .push_back(PlotPoint::new(now, ctr_both_20));

        // OTR Top-1
        let otr_bid_1 = safe_div(self.inflows_bid_top1, self.fills_bid_top1);
        let otr_ask_1 = safe_div(self.inflows_ask_top1, self.fills_ask_top1);
        let otr_both_1 = safe_div(
            self.inflows_bid_top1 + self.inflows_ask_top1,
            self.fills_bid_top1 + self.fills_ask_top1,
        );
        self.otr_history_bid_top1
            .push_back(PlotPoint::new(now, otr_bid_1));
        self.otr_history_ask_top1
            .push_back(PlotPoint::new(now, otr_ask_1));
        self.otr_history_both_top1
            .push_back(PlotPoint::new(now, otr_both_1));

        // OTR Top-20
        let otr_bid_20 = safe_div(self.inflows_bid_top20, self.fills_bid_top20);
        let otr_ask_20 = safe_div(self.inflows_ask_top20, self.fills_ask_top20);
        let otr_both_20 = safe_div(
            self.inflows_bid_top20 + self.inflows_ask_top20,
            self.fills_bid_top20 + self.fills_ask_top20,
        );
        self.otr_history_bid_top20
            .push_back(PlotPoint::new(now, otr_bid_20));
        self.otr_history_ask_top20
            .push_back(PlotPoint::new(now, otr_ask_20));
        self.otr_history_both_top20
            .push_back(PlotPoint::new(now, otr_both_20));
    }

    /// Remove history points older than `limit` seconds.
    pub fn prune(&mut self, limit: f64) {
        prune_deque(&mut self.ctr_history_bid_top1, limit);
        prune_deque(&mut self.ctr_history_ask_top1, limit);
        prune_deque(&mut self.ctr_history_both_top1, limit);
        prune_deque(&mut self.ctr_history_bid_top20, limit);
        prune_deque(&mut self.ctr_history_ask_top20, limit);
        prune_deque(&mut self.ctr_history_both_top20, limit);
        prune_deque(&mut self.otr_history_bid_top1, limit);
        prune_deque(&mut self.otr_history_ask_top1, limit);
        prune_deque(&mut self.otr_history_both_top1, limit);
        prune_deque(&mut self.otr_history_bid_top20, limit);
        prune_deque(&mut self.otr_history_ask_top20, limit);
        prune_deque(&mut self.otr_history_both_top20, limit);
    }
}

// ── Private helpers ────────────────────────────────────────────────────────────

#[inline]
fn safe_div(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn prune_deque(deq: &mut VecDeque<PlotPoint>, limit: f64) {
    while deq.front().map_or(false, |p| p.x < limit) {
        deq.pop_front();
    }
}
