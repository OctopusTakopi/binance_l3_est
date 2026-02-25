//! TWAP detector: time-binned trade-count accumulator + FFT-based periodicity
//! detection, split by taker side (buy vs sell).

use crate::types::Trade;
use num_complex::Complex;
use rust_decimal::prelude::ToPrimitive;
use rustfft::FftPlanner;
use std::collections::VecDeque;

/// Full TWAP (Time-Weighted Average Price) detector state.
///
/// Trades are accumulated into fixed-width time bins; every `batch_bins` new
/// bins the FFT pipeline runs and updates the stored PSD + peak results.
pub struct TwapDetector {
    // ── Configuration ──────────────────────────────────────────────────────
    /// Width of each time bin in milliseconds.
    pub bin_ms: u64,
    /// Number of bins kept in the sliding window for FFT.
    pub window_bins: usize,
    /// Peak detection threshold (multiples of spectral noise std).
    pub threshold_sigma: f64,

    // ── Current open bin ───────────────────────────────────────────────────
    pub current_bin_start: u64,
    pub current_bin_buy: u64,
    pub current_bin_sell: u64,
    pub current_vol_buy: f64,
    pub current_vol_sell: f64,

    // ── Completed bins (sliding window) ────────────────────────────────────
    /// Taker-buy trade-count per bin.
    pub bins_buy: VecDeque<f64>,
    /// Taker-sell trade-count per bin.
    pub bins_sell: VecDeque<f64>,
    /// Taker-buy USD notional per bin.
    pub vol_bins_buy: VecDeque<f64>,
    /// Taker-sell USD notional per bin.
    pub vol_bins_sell: VecDeque<f64>,

    /// How many bins have been completed since the last FFT run.
    pub bins_since_fft: usize,

    // ── FFT results ────────────────────────────────────────────────────────
    /// One-sided PSD for taker-buy count signal: `[freq_hz, power]` pairs.
    pub psd_buy: Vec<[f64; 2]>,
    /// One-sided PSD for taker-sell count signal.
    pub psd_sell: Vec<[f64; 2]>,
    /// Detected peaks (freq, power) in the buy count PSD.
    pub peaks_buy: Vec<(f64, f64)>,
    /// Detected peaks in the sell count PSD.
    pub peaks_sell: Vec<(f64, f64)>,
    /// One-sided PSD for taker-buy USD volume signal (amplitude estimation).
    pub psd_vol_buy: Vec<[f64; 2]>,
    pub psd_vol_sell: Vec<[f64; 2]>,

    // ── Internal compute state ─────────────────────────────────────────────
    /// Planners are expensive to allocate (twiddle factors).
    pub planner: FftPlanner<f64>,
    /// Pre-allocated buffer for FFT complex values to avoid allocation in hot path.
    pub fft_buffer: Vec<Complex<f64>>,
}

impl TwapDetector {
    /// Create with sensible defaults.
    pub fn new() -> Self {
        Self {
            bin_ms: 500,
            window_bins: 512,
            threshold_sigma: 3.0,

            current_bin_start: 0,
            current_bin_buy: 0,
            current_bin_sell: 0,
            current_vol_buy: 0.0,
            current_vol_sell: 0.0,

            bins_buy: VecDeque::with_capacity(1024),
            bins_sell: VecDeque::with_capacity(1024),
            vol_bins_buy: VecDeque::with_capacity(1024),
            vol_bins_sell: VecDeque::with_capacity(1024),
            bins_since_fft: 0,

            psd_buy: Vec::new(),
            psd_sell: Vec::new(),
            peaks_buy: Vec::new(),
            peaks_sell: Vec::new(),
            psd_vol_buy: Vec::new(),
            psd_vol_sell: Vec::new(),
            planner: FftPlanner::<f64>::new(),
            fft_buffer: Vec::with_capacity(1024),
        }
    }

    /// Reset all state (called on snapshot or symbol change).
    pub fn reset(&mut self) {
        self.bins_buy.clear();
        self.bins_sell.clear();
        self.vol_bins_buy.clear();
        self.vol_bins_sell.clear();
        self.psd_buy.clear();
        self.psd_sell.clear();
        self.peaks_buy.clear();
        self.peaks_sell.clear();
        self.psd_vol_buy.clear();
        self.psd_vol_sell.clear();
        self.current_bin_start = 0;
        self.current_bin_buy = 0;
        self.current_bin_sell = 0;
        self.current_vol_buy = 0.0;
        self.current_vol_sell = 0.0;
        self.bins_since_fft = 0;
    }

    /// Reset only the bin-parameter-sensitive fields (called when `bin_ms` slider
    /// changes in the UI).
    pub fn reset_bins(&mut self) {
        self.bins_buy.clear();
        self.bins_sell.clear();
        self.psd_buy.clear();
        self.psd_sell.clear();
        self.peaks_buy.clear();
        self.peaks_sell.clear();
        self.current_bin_start = 0;
        self.current_bin_buy = 0;
        self.current_bin_sell = 0;
        self.bins_since_fft = 0;
    }

    // ── Trade ingestion ────────────────────────────────────────────────────────

    /// Process one trade: accumulate into the current bin, closing it if its
    /// time window has elapsed, then optionally run the FFT pipeline.
    pub fn on_trade(&mut self, trade: &Trade) {
        let ts = trade.transaction_time;
        if self.current_bin_start == 0 {
            self.current_bin_start = ts;
        }

        let notional = trade.quantity.to_f64().unwrap_or(0.0) * trade.price.to_f64().unwrap_or(0.0);

        if trade.is_buyer_maker {
            self.current_bin_sell += 1;
            self.current_vol_sell += notional;
        } else {
            self.current_bin_buy += 1;
            self.current_vol_buy += notional;
        }

        if ts.saturating_sub(self.current_bin_start) >= self.bin_ms {
            self.close_bin();
        }
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    fn close_bin(&mut self) {
        push_capped(
            &mut self.bins_buy,
            self.current_bin_buy as f64,
            self.window_bins,
        );
        push_capped(
            &mut self.bins_sell,
            self.current_bin_sell as f64,
            self.window_bins,
        );
        push_capped(
            &mut self.vol_bins_buy,
            self.current_vol_buy,
            self.window_bins,
        );
        push_capped(
            &mut self.vol_bins_sell,
            self.current_vol_sell,
            self.window_bins,
        );

        self.current_bin_buy = 0;
        self.current_bin_sell = 0;
        self.current_vol_buy = 0.0;
        self.current_vol_sell = 0.0;
        self.current_bin_start = 0; // Will be re-set on next trade

        self.bins_since_fft += 1;
        if self.bins_since_fft >= 32 || (self.psd_buy.is_empty() && self.bins_buy.len() >= 64) {
            self.compute_fft();
            self.bins_since_fft = 0;
        }
    }

    pub fn compute_fft(&mut self) {
        let sigma = self.threshold_sigma;
        let bin_ms = self.bin_ms;

        Self::run_fft_pipeline(
            &self.bins_buy,
            bin_ms,
            sigma,
            &mut self.planner,
            &mut self.fft_buffer,
            &mut self.psd_buy,
            Some(&mut self.peaks_buy),
        );
        Self::run_fft_pipeline(
            &self.bins_sell,
            bin_ms,
            sigma,
            &mut self.planner,
            &mut self.fft_buffer,
            &mut self.psd_sell,
            Some(&mut self.peaks_sell),
        );
        Self::run_fft_pipeline(
            &self.vol_bins_buy,
            bin_ms,
            sigma,
            &mut self.planner,
            &mut self.fft_buffer,
            &mut self.psd_vol_buy,
            None,
        );
        Self::run_fft_pipeline(
            &self.vol_bins_sell,
            bin_ms,
            sigma,
            &mut self.planner,
            &mut self.fft_buffer,
            &mut self.psd_vol_sell,
            None,
        );
    }

    /// Pure FFT pipeline: linear detrend → Hanning window → FFT → one-sided PSD →
    /// peak detection. Populates outputs in place to avoid allocations.
    pub fn run_fft_pipeline(
        bins: &VecDeque<f64>,
        bin_ms: u64,
        sigma: f64,
        planner: &mut FftPlanner<f64>,
        buffer: &mut Vec<Complex<f64>>,
        psd_out: &mut Vec<[f64; 2]>,
        mut peaks_out: Option<&mut Vec<(f64, f64)>>,
    ) {
        let n = bins.len();
        psd_out.clear();
        if let Some(ref mut p) = peaks_out {
            p.clear();
        }

        if n < 64 {
            return;
        }

        let n_f = n as f64;

        // Least-squares linear detrending: y = mx + c
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, &y) in bins.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        // m = (N*Σ(xy) - Σx*Σy) / (N*Σ(x^2) - (Σx)^2)
        let denominator = n_f * sum_xx - sum_x * sum_x;
        let slope = if denominator == 0.0 {
            0.0
        } else {
            (n_f * sum_xy - sum_x * sum_y) / denominator
        };

        // c = (Σy - m*Σx) / N
        let intercept = (sum_y - slope * sum_x) / n_f;

        buffer.clear();
        buffer.extend(bins.iter().enumerate().map(|(i, &x)| {
            let x_detrended = x - (slope * i as f64 + intercept);
            let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n_f - 1.0)).cos());
            Complex {
                re: x_detrended * w,
                im: 0.0,
            }
        }));

        let fft = planner.plan_fft_forward(n);
        fft.process(buffer);

        let bin_sec = bin_ms as f64 / 1000.0;
        let half = n / 2;

        psd_out.extend((1..=half).map(|k| {
            let freq = k as f64 / (n_f * bin_sec);
            let power = 2.0 * buffer[k].norm_sqr() / (n_f * n_f);
            [freq, power]
        }));

        if let Some(peaks_out) = peaks_out {
            let n_psd = psd_out.len() as f64;
            let mut p_sum = 0.0;
            for p in psd_out.iter() {
                p_sum += p[1];
            }
            let p_mean = p_sum / n_psd;

            let mut p_var_sum = 0.0;
            for p in psd_out.iter() {
                p_var_sum += (p[1] - p_mean).powi(2);
            }
            let p_var = p_var_sum / n_psd;
            let p_std = p_var.sqrt();
            let threshold = p_mean + sigma * p_std;

            peaks_out.extend(
                psd_out
                    .iter()
                    .filter(|p| p[1] > threshold)
                    .map(|p| (p[0], p[1])),
            );
            peaks_out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            peaks_out.truncate(5);
        }
    }
}

impl Default for TwapDetector {
    fn default() -> Self {
        Self::new()
    }
}

fn push_capped(deq: &mut VecDeque<f64>, value: f64, cap: usize) {
    deq.push_back(value);
    if deq.len() > cap {
        deq.pop_front();
    }
}
