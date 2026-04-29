use crate::engine::simd_queue::{GlobalOrderPool, NULL_BLOCK, QueuePtrs};

pub const WINDOW_SIZE: usize = 1 << 16; // 65536 ticks
pub const QTY_MULTIPLIER: f64 = 1e8;

/// Hierarchical bitset for fast BBO discovery.
#[derive(Clone)]
pub struct Bitset {
    pub l2: [u64; 1024],
    pub l1: [u64; 16],
    pub l0: u64,
}

impl Default for Bitset {
    fn default() -> Self {
        Self::new()
    }
}

impl Bitset {
    pub fn new() -> Self {
        Self {
            l2: [0; 1024],
            l1: [0; 16],
            l0: 0,
        }
    }

    #[inline(always)]
    pub fn set(&mut self, idx: usize) {
        let i2 = idx >> 6;
        let b2 = 1u64 << (idx & 63);
        self.l2[i2] |= b2;

        let i1 = i2 >> 6;
        let b1 = 1u64 << (i2 & 63);
        self.l1[i1] |= b1;

        self.l0 |= 1u64 << i1;
    }

    #[inline(always)]
    pub fn clear(&mut self, idx: usize) {
        let i2 = idx >> 6;
        let b2 = 1u64 << (idx & 63);
        self.l2[i2] &= !b2;

        if self.l2[i2] == 0 {
            let i1 = i2 >> 6;
            let b1 = 1u64 << (i2 & 63);
            self.l1[i1] &= !b1;

            if self.l1[i1] == 0 {
                self.l0 &= !(1u64 << i1);
            }
        }
    }

    #[inline(always)]
    pub fn find_first(&self) -> Option<usize> {
        if self.l0 == 0 {
            return None;
        }
        let i1 = self.l0.trailing_zeros() as usize;
        let i2_base = self.l1[i1].trailing_zeros() as usize;
        let i2 = (i1 << 6) + i2_base;
        let bit = self.l2[i2].trailing_zeros() as usize;
        Some((i2 << 6) + bit)
    }

    #[inline(always)]
    pub fn find_last(&self) -> Option<usize> {
        if self.l0 == 0 {
            return None;
        }
        let i1 = 63 - self.l0.leading_zeros() as usize;
        let i2_base = 63 - self.l1[i1].leading_zeros() as usize;
        let i2 = (i1 << 6) + i2_base;
        let bit = 63 - self.l2[i2].leading_zeros() as usize;
        Some((i2 << 6) + bit)
    }

    #[inline(always)]
    pub fn is_set(&self, idx: usize) -> bool {
        if idx >= WINDOW_SIZE {
            return false;
        }
        let i2 = idx >> 6;
        (self.l2[i2] & (1u64 << (idx & 63))) != 0
    }

    #[inline(always)]
    pub fn find_next(&self, start_idx: usize) -> Option<usize> {
        if start_idx >= WINDOW_SIZE {
            return None;
        }

        let i2_idx = start_idx >> 6;
        let bit_in_i2 = start_idx & 63;

        // 1. Check current L2 word
        let word2 = self.l2[i2_idx] & (!0u64 << bit_in_i2);
        if word2 != 0 {
            return Some((i2_idx << 6) + word2.trailing_zeros() as usize);
        }

        // 2. Search hierarchy
        let i1_idx = i2_idx >> 6;
        let bit_in_i1 = i2_idx & 63;

        // Next words in current L1
        if bit_in_i1 < 63 {
            let word1 = self.l1[i1_idx] & (!0u64 << (bit_in_i1 + 1));
            if word1 != 0 {
                let next_i2 = (i1_idx << 6) + word1.trailing_zeros() as usize;
                return Some((next_i2 << 6) + self.l2[next_i2].trailing_zeros() as usize);
            }
        }

        // Next blocks in L0
        if i1_idx < 15 {
            let word0 = self.l0 & (!0u64 << (i1_idx + 1));
            if word0 != 0 {
                let next_i1 = word0.trailing_zeros() as usize;
                let next_i2_base = self.l1[next_i1].trailing_zeros() as usize;
                let next_i2 = (next_i1 << 6) + next_i2_base;
                return Some((next_i2 << 6) + self.l2[next_i2].trailing_zeros() as usize);
            }
        }

        None
    }

    #[inline(always)]
    pub fn find_prev(&self, start_idx: usize) -> Option<usize> {
        if start_idx >= WINDOW_SIZE {
            return self.find_last();
        }
        let i2_idx = start_idx >> 6;
        let bit_in_i2 = start_idx & 63;

        // 1. Check current L2 word
        let mask2 = if bit_in_i2 == 63 {
            !0u64
        } else {
            (1u64 << (bit_in_i2 + 1)) - 1
        };
        let word2 = self.l2[i2_idx] & mask2;
        if word2 != 0 {
            return Some((i2_idx << 6) + 63 - (word2.leading_zeros() as usize));
        }

        // 2. Search hierarchy
        let i1_idx = i2_idx >> 6;
        let bit_in_i1 = i2_idx & 63;

        // Previous words in current L1
        if bit_in_i1 > 0 {
            let mask1 = (1u64 << bit_in_i1) - 1;
            let word1 = self.l1[i1_idx] & mask1;
            if word1 != 0 {
                let prev_i2 = (i1_idx << 6) + 63 - (word1.leading_zeros() as usize);
                return Some((prev_i2 << 6) + 63 - (self.l2[prev_i2].leading_zeros() as usize));
            }
        }

        // Previous blocks in L0
        if i1_idx > 0 {
            let mask0 = (1u64 << i1_idx) - 1;
            let word0 = self.l0 & mask0;
            if word0 != 0 {
                let prev_i1 = 63 - (word0.leading_zeros() as usize);
                let prev_i2_base = 63 - (self.l1[prev_i1].leading_zeros() as usize);
                let prev_i2 = (prev_i1 << 6) + prev_i2_base;
                return Some((prev_i2 << 6) + 63 - (self.l2[prev_i2].leading_zeros() as usize));
            }
        }

        None
    }

    pub fn reset(&mut self) {
        self.l2.fill(0);
        self.l1.fill(0);
        self.l0 = 0;
    }

    pub fn rebuild_hierarchies(&mut self) {
        self.l1.fill(0);
        self.l0 = 0;
        for i2 in 0..1024 {
            if self.l2[i2] != 0 {
                let i1 = i2 >> 6;
                self.l1[i1] |= 1u64 << (i2 & 63);
                self.l0 |= 1u64 << i1;
            }
        }
    }
}

pub struct FastOrderBook {
    pub queues: Vec<QueuePtrs>,
    pub total_qtys: Vec<u64>,
    pub bids_bitset: Bitset,
    pub asks_bitset: Bitset,
    pub base_price_ticks: i64,
    pub pool: GlobalOrderPool,
    pub tick_size: f64,
    pub initialized: bool,
}

impl FastOrderBook {
    pub fn new(tick_size: f64, base_price: f64) -> Self {
        let base_ticks = (base_price / tick_size).round() as i64 - (WINDOW_SIZE as i64 / 2);
        Self {
            queues: vec![QueuePtrs::default(); WINDOW_SIZE],
            total_qtys: vec![0; WINDOW_SIZE],
            bids_bitset: Bitset::new(),
            asks_bitset: Bitset::new(),
            base_price_ticks: base_ticks,
            pool: GlobalOrderPool::new(100000),
            tick_size,
            initialized: false,
        }
    }

    /// Reset the level at `idx` if the opposite side currently owns it.
    /// Prevents stale qty/queue from leaking across a bid<->ask transition.
    #[inline(always)]
    pub fn ensure_side(&mut self, idx: usize, is_bid: bool) {
        let opposite_set = if is_bid {
            self.asks_bitset.is_set(idx)
        } else {
            self.bids_bitset.is_set(idx)
        };
        if opposite_set {
            self.bids_bitset.clear(idx);
            self.asks_bitset.clear(idx);
            self.total_qtys[idx] = 0;
            self.clear_queue(idx);
        }
    }

    #[inline(always)]
    pub fn f64_to_u64(&self, val: f64) -> u64 {
        (val * QTY_MULTIPLIER).round() as u64
    }

    #[inline(always)]
    pub fn u64_to_f64(&self, val: u64) -> f64 {
        val as f64 / QTY_MULTIPLIER
    }

    #[inline(always)]
    pub fn price_to_ticks(&self, price: f64) -> i64 {
        (price / self.tick_size).round() as i64
    }

    #[inline(always)]
    pub fn ticks_to_price(&self, ticks: i64) -> f64 {
        ticks as f64 * self.tick_size
    }

    #[inline(always)]
    pub fn get_idx(&self, ticks: i64) -> Option<usize> {
        let rel = ticks - self.base_price_ticks;
        if rel >= 0 && rel < WINDOW_SIZE as i64 {
            Some(rel as usize)
        } else {
            None
        }
    }

    pub fn process_update(&mut self, price_ticks: i64, new_total_f64: f64, is_bid: bool) {
        let new_total_u64 = self.f64_to_u64(new_total_f64);
        let idx = match self.get_idx(price_ticks) {
            Some(i) => i,
            None => return,
        };

        self.ensure_side(idx, is_bid);

        let old_total = self.total_qtys[idx];
        if new_total_u64 > old_total {
            let diff = new_total_u64 - old_total;
            if self.queues[idx].head_block == NULL_BLOCK {
                self.queues[idx] = QueuePtrs::new(&mut self.pool);
            }
            self.queues[idx].push_back(&mut self.pool, diff);
            self.total_qtys[idx] = new_total_u64;
            if is_bid {
                self.bids_bitset.set(idx);
            } else {
                self.asks_bitset.set(idx);
            }
        } else if new_total_u64 < old_total {
            let mut to_remove = old_total - new_total_u64;
            while to_remove > 0 && !self.queues[idx].is_empty(&mut self.pool) {
                let first = self.queues[idx].first_size(&mut self.pool);
                if first <= to_remove {
                    self.queues[idx].pop_front(&mut self.pool);
                    to_remove -= first;
                } else {
                    self.queues[idx].mut_first_size(&mut self.pool, |s| *s -= to_remove);
                    to_remove = 0;
                }
            }
            self.total_qtys[idx] = new_total_u64;
            if new_total_u64 == 0 {
                if is_bid {
                    self.bids_bitset.clear(idx);
                } else {
                    self.asks_bitset.clear(idx);
                }
                self.clear_queue(idx);
            }
        }
    }

    pub fn slide_window(&mut self, target_ticks: i64, force: bool) {
        let current_center = self.base_price_ticks + (WINDOW_SIZE as i64 / 2);
        let diff = (target_ticks - current_center).abs();

        if !force && self.initialized && diff < (WINDOW_SIZE as i64 / 4) {
            return;
        }

        let new_base = target_ticks - (WINDOW_SIZE as i64 / 2);
        let shift = new_base - self.base_price_ticks;

        if shift == 0 {
            self.initialized = true;
            return;
        }

        if shift.abs() >= WINDOW_SIZE as i64 {
            self.reset_with_base(new_base);
            return;
        }

        let shift_abs = shift.unsigned_abs() as usize;
        if shift > 0 {
            for i in 0..shift_abs {
                self.clear_queue(i);
            }
            self.total_qtys.rotate_left(shift_abs);
            self.queues.rotate_left(shift_abs);

            for i in (WINDOW_SIZE - shift_abs)..WINDOW_SIZE {
                self.total_qtys[i] = 0;
                self.queues[i] = QueuePtrs::default();
            }

            Self::shift_bitset_left(&mut self.bids_bitset, shift_abs);
            Self::shift_bitset_left(&mut self.asks_bitset, shift_abs);
        } else {
            for i in (WINDOW_SIZE - shift_abs)..WINDOW_SIZE {
                self.clear_queue(i);
            }
            self.total_qtys.rotate_right(shift_abs);
            self.queues.rotate_right(shift_abs);

            for i in 0..shift_abs {
                self.total_qtys[i] = 0;
                self.queues[i] = QueuePtrs::default();
            }

            Self::shift_bitset_right(&mut self.bids_bitset, shift_abs);
            Self::shift_bitset_right(&mut self.asks_bitset, shift_abs);
        }

        self.base_price_ticks = new_base;
        self.bids_bitset.rebuild_hierarchies();
        self.asks_bitset.rebuild_hierarchies();
        self.initialized = true;
    }

    fn shift_bitset_left(bs: &mut Bitset, amount: usize) {
        let u64_shift = amount >> 6;
        let bit_shift = amount & 63;
        if bit_shift == 0 {
            bs.l2.copy_within(u64_shift..1024, 0);
            bs.l2[1024 - u64_shift..].fill(0);
        } else {
            let inv_bit_shift = 64 - bit_shift;
            for i in 0..(1024 - u64_shift - 1) {
                bs.l2[i] = (bs.l2[i + u64_shift] >> bit_shift)
                    | (bs.l2[i + u64_shift + 1] << inv_bit_shift);
            }
            if 1024 > u64_shift {
                bs.l2[1024 - u64_shift - 1] = bs.l2[1023] >> bit_shift;
            }
            bs.l2[1024 - u64_shift..].fill(0);
        }
    }

    fn shift_bitset_right(bs: &mut Bitset, amount: usize) {
        let u64_shift = amount >> 6;
        let bit_shift = amount & 63;
        if bit_shift == 0 {
            bs.l2.copy_within(0..(1024 - u64_shift), u64_shift);
            bs.l2[..u64_shift].fill(0);
        } else {
            let inv_bit_shift = 64 - bit_shift;
            for i in (u64_shift + 1..1024).rev() {
                bs.l2[i] = (bs.l2[i - u64_shift] << bit_shift)
                    | (bs.l2[i - u64_shift - 1] >> inv_bit_shift);
            }
            bs.l2[u64_shift] = bs.l2[0] << bit_shift;
            bs.l2[..u64_shift].fill(0);
        }
    }

    pub fn clear_queue(&mut self, idx: usize) {
        let q = self.queues[idx];
        if q.head_block != NULL_BLOCK {
            let mut curr = q.head_block;
            while curr != NULL_BLOCK {
                let next = self.pool.get_block(curr).next_block_idx;
                self.pool.free_block(curr);
                if curr == q.tail_block {
                    break;
                }
                curr = next;
            }
            self.queues[idx] = QueuePtrs::default();
        }
    }

    pub fn reset_with_base(&mut self, new_base: i64) {
        self.base_price_ticks = new_base;
        self.bids_bitset.reset();
        self.asks_bitset.reset();
        for i in 0..WINDOW_SIZE {
            self.clear_queue(i);
        }
        self.total_qtys.fill(0);
        self.initialized = true;
    }

    pub fn best_bid(&self) -> Option<i64> {
        self.bids_bitset
            .find_last()
            .map(|idx| self.base_price_ticks + idx as i64)
    }

    pub fn best_ask(&self) -> Option<i64> {
        self.asks_bitset
            .find_first()
            .map(|idx| self.base_price_ticks + idx as i64)
    }
}
