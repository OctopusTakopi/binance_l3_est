use std::arch::x86_64::*;

pub const NULL_BLOCK: u32 = u32::MAX;

#[repr(align(32))]
#[derive(Clone, Copy, Debug)]
pub struct OrderBlock {
    pub sizes: [u64; 4],
    pub next_block_idx: u32,
    pub prev_block_idx: u32,
}

impl Default for OrderBlock {
    fn default() -> Self {
        Self {
            sizes: [0; 4],
            next_block_idx: NULL_BLOCK,
            prev_block_idx: NULL_BLOCK,
        }
    }
}

pub struct GlobalOrderPool {
    pub pool: Vec<OrderBlock>,
    free_head: u32,
}

impl GlobalOrderPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            pool: Vec::with_capacity(capacity),
            free_head: NULL_BLOCK,
        }
    }

    pub fn alloc_block(&mut self) -> u32 {
        if self.free_head != NULL_BLOCK {
            let idx = self.free_head;
            self.free_head = self.pool[idx as usize].next_block_idx;
            self.pool[idx as usize] = OrderBlock::default();
            idx
        } else {
            let idx = self.pool.len() as u32;
            self.pool.push(OrderBlock::default());
            idx
        }
    }

    pub fn free_block(&mut self, idx: u32) {
        self.pool[idx as usize].next_block_idx = self.free_head;
        self.free_head = idx;
    }

    #[inline(always)]
    pub fn get_block_mut(&mut self, idx: u32) -> &mut OrderBlock {
        &mut self.pool[idx as usize]
    }

    #[inline(always)]
    pub fn get_block(&self, idx: u32) -> &OrderBlock {
        &self.pool[idx as usize]
    }
}

#[derive(Clone, Copy)]
pub struct QueuePtrs {
    pub head_block: u32,
    pub head_offset: u32,
    pub tail_block: u32,
    pub tail_offset: u32,
}

impl Default for QueuePtrs {
    fn default() -> Self {
        Self {
            head_block: NULL_BLOCK,
            head_offset: 0,
            tail_block: NULL_BLOCK,
            tail_offset: 0,
        }
    }
}

impl QueuePtrs {
    pub fn new(pool: &mut GlobalOrderPool) -> Self {
        let block_idx = pool.alloc_block();
        Self {
            head_block: block_idx,
            head_offset: 0,
            tail_block: block_idx,
            tail_offset: 0,
        }
    }

    #[inline(always)]
    pub fn is_empty(&mut self, pool: &mut GlobalOrderPool) -> bool {
        self.first_size(pool);
        self.head_block == NULL_BLOCK
            || (self.head_block == self.tail_block && self.head_offset == self.tail_offset)
    }

    #[inline(always)]
    pub fn push_back(&mut self, pool: &mut GlobalOrderPool, qty: u64) {
        if self.head_block == NULL_BLOCK {
            *self = Self::new(pool);
        }

        let block = pool.get_block_mut(self.tail_block);
        block.sizes[self.tail_offset as usize] = qty;
        self.tail_offset += 1;

        if self.tail_offset == 4 {
            let next_idx = pool.alloc_block();
            let old_tail = self.tail_block;
            pool.get_block_mut(old_tail).next_block_idx = next_idx;
            pool.get_block_mut(next_idx).prev_block_idx = old_tail;
            self.tail_block = next_idx;
            self.tail_offset = 0;
        }
    }

    #[inline(always)]
    pub fn first_size(&mut self, pool: &mut GlobalOrderPool) -> u64 {
        self.mut_first_size(pool, |_| {}).unwrap_or(0)
    }

    #[inline(always)]
    pub fn mut_first_size<F>(&mut self, pool: &mut GlobalOrderPool, mut f: F) -> Option<u64>
    where
        F: FnMut(&mut u64),
    {
        let mut curr_block = self.head_block;
        let mut curr_offset = self.head_offset;

        while curr_block != NULL_BLOCK
            && !(curr_block == self.tail_block && curr_offset == self.tail_offset)
        {
            let size = pool.get_block(curr_block).sizes[curr_offset as usize];
            if size > 0 {
                let mut sz = size;
                f(&mut sz);
                pool.get_block_mut(curr_block).sizes[curr_offset as usize] = sz;

                self.head_block = curr_block;
                self.head_offset = curr_offset;
                return Some(size);
            }
            curr_offset += 1;
            if curr_offset == 4 {
                let next_b = pool.get_block(curr_block).next_block_idx;
                if next_b != NULL_BLOCK {
                    pool.get_block_mut(next_b).prev_block_idx = NULL_BLOCK;
                }
                pool.free_block(curr_block);
                curr_block = next_b;
                curr_offset = 0;
            }
        }
        self.head_block = curr_block;
        self.head_offset = curr_offset;
        None
    }

    #[inline(always)]
    pub fn pop_front(&mut self, pool: &mut GlobalOrderPool) {
        self.mut_first_size(pool, |s| *s = 0);
    }

    #[inline(always)]
    pub fn pop_back(&mut self, pool: &mut GlobalOrderPool) -> Option<u64> {
        while self.head_block != NULL_BLOCK
            && !(self.head_block == self.tail_block && self.head_offset == self.tail_offset)
        {
            if self.tail_offset > 0 {
                self.tail_offset -= 1;
                let val = pool.get_block(self.tail_block).sizes[self.tail_offset as usize];
                if val > 0 {
                    pool.get_block_mut(self.tail_block).sizes[self.tail_offset as usize] = 0;
                    return Some(val);
                }
            } else {
                let prev = pool.get_block(self.tail_block).prev_block_idx;
                if prev != NULL_BLOCK {
                    pool.free_block(self.tail_block);
                    self.tail_block = prev;
                    pool.get_block_mut(self.tail_block).next_block_idx = NULL_BLOCK;
                    self.tail_offset = 4;
                } else {
                    break;
                }
            }
        }
        None
    }
}

pub struct QueueIter<'a> {
    pool: &'a GlobalOrderPool,
    curr_block: u32,
    curr_offset: u32,
    tail_block: u32,
    tail_offset: u32,
}

impl<'a> Iterator for QueueIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr_block != NULL_BLOCK
            && !(self.curr_block == self.tail_block && self.curr_offset == self.tail_offset)
        {
            let size = self.pool.get_block(self.curr_block).sizes[self.curr_offset as usize];
            self.curr_offset += 1;
            if self.curr_offset == 4 {
                self.curr_block = self.pool.get_block(self.curr_block).next_block_idx;
                self.curr_offset = 0;
            }
            if size > 0 {
                return Some(size);
            }
        }
        None
    }
}

impl QueuePtrs {
    pub fn iter<'a>(&self, pool: &'a GlobalOrderPool) -> QueueIter<'a> {
        QueueIter {
            pool,
            curr_block: self.head_block,
            curr_offset: self.head_offset,
            tail_block: self.tail_block,
            tail_offset: self.tail_offset,
        }
    }

    /// # Safety
    ///
    /// The CPU must support the AVX2 + BMI1 target features (this fn is gated by
    /// `#[target_feature]` and can only be called on hardware where those flags
    /// are present). The caller is also responsible for not aliasing `pool`
    /// elsewhere for the duration of the call.
    #[target_feature(enable = "avx2,bmi1")]
    pub unsafe fn remove_order_by_size_simd(
        &self,
        pool: &mut GlobalOrderPool,
        target_size: u64,
    ) -> bool {
        let mut curr_block_idx = self.head_block;
        let target_vec = _mm256_set1_epi64x(target_size as i64);

        while curr_block_idx != NULL_BLOCK {
            let block = pool.get_block_mut(curr_block_idx);
            let sizes_ptr = block.sizes.as_ptr() as *const __m256i;
            let sizes_vec = unsafe { _mm256_loadu_si256(sizes_ptr) };

            let cmp_mask = _mm256_cmpeq_epi64(sizes_vec, target_vec);
            let match_bits = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_mask));

            let mut bits = match_bits as u32;
            while bits != 0 {
                let match_idx = bits.trailing_zeros();
                bits &= bits - 1;

                let mut valid = true;
                if curr_block_idx == self.head_block && match_idx < self.head_offset {
                    valid = false;
                }
                if curr_block_idx == self.tail_block && match_idx >= self.tail_offset {
                    valid = false;
                }

                if valid && block.sizes[match_idx as usize] > 0 {
                    block.sizes[match_idx as usize] = 0;
                    return true;
                }
            }

            if curr_block_idx == self.tail_block {
                break;
            }
            curr_block_idx = block.next_block_idx;
        }
        false
    }
}
