use std::arch::x86_64::*;

#[repr(align(32))]
#[derive(Clone, Copy, Debug)]
pub struct OrderBlock {
    pub sizes: [f64; 4],
    pub next_block_idx: u32,
    pub is_our_order_mask: u8,
    pub _pad: [u8; 27],
}

#[target_feature(enable = "avx2")]
pub unsafe fn test_simd() {
    let mut block = OrderBlock {
        sizes: [1.0, 2.0, 3.0, 0.0],
        next_block_idx: u32::MAX,
        is_our_order_mask: 0,
        _pad: [0; 27],
    };
    
    let target = _mm256_set1_pd(2.0);
    let sizes_ptr = block.sizes.as_ptr();
    let sizes_vec = _mm256_load_pd(sizes_ptr);
    
    let cmp = _mm256_cmp_pd::<_CMP_EQ_OQ>(sizes_vec, target);
    let match_bits = _mm256_movemask_pd(cmp);
    
    if match_bits != 0 {
        let idx = match_bits.trailing_zeros();
        block.sizes[idx as usize] = 0.0;
    }
    
    assert_eq!(block.sizes, [1.0, 0.0, 3.0, 0.0]);
}

fn main() {
    unsafe { test_simd(); }
}
