use crate::types::{BookTicker, DepthUpdate, Trade};
use crate::utils::SymbolStr;
use std::mem::size_of;
use zerocopy::byteorder::LittleEndian;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

type U16LE = zerocopy::byteorder::U16<LittleEndian>;
type U32LE = zerocopy::byteorder::U32<LittleEndian>;
type I64LE = zerocopy::byteorder::I64<LittleEndian>;

const TEMPLATE_TRADES: u16 = 10000;
const TEMPLATE_BEST_BID_ASK: u16 = 10001;
const TEMPLATE_DEPTH_DIFF: u16 = 10003;

#[derive(Debug)]
pub enum SpotSbeEvent {
    Trades(Vec<Trade>),
    BestBidAsk(BookTicker),
    DepthDiff(DepthUpdate),
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct MessageHeader {
    block_length: U16LE,
    template_id: U16LE,
    schema_id: U16LE,
    version: U16LE,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct GroupSizeEncoding {
    block_length: U16LE,
    num_in_group: U32LE,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct GroupSize16Encoding {
    block_length: U16LE,
    num_in_group: U16LE,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct TradesBlock {
    event_time: I64LE,
    transact_time: I64LE,
    price_exponent: i8,
    qty_exponent: i8,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct TradeEntry {
    id: I64LE,
    price: I64LE,
    qty: I64LE,
    is_buyer_maker: u8,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct BestBidAskBlock {
    event_time: I64LE,
    book_update_id: I64LE,
    price_exponent: i8,
    qty_exponent: i8,
    bid_price: I64LE,
    bid_qty: I64LE,
    ask_price: I64LE,
    ask_qty: I64LE,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct DepthDiffBlock {
    event_time: I64LE,
    first_book_update_id: I64LE,
    last_book_update_id: I64LE,
    price_exponent: i8,
    qty_exponent: i8,
}

#[derive(FromBytes, IntoBytes, Immutable, KnownLayout, Unaligned, Debug, Copy, Clone)]
#[repr(C)]
struct DepthEntry {
    price: I64LE,
    qty: I64LE,
}

pub fn parse_spot_sbe_frame(payload: &[u8]) -> Option<SpotSbeEvent> {
    let (header, rest) = MessageHeader::read_from_prefix(payload).ok()?;
    let block_length = header.block_length.get() as usize;

    match header.template_id.get() {
        TEMPLATE_TRADES => parse_trades(rest, block_length),
        TEMPLATE_BEST_BID_ASK => parse_best_bid_ask(rest, block_length),
        TEMPLATE_DEPTH_DIFF => parse_depth_diff(rest, block_length),
        _ => None,
    }
}

fn parse_trades(payload: &[u8], block_length: usize) -> Option<SpotSbeEvent> {
    let (block, rest) = read_root_block::<TradesBlock>(payload, block_length)?;
    let (group_header, rest) = GroupSizeEncoding::read_from_prefix(rest).ok()?;
    let entry_count = group_header.num_in_group.get() as usize;
    let group_block_length = group_header.block_length.get() as usize;
    if group_block_length < size_of::<TradeEntry>() {
        return None;
    }

    let entries_len = entry_count.checked_mul(group_block_length)?;
    let (entries, rest) = rest.split_at_checked(entries_len)?;
    let symbol = parse_symbol(rest)?;
    let mut trades = Vec::with_capacity(entry_count);

    for chunk in entries.chunks_exact(group_block_length) {
        let entry = TradeEntry::read_from_prefix(chunk).ok()?.0;
        trades.push(Trade {
            event_type: "trade".to_string(),
            event_time: us_to_ms(block.event_time.get()),
            symbol: symbol.clone(),
            trade_id: entry.id.get() as u64,
            price: mantissa64_to_f64(entry.price.get(), block.price_exponent),
            quantity: mantissa64_to_f64(entry.qty.get(), block.qty_exponent),
            order_type: "LIMIT".to_string(),
            transaction_time: us_to_ms(block.transact_time.get()),
            is_buyer_maker: entry.is_buyer_maker != 0,
        });
    }

    Some(SpotSbeEvent::Trades(trades))
}

fn parse_best_bid_ask(payload: &[u8], block_length: usize) -> Option<SpotSbeEvent> {
    let (block, rest) = read_root_block::<BestBidAskBlock>(payload, block_length)?;
    let symbol = parse_symbol(rest)?;
    Some(SpotSbeEvent::BestBidAsk(BookTicker {
        update_id: block.book_update_id.get() as u64,
        symbol,
        best_bid_price: mantissa64_to_f64(block.bid_price.get(), block.price_exponent),
        best_bid_qty: mantissa64_to_f64(block.bid_qty.get(), block.qty_exponent),
        best_ask_price: mantissa64_to_f64(block.ask_price.get(), block.price_exponent),
        best_ask_qty: mantissa64_to_f64(block.ask_qty.get(), block.qty_exponent),
        transaction_time: us_to_ms(block.event_time.get()),
        event_time: us_to_ms(block.event_time.get()),
    }))
}

fn parse_depth_diff(payload: &[u8], block_length: usize) -> Option<SpotSbeEvent> {
    let (block, rest) = read_root_block::<DepthDiffBlock>(payload, block_length)?;
    let (bids, rest) = parse_depth_levels(rest, block.price_exponent, block.qty_exponent)?;
    let (asks, rest) = parse_depth_levels(rest, block.price_exponent, block.qty_exponent)?;
    let symbol = parse_symbol(rest)?;

    Some(SpotSbeEvent::DepthDiff(DepthUpdate {
        event_type: "depthUpdate".to_string(),
        event_time: us_to_ms(block.event_time.get()),
        transaction_time: us_to_ms(block.event_time.get()),
        symbol,
        capital_u: block.first_book_update_id.get() as u64,
        small_u: block.last_book_update_id.get() as u64,
        pu: None,
        b: bids,
        a: asks,
    }))
}

fn parse_depth_levels(payload: &[u8], price_e: i8, qty_e: i8) -> Option<(Vec<[f64; 2]>, &[u8])> {
    let (group_header, rest) = GroupSize16Encoding::read_from_prefix(payload).ok()?;
    let entry_count = group_header.num_in_group.get() as usize;
    let block_length = group_header.block_length.get() as usize;
    if entry_count == 0 {
        return Some((Vec::new(), rest));
    }
    if block_length < size_of::<DepthEntry>() {
        return None;
    }

    let entries_len = entry_count.checked_mul(block_length)?;
    let (entries, rest) = rest.split_at_checked(entries_len)?;
    let mut levels = Vec::with_capacity(entry_count);

    for chunk in entries.chunks_exact(block_length) {
        let entry = DepthEntry::read_from_prefix(chunk).ok()?.0;
        levels.push([
            mantissa64_to_f64(entry.price.get(), price_e),
            mantissa64_to_f64(entry.qty.get(), qty_e),
        ]);
    }

    Some((levels, rest))
}

fn parse_symbol(payload: &[u8]) -> Option<SymbolStr> {
    let mut offset = 0;
    let symbol = get_var_str_u8(payload, &mut offset);
    if symbol.is_empty() {
        return None;
    }
    let symbol = std::str::from_utf8(symbol).ok()?;
    Some(SymbolStr::from(symbol))
}

fn read_root_block<T: FromBytes>(payload: &[u8], block_length: usize) -> Option<(T, &[u8])> {
    let (block, rest) = payload.split_at_checked(block_length)?;
    let parsed = T::read_from_prefix(block).ok()?.0;
    Some((parsed, rest))
}

fn get_var_str_u8<'a>(data: &'a [u8], offset: &mut usize) -> &'a [u8] {
    if *offset + 1 > data.len() {
        return &[];
    }
    let len = data[*offset] as usize;
    *offset += 1;
    if *offset + len > data.len() {
        return &[];
    }
    let s = &data[*offset..*offset + len];
    *offset += len;
    s
}

fn mantissa64_to_f64(mantissa: i64, exponent: i8) -> f64 {
    (mantissa as f64) * 10f64.powi(exponent as i32)
}

fn us_to_ms(value: i64) -> u64 {
    value.max(0) as u64 / 1_000
}
