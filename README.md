# BINANCE L3 Order Book Estimator

![Demo GIF](demo2.gif)

This project is a high-performance real-time visualization tool for the Binance order book. It estimates a Level 3 (L3) order book from Level 2 (L2) data streams to give you deep insights into market liquidity, order queue seniority, and participant behavior.

## Key Features

*   **Real-time Order Book Heatmap**
    Visualize market depth with a dynamic heatmap. Easily spot liquidity clusters and identify major support and resistance levels.
    
*   **Interactive Analytics & Zooming**
    Analyze the market with multi-axis rendering. Right-click and drag on liquidity charts to zoom into specific price ranges for precise inspection of the order book.

*   **Microstructure Health Dashboard**
    Monitor market health in real-time with a dedicated dashboard displaying key indicators:
    *   **Order-to-Trade Ratio (OTR)** to track liquidity provision density.
    *   **Cancellation-to-Trade Ratio (CTR)** to spot cancellation velocity and potential spoofing.

*   **Market Participant Clustering**
    Automatically classify market participants in real-time. The tool uses K-Means clustering based on order sizes and trading frequencies to help you identify distinct trading behaviors.

*   **Algorithmic TWAP Detector**
    Spot hidden execution algorithms (Time-Weighted Average Price bots) in the market. The tool analyzes trade streams to identify distinct buy and sell side periodic trading patterns, revealing execution frequency and estimated volume limits.

*   **Advanced Order Queue Tracking**
    Get a highly accurate representation of the L3 order book. The tool intelligently tracks order priorities, partial fills, multi-order cancellations, and explicitly highlights massive "Whale" orders.

## Performance

The L3 estimator is designed for sustained multi-thousand-message-per-second
depth streams, using a flat, cache-friendly layout instead of a tree.

*   **Flat ring-buffer storage.**
    Per-level state lives in a fixed 65,536-tick (`1 << 16`) power-of-two ring
    buffer addressed directly by `price_ticks - base_price_ticks`. Every level
    lookup is a subtraction and a load — no pointer chasing, no per-node
    allocation, no B-tree rebalancing. The window slides (rotating the backing
    arrays in place) only when the mid-price drifts past a quarter of the
    window, so steady-state updates never rebuild the book.

*   **O(1) best-bid/ask and top-N ranking.**
    Each side carries a three-level hierarchical bitset (one bit per tick,
    summarised into 16 words and a single root word). Best bid/ask discovery is
    a handful of trailing/leading-zero instructions, and *"is this price in the
    top-N?"* is answered by a bounded hardware `POPCNT` over the bitmap instead
    of an O(N) walk of the price levels.

*   **Bounded-latency L3 maintenance.**
    Individual orders live in a pooled, arena-allocated block queue (AVX2
    exact-size matching on x86-64, portable scalar fallback elsewhere). Order
    cancellation searches only the most recent blocks, so cancel latency stays
    flat regardless of how long a level has been alive — the worst case is
    bounded, not proportional to queue age. Snapshot and gap recovery clear only
    the occupied levels by scanning the occupancy bitmap rather than all 65,536
    slots.

*   **Measured throughput.**
    On the bundled `criterion` benchmark, a fully applied depth update —
    including top-N ranking and L3 queue maintenance — runs in **~500 ns**,
    roughly **15× faster** than the previous iteration and within ~1.4× of a
    plain L2 `BTreeMap` that performs none of the L3 work. Per-update latency is
    now bounded rather than growing with queue depth.

*   **GPU-accelerated rendering.**
    The UI runs on eframe's wgpu backend (Vulkan on Linux/Windows, Metal on
    macOS) with a 75 FPS repaint throttle that decouples the draw cadence from
    the network ingest rate. Heatmap textures, liquidity-curve plots, and
    K-Means clustering results are cached and reused across frames.

## Usage

#### From Source

Ensure you have Rust installed ([rust-lang.org](https://www.rust-lang.org)).

1. Clone the repository:
   ```bash
   git clone https://github.com/OctopusTakopi/binance_l3_est.git
   cd binance_l3_est
   ```

2. Run the project:
   ```bash
   cargo run -r
   ```

The desktop UI uses WGPU hardware acceleration: Metal on macOS, Vulkan on
Linux, and Direct3D 12 on Windows. Tagged releases (`v*`) build and publish a
native binary for all three platforms.

#### UI Controls
*   **Microstructure Toggle**: Use the UI panel to enable/disable the OTR/CTR dashboard.
*   **TWAP Toggle**: Open the TWAP Detector window to monitor periodic trading activity.
*   **Heatmap Z-Score**: Adjust the standardization slider to highlight liquidity outliers.
*   **Zoom**: Right-click and drag on the liquidity charts to inspect specific price ranges.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
