# BINANCE PERP L3 Order Book Estimator

![Demo GIF](demo2.gif)

This project is a high-performance real-time visualization tool for the Binance perpetual swap order book. It leverages **Execution-Aware Queue Dynamics (EAQD)** to estimate a Level 3 (L3) order book microstructure from Level 2 (L2) events and real-time trade streams, providing deep insights into order queue seniority and market participant behavior.

## Key Features

*   **Real-time Data Architecture**: Low-latency streaming of order book depth and trade events via Binance WebSocket API.
*   **Advanced L3 Estimation**: Moves beyond naive estimation to account for execution priority, cancellation behavior, and market regime.
*   **Microstructure Metrics Panel**: Dedicated real-time dashboard for high-resolution microstructure health indicators.
    *   **Order-to-Trade Ratio (OTR)**: Tracks liquidity provision intensity at Top-1 and Top-20 levels.
    *   **Cancellation-to-Trade Ratio (CTR)**: Monitors cancellation/spoofing velocity across sides.
*   **Dynamic Heatmap Visualization**: Standardized depth heatmap using Z-score normalization with interactive controls.
*   **Interactive Analytics**: Sweep/Liquidity-Cost windows with right-click drag-to-zoom and multi-axis rendering.
*   **K-Means Clustering**: Real-time classification of market participants based on order size and arrival patterns.

## Technical Core

The estimator utilizes several proprietary techniques to maintain a ground-truth-aligned L3 view:

### 1. Execution-Aware Queue Dynamics (EAQD)
Refined logic for inflow and outflow handling. Unlike static models, EAQD understands the difference between fills (FIFO consumption) and cancellations (LIFO/Priority-based reduction).

### 2. Deep Depth Fragmentation (DWF)
Also known as Statistical Order Flow Profiling (SOFP). This system fragments large L2 liquidity additions into multiple virtual orders based on a rolling distribution of market trade sizes.
*   **Whale Bypass**: If an addition is extremely large (e.g., > 20x average trade size), the system bypasses fragmentation, treating it as a single high-conviction "Whale" order.
*   **Robust Multi-Order Cancellation**: Combined with EAQD, the system now handles large LIFO cancellations that span multiple fragments, ensuring accurate queue reduction even when fragmented blocks are removed.

### 3. Marker-Triggered Queue Refining (MTQR)
Integrates the `@trade` stream as a synchronization pulse. When a trade occurs at a specific price, MTQR validates the maker's size against our estimated queue. If a trade exceeds our front-of-queue estimate, the model "snaps" to the ground truth and adjusts seniority accordingly.

### 4. Seniority Decay & Priority Reset
Handles partial fills and order modifications. Partial fills retain their queue position, while modifications that increase size or change price trigger a priority reset, accurately reflecting exchange matching engine logic.

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

#### UI Controls
*   **Microstructure Toggle**: Use the UI panel to enable/disable the OTR/CTR dashboard.
*   **Heatmap Z-Score**: Adjust the standardization slider to highlight liquidity outliers.
*   **Zoom**: Right-click and drag on the liquidity charts to inspect specific price ranges.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
