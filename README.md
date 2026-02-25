# BINANCE PERP L3 Order Book Estimator

![Demo GIF](demo2.gif)

This project is a high-performance real-time visualization tool for the Binance perpetual swap order book. It estimates a Level 3 (L3) order book from Level 2 (L2) data streams to give you deep insights into market liquidity, order queue seniority, and participant behavior.

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
*   **TWAP Toggle**: Open the TWAP Detector window to monitor periodic trading activity.
*   **Heatmap Z-Score**: Adjust the standardization slider to highlight liquidity outliers.
*   **Zoom**: Right-click and drag on the liquidity charts to inspect specific price ranges.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
