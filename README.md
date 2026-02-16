# NIFTY 50 Trading Indicators Analyzer

A comprehensive Python tool that analyzes all NIFTY 50 stocks for trading opportunities based on technical indicators, chart patterns, resistance levels, and 52-week highs.

## Features

- **Real-time Data Fetching**: Pulls latest NIFTY 50 stock data using yfinance
- **Multiple Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Stochastic Oscillator
  - ADX (Average Directional Index)
  - ATR (Average True Range)
  - Moving Averages (SMA 20, 50, 200)
  
- **Advanced Analysis**:
  - 52-week high/low tracking
  - Resistance level detection
  - Chart pattern recognition
  - Trend analysis

- **Stock Filtering**: Identifies stocks meeting buy criteria with scoring system
- **Flyers Integration**: Ready to connect with Flyers trading account for execution

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Flyers credentials
```

## Usage

Run the analyzer:
```bash
python main.py
```

This will:
1. Fetch data for all 50 NIFTY stocks
2. Calculate technical indicators
3. Apply buy signal logic
4. Display qualified stocks sorted by score

## Output

The tool displays:
- Stock symbol and current price
- Analysis score (based on multiple indicators)
- Reasons for qualification (specific signals detected)

Example:
```
RELIANCE.NS    | Price:    2850.50 | Score: 7
  • RSI Oversold: 28.45
  • MACD Bullish
  • Golden Cross (20>50>200)
  • Near 52W High (2.5% away)
```

## Configuration

Edit `main.py` to adjust:
- `MIN_SCORE_THRESHOLD`: Minimum score for stock qualification (default: 4)
- Indicator timeperiods (RSI, MACD, Bollinger Bands, etc.)
- Buy signal logic conditions

## Next Steps

1. Add Flyers API integration for automated trading
2. Implement real-time alerts
3. Add backtesting module
4. Create web dashboard for visualization

## Requirements

- Python 3.8+
- See requirements.txt for dependencies

## Disclaimer

This tool is for educational purposes. Always do your own research before trading.
