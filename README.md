# NIFTY-50-Prediction-Model Using Machine Learning 
This workspace contains a Python script that fetches live NIFTY 50 data, computes:
- 50-day EMA
- 100-day EMA
- 200-day EMA
- 14-day RSI
- volume-based features

It then trains a simple ridge-regression model on historical daily data to estimate:
- 1-month forward return and price
- 1-year forward return and price

Using live `^NSEI` data fetched on `2026-04-19` for the latest trading session `2026-04-17`:
- Close: `24,353.55`
- EMA 50: `24,200.79`
- EMA 100: `24,682.41`
- EMA 200: `24,820.90`
- RSI 14: `57.11`
- Volume ratio vs 21-day average: `0.955`

Sample model output from that same run:
- 1-month prediction: `24,975.58` (`+2.55%`)
- 1-year prediction: `29,612.22` (`+21.59%`)

## Notes
- The script uses the Yahoo Finance chart endpoint for live data.
- The machine-learning model is a simple `numpy` ridge regression, not a trading system.
- This is not investment advice.
- Long-horizon accuracy numbers can look better than they really are because daily forward targets overlap.
- Index volume is a proxy and may differ from cash-market stock volume measurements
