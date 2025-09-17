# Professional Stockbroker Assistant

*A practical, explainable DS/ML engine for forecasting, ML signals, portfolio optimization, RL trading simulation, and stress testing — wrapped in a clean Gradio UI.*

## What this is

An end-to-end quantitative assistant that **ingests market data**, **engineers features**, **forecasts prices**, **learns directional signals**, **optimizes a portfolio**, **simulates trading with an RL-style sandbox**, **stress-tests risk**, and **explains itself** with clear charts and reasoning. It’s built to be math-first and UI-second: the interface is deliberately minimal; the engine does the heavy lifting.

## Why it’s different

Most “AI trading” demos are UI-led and opaque. This project flips that: each block is transparent, modular, and independently testable. You can read the numbers behind every decision—forecasts, feature importances, allocation rationales, scenario impacts—so it’s clear **what** the system is saying and **why**.

## Core pipeline

### 1) Data ingestion & cleaning

* Pulls OHLCV with `yfinance`, auto-aligns dates, forward/back-fills small gaps, and computes daily returns (`Returns`) as the backbone for downstream stats.
* Caches intelligently to avoid repeated downloads in a session.

### 2) Feature engineering

* Technical set includes: `RSI`, `MACD`, `MACD_Signal`, `MACD_Hist`, moving averages (5/20/50), Bollinger Bands, rate of change (`ROC_5`, `ROC_20`), short/long vol (`Volatility_5`, `Volatility_20`), volume ratio, **market correlation** and **regime** flags.
* All features are forward-filled/back-filled and aligned to avoid leakage.

### 3) Forecasting (next-5-day horizon)

* **Trend Forecast (realistic):** 30-day linear regression with R²-weighted slope and volatility-scaled noise; bounds prevent absurd paths.
* **Mean Reversion:** Exponential pull to 60-day mean.
* **Exponential Smoothing (double):** Level + trend updates with α/β.
* **MA-Momentum:** Damped drift from MA(5) vs MA(20) spread.
* **ARIMA:** Safe `pmdarima.auto_arima` on recent 100 points when available.
* **Ensemble:** Weighted average across methods for a stable headline path with confidence shading.

### 4) ML predictions

* **Targets:** next-day price (regression) and direction (classification).
* **Models:** Random Forest (regressor + classifier) with scaling; Neural Network with LSTM/Attention and LightGBM ensemble.
* **Validation:** time-ordered split; report RMSE, accuracy, and **feature importances**; provide class probabilities to drive conviction.
* **Robustness:** strict NaN handling, feature availability checks, minimum sample rules.

### 5) Portfolio optimization

* Builds returns matrix; computes annualized return/vol, Sharpe, and drawdown per asset.
* **Optimizers:**

  * CVXPY mean-variance (if installed) with min/max weights and diversification constraints.
  * Scipy Sharpe maximization fallback with concentration penalty and multiple restarts.
  * Risk-parity / equal-weight as final safety net.
* **Outputs:** target weights, share counts, invested vs target \$, efficiency, portfolio-level expected return, vol, Sharpe, concentration (HHI), plus **plain-English reasoning** per asset.

### 6) RL-style trading sandbox (simulated)

* Constructs a state from key indicators (RSI, MACD, vol, ROC, regime, price vs MA, BB position).
* Simulates **conservative / moderate / aggressive** policies with buy/hold/sell logic, take-profit, stop-loss, holding period, and volume cues.
* Reports trades, win-rate, total return, avg win/loss, and a **confidence timeline**; recommendations summarize recent signals and backtest performance.

### 7) Stress testing

* Scenarios include **market crash (-20%)**, **volatility spike (3×)**, **mild correction (-10%)**, **bull rally (+15%)**, **stagflation**.
* Shocks mean/vol, recomputes portfolio return, vol, Sharpe, and VaR-style cuts; assigns an **impact severity** and explains scenario.

### 8) Advisory layer

* Combines forecasts, ML probabilities, RL signals, and stress results into per-ticker recommendations (BUY / HOLD / SELL variants) with confidence and rationale.
* Produces a portfolio-level summary: expected return, risk, Sharpe, diversification score, stress resilience, sentiment breakdown, and key insights.

## What you’ll see in the UI

* **Price & TA dashboard:** Candles, MA20/50, Bollinger Bands, RSI, MACD, volume.
* **Forecasts:** last 50 actuals + 5-day ensemble with volatility-scaled confidence band.
* **Portfolio:** pie of weights, \$ invested vs target, risk-return bubble chart, and a table with shares/price/efficiency—plus textual “why these allocations” notes.
* **RL trading:** price with buy/sell markers, position size area, cumulative P\&L, and a separate confidence panel.
* **Stress tests:** scenario bar charts and a risk-return map with VaR-scaled bubbles and a resilience summary.

## Tech stack

`Python`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `pmdarima`, `lightgbm` (optional), `plotly`, `gradio`, `yfinance`, and optional `cvxpy`. The app is designed to degrade gracefully (e.g., ARIMA/CVXPY off → safe fallbacks).

## Notes & guardrails

* **Education only.** Not investment advice. Markets are stochastic; all models are approximations.
* **Data caveats.** Yahoo! data can have gaps and corporate action quirks; the loader mitigates small issues but cannot eliminate vendor noise.
* **Performance.** Heavy features/optimizers may scale with the number of tickers and date span; the UI streams progress and keeps charts responsive.

## Acknowledgments

Built to be interrogable: every major decision is traceable back to a simple, testable component.

## Try it live

**Playground on Hugging Face Spaces:**
[https://huggingface.co/spaces/deadven7/stockbroker-assist](https://huggingface.co/spaces/deadven7/stockbroker-assist)
