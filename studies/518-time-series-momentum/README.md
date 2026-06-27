# Study 518 -- Time-Series-Momentum

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> "Trend everywhere." An asset's OWN trailing 12-month return sign predicts its next month --
> in equities, bonds, commodities, FX and gold alike. Buy what is going up, short what is going
> down, risk-balance, diversify. Does it still pay on a small modern ETF basket?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- does an asset's own trend predict its next month? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The vol-scaled cross-asset book earns **+3.09%/yr** gross at HAC *t* = **+1.611**, and the sign-shuffle placebo *p* = **0.005** says the own-trend link is *real structure* -- far better than randomly flipping each position's sign. But the canonical 12-month one-sample *t* **does not clear the desk's bar of 2**, and it is fragile to the lookback (2.02 / 1.61 / 0.45 at 9 / 12 / 15 months). Strong literature plus a sub-2 *t* is **WEAK**, never REAL. **Survivorship is named here**: the basket is ETFs *still trading in 2026* -- though TSMOM's own-sign book shorts a falling ETF rather than deleting it, so the bias is far milder than a cross-sectional equity sort. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Net of 10bps x turnover + 50bps/yr borrow the book still earns **+2.68%/yr** at a **0.34 net Sharpe** with a contained **-19.5%** drawdown -- genuinely positive net, unlike most factors here. But the edge rests on persistent shorting (~50% short notional + recall risk), leveraged inverse-vol sizing, and 12-asset diversification a retail book cannot cheaply hold; with a signal *t* below 2 it is a thin, fragile carry, not a robustly investable one. |
| **Vol-scaling vs equal-notional?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Inverse-vol risk-balancing roughly **doubles the Sharpe** (0.39 vs 0.20) and the *t* (1.61 vs 0.79) and **halves the drawdown** (-19% vs -25%) versus a naive sign-only book. The Moskowitz-Ooi-Pedersen insight -- risk-balancing across asset classes turns own-asset trend into a diversified premium -- holds on this tape. |

> **In one sentence:** Moskowitz-Ooi-Pedersen time-series momentum -- own-asset trend, vol-scaled
> and diversified across equity / bond / commodity / FX / gold ETFs -- shows *real* structure
> (placebo *p* = 0.005) and a positive net return (+2.68%/yr, 0.34 Sharpe), but the canonical
> 12-month one-sample HAC *t* of 1.61 does not clear the desk's bar of 2 and wobbles with the
> lookback, so on 213 modern months it lands Weak, not Real.

## What we tested

Moskowitz, Ooi & Pedersen (2012): at each month-end, take the SIGN of each asset's own trailing
12-month return (+1 if up, -1 if down), size the position inverse to its recent realised
volatility toward a common risk target (capped leverage), and average across a diversified
**12-ETF cross-asset basket** -- equity (SPY, EFA, EEM, IWM), bonds (TLT, IEF, LQD), commodities
(DBC, USO), gold (GLD) and FX (UUP, FXE). We prove the apparatus on a deterministic synthetic
panel with a *baked-in* own-asset trend (and a no-trend null that earns nothing), then run the
book on **213 holding months** (yfinance daily adjusted-close, 2008--2025) at both the vol-scaled
and the equal-notional sizing. One execution lag (form on the month-end close, hold the next
month -- no same-bar fill, no look-ahead; the vol estimate uses only past data); costs of 10bps x
turnover per rebalance plus a 50bps/yr borrow on the net short leg; a sign-shuffle placebo null;
survivorship named on the **signal** axis (mild here, by construction).

This is **distinct** from [Study 507 -- Cross-Sectional-Momentum](../507-cross-sectional-momentum)
(rank assets *against each other*) and from [Study 110 -- Faber-Timing](../110-faber-timing)
(price-vs-SMA, *long-or-flat*, single asset). Here the signal is the *sign of the asset's own
trailing return*, the book is *long AND short*, and positions are *vol-scaled and diversified*.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | trend in plain language, why "buy what's going up across everything" diversifies, the synthetic control, the real result, and the honest Weak verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the own-sign signal, inverse-vol construction, HAC inference, the sign-shuffle placebo, the lookback-fragility sweep, vol-scaled-vs-equal-notional, year-by-year crisis alpha, equity curve and drawdown, gross-vs-net |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`time_series_momentum/`](time_series_momentum/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
