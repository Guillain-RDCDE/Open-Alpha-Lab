# Study 429 — Mass Index 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the bulge predict a reversal? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across **36 bulges** in 21.4y of SPY, the raw forward return **matches the base rate** at every horizon (it even *trails* it at 5/10/20 days; the lone 40-day *t* = 2.30 is just market drift, *t* vs base +0.59). The directional trend-fade is suggestive (5-day *t* = **+1.95**) but **never clears t ≥ 2**, sits near a coin-flip win-rate, and **flips sign across the panel** (QQQ −0.62%, GLD −1.24%). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The post-bulge fade scores net Sharpe **+0.75** vs buy-and-hold **+0.64**, but the HAC *t* of the daily difference is only **+0.67** — insignificant. The rule is long **90%** of the time and trades ~2.5×/yr: it's buy-and-hold with an occasional defensive tilt, not a separable edge. |
| **"The range bulge calls a reversal"?** | ![Busted](https://img.shields.io/badge/Calls_reversals%3F-Busted-8b949e?style=flat-square) | The bulge is **volatility clustering**, not a reversal oracle: forward returns after it are statistically indistinguishable from the base rate and the directional version doesn't clear the bar — while a synthetic control with a *planted* reversal lights up (*t* = +3.94), proving the harness would have caught a real one. |

> **In one sentence:** Donald Dorsey's Mass Index "reversal bulge" (rise above 27, fall back below 26.5) fires only ~1.7 times a year on SPY, and when it does the next 5–40 days look exactly like any other stretch of tape — the raw post-bulge return matches the base rate, the trend-fade version peaks at a sub-significant *t* = 1.95 and flips sign on QQQ and gold, and turning it into a timing rule just reproduces buy-and-hold (HAC *t* = +0.67) — so the bulge is a volatility-clustering detector wearing a reversal-oracle label, even though a synthetic positive control proves the harness *would* have banked a real reversal.

## What we tested

We compute Dorsey's **Mass Index** — the 25-day sum of EMA₉(high−low) / EMA₉(EMA₉(high−low)) — on daily SPY OHLCV (yfinance, auto-adjusted, 2005→2026, cache-first), and fire the folk **reversal bulge** the day the index rises above 27 and falls back below 26.5. We test it two ways. **(1) An event study:** the forward **5 / 10 / 20 / 40-day** return after each of the 36 bulges, measured against the **unconditional base rate**, both raw and **signed against the prevailing trend** (a real reversal makes the trend-fade pay), with a one-sample *t*, a Welch *t* vs the base rate, and a 5,000-draw equal-size random-trigger placebo. **(2) A timing rule:** a daily **post-bulge fade** raced **NET** of one-way costs × NAV against buy-and-hold, excess-of-cash to excess-of-cash, with a one-day execution lag and a HAC (Newey-West) *t* on the daily difference. A deterministic synthetic panel with a *planted* post-bulge reversal confirms the engine lights up when the effect is real and stays dark when it isn't — so the flat SPY result is a true negative, not a broken pipeline.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Mass Index measures, why a "range bulge" is really just volatility clustering, what actually happens in the 5–40 days after a bulge vs a normal stretch, and why a timing rule on it just turns back into buy-and-hold — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the bulge event study (forward return vs base rate, raw & trend-signed), one-sample & Welch *t* + a rare-event placebo, the net-Sharpe fade race with a HAC *t* of the daily difference, the cost/variant sweeps, the 5-ETF panel, and a synthetic planted-reversal control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`mass_index/`](mass_index/). Real tape is SPY daily OHLCV (auto-adjusted = total-return proxy); the timing race is **net** of 1 bp one-way × NAV turnover, **excess-of-cash vs excess-of-cash**, one-day lag. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
