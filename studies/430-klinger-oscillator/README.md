# Study 430 — Klinger Oscillator 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the volume oscillator time SPY? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Klinger KVO(34,55,13) long/flat on SPY scores net Sharpe **+0.38** (excess-of-cash) vs buy-and-hold **+0.65** over 21.4y. The HAC *t* of the daily difference is **−2.87** — significantly **worse**, not better. Every variant (signal-line, long/short) is worse still; the long/short legs go negative. Nowhere near the desk's **t ≥ 2** bar. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | It loses the race at **every** cost level (0–5 bps) while trading ~22×/yr, and is beaten by a one-line **200-day SMA** (Sharpe **+0.78**, far smaller drawdown, a quarter of the turnover). There is no residual edge to deploy. |
| **"Volume leads price"?** | ![Busted](https://img.shields.io/badge/Volume_leads_price%3F-Busted-8b949e?style=flat-square) | A block-shuffled **random schedule** with the same time-in-market beats Klinger's actual timing **67%** of the time (median random Sharpe **+0.44** > Klinger's **+0.38**). The oscillator is a lagging difference-of-EMAs of volume — a filter that delays, not leads. |

> **In one sentence:** Klinger's "volume leads price" oscillator, turned into a long/flat timing rule on SPY over 21 years, lands **net Sharpe +0.38 against buy-and-hold's +0.65** (HAC *t* of the difference = −2.87, i.e. reliably worse), is beaten by a dumb 200-day moving average, and times the market no better than a coin with the same exposure — so the leading-indicator claim is busted on the tape, while a synthetic control proves the harness *would* have caught a real volume-leads-price effect.

## What we tested

We compute Stephen Klinger's **Volume Force** oscillator — KVO = EMA₃₄(VF) − EMA₅₅(VF), with a 13-EMA trigger — on daily SPY OHLCV (yfinance, auto-adjusted, 2005→2026, cache-first), and turn it into the folk **be-long when KVO > 0, be-flat otherwise** rule (plus signal-line and long/short variants). The position is read at the close of *t* and held over *t+1* (one execution lag); we book everything **excess of cash** and **net** of one-way costs × NAV turnover, then race the net Sharpe against **buy-and-hold** and the obvious simpler competitors (**200-day SMA**, **MACD**) so the "Klinger is better" claim is actually tested. The Signal axis uses a HAC (Newey-West) *t* of the daily strategy-minus-benchmark difference; the myth axis uses a 3,000-draw block-permutation placebo (shuffle *when* the rule is long, same time-in-market). A deterministic synthetic panel with a *planted* volume-leads-price effect confirms the engine lights up when the effect is real and stays dark when it isn't — so the dead result on SPY is a true negative, not a broken pipeline.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "volume leads price" means, the equity-curve race vs buy-and-hold and a dumb 200-day average, the coin-beats-it luck test, and why costs only make it worse — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | KVO(34,55,13) long/flat & long/short, net-Sharpe race excess-vs-excess, HAC *t* of the daily difference, a block-permutation placebo, the SMA/MACD head-to-head, the 6-ETF panel, and a synthetic volume-leads-price control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`klinger_oscillator/`](klinger_oscillator/). Real tape is SPY daily OHLCV (auto-adjusted = total-return proxy); the race is **net** of 1 bp one-way × NAV turnover, **excess-of-cash vs excess-of-cash**. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
