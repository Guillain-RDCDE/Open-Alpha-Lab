# Study 680 — Disparity-Index

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | DI < 95 beats neutral by +73.4 bps/5d (Welch *t* = **+3.84**, NW *t* = **+2.50**) — clears the bar raw — but the DI > 105 leg has the **wrong sign** (+99.6 bps/5d, *t* = +4.30, i.e. overbought predicts *more* upside), and against a same-ticker random-*day* baseline the surviving oversold edge falls to *t* = **+1.42**. DI correlates **r = 0.84** with the plain trailing return — it is captured drift, not information. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The literal rule (buy oversold, short overbought) **loses money** net of costs: −17.0 bps/trade at 5 bps (HAC *t* = −0.94), not distinguishable from a random-direction coin on the same entries (delta *t* = −1.09). |
| **"More than plain short-term reversal?"** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | DI is a smoothed transform of the trailing N-day return (r = 0.84); buying the dip does not beat buying a random day of the same drifting stock (*t* = +1.42), and shorting the rip fights a real momentum effect. |

> **In one sentence:** the Disparity Index looks real at first glance — both its oversold *and* overbought extremes beat a neutral day (both *t* > 3.8) — but the overbought half has the wrong sign, the surviving oversold half evaporates against a random-day-in-the-same-stock control (*t* = 1.42), and the whole indicator is 84%-correlated with the plain trailing return it's dressed up to replace: a mirage built on captured bull-market drift, not a rubber band that snaps back.

## What we tested

DI(N) = 100 x Close / SMA(N), a Korean/Japanese technical-analysis staple: buy when the
close has stretched 5%+ below its 10-day average, sell/short when it has stretched 5%+
above. We test it on SPY + a five-name basket (QQQ, IWM, AAPL, TSLA, NVDA), 2003–2026,
with a conditional forward-return split (Welch + Newey-West), a zone-trigger trade
ledger vs a random-direction coin, a parameter grid, and the control that actually
decides it: **a same-ticker random-*day* baseline**, because this pooled universe carries
two decades of real drift a naive "beats neutral" split can't separate from genuine
reversion. **Dedup:** [329-one-month-reversal](../329-one-month-reversal/) is the direct
ancestor DI turns out to be a smoothed relabeling of (r = 0.84); it already found the
effect is bid-ask-bounce microstructure, dead since 2002.
[104-bollinger-reversion](../104-bollinger-reversion/) is the Western band-distance
cousin, found to have the identical drift-contamination pattern independently;
[137-mansfield-rs](../137-mansfield-rs/) is the opposite (trend-following) philosophy;
[679-psychological-line](../679-psychological-line/) is the nearest sibling by protocol
design (same universe, same conditional-split + zone-ledger shape, a different
oscillator).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "price is far from its average" sounds like a law of physics, and why buying the dip on a stock that's already rocketing isn't actually contrarian |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/NW conditional split, the wrong-signed overbought leg, the random-day drift control, the parameter grid, the reversal-correlation diagnostic, the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`disparity_index/`](disparity_index/). SPY/QQQ/IWM carry no survivorship (index
ETFs); the AAPL/TSLA/NVDA sleeve is a name-recognition pick, not a systematic historical
panel — named on the data layer. **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
