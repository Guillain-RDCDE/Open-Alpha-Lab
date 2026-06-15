# Study 181 — Ultimate-Oscillator

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Threshold rule (UO<30 / UO>70): +14.4 bps/trade gross at HAC *t* = +1.50 over 681 pooled entries — positive but below the |*t*|≥2 bar. Best hold period (1d, *t* = +2.14) does not survive Bonferroni. Divergence rule: −7.1 bps, *t* = −1.80, actively negative. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Low turnover (≈7 signals/ticker/year) keeps costs manageable, but the signal is statistically uncertified. Any trading on an unproven edge is speculation. |
| **Low signal-to-noise?** | ![Confirmed](https://img.shields.io/badge/Low_signal--to--noise-8b949e?style=flat-square) | 681 entries over 5 tickers × 20 years; signals cluster in crisis episodes (2008, 2020); divergence rule fires 7.8× more often but is negative. |

> **In one sentence:** Williams' Ultimate Oscillator shows a positive but sub-threshold mean-reversion signal on its threshold rule and a negative result on its more famous divergence rule — weak evidence scattered across a sparse, crisis-clustered entry set that cannot be certified as real at any conventional significance level.

## What we tested

Larry Williams (1985) designed the Ultimate Oscillator (UO) to avoid the single-look-back problem
of earlier oscillators by weighting three periods (7/14/28 bars) of buying pressure (BP) relative
to true range (TR).  The folk rules: buy when UO < 30 ("oversold"), short when UO > 70
("overbought"), or trade *bullish/bearish divergences* (price confirms a new extreme but UO does
not).  We test both framings vs a **random-direction control on the same entries** — the only
honest answer to "does the oscillator add directional information over a coin?" — across five
liquid ETFs (SPY, QQQ, IWM, GLD, TLT), 20 years of daily data, sweeping five hold periods (1–20
days) with a Bonferroni correction for multiple comparisons.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the oscillator's logic in plain English, the threshold rule vs the coin, why the divergence rule disappoints, the hold-period trap |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-ticker HAC *t*, Bonferroni correction, hold-period sweep, cost analysis, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`ultimate_oscillator/`](ultimate_oscillator/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
