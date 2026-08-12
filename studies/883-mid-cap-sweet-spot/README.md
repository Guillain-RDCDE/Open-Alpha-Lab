# Study 883 — Mid-Cap Sweet Spot 🎯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is mid-cap a risk-adjusted sweet spot, better than BOTH large & small? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Fails the "over BOTH" bar. On the excess-of-cash Sharpe race (2007-2026, BIL cash) mid sits **in the middle**: IJH **0.453**, *below* large SPY **0.542** and only just above small IWM **0.394** — both bootstrap Sharpe-advantage CIs span zero ([−0.225,+0.050] vs SPY, [−0.038,+0.169] vs IWM). The long-run return tilt is **sign-correct** (mid out-returned both: MDY − SPY **+0.95%/yr** over 1995-2026, IJH − SPY **+1.74%/yr**) but **never clears HAC *t* = 2** (best *t* = +1.19) and it **reverses** in the mega-cap era (MDY − SPY = **−3.5%/yr** in 2017-2026). A real but fragile, era-dependent tilt. *Short cash history: the Sharpe race misses the 1995-2006 mid heyday — named on Signal.* |
| **Tradability** — can you bank the sweet spot? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No costed dollar-neutral spread clears the bar. Long-mid/short-large nets **+1.00%/yr** but at net HAC *t* = **+0.68** — and it is exactly the leg that inverted post-2017; long-mid/short-small nets **−0.30%/yr** after 50 bps borrow + penny-ETF spreads. The Sharpe "edge" over small is a lower-return / similar-vol artifact, not a paid premium. |

> **In one sentence:** the "forgotten middle" is real folklore with a real *sign* — mid-caps
> did out-return both large and small over the long run — but the advantage **never reaches
> significance, sits below large-cap on a modern Sharpe basis, reverses in the last decade,
> and dies in costs**, so the honest read is a weak tilt, not a bankable sweet spot.

## What we tested

The claim: **mid-caps are the risk-adjusted sweet spot — a higher Sharpe than BOTH large
(SPY) and small (IWM)**. We race the mid-cap ETF (**IJH**, and **MDY** for the longer S&P
MidCap 400 tape) against SPY and IWM on **excess-of-cash** Sharpe (cash = **BIL** T-bills),
run a **Newey-West HAC *t*** on the pairwise daily return difference (the cash leg cancels in
a difference, so this reaches the full 1995→2026 MDY tape), put a **paired circular-block-
bootstrap** CI on the Sharpe advantage, cut the tape into **four eras**, and charge a
dollar-neutral **long-mid / short-neighbour** spread real one-way spreads + borrow. Prices
are yfinance total-return (`auto_adjust=True`); BIL's 2007-05 start caps the Sharpe-race
window (named on **Signal**). A seeded synthetic world with a **planted mid Sharpe edge**
(null at 0) proves the detector recovers a real advantage and stays quiet on the null.
**Dedup:** [513-size-effect](../513-size-effect/) is **small-minus-big** (SMB), not the
middle-beats-both claim; [177-megacap-concentration](../177-megacap-concentration/) is the
top-heaviness *within* the large-cap index; [94-level-pegging](../94-level-pegging/)
**equal-weights** one universe rather than comparing separate size bands;
[657-larry-portfolio](../657-larry-portfolio/) is the **small-value** tilt, not mid-cap
blend. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the "forgotten middle" *should* be a sweet spot, and what the tape actually shows — mid in the middle, the tilt that reversed |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-vs-excess Sharpe race, the paired-bootstrap advantage CIs, the HAC *t* on the pairwise difference, the four-era myth-check, the costed spread, and the planted-edge synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`midcap/`](midcap/). Excess-of-cash Sharpe (cash = BIL); the mid-vs-benchmark
difference is cash-independent, so the era cut uses the full IJH/MDY tape. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
