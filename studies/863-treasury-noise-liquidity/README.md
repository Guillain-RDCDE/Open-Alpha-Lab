# Study 863 — Treasury Noise Liquidity 🌊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does Treasury-curve **roughness** precede lower equity / wider credit (Hu-Pan-Wang noise)? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The noise measure has the **right sign** (a rough curve precedes weaker SPY and wider HYG−IEF) and is **full-tape significant and placebo-real at the short horizon**: forward 5-day SPY return loads **−0.171%/1σ**, Newey-West *t* = **−2.20**, sitting **−2.39σ** into a 3,000-draw block-rotation placebo's left tail (21-day a near-miss, *t* = −1.97). But the edge is **concentrated in the crisis-heavy 2007–2015 era** (SPY 5d *t* = **−2.42**) and **decays to insignificance after 2016** (*t* = **−1.14**); **no sub-era clears \|*t*\| ≥ 2 at the monthly horizon**, and the credit leg fires **only in the GFC** (−2.54 → +0.14). A 20-seed synthetic control recovers a *planted* relation cleanly (*t* = −7.33, fires **0/20** on nulls), so the decay is real, not machinery. *No survivorship (CMT indices + ETFs are continuously listed); risk-free leg proxied at 0 — named on the Signal axis.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A noise-conditioned long/flat SPY timer **loses to buy-and-hold** (Sharpe **0.31 vs 0.53**; **−2.53 bps/day**, NW *t* = −2.14) while churning 12.7×/yr — the noise spikes *coincide* with the drawdown rather than leading it, so stepping out on rough days forfeits the rebound. |
| **In-sample vs out-of-sample** | ![Decayed](https://img.shields.io/badge/In--sample_vs_out--of--sample-Decayed-8b949e?style=flat-square) | A ~2.4-*t* crisis-era edge (2007–2015) collapses to an insignificant, near-zero −1.1 in 2016–2026 — a post-publication / post-QE fade. |

> **In one sentence:** the Hu-Pan-Wang Treasury-noise measure is a **genuine stress thermometer
> that points the right way** — a rougher curve really did precede weaker stocks and wider credit,
> sharply so in 2007–2015 (SPY 5-day NW *t* = −2.2, placebo-real) — but the edge **lives in the
> crises, decays to a whisper after 2016, and no noise-timer beats simply holding the index**, so
> the honest read is **a real-in-stress signal that faded, not a paycheck**.

## What we tested

Hu, Pan & Wang (2013), **"Noise as Information for Illiquidity"**: the cross-maturity
**roughness** of the Treasury yield curve (RMS deviation of yields from a smooth fit) is a
market-wide illiquidity gauge — scarce arbitrage capital lets maturities wander off the curve —
and high noise is said to **precede lower equity returns and wider credit spreads**. We build the
self-contained daily version from **four CMT yields (`^IRX/^FVX/^TNX/^TYX`, 13w/5y/10y/30y) plus
`SPY`, `HYG`, `IEF` (yfinance, 2007-04-12 → 2026-06-30, 4,833 signal days)**: `noise` = RMS
residual of a **quadratic-in-maturity** fit (one constant projection matrix, fully vectorised),
regressed against the **forward SPY** and **forward HYG − IEF** returns at 5/21/63-day horizons
with a **Newey-West slope *t***, a 3,000-draw block-rotation placebo, a two-era cut, a costed
long/flat timer, and a 20-seed synthetic positive control. Point-in-time (noise known at close
`t`, held `t → t+h`, one lag, zero look-ahead); the CMT indices and ETFs are continuously listed
(no survivorship); the risk-free leg is proxied at 0 — named on the **Signal** axis. **Dedup:**
[112-move-index](../112-move-index/) tests option-**implied vol** of Treasuries (MOVE), not the
realized curve residual; [383-sofr-repo-stress](../383-sofr-repo-stress/) tests discrete
**repo/SOFR funding spikes**, not a whole-curve fitting residual;
[386-nfci-conditions](../386-nfci-conditions/) tests a broad **financial-conditions** blend, not
a Treasury-only residual; [581-term-premium](../581-term-premium/) tests the **level** of the
term premium (the curve's *shape*), not the *deviation from* a smooth shape. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a "rough" curve means in one picture, why scarce arbitrage capital roughens it — and why the warning fired in the crises but faded after 2016 |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the roughness projection, the forward SPY / HYG−IEF Newey-West slope *t*, the *t*-by-horizon curve, the two-era cut, the 3,000-draw placebo, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`treasury_noise/`](treasury_noise/). Noise = RMS deviation of four CMT yields from a
quadratic-in-maturity fit; forward SPY / HYG−IEF returns via a Newey-West predictive regression.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
