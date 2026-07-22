# Study 795 — Corporate-Bond-Momentum 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do past-winner bond ETFs keep beating past losers? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Ranking an 11-name credit + Treasury bond-ETF basket on trailing 6-month total return and going long the top third / short the bottom third earns **+1.63%/yr** — but at **HAC *t* = +0.80** (plain *t* = +0.67), a coin-flip hit rate (**51.4%**, Wilson [44.8%, 58.0%] straddles 50%), and a rank-shuffle placebo **p = 0.152** (a random ranking beats the momentum ranking ~1 time in 7). The sign **flips negative** at the 12-month formation the claim also names (*t* = −0.54). No subperiod certifies it. |
| **Tradability** — is there an edge to charge costs against? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Break-even ≈ **27 bps** one-way against 50%/month turnover; the net mean is a rounding error at 5-10 bps (*t* ≈ 0.3-0.4) and **negative** at 20 bps. Gross Sharpe **0.16**, a **−24%** drawdown and a single **−18.9%** month. There is no residual to trade — and the dollar-neutral spread is designed to strip out the one real premium here (long HYG: HAC *t* = +2.47). |
| **Stable across the 6-12m the claim names?** | ![Not supported](https://img.shields.io/badge/Lookback_stable%3F-Not_supported-8b949e?style=flat-square) | The point estimate slides from faintly positive at 6m (*t* = +0.80) to faintly **negative** at 12m and 12-1 (*t* = −0.54) — the opposite of a robust anomaly. Consistent with Jostova et al.'s own finding that bond momentum lives in the *single-name high-yield* cross-section, which a coarse, IG-and-Treasury-heavy ETF panel simply cannot hold. |

> **In one sentence:** cross-sectional momentum is bulletproof in stocks, and Jostova et al.
> (2013) found it in *single high-yield bonds* — but ranking a basket of credit and Treasury
> **ETFs** on trailing return earns a statistically invisible **+1.6%/yr** (HAC *t* = 0.80)
> that flips sign across the very 6-12m window the claim names and dies after costs, because
> one ETF per credit sleeve can't reproduce the within-high-yield dispersion the effect lives
> in. **No signal, no paycheck.**

## What we tested

The claim, steelmanned: *"momentum works in bonds too — rank on trailing 6-12 month total
return, buy the winners, short the losers"* (Jostova, Nikolova, Philipov & Stahel 2013,
*Momentum in Corporate Bond Returns*, RFS — who find it concentrated in **high-yield** single
names). We take it to the tradable ETF panel: **11 credit + Treasury bond ETFs** (LQD, VCLT,
VCSH, HYG, JNK, EMB, BKLN, ANGL, SHY, IEF, TLT) on daily total-return yfinance data,
**2007-2026**. Each month-end we rank on trailing 6-month return, go long the top third /
short the bottom third (equal-weight, dollar-neutral), form on the close and earn the *next*
month (one execution `shift`, zero look-ahead), and grade the monthly winners-minus-losers
spread with a **Newey-West HAC one-sample *t***, a Wilson hit rate, a **2,000-permutation
rank-shuffle placebo**, a 6-12m lookback sweep, an era split, and a costed timer (one-way ×
NAV, the short leg pays borrow). A deterministic synthetic panel with a planted momentum knob
proves the engine is faithful and powered (recovers a planted effect at *t* up to +8.6, scores
the null at *t* = +0.14 over 20 seeds). **Dedup:** this is **cross-sectional** (rank assets
against each other), unlike [518-time-series-momentum](../518-time-series-momentum/) (each
asset on its *own* trailing sign); it is a trailing-**return rank**, not the calendar signal
of [247-bond-seasonality](../247-bond-seasonality/); and it is a *change* signal, orthogonal
to the *level*/carry claims of [611-mreit-carry](../611-mreit-carry/) and
[612-em-debt-carry](../612-em-debt-carry/). Survivorship (current-membership ETF basket) is
named on the **Signal** axis — but a *null* is not manufactured by it. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "momentum works in bonds too" is real for single high-yield names but vanishes on an ETF basket, why the winner/loser gap is a coin flip, and why the one real edge here is just owning credit |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the WML HAC *t*, the 2,000-permutation rank-shuffle placebo, the 6-12m lookback-sign sweep, the era split, the honest cost/borrow sweep and break-even, and the planted-momentum synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (2007-01 → 2026-06, fp `1f2efa58efab`): [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py).

---

*Engine: [`bond_momentum/`](bond_momentum/). The signal is the monthly winners-minus-losers spread's HAC *t*; the myth-check is the 6-12m lookback-sign sweep. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
