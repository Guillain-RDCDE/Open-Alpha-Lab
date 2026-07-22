# Study 796 — Corporate-Bond-Low-Risk 🐢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low-vol bond ETFs earn a higher risk-adjusted return than high-vol ones? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Ranking an 11-name credit + Treasury bond-ETF basket on trailing 1-year volatility and building a risk-matched **low-minus-high** spread (safe leg levered up, risky leg levered down) earns **+0.84%/yr** — but at **HAC *t* = +0.51** (plain +0.57), and a vol-rank-shuffle placebo is *damning*: a **random** volatility ranking earns **+4.87%/yr** vs the real **+1.29%/yr** (**p = 0.989**). The low-vol basket does have a slightly higher Sharpe (**0.74** vs **0.63**), but that whisker doesn't survive being turned into a trade, is flat across vol-windows and the 2022 cut, and flips sign across eras. |
| **Tradability** — is there an edge to charge costs against? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Turnover break-even ≈ **29 bps** one-way, but the safe leg is levered **~2.6x**, so financing pushes the net **negative** everywhere: **−0.5%/yr** at 5 bps + 50 fin, **−1.2%** at 10 bps + 75, **−2.1%** at 20 bps + 100. Gross Sharpe **0.13**, a **−26%** drawdown and a single **−14.8%** month. There is no residual to trade. |
| **Is it the *real* volatility ranking that pays?** | ![Not supported](https://img.shields.io/badge/Ranked_on_real_vol%3F-Not_supported-8b949e?style=flat-square) | A **random** relabelling of which ETF is "calm" vs "wild" earns *more* than the true low-vol sort (placebo **p = 0.989**). The spread's return is a **leverage artifact**, not a low-risk signal — the genuinely calm names (short-duration Treasuries) are among the worst things to lever. |

> **In one sentence:** the low-risk / betting-against-beta anomaly is bulletproof in stocks,
> but ranking a basket of credit and Treasury **ETFs** on volatility and levering the safe leg
> earns a statistically invisible **+0.8%/yr** (HAC *t* = 0.51) that a *random* ranking beats
> outright and that dies once you finance the leverage — because a dozen broad ETFs can't
> reproduce the single-name risk cross-section the anomaly lives in. **No signal, no paycheck.**

## What we tested

The claim, steelmanned: *"safe assets are underpriced — the lowest-risk exposures earn the
highest risk-adjusted returns, so lever up the safe leg and short the risky leg"* (Frazzini &
Pedersen 2014, *Betting Against Beta*, JFE — who find it across equities *and* fixed income). We
take it to the tradable ETF panel: **11 credit + Treasury bond ETFs** (SHY, VCSH, LQD, IEF, BKLN,
VCLT, TLT, HYG, JNK, ANGL, EMB) on daily total-return yfinance data, **2007-2026**. Each month-end
we rank on trailing 1-year volatility, go long the calm bottom third / short the wild top third,
**scale each leg to a common ex-ante 6%/yr risk** (avg leverage low 2.6x / high 0.8x), form on the
close and earn the *next* month (one execution `shift`, zero look-ahead), and grade the monthly
low-minus-high spread with a **Newey-West HAC one-sample *t***, a Wilson hit rate, a
**2,000-permutation vol-rank-shuffle placebo**, a vol-window sweep, a 2022 myth-check, an era split,
and a costed timer (one-way × NAV, financing on the levered notional). A deterministic synthetic
panel with a planted low-risk knob proves the engine is faithful and powered (recovers a planted
effect at *t* up to +4.4, scores the null at *t* = +0.17 over 20 seeds, fires 1/20). **Dedup:** this
is the **bond, volatility-sorted, risk-level** leg — unlike [238-betting-against-beta](../238-betting-against-beta/)
(BAB in **equities**, single-stock beta) and [330-low-volatility-anomaly](../330-low-volatility-anomaly/)
(the low-vol anomaly in **stocks**); and it is a **risk-level** signal, orthogonal to the *return*
rank of [795-corporate-bond-momentum](../795-corporate-bond-momentum/) (its momentum sibling on the
identical tape). Survivorship (current-membership ETF basket) is named on the **Signal** axis — and
would *understate*, not manufacture, a low-risk premium. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "boring low-risk wins" is real in stocks but vanishes on a bond-ETF basket, why the calm basket's Sharpe edge is a whisker, and why a random ranking beats the real one |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the low-vs-high basket Sharpe, the vol-scaled BAB HAC *t*, the 2,000-permutation vol-rank-shuffle placebo, the vol-window sweep, the 2022 myth-check, the era split, the honest cost/financing sweep, and the planted-low-risk synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (2007-01 → 2026-06, fp `1f2efa58efab`): [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py).

---

*Engine: [`bond_low_risk/`](bond_low_risk/). The signal is the monthly vol-scaled low-minus-high spread's HAC *t*; the myth-check is the vol-rank-shuffle placebo. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
