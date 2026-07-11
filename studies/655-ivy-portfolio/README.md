# Study 655 — Ivy-Portfolio 🌿

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 5-asset endowment mix out-Sharpe a 60/40, timed or not? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the SMA timer's drawdown cut · None on the static diversification's Sharpe claim.* Static Ivy's Sharpe is **robustly worse** than 60/40 — bootstrap 95% CI **[−0.470, −0.018]**, entirely negative. The 10-month timer cuts drawdown **−44.8% → −13.3%**, validated against a matched-exposure random-timing control (beats **100%/40** shuffles on drawdown, 95%/40 on Sharpe). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Cheap either way (turnover 28%/yr static, 361%/yr timed; costs ≤36 bps/yr) — but nothing survives to trade: static Ivy loses to a free 60/40 on Sharpe, CAGR and drawdown; timed Ivy's drawdown win never converts into a certified Sharpe edge over 60/40 (CI **[−0.589, +0.166]**) while its return is **significantly worse** (HAC *t* = **−2.26**). |
| **"Risk reduction or alpha (the 10-month SMA)?"** | ![Confirmed](https://img.shields.io/badge/Risk_reduction%3F-Confirmed-8b949e?style=flat-square) | Drawdown protection is genuine and repeats ex-GFC (−19.3% → −13.1%). The return cost is genuine too: timed Ivy trails 60/40 by **−3.84%/yr** (*t* = −2.26). A crash shield, not a return engine — exactly Faber's own single-asset finding ([Study 110](../110-faber-timing/)), now shown across a 5-asset composite. |

> **In one sentence:** Faber's 20%-each US/foreign-equity/REITs/bonds/commodities mix does **not**
> out-Sharpe a plain 60/40 on the 2007–2026 tape (bootstrap CI entirely negative — commodities and
> REITs were themselves disasters over this stretch), but bolting on the 10-month SMA timer, one
> sleeve at a time, genuinely and repeatedly cuts drawdown (validated against random-timing) at
> the cost of return — a real risk-reduction tool wearing a diversification story that doesn't
> hold up.

## What we tested

Mebane Faber's **Ivy Portfolio**: 20% each in VTI (US equity), VEU (foreign equity), VNQ
(REITs), AGG (bonds) and DBC (commodities), monthly-rebalanced — and, separately, the same
five sleeves each timed by their own **10-month simple moving average** (hold the asset above
its 10-month average, else park that 20% in T-bills). Real daily closes since 2007 (bound by
BIL's inception), Sharpe excess-of-BIL throughout, one documented execution lag, costs one-way
× NAV on rebalance turnover. We test the two claims **separately** — is the 5-asset mix a
better risk-adjusted diversifier than 60/40 (bootstrap Sharpe-difference CI), and does the SMA
timer add alpha or just cut risk (matched-exposure random-timing control, ex-GFC sub-period,
and a synthetic null/planted machinery check)? Dedup: distinct from
[68-all-weather](../68-all-weather/) (risk parity, different quartet),
[110-faber-timing](../110-faber-timing/) (the same SMA rule, single-asset),
[144-permanent-portfolio](../144-permanent-portfolio/) and
[203-golden-butterfly](../203-golden-butterfly/) (no timing overlay, different legs), and
[592-dual-momentum-gem](../592-dual-momentum-gem/) (a relative-momentum switcher, not a
diversified equal-weight blend). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why endowments diversify, which two of Faber's five legs quietly wrecked the mix, and what the 10-month rule actually buys you |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the bootstrap Sharpe-difference CIs, the matched-exposure random-timing control, the ex-GFC robustness check, costs/turnover, and the synthetic null/planted control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ivy_portfolio/`](ivy_portfolio/). VTI/VEU/VNQ/AGG/DBC/BIL are broad, still-listed
ETFs — no survivorship. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
