# Study 832 — High-Yield Credit Momentum 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the credit **trend** (HYG excess over IEF) time equity risk-on/off? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The trailing 3-/6-month duration-hedged credit trend does **not** discriminate the day-`t` equity excess `r_SPY − r_IEF`. Risk-on (positive-trend) days actually earned a *lower* excess than risk-off days — the **wrong sign** — and negligibly so: **−1.62 bps/day** (Newey-West *t* = **−0.37**) at 6 months, −2.55 bps (*t* = −0.61) at 3 months. The observed value sits **inside** a 1,000-draw label-shift placebo (−0.4σ, p = 0.65) and the sign **flips across eras** (+4.80 → −12.27 bps, |*t*| < 2 in both halves). A 20-seed synthetic control recovers a *planted* credit→equity effect cleanly (fires on **1/20** nulls), so the flat real-tape result is a true absence, not a broken engine. *Regime caveat on the Signal axis: one US 2007-2026 credit history.* |
| **Tradability** — can the SPY↔IEF switch beat buy-and-hold? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The costed timer never robustly beats a 100%-SPY buy-and-hold: net Sharpe **0.627 vs 0.619** at 1 bp/leg is an insignificant hair (active return *negative*, NW *t* = **−1.26**), it forfeits **3.5%/yr of CAGR** (7.3% vs 10.9%), and by 5 bps/leg its Sharpe (0.558) falls **below** buy-and-hold. Its one honest merit — a **−41% vs −55%** max drawdown — is insurance paid for in forgone return, not a paycheck. |

> **In one sentence:** the popular "ride high-yield credit momentum into stocks" rule
> **does not survive an honest test** — the credit trend fails to predict the equity leg
> (wrong sign, *t* = −0.37, indistinguishable from a label shuffle, sign-flipping across
> eras), and the SPY↔IEF timer only trades away 3.5%/yr of return for a lower drawdown, so
> the read is **claimed signal absent, paycheck a mirage**.

## What we tested

The practitioner claim: the **trend** (trailing 3-6-month total return) of **duration-hedged
high-yield credit** — HYG in excess of IEF — is a risk-on/off timing signal for equities;
positive trend → long SPY, negative → de-risk to IEF. We take the self-contained daily
version on four total-return ETFs (**yfinance, `auto_adjust=True`, HYG/IEF/LQD/SPY,
2007-05-01 → 2026-06-30, 4,822 rows**): the credit trend known at the close of `t−1` (one
shift, zero look-ahead), a Newey-West *t* on whether it discriminates the day-`t`
equity-excess `r_SPY − r_IEF` (via a time-ordered regime-contrast series), a 1,000-draw
label-shift placebo, a two-era cut, a **costed SPY↔IEF timer vs a 100%-SPY buy-and-hold**
(net Sharpe / CAGR / max-drawdown, NW *t* on the active return), and a 20-seed synthetic
positive control. The ETFs are all live (no delisting bias); the Signal-axis caveat is
single-regime overfit (one US credit history). **Dedup:**
[115-credit-spreads](../115-credit-spreads/) reads the credit-spread **level** as a stress
*warning*, not the **trend** as a timing switch; [795-corporate-bond-momentum](../795-corporate-bond-momentum/)
is **cross-sectional** momentum *within* the bond universe, not a single time-series trend
timing *equities*; [131-utilities-canary](../131-utilities-canary/) uses **utilities**
relative strength as the risk canary, a different asset. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why credit momentum *should* lead stocks — and why here it doesn't, in one picture |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the discrimination NW *t*, the label-shift placebo, the two-era cut, the costed timer vs buy-and-hold, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`hy_credit_momentum/`](hy_credit_momentum/). Four live ETFs pulled once via
yfinance and cached under `_cache/`; all analysis is offline & deterministic thereafter.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
