# Study 865 — Credit → Equity Lead-Lag 🔗

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the trailing HY-excess return **lead** next-week SPY? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The Granger-style predictive regression `r_SPY[t+1] ~ trailing HY-excess trend[t]` has the **wrong sign** and is **insignificant** at every horizon: 4-week slope **−19.6 bps per 1σ trend** (Newey-West *t* = **−1.70**, R² 0.60%), 1-week *t* = −1.58, 2-week *t* = −0.23. The risk-on−risk-off next-week difference (−7.5 bps/wk, *t* = −0.48) sits **inside** a 1,000-draw label-shift placebo (−0.4σ, p = 0.66), and the wrong sign is **consistent across eras** (both halves negative, |*t*| < 2). If anything credit's recent strength faintly foreshadows equity *mean-reverting* — the reverse of the claim. A 20-seed synthetic control recovers a *planted* one-week lead cleanly (fires on **0/20** nulls), so the flat real-tape result is a true absence, not a broken engine. *Regime caveat on the Signal axis: one US 2007-2026 credit history.* |
| **Tradability** — can the SPY↔IEF overlay beat buy-and-hold? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The weekly overlay never beats a 100%-SPY buy-and-hold: net Sharpe **0.644 vs 0.645** at 1 bp/leg is an insignificant hair *below* it (active return *negative*, NW *t* = **−1.22**), it forfeits **3.3%/yr of CAGR** (7.4% vs 10.6%), and by 5 bps/leg its Sharpe (0.569) falls further behind. Its one honest merit — a **−33% vs −55%** max drawdown — is insurance paid for in forgone return, not a paycheck. |

> **In one sentence:** the popular "credit leads equity" rule **does not survive an honest
> test** — the trailing HY-excess return fails to predict next-week SPY (wrong sign,
> *t* = −1.70, indistinguishable from a label shuffle, consistently wrong-signed across
> eras), and the SPY↔IEF overlay only trades 3.3%/yr of return for a lower drawdown, so the
> read is **claimed lead absent, paycheck a mirage**.

## What we tested

The practitioner claim: high-yield credit **turns before** stocks, so the trailing 1-4-week
**duration-hedged HY-excess return** — HYG in excess of IEF — should **predict the next
week's SPY return** (a Granger-style lead-lag). We take the self-contained weekly version on
four total-return ETFs (**yfinance, `auto_adjust=True`, HYG/IEF/LQD/SPY, 2007-05-01 →
2026-06-30, 4,822 daily rows → 1,001 weekly closes**): a **predictive regression** of
next-week SPY on the trailing HY-excess trend known at the Friday close (one shift, zero
look-ahead) with a Newey-West *t* on the slope, a companion risk-on/off discrimination, a
1,000-draw label-shift placebo, a two-era cut, a **costed SPY↔IEF timing overlay vs a
100%-SPY buy-and-hold** (net Sharpe / CAGR / max-drawdown, NW *t* on the active return), and
a 20-seed synthetic positive control. The ETFs are all live (no delisting bias); the
Signal-axis caveat is single-regime overfit (one US credit history). **Dedup:**
[115-credit-spreads](../115-credit-spreads/) reads the credit-spread **level** as a stress
*warning*, not the trend as a forward *predictor*;
[832-high-yield-credit-momentum](../832-high-yield-credit-momentum/) grades the credit trend
as its own **daily SPY↔IEF timer on the trend *sign***, not the explicit **weekly lead-lag
regression** `r_SPY[t+1] ~ trend[t]`; [131-utilities-canary](../131-utilities-canary/) uses
**utilities** relative strength as the risk canary, a different asset;
[379-etf-lead-lag](../379-etf-lead-lag/) is a **generic cross-ETF** lead-lag grid, not the
specific duration-hedged credit→equity lead. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why credit *should* lead stocks — and why here it doesn't (and even points the wrong way), in one picture |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the predictive-regression NW *t*, the risk-on/off discrimination, the label-shift placebo, the two-era cut, the costed overlay vs buy-and-hold, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`credit_lead/`](credit_lead/). Four live ETFs pulled once via yfinance and cached
under `_cache/`; all analysis is offline & deterministic thereafter. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
