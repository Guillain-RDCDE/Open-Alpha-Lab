# Study 880 — Aggregate Short Interest 🐻

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is market-wide short interest "the strongest known predictor" of the market (Rapach-Ringgenberg-Zhou)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The RRZ aggregate-short-interest predictor **fails to replicate** on a 2017–2026 FINRA-built, 50-mega-cap, days-to-cover index. The forward-SPY-return slope has the **right sign** (**−17.2 bps per 1σ**, negative at *every* horizon out to 3 months, negative in **both** eras) but is **statistically absent**: Newey-West *t* = **−0.66**, R² = **0.28%**, permutation placebo *p* = 0.22, high-vs-low tercile Welch *t* = −0.75. A 20-seed synthetic control fires on a *planted* relation (*t* = −4.10) and stays quiet on the null, so the flat real result is genuine, not a broken engine. *Days-to-cover proxy (FINRA has no shares-outstanding); current-membership mega-caps (survivorship) — both named on the Signal axis.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The de-risk-to-cash-when-shorts-are-crowded overlay earns **+8.5%/yr** net (1 bp one-way) versus **+13.2%/yr** buy-and-hold — a weak, wrong-way-for-a-bull-market timing rule that *destroys* return. Mirage at any cost. |

> **In one sentence:** the celebrated aggregate-short-interest predictor — high market-wide
> short interest should forecast lower market returns — is **directionally right but
> statistically absent** on a modern bi-monthly FINRA index (NW *t* = −0.66, R² = 0.28%), and
> the timing overlay loses to buy-and-hold, so the honest read is **claimed signal absent,
> paycheck a mirage**.

## What we tested

Rapach, Ringgenberg & Zhou (2016), **"Short Interest and Aggregate Stock Returns"**: build a
**market-wide** short-interest index, detrend it, and it predicts the aggregate equity return
with a strong *negative* slope — "arguably the strongest known predictor" of the market. We
rebuild the index from the **FINRA Consolidated Short Interest** file (the official bi-monthly
settlement-date report, pulled per-name from the public FINRA Query API) for a **50-name
liquid US panel** as the equal-weight mean **days-to-cover**, detrend its log against a linear
trend (the RRZ step), and regress forward **SPY total-return** on it (**205 bi-monthly prints,
2017-12-29 → 2026-06-30**) with one documented publication lag (signal at settlement `t` acted
on the next settlement `t+1`, zero look-ahead), a Newey-West slope *t*, a 5,000-draw
permutation placebo, a two-era cut, a costed timing overlay, and a 20-seed synthetic positive
control. Honesty rails travel with every number: aggregate SI is **bi-monthly** with an ~8-day
publication lag (not daily); the index is a **days-to-cover** average, not the paper's
shares-outstanding ratio; the panel is **current-membership** mega-caps (survivorship, on the
**Signal** axis); and the sample is short and mostly a bull market. **Dedup:**
[262-short-interest](../262-short-interest/) is the **cross-sectional** name-by-name sort (this
is the aggregate/time-series cousin); [557-borrow-fee-signal](../557-borrow-fee-signal/) uses
the **cost to borrow**, not the short-position ratio; [558-failures-to-deliver](../558-failures-to-deliver/)
uses **settlement failures**; [260-margin-debt](../260-margin-debt/) is the mirror on the
**long-leverage** side. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why crowded market-wide shorts *should* forecast a weaker market — and why on a modern mega-cap index the signal is directionally right but statistically flat |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the horizon sweep, the Newey-West slope *t*, the tercile Welch test, the 5,000-draw placebo, the two-era cut, the costed timer, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`agg_short/`](agg_short/). Aggregate index rebuilt from the public FINRA
Consolidated Short Interest file over a current-membership mega-cap panel (a days-to-cover
proxy → magnitudes are an approximation). **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
