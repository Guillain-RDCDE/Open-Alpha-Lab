# Study 828 — FX Dollar Factor 💵

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the dollar factor DOL earn a premium, and does the avg. forward discount time it (Lustig-Roussanov-Verdelhan)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The **unconditional** dollar factor earns **+0.06%/yr** (Newey-West *t* = **+0.04**) on a 7-currency G10 basket, 2004–2026 — indistinguishable from zero (which is LRV's own premise). The **dollar-timing** test, run with the only conditioning variable buildable from yfinance spot (a trailing-dollar-trend proxy for the average forward discount), is **insignificant and wrong-signed** (NW *t* = **−1.46**, block-shuffle placebo p = 0.084); the premium even **flips sign** across eras (+1.07% → −0.90%/yr). A 20-seed synthetic control recovers a *planted* premium (*t* = +3.63) and *planted* timing (*t* = +5.23) cleanly and fires on **0/20** premium nulls — so the flat tape is a real absence, not machinery. *Caveat: the true rate-based average forward discount is not reconstructable from spot — a data limit, flagged. Survivorship: current-membership G10 — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Both books lose money net: the static long basket is **−0.41%/yr** and the timed overlay **−0.96%/yr** at 5 bps one-way — there is no premium to pay for the friction. |

> **In one sentence:** the celebrated dollar factor DOL earns **no unconditional premium**
> (NW *t* = +0.04) on a G10 basket, its forward-discount *timing* is absent with the spot-only
> proxy available here (NW *t* = −1.46, wrong sign), and no book survives costs — **claimed
> signal absent, paycheck a mirage**.

## What we tested

Lustig, Roussanov & Verdelhan (2011), **"Common Risk Factors in Currency Markets"**: the
**dollar factor** `DOL` — the equal-weight average excess return of a basket of foreign
currencies vs the USD — is a candidate priced risk factor, and the **average forward discount**
is said to *time* it. We build DOL as the equal-weight mean of **7 G10 currency spot returns vs
USD** (yfinance daily FX, month-end, 2003-12-31 → 2026-06-30), carefully normalising every pair
to **USD-per-foreign** (inverting the USD-base quotes `JPY=X / CAD=X / CHF=X`), test the premium
with a Newey-West *t*, and run a **dollar-timing predictive regression** of next-month DOL on a
trailing-dollar-trend conditioning variable — a spot-only **proxy** for the average forward
discount, since true rate differentials are not on yfinance. One documented lag (signal at close
`t`, hold `t+1`), a 1,000-rotation block-shuffle placebo, a two-era cut, a costed static/timed
book, and a 20-seed synthetic positive control. The basket is a **current-membership** G10 set —
survivorship named on the **Signal** axis. **Dedup:** [364-fx-carry-trade](../364-fx-carry-trade/)
is the **HML_FX carry** slope (long-high/short-low), not the DOL level factor;
[797-fx-value-ppp](../797-fx-value-ppp/) is **PPP value**; [147-fx-momentum](../147-fx-momentum/)
is **cross-sectional FX momentum**; [36-greenback](../36-greenback/) is broad dollar-strength — none
tests DOL and its forward-discount timing. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the "dollar factor" is, why its unconditional premium is ~zero, and why the forward-discount timing doesn't show up on the data we can build |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the DOL premium NW *t*, the timing regression + block-shuffle placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dollar_factor/`](dollar_factor/). FX spot pulled once via yfinance into this study's own
`_cache/` (month-end parquet, USD-per-foreign). **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
