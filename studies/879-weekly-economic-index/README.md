# Study 879 — Weekly Economic Index 📅

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the WEI level / weekly change time forward SPY & the XLY−XLP rotation? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A weekly growth nowcast does **not** beat the monthly tape. The WEI **level** does not predict forward SPY (Newey-West *t* = **−1.12**, wrong sign) and is era-unstable (late-era *t* = −2.69); the **weekly change** has the right sign but is **insignificant** (*t* = +1.34), its only \|t\| ≥ 2 hit (SPY, early era, +2.26) a one-off of the 2008–09 recession/recovery that **vanishes post-2017** (+0.24). The single overall \|t\| ≥ 2 slope — rotation level, 4-week, **−2.24** (placebo p = 0.026) — is **wrong-signed** (strong growth → cyclical *under*-performance, a mean-reversion), so it cannot support the claim. A 20-seed synthetic control recovers a *planted* edge cleanly (*t* = +16.7) and fires on the null at ~5%, so the null is genuine, not a broken test. *Revision caveat: the level uses the revised WEI vintage — named on the Signal axis.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The costed long-cyclical / short-defensive (XLY−XLP) overlay **loses to simply holding the rotation** (net Sharpe **−0.16** for the level rule, **+0.02** for the weekly-change rule, vs **+0.28** always-hold). The thin weekly-change edge (gross Sharpe +0.21) is wiped out by 1,123 weekly turns. |

> **In one sentence:** the celebrated real-time Weekly Economic Index — ten weekly series
> blended into a growth nowcast — **does not time the market**: its level is
> insignificant/wrong-signed on forward SPY, its weekly change is a fragile 2008–09-recovery
> whiff that dies after 2017, the only significant slope is a *wrong-signed* rotation
> mean-reversion, and no overlay beats buy-and-hold — **claimed signal absent, paycheck a
> mirage**.

## What we tested

Lewis, Mertens & Stock (2020), the **Weekly Economic Index (WEI)**: ten weekly activity
series (Redbook retail, initial & continuing jobless claims, tax withholding, rail traffic,
fuel sales, temp-staffing, steel, electricity, consumer confidence) blended into a
real-time nowcast of U.S. year-over-year growth, published every week by the **Dallas Fed**.
The claim: because it is *higher-frequency* than the monthly macro tape, its **level** and
**weekly change** should predict **forward SPY** and the **cyclical-vs-defensive rotation**
(consumer-discretionary `XLY` vs consumer-staples `XLP`). We take the real Dallas Fed
workbook history (**weekly, 2008-01 → 2026-06**, 962 aligned weeks) plus SPY/XLY/XLP daily
total-return closes, run a **predictive regression with a Newey-West (8-lag) HAC *t*** on the
standardized level & weekly change, a **two-era cut** (split 2017-01), a **2,000-draw
permutation placebo**, a **costed rotation overlay**, and a **20-seed synthetic positive
control**. One documented lag: a week-ending nowcast is only published the next week, so
every forward return is anchored **one trading week later** (zero look-ahead); the level
carries a **revision** caveat (we use the revised vintage) — named on the **Signal** axis.
**Dedup:** [384-ism-pmi-regime](../384-ism-pmi-regime/) is the **monthly** PMI regime;
[387-economic-surprise-index](../387-economic-surprise-index/) is a **surprise-vs-consensus**
gap, not a level nowcast; [626-unemployment-trend-timing](../626-unemployment-trend-timing/)
is the single **monthly unemployment trend**; [757-cass-freight](../757-cass-freight/) is the
single **monthly** freight index — none tests the **weekly ten-series composite** against the
**XLY−XLP rotation**. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a weekly nowcast *should* beat monthly data — and why on the real tape it doesn't time SPY or the rotation |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the predictive-regression HAC *t*, the two-era cut, the permutation placebo, the costed rotation overlay, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`wei/`](wei/). Nowcast from the **Dallas Fed** WEI workbook (current vintage →
level magnitudes are an upper bound); returns from yfinance total-return closes.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
