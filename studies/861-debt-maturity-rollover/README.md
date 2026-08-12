# Study 861 — Debt-Maturity Rollover Risk ⏳

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do high-short-term-debt firms under-earn? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | A monthly tercile long-short (long the low-share / short the high-share names) earns **+46.7 bps/mo (+5.6%/yr gross)** with a robust **Newey-West *t* = +3.22** (one-sample +2.65). It is **right-signed** (rollover-risk names under-earn, as claimed), **survives a drop-one-name jackknife** (all 32 refits keep NW *t* > 2, min +2.14), holds under half/quartile/quintile sorts (+3.87 / +2.69 / +2.27) and the asset-scaled variant (+2.67), and is confirmed by a pooled event drift (63d *t* +2.02, 126d *t* +2.85, placebo *p* ≤ 0.02). A rare genuine positive on this desk. |
| **Tradability** — can you get paid for it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Turnover is low (~0.10/mo), so net of **20 bps + 100 bps borrow** it survives (**+4.1%/yr, NW *t* = +2.36, Sharpe 0.47**) — but the net Sharpe is modest and the significance **breaks under a harsher 30 bps + 200 bps** assumption (NW *t* = +1.64). A real edge on the base cost model that is one cost-bump from vanishing. |
| **Rate-era bite?** | ![Confirmed](https://img.shields.io/badge/Rate--era_bite%3F-Confirmed-8b949e?style=flat-square) | The penalty is **~2.6× larger in the 2022+ hiking cycle** (+85.7 bps/mo, NW *t* = +2.95) than before (+32.6 bps/mo, NW *t* = +2.10). Rollover risk really does bite harder when rates rise — the claim's own prediction, confirmed. |

> **In one sentence:** firms funded with a big slice of short-term debt — a maturity wall they
> must refinance soon — **under-earn the termed-out names by ~+5.6%/yr gross** (robust *t* =
> +3.22, jackknife- and spec-robust), and the penalty **more than doubles in the 2022+ rising-rate
> era** exactly as the rollover story predicts; but the tradable residual after realistic costs is
> thin (Sharpe ~0.47) and part of the win is simply the safe, big-cap leg being a good hiding
> place: **a real, rate-conditional risk penalty; a fragile trade.**

## What we tested

The corporate-finance staple, stated the way its believers state it: *"a firm with a big chunk of
debt maturing within a year has to roll it over at whatever rates prevail — so bet against the
names with the biggest short-term maturity wall, especially when rates are rising."* We take it
literally on **32 large, debt-carrying US filers that report a clean maturity split on EDGAR**
(`DebtCurrent` + `LongTermDebtCurrent` over total debt including `LongTermDebtNoncurrent`),
2008→2026, ranked **point-in-time on the 10-Q/10-K filing date** (zero look-ahead). The primary is
a monthly tercile **long-short** (long low-share / short high-share) held one month forward, graded
on an autocorrelation-robust **Newey-West *t***, and — because it comes back *positive* — held to a
higher bar: a **drop-one-name jackknife**, half/quartile/quintile re-sorts, a staleness and an
asset-scaled variant, a pooled **event drift + label-shuffle placebo**, an explicit **2022 rate-era
cut**, a four-point **cost/borrow stress**, and a 12-seed synthetic control. **Coverage is thin and
uneven** — only 32 of a 48-name basket tag all three legs, the cross-section is sparser pre-2010,
and much of the spread rides the safe leg — and we say so throughout.
**Dedup:** [540-distress-risk-anomaly](../540-distress-risk-anomaly/), [123-altman-z](../123-altman-z/)
and [230-ohlson-o-score](../230-ohlson-o-score/) rank on default *probability* (a composite hazard
score); [154-leverage-anomaly](../154-leverage-anomaly/) ranks on the *amount* of leverage. None
ranks on the debt's **maturity composition** — *when* it comes due, not *how much* there is — this
study's own axis. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the maturity wall is a real, priced-too-slowly risk, why it bites hardest in the hiking cycle, and why the tradable slice is thinner than it looks |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the calendar-time Newey-West long-short, the drop-one jackknife, the 2022 rate-era cut, the pooled event drift + placebo + monotonicity, the cost/borrow stress, and the 12-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`debt_maturity/`](debt_maturity/). EDGAR XBRL `companyconcept` (DebtCurrent,
LongTermDebtCurrent, LongTermDebtNoncurrent, Assets) + yfinance adjusted closes; a
**current-survivors** basket — survivorship named on the Signal axis (it drops the actual rollover
blow-ups, biasing the penalty conservatively). **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
