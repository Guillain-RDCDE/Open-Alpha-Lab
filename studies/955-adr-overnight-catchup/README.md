# Study 955 — ADR Catch-Up 🌏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Pooled over eight ADRs, yesterday's home move loads **+0.021** (HAC *t* = +1.15) — nothing, and what little there is comes from the *FX* leg (*t* = +2.79) not the closed index (*t* = +0.12). **Japan** — the one home market that shuts 13 hours before the ADR — gives **+0.081** (*t* = +3.25) where the clock says it should, and the UK, closing *inside* the US session, gives −0.048. But that one number does not survive the knife: deleting the 48 largest lagged home moves (**0.5%** of rows) takes it to +0.017 (*t* = +0.67), where the same trim moves a *synthetic* linear planted lag by 1%; winsorized it is +0.060 (*t* = +2.62), and the bucket sort pays **+1.6 bp** Q5 − Q1 with no gradient. A tail whiff on huge Tokyo nights, on two survivorship-picked names — not an established effect. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The catch-up book is negative gross (−0.92%/yr, Sharpe −0.049) with a **negative breakeven cost**; Japan-only breaks even at **0.85 bps** one-way, inside a real ADR spread. It turns over 108% of NAV a day, so 5 bps costs it 12–15%/yr. A control book that never opens the home tape *beats* it gross (+0.326 vs −0.049 Sharpe). |

> **In one sentence:** The clock's prediction does show up — Tokyo, which shuts long before
> New York, is the only home market whose yesterday still loads on the ADR, and London,
> closing mid-session, has nothing left over — but that loading lives almost entirely in
> the 0.5% of nights when Tokyo moved enormously, the coefficient a trader could bet on
> clears the bar nowhere, and the book breaks even under one basis point of cost.

## What we tested

Eight cross-listed ADRs (**TM, SONY, SAP, NVO, SHEL, BP, HSBC, RIO**) against their home
index (`^N225`, `^GDAXI`, `^FTSE`) and the FX that turns a home move into dollars,
2004-01-05 → 2026-06-30 (41,272 name-days). Four nested tests: does *yesterday's* home
dollar move still load on today's ADR return (HAC, pooled and by region); is that loading
linear or tail-carried (trim/winsorize sweep, bucket sort and block bootstrap, calibrated
against a synthetic linear lag); does the rolling-beta residual mean-revert (Fama-MacBeth,
rank-standardised); and is anything left once the ADR's own move is controlled for. Then a
costed long/short book, one execution lag, cost and borrow sweeps, an era cut, and a
control that uses no home data at all. Every number is close-to-close — a **coarse proxy**
for a genuinely intraday effect, said plainly throughout. **Dedup:** distinct from
**01-overnight-anomaly** / **788-overnight-intraday-tug-of-war** (one tape split by session,
not two markets on two clocks), **379-etf-lead-lag** / **865-credit-equity-lead-lag** /
**870-industry-leader-lead-lag** (informational lead-lag *within* one session, not a
physically shut market), **613-currency-hedged-etf-carry** / **634-us-leads-the-world**
(same home-index/FX plumbing, a carry identity and a global lead), and
**916-withholding-drag-international** (a tax wedge in the dividend, not a timing wedge in
the price).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an ADR should owe its home market a move, what the clock predicts, why Tokyo is different from London, why the one positive number melts when you remove 48 days, and why a real effect can still be untradable |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the stale-price HAC regression, the index/FX leg split, the trim/winsorize tail sweep against a synthetic linear calibration, `b1` vs the bettable γ and the autocorrelation that separates them, the region and era cuts, the costed books with breakeven costs, and the live synthetic control including the bounce confound |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`adr_catchup/`](adr_catchup/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
