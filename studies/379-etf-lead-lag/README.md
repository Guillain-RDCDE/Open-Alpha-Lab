# Study 379 — ETF-Lead-Lag 🔗

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the leader predict the next member move? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The only large correlation is **contemporaneous** (**+0.87** at lag 0) and untradable. The one-day lead the claim needs is **absent**: cross-corr at k=+1 is **−0.05**, the HAC lag-1 slope is **−0.058 (t = −1.88)**, and the conditional next-day return is *below* the base rate (**t = −1.34**, placebo *p* = **0.97**). No predictive lead on daily bars. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The next-bar "buy the laggards after the leader pops" rule is **negative gross** (**−0.078%**/trade) and worse net of a 10 bps round-turn (**−0.178%**). An edge that points the wrong way before costs can't be deployed. |
| **Free lunch?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | The co-movement that *looks* like a free lunch is **same-day** — by the time you can act, the move already happened. The only tradable piece (the next-day lead) is **zero** on daily bars. Real intraday, **gone by the daily close**. |

> **In one sentence:** the big ETF and its slow members move together enormously *the same day* (corr 0.87), but that's untradable, and the one-day lead the folklore promises is statistically absent on daily bars (HAC t = −1.88, the wrong sign) — so a next-bar "buy the laggards" rule loses money even before costs, and the apparent free lunch is just same-day co-movement you can't act on.

## What we tested

True ETF/constituent lead-lag is an **intraday** phenomenon and tick data isn't free, so we test the **daily-bar** version believers point to: using **SPY** as the leader and an equal-weight basket of **37 smaller, less-liquid US members** as the laggards, does *yesterday's* leader return predict *today's* member return? We build the full lead-lag cross-correlation profile, an HAC-robust one-day slope, and a tradable next-bar rule (long the laggards the day after a top-decile leader day) net of one-way costs — with a 20,000-draw placebo null and a synthetic control whose planted lead-knob proves the engine would catch a real lead if one existed. (Daily-bar **proxy** for intraday lead-lag, and a mildly survivorship-tilted surviving-names basket, both named on the Signal axis.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "the leader leads the members" really means, why same-day co-movement isn't a free lunch, and why the next-day version pays nothing — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the lead-lag cross-correlation profile, an HAC-robust one-day slope, a next-bar rule with a Welch *t* + placebo null, costs, and a synthetic planted-lead / faithful-engine control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`etf_lead_lag/`](etf_lead_lag/). Lead-lag here is measured on **daily** bars (an explicit **proxy** for the intraday phenomenon), on a fixed surviving-names member basket. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
