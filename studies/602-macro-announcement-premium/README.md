# Study 602 — Macro-Announcement-Day Premium 📅

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the equity premium earned on scheduled CPI/NFP/FOMC days? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | On 29.5 years of SPY the 923 announcement days earn **+10.63 bps/day vs +3.69** the rest — **12.4% of sessions carry 32.1% of the cumulative return** — but the pooled premium is only **Welch *t* = 1.57** (placebo *p* = 0.056), below the **t ≥ 2** bar. The only leg that clears is **FOMC** (*t* = 2.25); CPI *t* = 0.12, NFP *t* = 1.38, and the pooled effect has been **absent since 2017** (diff −3.2 bps/day). The literature (1958-2009) says real; this tape alone can't certify it. |
| **Tradability** — can you harvest it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Long SPY only on A-days (31.4 round trips/yr, 12.4% time in market): even at an institutional **1 bp/leg** it nets **+2.48%/yr vs +10.05%** buy & hold, and at a realistic **5 bps it is dead** (net −0.05%/yr). You give up three-quarters of the market return to time a calendar that costs eat. |
| **"Is it all just FOMC?"** | ![Confirmed](https://img.shields.io/badge/All_just_FOMC%3F-Confirmed-8b949e?style=flat-square) | Strip the 236 FOMC sessions out and the remaining CPI/NFP announcement days earn **+2.97 bps/day over ordinary days (*t* = 0.58, placebo *p* = 0.32)** — a statistical zero. The pooled "macro-announcement premium" is the **FOMC-day premium wearing a bigger calendar** — and even that engine (2007-2016 *t* = 2.90) faded after 2017 (*t* = −0.41). |

> **In one sentence:** Savor-Wilson's macro-announcement-day premium shows its silhouette on the modern tape — 12% of sessions carry a third of SPY's return — but the pooled CPI+NFP+FOMC premium never clears *t* = 2 (Welch *t* = 1.57), every basis point of it traces back to the FOMC days alone (ex-FOMC *t* = 0.58), the engine has been flat since 2017, and the A-day-only overlay loses to buy & hold at any cost level — **Weak, and a Mirage to trade**.

## What we tested

We rebuild Savor & Wilson (2013) on SPY daily total-return closes, 1997-01 → 2026-06, against a
**hardcoded, source-documented release calendar**: 236 scheduled FOMC decision days (Fed
historical calendars) plus 353 CPI and 353 NFP **actual release dates** scraped from the BLS
archived-news-release indexes and cross-checked against the official `histreleasedates.pdf`
(19/19 overlapping dates agree; shutdown gaps and holiday mappings documented). The Signal axis
pools the 923 announcement sessions against the other 6,496 (Welch *t* + a 20,000-draw
same-density random-calendar placebo), splits by type and by decade, and checks TLT. Tradability
charges one-way costs × NAV on a prior-close-entry / A-day-close-exit overlay (one execution
lag; the schedule is public in advance). The third axis re-runs the pooled test **without** the
FOMC sessions — the sharpest cut against its desk siblings [517-pre-fomc-drift](../517-pre-fomc-drift/)
(pre-FOMC drift only), [67-fed-drift](../67-fed-drift/) and [135-fomc-cycle](../135-fomc-cycle/)
(cycle weeks): here the claim is the **pooled macro-day premium including CPI/NFP**. A
seed-averaged synthetic control (100 seeds) proves the machinery detects a planted premium and
manufactures nothing under the null. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "announcement days" are, why a third of the market's return lands on an eighth of its days, why that sounds like a money machine and isn't — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | pooled Welch *t* + random-calendar placebo, per-type and per-decade splits, the TLT check, costs × turnover on the A-day overlay, the ex-FOMC third axis, and a 100-seed synthetic power/null control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`macro_announcement_premium/`](macro_announcement_premium/). The calendar is actual
release dates (not a weekday pattern), hardcoded with sources in `data.py`. SPY/TLT are
survivorship-clean index vehicles. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
