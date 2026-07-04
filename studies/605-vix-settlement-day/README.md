# Study 605 — VIX Settlement Day 🎯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the settlement Wednesday leave tracks in ^VIX? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the structure, none on the level.* On normal Wednesdays the VIX morning gap **mean-reverts** into the close (slope −0.18, *t* = −2.3); on the 270 settlement Wednesdays it **flips to continuation** — interaction **+0.427** (White *t* = **+2.56**), **+0.497** (*t* = **+2.86**) ex-FOMC, random-calendar placebo **p = 0.0025**, robust to the Yahoo stale-open screen and winsorising. But every **level** test (open jump, day volatility, range, SPX drift) **dies once the 40 settlement-FOMC overlap days are stripped** (all ex-FOMC *t* < 2) — the "louder day" was mostly the Fed. |
| **Tradability** — can you cash the track? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The VIX **index is not tradable**; the sign-rule paper value is **−0.02%/event gross** (*t* ≈ 0) and **−6.3%/yr** net of futures-level costs (2 × 25 bps, 12 events/yr) — and the *expiring* contract has already printed its SOQ when the signal exists. An OLS moment, not a trade. |
| **"Did it fade after the 2018 paper + lawsuits?"** | ![Mixed](https://img.shields.io/badge/Faded_after_2018%3F-Mixed-8b949e?style=flat-square) | The extra-volatility level effect vanished (\|Δclose\| Welch *t* **+3.10 → +0.18**) — but the continuation signature's point estimate **did not budge** (+0.468 → +0.490; *t* = 1.53 on the 82 post-2018 settlements, under-powered, not gone). |

> **In one sentence:** Griffin-Shams' settlement-morning pressure does leave a measurable
> track in the daily ^VIX tape — settlement Wednesdays are the one day the morning gap
> *carries into the close* instead of fading (interaction *t* = +2.9 ex-FOMC, placebo
> p = 0.0025) — but the loud "wilder day" story is mostly the FOMC confounder, the footprint
> is uncashable (Mirage), and it never quite went away after the lawsuits.

## What we tested

We rebuild the monthly VIX-futures **final-settlement calendar by rule** (the Wednesday
30 days before the following month's S&P-500 option expiry, holiday-adjusted — asserted
against 18 known CBOE dates, including all 7 holiday-shifted Tuesdays), then ask the daily
^VIX/^GSPC tape (2004-2026, 270 settlements vs 899 other Wednesdays) whether settlement
Wednesdays behave differently: Welch *t* on the planned level metrics (overnight jump,
intraday move, day range, SPX return), a White-robust **settlement × gap interaction** for
the *structural* signature (does the distorted morning print carry into the day?), a
2,000-draw / 25-seed random-calendar placebo, an explicit **FOMC-overlap control** (40 of 270
settlements are statement days), and a pre/post-2018 fade test. Tradability charges 25 bps
one-way × 2 on the only sign-tradable expression. A deterministic synthetic world with a
planted continuation slope proves the machinery (null quiet, planted +0.50 recovered at
*t* = +5.2). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the VIX settlement auction is, why a manipulator would push it, what "the gap that refuses to fade" looks like, and why you still can't trade it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch level table with the FOMC confounder exposed, the interaction regression + placebo, stale-open and winsorising robustness, the 2018 fade split, and the synthetic faithful-engine control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).
Siblings: this is the **settlement-day microstructure event** — distinct from
[111-vix-term-structure](../111-vix-term-structure/) (curve shape/carry) and
[375-vxx-roll-decay](../375-vxx-roll-decay/) (ETP roll drag).

---

*Engine: [`vix_settlement_day/`](vix_settlement_day/). The signal is the rule-built settlement
calendar × the daily bar structure; no survivorship (indices). **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
