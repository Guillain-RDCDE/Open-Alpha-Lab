# Study 383 — SOFR-Repo-Stress 🚨

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a repo spike warn risk assets? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | There's a faint, correctly-signed 5-day dip (SPY cond **−0.71%** vs base **+0.29%**), but it **fails \|t\| ≥ 2** (best is LQD **t = −1.65**) and **flips positive within a month** — SPY is **+1.95% / +5.21%** at 20/60 days. On the *systemic-only* subset the only significant result is **+12.2% at 60d** (*t* = +2.18, the **wrong** sign). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | "De-risk on the spike" earns a noise-level **+0.66%** net at 5 days, then **loses 2–5%** at 20/60 days (you sell into the rebound). **13 trades in 16.5 years** — wrong-signed where it matters, and never NAV-scale. |
| **"Repo spike = flee"?** | ![Busted](https://img.shields.io/badge/Repo_spike_%3D_flee%3F-Busted-8b949e?style=flat-square) | A **sample-size + confounding mirage**: the iconic **Sept 2019** spike was followed by SPY ≈ **flat**; **March 2020** and **SVB 2023** were *bottoms* (+10.5% / +6.7%). The one crash everyone cites is **COVID, not repo**. |

> **In one sentence:** the "watch the repo market" oracle — born from the September 2019 spike — is what a rare, vivid, confounded event-list looks like in a market that gets backstopped most times the plumbing seizes: on 13 named episodes the forward reaction is a faint sub-significant 5-day dip that **inverts into a relief rally** by 20–60 days, so it is no signal, a money-losing trade, and a busted myth (its one "proof" crash, March 2020, is COVID).

## What we tested

There's no free, clean daily SOFR / repo-spread tape on yfinance, so we test the folklore against a **hardcoded, sourced table of 13 named repo-stress episodes** — September 2019, the quarter- and year-end funding turns, March 2020, SVB 2023, and the rest — and measure the forward reaction of the assets it's said to warn: **SPY** (equities) and **HYG / LQD** (credit), over **16.5 years** (2010–2026). Each episode enters one trading day after the spike (no look-ahead), held 5 / 20 / 60 days, against the unconditional base rate, with a Welch *t* and a 20,000-draw placebo null sized to the event count. A deterministic synthetic control with an *injected* bearish edge confirms the engine is faithful **and** that ~a dozen events can't reach significance unless the planted edge is implausibly large. (Same rare-event / confounding pathology as the desk's other "stress oracle" studies.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "watch the repo market" feels prophetic, what Sept 2019 actually was, and why the crash you remember (March 2020) is the opposite of the lesson — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | event-window forward returns on SPY/HYG/LQD, conditional vs unconditional means, a Welch *t* + placebo randomization null, the sign-flip by horizon, costs, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sofr_repo_stress/`](sofr_repo_stress/). The repo-stress signal here is an explicit **hardcoded event list** (no free SOFR feed), not a live funding-stress tape. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
