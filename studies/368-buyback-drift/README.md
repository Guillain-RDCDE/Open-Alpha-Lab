# Study 368 — Buyback-Drift 🔁

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a stock drift up for months after a big buyback authorization? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The abnormal (stock − SPY) drift is **positive at every horizon** (+0.5 / +2.4 / +4.5 / +5.3% at 1 / 3 / 6 / 12m) — the direction the folklore and the academic underreaction story predict — but it **fails t ≥ 2 everywhere** (best **t = 1.21**, placebo *p* = **0.17**), the *median* event drifts only ~1%, and the win-rate sits at a single-name **coin-flip**. A positive-but-insignificant estimate dominated by single-stock variance, on a survivorship- and visibility-selected sample. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Costs are negligible (10 bps off a months-long hold), but the per-trade drift is **inside its own error bar**, and "bigger buyback ⇒ bigger drift" is **false** (the *bigger* programs drift *less*, t = 0.52). A drift you can't distinguish from zero on **32** events is not a NAV-scale strategy. |
| **"Free lunch"?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | A **base-rate + small-sample mirage**: the real, documented move is the **announcement-day jump** (~2–3%); enter the day *after* and strip the market, and the leftover months-long "drift" is what a **same-names coin-flip** matches ~1 time in 5. The jump is real; the tradable drift is not certified. |

> **In one sentence:** big share-buyback *authorizations* do leave a faintly positive abnormal drift — the sign the folklore and the 1995 underreaction literature predict — but across **32** notable mega-cap announcements it never clears **t = 2** (best t = 1.21), the typical event drifts ~1%, "bigger program ⇒ bigger drift" is backwards, and a same-names placebo can't reject luck, so the months-long drift is real-as-lore, weak-as-edge, and undeployable — the genuine move is the announcement-day pop, not the drift after it.

## What we tested

Clean, point-in-time feeds of repurchase **authorizations** aren't free on yfinance, so we hard-code a **transparent table of 32 notable mega-cap buyback authorizations** (ticker, announcement date, headline $bn size) and measure the **abnormal** drift — the event stock's forward return **minus SPY** over 1 / 3 / 6 / 12 months, so a market rally can't masquerade as buyback drift. We deliberately separate the **authorization** (the board/press-release OK of a $X program — what the headline reacts to) from **execution** (actual repurchases, which trickle out over years), and we enter **one day after** the announcement, so we measure the *drift* the claim is about, not the documented announcement-day jump. Inference is a one-sample *t* against zero plus a **same-names placebo null** (re-enter each name on random dates), with one-day lag and one-way costs. A deterministic synthetic control with an *injected* drift confirms the engine is faithful **and** that ~30 single-name events can't reach significance unless the planted drift is implausibly large.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an authorization is (and isn't), why "the stock drifts up for months" is mostly the market plus a few lucky names, and why 30-odd events can't be a strategy — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | abnormal-return event study, forward drift vs zero, a one-sample *t* + same-names placebo null, a size-split robustness cut, costs, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`buyback_drift/`](buyback_drift/). The event set is an explicit **hand-curated sample** of notable authorizations, not a point-in-time universe (survivorship + visibility named on the Signal axis). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
