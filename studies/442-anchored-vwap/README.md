# Study 442 — Anchored VWAP 🧲

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does price respect the level? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The post-touch reversion of the anchored VWAP is **+0.19 bp** at 30 minutes (one-sample *t* = **0.19**, HAC *t* = **0.29**), flat across all horizons, win-rate **50%** — on the **most liquid US tape** (SPY/QQQ/AAPL/MSFT/NVDA, 5-minute bars). Nowhere near the **t ≥ 2** bar. Carries an explicit **short-span** caveat (~59 sessions, the ~60-day yfinance 5m limit). |
| **Tradability** — can you trade it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of a 1-bp half-spread × 2 legs every horizon is **negative** (6-bar net **−1.81 bp**); even a 0.5-bp half-spread nets **−0.81 bp**. The gross reaction is thinner than the spread you must cross to capture it. |
| **"Anchored VWAP is a price magnet"?** | ![Not supported](https://img.shields.io/badge/Price_magnet%3F-Not_supported-8b949e?style=flat-square) | A **randomly placed** horizontal level reacts just as strongly (control **+0.35 bp** vs AVWAP +0.19 bp; Welch *t* = **−0.16**) and matches/beats the AVWAP **71%** of the time (placebo *p* = **0.711**). The line is an *average*, not a magnet. |

> **In one sentence:** the anchored-VWAP "magnet" is an optical illusion of selection — on 59 sessions of 5-minute tape for the five most-liquid US names, a touch of the AVWAP is followed by a +0.19 bp coin-flip (HAC *t* = 0.29), the line is no more respected than a level drawn at **random** (placebo *p* = 0.71), and even that sliver is eaten by the spread (net −1.81 bp), while a deterministic synthetic control with a *planted* magnet lights the detector up to *t* = 4.7 — so the flat tape is a genuine "no," not a broken test.

## What we tested

We rebuild the strongest, cleanest version of the AVWAP claim on intraday tape: anchor the VWAP to the **09:30 session open** (a rule-based anchor — the discretionary versions hand-pick the anchor, which is where the selection bias lives), detect every bar where price **crosses** the line, and measure the reversion-oriented reaction over the next **1 / 3 / 6 / 12** five-minute bars with a one-bar execution lag (no look-ahead). Because a running VWAP is *an average* — price crosses it constantly and "respects" it trivially — the decisive test is whether the AVWAP beats a **horizontal level placed at random** in the same session range: a one-sample / **HAC** *t* against zero, a Welch *t* of AVWAP minus control, and a 2,000-draw random-level placebo, then the **bid-ask spread** charged on both legs. A deterministic synthetic tape with a *planted* magnet confirms the detector fires when there is something to find. Data: **yfinance 5-minute bars** (SPY/QQQ/AAPL/MSFT/NVDA, 2026-03-30 → 2026-06-23) — a short, friendliest-spread window, stated loudly as a power/capacity caveat.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what anchored VWAP is, why the screenshots fool you (any average gets touched a lot), the random-line test, and why the spread eats it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the touch detector, post-touch reversion by horizon, one-sample + HAC *t*, the random-level placebo, per-name dispersion, a cost sweep, and a planted-magnet synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`anchored_vwap/`](anchored_vwap/). Anchor = the 09:30 session open. Real tape is yfinance 5-minute bars on the 5 most-liquid US names — a deliberately **short** (~60-day) window, named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
