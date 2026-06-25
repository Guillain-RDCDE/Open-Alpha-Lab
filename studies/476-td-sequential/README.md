# Study 476 — TD Sequential (DeMark 9-13) 🔢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a completed 9 call the bottom? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The buy setup-9 *hints* at a 20-day edge over a drift-matched **random-entry** baseline (Δ = **+91 bps**), but it does **not** survive a properly sampled baseline: empirical *p* = **0.060** at 20d (the closest horizon), and the **seed-averaged** Welch *t* is only **+2.07** at 20d / **+0.82** at 60d. The eye-catching single-seed Welch *t* (+2.91 / +3.31) and one-sample *t* (+5.08 / +5.32) are mostly **beta** — the upward drift every dip-buy inherits. DeMark's deeper "13" is *weaker*, not stronger. A near-miss, not a confirmed signal. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Whatever faint tilt exists is generic dip-buying drift, not a scalable edge; costs only deepen the hole, and 239 setups over 21 years (~11/yr) give no capacity. You'd capture the same drift more cheaply by **holding the index**. |
| **"Does the 9-13 count forecast exhaustion?"** | ![Busted](https://img.shields.io/badge/Forecasts_exhaustion%3F-Busted-8b949e?style=flat-square) | Replace DeMark's exact 4-bar count with any other lookback (scrambled-lookback placebo) and the result barely moves: **67%** of scrambled counts match or beat the canonical one (*p* = **0.673**). And the "stronger" 13-countdown is the *weaker* signal. The specific 9-13 geometry carries no exhaustion information. |

> **In one sentence:** TD Sequential's 9-13 count *almost* clears the bar — fire the "go long on a completed buy setup" rule 239 times across 5 indices over 21 years and at 20 days it beats random by +91 bps — but the win evaporates under a properly resampled baseline (empirical *p* = 0.06, never < 0.05), the scrambled-lookback placebo leaves it untouched (*p* = 0.67), and DeMark's deeper "13" is the *weaker* signal: a near-miss that is mostly drift, not a forecast of exhaustion.

## What we tested

We encode the canonical DeMark mechanics verbatim (no eyeballing — the count is fully algorithmic). A **TD Buy Setup** is nine consecutive closes each strictly below the close **four bars earlier**; its completion (the "9") fires a long, entered at the **next close** (one documented lag). A **TD Buy Countdown** runs to thirteen "close ≤ low two bars earlier" rungs (the "13"), with standard recycling on a fresh setup. We measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **setup-9 vs a drift-matched random-entry baseline** — sampled over **200 seeds**, because a single random draw can flatter or flatten the Welch *t* by ±3 — plus a **scrambled-lookback placebo** that swaps DeMark's exact 4-bar comparison for other offsets, the direct test of "does the specific count matter?". Tradability charges costs on every signal. A deterministic synthetic control with a *planted* post-setup exhaustion bounce proves the detector is live (edge 0 → *t* = +1.28; planted bounce → *t* = +6.55, win 79%), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what TD Sequential's 9-13 count is, why a dip-buy on a rising market always looks good, the setup-vs-random race, why a lucky seed faked significance, and the lookback scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the mechanical 9-13 count, one-sample HAC *t* vs the beta trap, the seed-averaged random-entry test, the empirical *p*, the scrambled-lookback placebo, the weaker "13", per-ticker, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`td_sequential/`](td_sequential/). Setup = 9 consecutive closes < close-4; countdown = 13 rungs of close ≤ low-2; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument timing study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
