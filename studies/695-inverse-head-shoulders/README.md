# Study 695 — Inverse Head-and-Shoulders 📈🗿

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the confirmed neckline break predict anything? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across **229** confirmed inverse-H&S breakouts on SPY + 29 large-caps (21.5y), the forward excess over base rate never clears **t ≥ 2** at any of four horizons (best: 10d, HAC **t = 1.56**), and every random-date placebo lands at **p ≈ 0.4–0.5**. Survives a detector-strictness sweep. Survivorship tilts *for* the figure — and it still fails. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A long-timer (hold to the measured-move target or 126-day timeout) shows a healthy-looking gross *t* ≈ 5 — against **zero**. Raced against a holding-period-matched base rate, the excess is **−0.15% at HAC t = −0.23**. Nothing to deploy; costs only make it worse. |
| **"Measured-move target forecasts the price"?** | ![Busted](https://img.shields.io/badge/Measured--move_target%3F-Busted-8b949e?style=flat-square) | The classic target (head-to-neckline height, projected up) is hit **74.2%** of the time — *below* a magnitude-matched random-entry placebo (**77.7%**). A random move of the same size gets there more often than the pattern does. |

> **In one sentence:** a clean, objective detector for the textbook "three troughs, deepest in the middle, confirmed neckline break" figure finds 229 inverse head-and-shoulders bottoms across two decades of large-caps, but the breakout beats the stock's own drift by nothing certifiable (best HAC *t* = 1.56 across four horizons, placebo *p* ≈ 0.4–0.5), the long-hold "measured-move" trade earns exactly what buying-and-holding for the same period earns (excess *t* = −0.23), and the target itself is hit *less* often than a random move of the same size — so the pattern is a shape the eye loves, a target the math doesn't back, and the tape ignores both.

## What we tested

Chart figures are **partly subjective**, so we wrote down the closest **mechanical** definition we could and said so: three swing-pivot troughs (symmetric shoulders, a strictly deeper head, a near-horizontal neckline through the two intervening swing highs), then a **confirmed close above the neckline** as the entry. Running it on a fixed **30-name large-cap basket + SPY** (yfinance auto-adjusted daily OHLC, 2005 → 2026-06, as-of 2026-06-30), we measure the forward **5/10/20/40-day** return after each breakout, entering the next day's close (no look-ahead), **net of each name's own base rate**. The Signal axis tests the pooled excess with one-sample and HAC *t* and a same-tape random-date placebo; Tradability runs a long timer that holds to the pattern's own **measured-move target** or a 126-day timeout, net of costs, raced against a holding-period-matched base rate; the myth-check asks whether that target is hit more often than a magnitude-matched random move. A deterministic synthetic control confirms the harness stays quiet on pure noise (20 seeds, |t| ≥ 2 in only 2/20) and lights up on a planted continuation edge (*t* = 8.23). Survivorship (a surviving-names basket, which tilts *for* the figure) is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an inverse head-and-shoulders is, a real detected example drawn by the code, why "three troughs" feels like a floor but mostly isn't, and why the measured-move target is weaker than a coin flip of the same reach — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the swing-pivot detector, forward 5/10/20/40-day excess over base rate, one-sample + HAC *t*, the same-tape placebo, a detector-strictness sweep, the magnitude-matched measured-move test, the long-timer race against a holding-matched base rate, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`inverse_head_shoulders/`](inverse_head_shoulders/). Detector is one mechanical definition of a partly-subjective figure — said loudly on the Signal axis. Basket is **survivors** (tilts *for* the figure) — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
