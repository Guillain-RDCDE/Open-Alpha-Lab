# Study 440 — Floor-Trader Pivot Points 📏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do pivots act as support/resistance? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The pivot touch→bounce rate is **0.467** — *below* both a randomly-placed control line (**0.501**) and a coin flip. The fade trade is **−1.12 bps**/touch at **HAC t = −1.04**, and a 300-draw permutation placebo gives **p = 1.00** (a random line is at least as good as the pivots). Short-span / 5-name survivorship caveat on this axis. |
| **Tradability** — can you fade the touch? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The fade is **−1.12 bps**/touch *gross* and **−3.12 bps** net of a generous 2-bp round trip. Negative before costs — nothing to size, nothing to scale. |
| **"Price respects the level"?** | ![Busted](https://img.shields.io/badge/Respects_the_level%3F-Busted-8b949e?style=flat-square) | An arbitrary horizontal line bounces *more often* than the floor-trader pivots. A synthetic positive control proves the harness detects a planted bounce at **t = 7.79**, so this is a true null, not a blind test. |

> **In one sentence:** the classic floor-trader pivots (P, R1–R3, S1–S3 from the prior session's H/L/C) bounce price *less* often than a line drawn at a random price on the same 5-minute tape (46.7% vs 50.1%, fade HAC t = −1.04, placebo p = 1.00) and the fade trade is negative before you even pay the spread — price respects the pivots no more than it respects a line you drew with your eyes shut.

## What we tested

We rebuild the floor-trader recipe exactly: each morning compute `P = (prior High + Low + Close)/3` and its `R1–R3 / S1–S3` extensions, then on **5-minute** bars for **5 of the most liquid US names** (SPY, QQQ, AAPL, MSFT, IWM, ~59 full sessions over ~12 weeks) we find every **touch** of a level (a bar straddling it within 5 bps) and ask whether price **bounces away** over the next 6 bars (30 min), entered one bar after the touch (no look-ahead). The only honest yardstick for a support/resistance claim is a **line drawn at a random price** on the same tape — because *any* line gets touched and bounced off some of the time — so we run the identical machinery on random control lines, score the fade trade with a HAC *t* against zero, and put a *p*-value on the bounce-rate gap with a 300-draw permutation placebo. A deterministic synthetic control with a *planted* bounce confirms the harness can detect a real level when one exists. **Loud caveats:** Yahoo caps 5-minute history at ~60 days (short span — enough to reject a tradable effect, not to certify a tiny one), and intraday fades die to the bid-ask **spread**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what pivot points are, why "it bounces half the time" is the sound of a coin, the per-level picture, and why the spread closes the case — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | touch→bounce hit-rate vs a random-line control, the fade-trade HAC *t*, a 300-draw permutation placebo, intraday costs, and a synthetic planted-bounce positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`pivot_points/`](pivot_points/). Levels are the classic floor-trader set from the prior session's H/L/C; the baseline is a uniform-random horizontal line on the same tape. Real tape is **5 surviving liquid names** on a ~12-week 5-minute window — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
