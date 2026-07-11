# Study 699 — Butterfly-Harmonic 🦋

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does price reverse where D overshoots X by 1.27-1.618x? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The pooled 5-day D-touch fade averages **+133.75 bps/event** (HAC *t* = **+2.28**; Welch *t* = **+2.18** vs a matched-direction random-day base rate) — clears the naive *t* ≥ 2 bar on its own. But it's built from only **50 touches over 21-25 years**, **0 of 7** Bonferroni-corrected pooled/per-ticker tests survive (critical \|*t*\| = 2.69), the 1-/10-day horizons show no persistence, SPY alone point-estimates *negative*, and it does **not** statistically beat a placebo extension projection (*t* = +0.55). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The pattern completes roughly **once every 4-5 years per ticker**; the apparent edge lives in a narrow 3-5 day hold and is gone by day 10; nothing survives once the six-ticker multiple-comparison penalty is charged. |
| **Beats a random extension-and-reversal placebo?** | ![Mixed](https://img.shields.io/badge/Beats_a_placebo%3F-Mixed-8b949e?style=flat-square) | Beats its own placebo control on 4 of 6 tickers and at the pooled short-horizon means — a numerical majority — but the pooled Welch *t* is only **+0.55**, far short of significance. |

> **In one sentence:** the Butterfly's signature — a D point extending **1.27-1.618×
> past** the original X point (not back toward it, the way Gartley/Bat/Crab do) —
> produces a pooled 5-day fade that *looks* real (uncorrected *t* ≈ 2.2-2.3 on just
> 50 touches across six tapes since 2001/2010) until you charge it for testing six
> tickers (**0/7 Bonferroni-corrected tests survive**) or race it against a placebo
> extension zone (**t = 0.55**) — a real-looking number that does not survive the
> desk's own robustness checks.

## What we tested

We encode the Butterfly mechanically off confirmed zig-zag pivots (no hand-labelling,
no look-ahead): four consecutive confirmed swings X, A, B, C where AB retraces XA by
**0.786 ± 0.06** and BC retraces AB in **0.382-0.886**, projecting **D = X − ext ×
(A − X)** with `ext` in the Butterfly's own **1.27-1.618** band — the ratio that sends
D *past* X, unlike every other member of the harmonic zoo. We scan forward for the
first touch of D and measure the fade's forward return on SPY, QQQ, AAPL, MSFT, TSLA
and NVDA daily bars (the identical basket as siblings
[468-gartley-harmonic](../468-gartley-harmonic/) and
[698-abcd-harmonic](../698-abcd-harmonic/)) — against **two** independent controls: a
random-day base rate matched on direction (kills the drift confound), Bonferroni-
corrected across the six-ticker breakdown, and a placebo arm that reruns the identical
pivots with randomized, off-Butterfly retrace/extension targets (kills the "any
equal-legged extension" confound). Only a result that clears **both** would be
evidence the specific 0.786/1.27-1.618 grid matters.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Butterfly pattern actually looks like on a chart, why D is supposed to be special, and what happens when you actually test the fade |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the pivot-confirmation mechanics, the HAC/Welch splits, the random-day base rate, the Bonferroni correction, the fade timer with costs, the placebo control, and the 20-seed synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`butterfly_harmonic/`](butterfly_harmonic/). No survivorship — six currently-listed,
individually named large-cap/ETF tickers, not a membership-conditioned panel.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
