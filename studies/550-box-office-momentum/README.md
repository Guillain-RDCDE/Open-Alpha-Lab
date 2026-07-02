# Study 550 — Box-Office-Momentum 🎬

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does box office lead media/market returns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The media-stock lead is **absent** (naive predictive *t* **+1.53**, below the bar; placebo *p* **0.139**), and it **collapses to *t* +0.65** once you control for the contemporaneous market. The broad-tape "signal" (*t* **+3.42**) is a **placebo-fooling common-factor artifact** (stays *t* +3.11 under an imperfect control). And there is **no survivorship-honest free box-office tape** — synthetic-only, so capped at `NONE`. |
| **Tradability** — does a signal-timer pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The long/flat timer beats buy-and-hold by only **+0.98 pp/yr** net (5 bps/switch, ~25 switches/yr) — the same factor-persistence artifact — and it **flips sign across thresholds** (−2.5 pp/yr at a looser cutoff). Nothing to harvest. |

> **In one sentence:** "busy cinemas foreshadow media-stock gains" is a `NONE` × `MIRAGE`: on an honest test the media lead vanishes once you control for the contemporaneous market (*t* +0.65), the broad-tape "signal" (*t* +3.42) is a persistent-common-factor artifact that even fools a naive placebo, and no survivorship-honest free box-office tape exists — while a seed-robust synthetic control proves the engine *would* catch a real lead if one were there.

## What we tested

The alt-data folklore: a strong **weekend box office** is a **consumer-sentiment leading
indicator** for media/studio stocks and even the broad tape. We build a box-office **momentum**
signal (this weekend's gross vs its trailing norm) and test whether it predicts *next-week* forward
returns via a **predictive regression**, a **circular-shift placebo** null, a
**confound-controlled** regression (adding the contemporaneous market to separate a genuine lead
from shared-factor co-movement), a **signal-timed long/flat strategy** net of costs, a
**threshold-robustness sweep**, and a **seed-robust synthetic positive control** (25 seeds) that
plants a real lead and proves the engine catches it while staying flat at the null. **Data caveat
(SIGNAL axis):** there is no free, survivorship-honest historical box-office API, and the studios you
would trade have been merged/restructured out of any clean panel — so this is a **synthetic-only**
study and can never earn `REAL`. *Distinct from the sentiment/alt-data neighbours —
[257 AAII-Sentiment](../257-aaii-sentiment/), [300 Sports-Sentiment](../300-sports-sentiment/),
[335 Buzz-Sentiment-ETF](../335-buzz-sentiment-etf/), [271 Cardboard-Box](../271-cardboard-box/) —
this one is the **box-office** variant, built around the common-consumer-factor confound.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "box office predicts stocks" means, why cinemas and markets rise together, and why that co-movement isn't a tradable lead |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the predictive regression + *t*, the circular-shift placebo, the confound-controlled slope, the SPY mirage, costs + threshold sweep, and the seed-robust synthetic positive control |

The fingerprinted synthetic headline run (null panel fp `985a49d136c2`, positive-control fp
`4efecda3764a`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery
runs on the deterministic synthetic world in
[`box_office_momentum/data.py`](box_office_momentum/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`box_office_momentum/`](box_office_momentum/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
