# Study 693 — Tasuki Gap 🕳️🕯️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the trend actually resume? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Entered next open, one lag, measured **fairly** against the basket's own direction-matched unconditional drift (Welch *t*, the decisive number) — the trend-signed return **never certifies a continuation** at 1/5/10/20 days (**+0.15 / −2.33 / −1.29 / −0.05**). The only reading past \|t\| ≥ 2 (5-day) points the **wrong way**: a mild reversal on the downside leg (*t* = −2.58), not a continuation. Mix-matched placebo puts it at **p = 0.995** — worse than 99.5% of random draws. **0/30 tickers** survive a Bonferroni correction individually. Neither the strict-gap nor the genuine-prior-trend myth check rescues it — the trend filter makes the dip *worse*. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Flat-to-negative **gross** at every horizon (**+0.01% to −0.26%**); net of a 5/10-bps round trip + 50-bps/yr short borrow it's worse everywhere (**−0.09% to −0.46%**). Nothing under the signal to charge costs against. |
| **"Predicts continuation?"** | ![Busted](https://img.shields.io/badge/Predicts_continuation%3F-Busted-8b949e?style=flat-square) | The "gap can't be closed, the trend has force left" reading does not survive contact with 21.5 years of tape once benchmarked fairly. If anything, the downside leg leans toward a faint (non-Bonferroni-robust) *reversal* at intermediate horizons — the opposite of the textbook claim. |

> **In one sentence:** the tasuki gap — a same-color gap pair, then an opposite-color
> pullback candle that can't quite close the gap — shows **no certifiable continuation
> edge** on 21.5 years of real tape once measured against what the same 30-name-+-SPY
> basket does on an ordinary day in the same direction (Welch *t* never certifies a
> continuation at any of 4 horizons, the loudest single reading points *backwards*,
> mix-matched placebo *p* = 0.995, 0/30 tickers survive a Bonferroni correction), it
> **loses money after costs** at every horizon, and neither a purist's full-range gap
> nor a genuine prior-trend-context filter saves it.

## What we tested

We rebuild the tasuki gap as a clean trend-**signed** event study on a fixed **30-name
liquid US large-cap + SPY** basket (yfinance daily OHLCV, 2005→2026, 162,180 bars). A
precise OHLC detector flags **both** shapes — the upside tasuki (two white bodies
gapping up, then a black body that opens inside the second body and closes back inside
the gap without filling it: long) and the downside tasuki (the exact mirror: two black
bodies gapping down, then a white pullback body: short); we wait for the confirming
close, enter the **next open** (one execution lag), and measure the forward **1 / 5 /
10 / 20-day** return signed toward the pattern's predicted trend direction. The Signal
axis's decisive number is a **direction-matched Welch *t*** against the basket's own
unconditional forward return (long events vs the plain pool, short events vs its
negation — not a plain *t*-vs-zero, which the synthetic control shows is contaminated by
the tape's own drift) plus a hit-rate-vs-base-rate and a mix-matched 5,000-draw coin
placebo; a **Bonferroni correction across the 30-ticker basket** checks whether any
single name quietly carries the effect; Tradability charges a 5/10-bps round-trip cost
+ short borrow on the down leg only. Two myth-checks ask whether a **strict full-range
gap** or a **genuine prior-trend-context** filter helps (neither does — the trend filter
makes the dip *worse*). A deterministic synthetic control with a *planted* post-pattern
continuation confirms the engine would catch a real one (it lights up at *t* = 8.1–13.9)
and that zero edge cannot fake significance under the fair comparison (1/20 null seeds
fire, in line with the ~5% nominal false-positive rate of a two-sigma test). Survivorship
— the basket excludes firms that never resumed and delisted, and (symmetrically) excludes
firms that never stopped falling — is named on the Signal axis. **Dedup:**
[74-mind-the-gap](../74-mind-the-gap/) is the general gap-**fill-rate** question across
all gap shapes, the opposite of this pattern's specific don't-fill-then-continue claim;
[455-three-methods](../455-three-methods/) is a five-candle continuation pattern with
**no gap** and three inside-range consolidation candles, not a gap-then-pullback shape;
[417-island-reversal](../417-island-reversal/) needs **two** opposite-direction gaps
predicting a **reversal** — the opposite claim — where this study has exactly one gap a
pullback candle fails to close, predicting continuation. None of them run this study's
specific two-same-color-gap-then-opposite-pullback, trend-signed, Bonferroni-corrected
bar.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the pattern is, why comparing to "zero" overstates the case, why the pullback doesn't reliably mean "resume," and why no stricter recipe saves it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the precise detector, the direction-matched Welch-*t* design, the signed 1/5/10/20-day event study, the mix-matched coin placebo, the Bonferroni correction across the basket, costs + borrow, the strict-gap & prior-trend myth checks, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`tasuki_gap/`](tasuki_gap/). Detector is the precise real-body tasuki gap
(strict full-range-gap + prior-trend-context variants for the myth-check). Basket is
**survivors** — named on the Signal axis. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
