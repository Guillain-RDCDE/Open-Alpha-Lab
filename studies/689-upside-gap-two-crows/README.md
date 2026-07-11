# Study 689 — Upside Gap Two Crows 🐦‍⬛🕳️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a bearish edge? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Shorting the upside gap two crows (enter next open, one lag), measured **fairly** against the basket's own unconditional drift (Welch *t*, the decisive number) — **never clears \|t\| ≥ 2** at 1/5/10 days (**+0.55 / +0.52 / −0.01**). Hit rates track the unconditional base rate. Coin-flip placebo *p* ≈ **0.44–0.95**. **0/26 tickers** survive a Bonferroni correction individually (and the two loudest survivors disagree on direction). Strict-gap and prior-uptrend myth checks don't rescue it. **Survivorship** caveat tilts the test *toward* the claim — and it still fails. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Flat-to-negative **gross** at every horizon (**+0.01% to −0.53%**); net of a 5/10-bps round trip + 50-bps/yr short borrow it's worse (**−0.09% to −0.65%**). Nothing under the signal to charge costs against. |
| **"Marks a top"?** | ![Busted](https://img.shields.io/badge/Marks_a_top%3F-Busted-8b949e?style=flat-square) | The "unfilled gap = exhaustion, the top is in" reading does not survive contact with 21.5 years of tape once benchmarked fairly. No single name in the basket carries a Bonferroni-robust version of the story, and the two textbook-purity filters (a strict full-range gap, a genuine prior uptrend) don't rescue it either. |

> **In one sentence:** one of the rarest, most dramatic-looking three-candle "tops" in
> the candlestick canon — a bullish day, a gap-up black crow, a second black crow that
> digs into the gap but can't close it — shows **no certifiable bearish edge** on 21.5
> years of real tape once measured against what the same 30-name-+-SPY basket does on an
> ordinary day (Welch *t* never reaches 2, coin-flip placebo *p* ≈ 0.44–0.95, 0/26
> tickers survive a Bonferroni correction), it **loses money after costs** at every
> horizon if shorted, and neither a purist's full-range gap nor a genuine prior-uptrend
> filter saves it — on a survivor basket deliberately stacked in the lore's favour.

## What we tested

We rebuild the upside gap two crows as a clean signed-**short** event study on a fixed
**30-name liquid US large-cap + SPY** basket (yfinance daily OHLCV, 2005→2026, 162,180
bars). A precise OHLC detector flags **every** occurrence (a bullish body, then a black
body gapping up from it, then a second black body that opens higher and closes lower
than the first — engulfing it from above — while staying above the first candle's
close, i.e. the gap is never fully filled); we wait for the confirming close, enter the
**next open** (one execution lag), and measure the forward **1 / 5 / 10-day** return
held short. The Signal axis's decisive number is a **drift-neutral Welch *t*** against
the basket's own unconditional forward return (not a plain *t*-vs-zero, which the
synthetic control shows is contaminated by the tape's own up-drift) plus a
hit-rate-vs-base-rate and a 5,000-draw coin-flip placebo; a **Bonferroni correction
across the 26-ticker-with-events basket** checks whether any single name quietly
carries the effect; Tradability charges a 5/10-bps round-trip cost + short borrow. Two
myth-checks ask whether the **strict full-range gap** or a **genuine prior uptrend**
filter helps. A deterministic synthetic control with a *planted* post-pattern crash
confirms the engine would catch a real one (it lights up at *t* = 2.4–4.9) and that zero
edge cannot fake significance under the fair comparison (2/20 null seeds fire, in line
with the ~5–10% nominal false-positive rate of a two-sigma test). Survivorship — the
basket excludes firms that actually topped, crashed and delisted, biasing the test
*toward* the claim — is named on the Signal axis. **Dedup:**
[408-three-black-crows](../408-three-black-crows/) is three consecutive red candles with
no gap requirement at all; [683-evening-star](../683-evening-star/) needs only one small
"star" body (any color) rather than two full black bodies straddling an unfilled gap;
[417-island-reversal](../417-island-reversal/) needs **two** opposite-direction gaps
sealing a stranded cluster, where this study has exactly one gap the crows fail to
close; [407-dark-cloud-piercing](../407-dark-cloud-piercing/) is a two-candle,
midpoint-penetration pair with the gap running *against* the trend, the opposite
geometry. None of them run this study's specific bullish-then-two-black-crows,
gap-not-closed detector against an unconditional-base, Bonferroni-corrected bar.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the pattern is, why comparing to "zero" overstates the case, why shorting it loses after costs, and why no stricter recipe saves it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the precise detector, the drift-neutral Welch-*t* design, the signed-short 1/5/10-day event study, the coin-flip placebo, the Bonferroni correction across the basket, costs + borrow, the strict-gap & prior-uptrend myth checks, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`upside_gap_two_crows/`](upside_gap_two_crows/). Detector is the precise
real-body upside-gap-two-crows (strict full-range-gap + prior-uptrend variants for the
myth-check). Basket is **survivors** — named on the Signal axis (and it cuts *toward*
the claim). **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
