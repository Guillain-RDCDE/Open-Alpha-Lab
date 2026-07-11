# Study 683 — Evening-Star 🌆

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a bearish edge? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Shorting the evening star (enter next open, one lag), measured **fairly** against the basket's own unconditional drift (Welch *t*, the decisive number — a plain *t*-vs-zero is contaminated by the tape's up-drift, proven on the synthetic control) — **never clears \|t\| ≥ 2** at 1/5/10 days (**−0.53 / −1.84 / −0.43**). Hit rates sit at or above the unconditional base rate. Coin-flip placebo *p* ≈ **0.84–1.00**. **0/30 tickers** survive a Bonferroni correction individually. Strict-gap and prior-uptrend myth checks don't rescue it. **Survivorship** caveat tilts the test *toward* the claim — and it still fails. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Negative **gross** at every horizon (**−0.06% to −0.61%**); net of a 5/10-bps round trip + 50-bps/yr short borrow it's worse (**−0.16% to −0.83%**). Nothing under the signal to charge costs against. |
| **"Crash incoming"?** | ![Busted](https://img.shields.io/badge/Crash_incoming%3F-Busted-8b949e?style=flat-square) | The "top of the uptrend, sell now" reading does not survive contact with 21.5 years of tape once benchmarked fairly. No single name in the basket carries a Bonferroni-robust version of the story, and the two textbook-purity filters (strict gap, genuine prior uptrend) make it *weaker*, not stronger. |

> **In one sentence:** the classic three-candle "top" signal — tall up day, small gapping
> star, tall down day — shows **no certifiable bearish edge** on 21.5 years of real tape
> once measured against what the same 30-name-+-SPY basket does on an ordinary day (Welch
> *t* never reaches 2, coin-flip placebo *p* ≈ 0.84–1.00, 0/30 tickers survive a Bonferroni
> correction), it **loses money before costs** if shorted, and neither a purist's strict
> gap nor a genuine prior-uptrend filter saves it — on a survivor basket deliberately
> stacked in the lore's favour.

## What we tested

We rebuild the evening star as a clean signed-**short** event study on a fixed **30-name
liquid US large-cap + SPY** basket (yfinance daily OHLCV, 2005→2026, 162,180 bars). A
precise OHLC detector flags **every** occurrence (a tall bullish body, then a small
"star" body gapping up from it, then a tall bearish body closing ≥50% back into the first
candle); we wait for the confirming close, enter the **next open** (one execution lag),
and measure the forward **1 / 5 / 10-day** return held short. The Signal axis's decisive
number is a **drift-neutral Welch *t*** against the basket's own unconditional forward
return (not a plain *t*-vs-zero, which the synthetic control shows is contaminated by the
tape's own up-drift) plus a hit-rate-vs-base-rate and a 5,000-draw coin-flip placebo; a
**Bonferroni correction across the 30-ticker basket** checks whether any single name
quietly carries the effect; Tradability charges a 5/10-bps round-trip cost + short borrow.
Two myth-checks ask whether the **strict textbook gap** or a **genuine prior uptrend**
filter helps. A deterministic synthetic control with a *planted* post-pattern crash
confirms the engine would catch a real one (it lights up at *t* = 3.5–5.7 over 20 null
seeds) and that zero edge cannot fake significance under the fair comparison.
Survivorship — the basket excludes firms that actually topped, crashed and delisted,
biasing the test *toward* the claim — is named on the Signal axis. **Dedup:**
[186-morning-star](../186-morning-star/) is the bullish mirror (and already runs a looser
evening-star arm that lands on the same null, *t* = −0.18, with a different control and a
smaller basket); [404-shooting-star](../404-shooting-star/) is a single-candle pattern
with no star/gap structure; [408-three-black-crows](../408-three-black-crows/) is three
tall red candles with no star or gap; [402-engulfing-pattern](../402-engulfing-pattern/)
is a two-candle reversal. None of them run this study's three-candle, gap-and-penetration
detector against an unconditional-base, Bonferroni-corrected bar.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the pattern is, why comparing to "zero" overstates the case, why shorting it loses, and why no stricter recipe saves it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the precise detector, the drift-neutral Welch-*t* design, the signed-short 1/5/10-day event study, the coin-flip placebo, the Bonferroni correction across the basket, costs + borrow, the strict-gap & prior-uptrend myth checks, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`evening_star/`](evening_star/). Detector is the precise real-body evening star
(strict-gap + prior-uptrend variants for the myth-check). Basket is **survivors** — named
on the Signal axis (and it cuts *toward* the claim). **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
