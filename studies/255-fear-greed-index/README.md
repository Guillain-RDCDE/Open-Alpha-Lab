# Study 255 -- Fear-Greed

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Contrarian fear-minus-greed spread HAC *t* = **+0.41** over 767 weeks; block-bootstrap Sharpe 95% CI = [-0.24, +0.45], 33% of resamples negative; the real spread sits at the **47.5th percentile** of a shuffled-sentiment null. The band pattern is the *opposite* of the claim -- forward returns *rise* with greed (Greed band *t* = +5.33, an artefact of the bull trend), and Extreme-Fear weeks (+14.2%/yr) merely match the +14.0%/yr unconditional drift. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | There is no gross edge to erode: the overlay is flat 89% of weeks and even gross-of-cost the spread is *t* = 0.41. "Extreme Fear is a buy" is a story told about a ~14%/yr bull market, not a timing signal. |
| **Data caveat** | ![Curated proxy / price-only](https://img.shields.io/badge/Curated--proxy%20%C2%B7%20price--only-8b949e?style=flat-square) | The Fear & Greed series is a hardcoded month-end anchor table (no clean public archive) interpolated to weekly; ^GSPC is split- but not dividend-adjusted (price-only). The dividend-neutral spread is unaffected by the latter. |

> **In one sentence:** CNN's Fear & Greed Index does *not* time the market -- the contrarian fear-minus-greed overlay earns a statistically invisible +0.7%/yr (*t* = 0.41) that is indistinguishable from randomly shuffling the sentiment column, while a synthetic positive control confirms the same engine clears *t* = 3.85 when a real contrarian effect is planted.

## The claim

> *Can CNN's Fear & Greed Index time the market?*

## What we tested

Each Friday we read the curated weekly CNN Fear & Greed Index (0--100) and act
on the published contrarian folk rule: **long ^GSPC in Extreme Fear (F&G < 25),
short in Extreme Greed (F&G >= 75), flat otherwise**, held one week with a
one-week execution lag (read Friday's close, trade Monday's open). We bin
forward weekly returns by the five published sentiment bands, build the
dividend-neutral fear-minus-greed spread, regress forward return on a linear
sentiment tilt, and pin all of it against (a) the unconditional weekly drift,
(b) a 2,000-draw shuffled-sentiment null, (c) a 2011-2026 sub-period breakdown,
and (d) a 5 bps + 50 bps/yr-borrow cost sweep. A deterministic synthetic
positive control confirms the engine recovers a planted contrarian premium.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the contrarian story in plain language, what the index actually did at the COVID low and the 2021 highs, and why "buy fear" just rode a bull market |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | band conditional means, HAC *t*-stats, bootstrap Sharpe CI, shuffled-sentiment null, sub-period slices, cost sweep, linear sentiment tilt, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fear_greed_index/`](fear_greed_index/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
