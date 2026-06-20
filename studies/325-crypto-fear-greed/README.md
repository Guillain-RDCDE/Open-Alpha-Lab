# Study 325 -- Crypto-Fear-Greed 😨

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The contrarian long-fear / short-greed BTC spread is **-10.7%/yr, HAC *t* = -0.86** over 2014--2026 -- the *wrong sign*; block-bootstrap Sharpe CI [-0.87, +0.36] is 79% negative; real spread sits at the **17.9th percentile** of a shuffled-gauge null. The single strongest band is Extreme **Greed** (+173.8%/yr, *t* = +3.58) -- the opposite of the claim, a crypto-momentum artefact. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No gross edge to erode -- it loses before costs, and crypto-realistic short funding (1000 bps/yr) deepens it to -15.6%/yr. "Buy crypto fear" is a story told about a trending asset, not a timing signal. |
| **Gauge construction** | ![Price--derived_proxy](https://img.shields.io/badge/Price--derived_proxy-8b949e?style=flat-square) | The live alternative.me index is network-blocked, so we rebuild a transparent gauge from BTC's own vol + drawdown + momentum -- the index's largest weights. We certify the *price-derived* contrarian claim, not the social/search index. |

> **In one sentence:** a price-derived crypto Fear & Greed gauge does *not* time Bitcoin -- buying fear and shorting greed is a wrong-sign, *t* = -0.86 loser, while a synthetic control confirms the same engine clears *t* = +5.4 when real mean reversion is planted.

## What we tested

The crypto **Fear & Greed Index** is a 0--100 mood ring for Bitcoin, and the folk rule -- a retail-friendly Buffett paraphrase -- says **buy when it's in Extreme Fear and sell when it's in Extreme Greed** ([alternative.me](https://alternative.me/crypto/fear-and-greed-index/)). Because the live API is network-blocked (and has no clean free archive), we reconstruct a transparent **proxy gauge** from BTC's own realised volatility, drawdown-from-trailing-high and trailing momentum -- the index's three heaviest ingredients -- and run the contrarian overlay on real daily BTC-USD (2014--2026): long Extreme Fear, short Extreme Greed, flat otherwise, one-day execution lag. We bin forward returns by band, build the dividend-neutral fear-minus-greed spread, and pin it against a shuffled-gauge null, a block-bootstrap Sharpe CI, sub-periods, crypto-realistic short funding, and a synthetic positive control. *(Distinct from [255-fear-greed-index](../255-fear-greed-index/), which tests CNN's curated gauge on equities.)*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the "buy fear" story in plain language, what the gauge did at the 2018 / 2022 crypto winters, and why greed actually paid more |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | band conditional means, HAC *t*-stats, block-bootstrap Sharpe CI, shuffled-gauge null, sub-periods, funding-cost sweep, linear gauge tilt, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`crypto_fear_greed/`](crypto_fear_greed/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
