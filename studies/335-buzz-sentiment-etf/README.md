# Study 335 — Buzz-Sentiment-ETF 📣

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Real-tape CAPM alpha **−9.7%/yr**, HAC *t* = **−1.26**; bootstrap P(alpha>0) = **9%** — no positive alpha clears the bar. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | BUZZ lagged SPY by ~5%/yr at **double the vol**, half the Sharpe (0.49 vs 0.96), a **−57%** drawdown. Dressed-up beta (β = 1.59). |
| **"AI sentiment beats the market"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The packaged product delivered worse risk-adjusted returns than the index it implicitly races. |

> **In one sentence:** An AI that reads the crowd to pick 75 winners turned out to be a high-beta, high-fee closet-index fund that lost to SPY on every risk-adjusted measure.

## What we tested

The VanEck Social Sentiment ETF (**BUZZ**, launched 2021) packages a natural-language "AI"
read of social media, news and online chatter into a 75-stock long-only US-equity basket,
reweighted monthly — sold on the premise that *crowd sentiment, read by machine, picks
market-beating winners* (it shot to fame when Dave Portnoy promoted it at launch). This is
the **tradeable-wrapper** cousin of [Study 254 — WSB-Mentions](../../254-wsb-mentions/) and
[Study 256 — Twitter-Mood](../../256-twitter-mood/), which tore down the raw signals; here we
grade the actual fund a buyer can click. The test is an **alpha** test, not a return test:
regress BUZZ's excess return on SPY's (the intercept is the skill), with a Newey-West *t*, a
block-bootstrap CI, and excess-vs-excess Sharpe — over BUZZ's full live history vs SPY.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the pitch vs the tape, beta-in-disguise, why "out-returned in a bull market" isn't skill |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | CAPM alpha + HAC *t*, block-bootstrap CI, information ratio, the synthetic positive control |

The pinned, fingerprinted real run is in [docs/results.md](docs/results.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
