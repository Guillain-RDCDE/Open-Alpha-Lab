# Study 760 — Michigan-Sentiment-Day 🙂

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) on the release-day · ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) on the level | Release day is an average day (**+6.2 bp** vs **+4.8 bp**, *t* = **+0.23**) and the surprise drift runs *backwards* (bigger after misses). "Low-then-rising marks bottoms" *looks* real at 12m (**+18.7%** vs **+12.0%**, naive *t* = **+3.55**) — but it's **21 clustered recoveries**; a 12-month block bootstrap can't clear it (*p* = **0.10**), and the sentiment **level alone is flat** (*t* ≈ 0). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The "buy the bottom" overlay is in the market **18%** of the time and earns **+3.1%/yr** (Sharpe **0.44**) vs buy-and-hold **+11.3%** (Sharpe **0.77**). You can't beat passive by being long a rare post-crash subset. |
| **Bottom-timer?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The one horizon where the *t* screams (12m, overlapping) is exactly where the statistic is least trustworthy; the autocorrelation-aware bootstrap and the episode count both say "a handful of recoveries," not a certifiable early tell. |

> **In one sentence:** the University of Michigan sentiment print doesn't move SPY on release day (and its "surprise" drifts the wrong way), and the beloved contrarian rule — *buy when sentiment is low and turning up* — is a naive-t mirage built on ~20 post-crash recoveries that a 12-month block bootstrap quietly refuses to certify.

## What we tested

Two claims about the most-watched U.S. sentiment release. **(A)** The release-day drift: the preliminary print (mid-month Friday, ~10:00 ET) is a market-mover, so SPY should react on the day and *drift* with the surprise. **(B)** The contrarian bottom-timer (Fisher–Statman 2003, *[Consumer Confidence and Stock Returns](https://www.pm-research.com/content/iijpormgmt/30/1/115)*): buy stocks when sentiment is **low and turning up** — "low-then-rising marks bottoms." We rebuild both on real tapes — a hardcoded, public monthly snapshot of FRED `UMCSENT` (FRED is firewalled here, so it's a labelled snapshot, like [Study 385](../385-jobless-claims-momentum/)) and cached yfinance SPY (daily for the event study, month-end for the regime test) — with a strict no-look-ahead lag, a Welch *t*, a **circular block bootstrap** that respects overlapping-return autocorrelation, an independent-episode count, a cost-net overlay, and a planted-edge synthetic control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "the market moves on Michigan" isn't true, why "buy cheap sentiment" is really "buy after crashes," and why a big *t*-stat can still be a mirage — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the release-day event study, the regime split, the decisive **naive-t vs block-bootstrap** gap, the 21-episode clustering, the buy-the-bottom overlay, robustness, and a planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`michigan_sentiment_day/`](michigan_sentiment_day/). Sentiment here is a hardcoded **snapshot** of FRED `UMCSENT` (final print, not the real-time vintage), named as such. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
