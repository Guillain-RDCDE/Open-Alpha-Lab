# Study 228 — Pre-Earnings Runup

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does price drift UP in the days BEFORE earnings, before the news even lands? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The equal-weight pre-event book earns a gross Sharpe of **+0.46** (Newey-West *t* **+1.68**) — below the |*t*| ≥ 2 bar. Worse, the passive equal-weight market earns **+0.97** over the same period; the book is almost entirely market beta with no distinct pre-earnings alpha visible on a liquid large-cap universe. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Break-even cost is **15.2 bp** on turnover of 0.25/day, but the book never outperforms the market on a gross basis, so even a zero-cost version of this strategy is a loss versus passive. Net @5 bp the Sharpe is **+0.31** vs the market's **+0.97**. |
| **Does the pre-event drift exist?** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | The raw per-day mean returns in the pre-event window are positive (+0.04% to +0.12%/day), consistent with a small drift. But they are not statistically distinguishable from the general market drift, and they do not show the expected monotone rise toward the announcement. `MIXED` — a faint signal drowned in beta. |

> **In one sentence:** the pre-earnings runup is a real but beta-dominated pattern on liquid large-cap stocks — the gross Sharpe (+0.46, *t* +1.68) is `WEAK` and the passive market (+0.97 Sharpe) crushes it, making it a tradability `MIRAGE` with the pre-event drift itself only `MIXED`.

> ✅ **Real-tape run, fingerprinted.** Run on yfinance adjusted closes + earnings dates for a fixed 100-name large-cap universe (98 names with events): **2345 events, 2014-01-03 → 2026-06-16, fingerprint `220e0562c461`**. Reproduce offline via [examples/verify.py](examples/verify.py) after caching: `python examples/verify.py --fetch`. **Caveat:** current-membership universe ⇒ survivorship bias (delisted losers absent), inflating magnitudes.

## What we tested

Do stocks drift up in the days before their scheduled earnings announcement, before the news even lands? The academic claim (Frazzini & Lamont 2007; Kim & Park 2005): informed traders and options-market participants accumulate long positions ahead of earnings, creating a systematic +0.5–1% pre-event premium over the 5 trading days before the release. We buy all large-cap names entering their 5-day pre-announcement window (strictly causal — one-day lag on the earnings calendar), equal-weighted, exiting before the announcement, and measure whether the return exceeds the passive market. The answer on a liquid large-cap universe in 2014–2026: the drift is present in raw form but is entirely explained by positive equity beta, not a distinct pre-earnings alpha.

**Distinct from study 34 (Aftershock / PEAD):** that study tests the POST-announcement drift (prices keep drifting in the surprise direction for weeks AFTER the news lands). This study tests the PRE-announcement window — a different mechanism (informed positioning vs. under-reaction to the surprise).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: why prices might drift before earnings, what informed trading looks like, and why on liquid names you cannot harvest the premium |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: gross vs market Sharpe, the Newey-West *t*, the break-even cost, the pre-day window sweep, the per-day drift curve |

The real run — every fingerprinted, as-of'd number — is in [docs/results.md](docs/results.md). Reproduce the real tape via [examples/verify.py](examples/verify.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
