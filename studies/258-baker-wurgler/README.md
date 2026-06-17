# Study 258 -- Baker-Wurgler

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The Baker-Wurgler contrarian *sign* is right -- low-sentiment months (+8.78%/yr) beat high-sentiment months (+7.16%/yr) -- and the literature (BW 2006/2007) is strong, but the predictive slope HAC *t* = **-1.29** and the low-minus-high spread *t* = **+0.32** never clear |t| >= 2; bootstrap Sharpe CI [-0.26, +0.36] straddles zero; the effect is concentrated in 1965-1989 and the sign **flips** post-2008. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A long-when-fearful timing overlay returns +2.9%/yr net vs **+8.39%/yr buy-and-hold** at a lower Sharpe (0.32 vs 0.56); shorting the greedy months is worse still. No tradable edge on the aggregate index. |
| **Price-only / reconstruction** | ![Named](https://img.shields.io/badge/Price--only-8b949e?style=flat-square) | ^GSPC returns are price-only (no dividends); the cross-regime spread is roughly dividend-neutral. The sentiment index is a hardcoded reconstruction of the documented BW regime structure, not the verbatim NYU file. |

> **In one sentence:** Baker-Wurgler captures a real psychological instinct -- buy fear, sell greed -- and the sign shows up faintly even on the broad S&P 500, but the aggregate-index effect is statistically insignificant, decays after publication, and a naive timing rule loses to buy-and-hold.

## The claim

> *Does the Baker-Wurgler sentiment index forecast returns?*

## What we tested

The Baker & Wurgler (2006, 2007) **contrarian** prediction: when investor sentiment is
*high*, subsequent returns are *low* (and the reverse), concentrated in speculative,
hard-to-value stocks. We hardcode a reconstruction of the monthly BW index, read it at
each month-end (a published, lagged figure), and predict the *next* month's ^GSPC return
(one-month execution lag). We (a) sort next-month returns by prior-sentiment tertile,
(b) regress next-month return on lagged sentiment with a Newey-West HAC *t*-stat,
(c) test the cross-sectional leg via small-minus-large (^RUT - ^GSPC), (d) split into
sub-periods, and (e) stress a tradable long-when-fearful timing overlay net of costs
against unconditional buy-and-hold. A deterministic synthetic positive control confirms
the engine recovers a planted contrarian effect (t = -2.58) and reads ~zero on the null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the sentiment tape, the fear-vs-greed bar chart, why the index-level effect is faint, and why a timing rule loses to buy-and-hold |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | predictive regression with HAC *t*, bootstrap Sharpe CI, sub-period decay and sign-flip, the small-minus-large cross-sectional leg, the timing overlay's costs, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`baker_wurgler/`](baker_wurgler/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
