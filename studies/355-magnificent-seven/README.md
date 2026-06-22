# Study 355 — Magnificent-Seven 🎬

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the Mag-7 basket genuinely out-earn the S&P 500? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | On the tape, **yes**: equal-weight Mag 7 beat SPY by **+19.9%/yr**, HAC **t = 3.41** (n = 130), and **t = 2.9** even after stripping the equal-vs-cap-weight effect. But the basket is **named because it won** — a look-ahead, survivorship selection. The synthetic null shows selection *alone* manufactures a spread with **no** true edge. Real recent spread, hindsight-selected. |
| **Tradability** — could you have captured it forward? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Costs are trivial (net CAGR unchanged). The mirage is that you **could not have named these seven in 2015**. A *blindly* chosen seven beat SPY by only **~4%/yr** (the Mag 7 sits at the **99.7th percentile** of 2,000 random baskets); the ex-post "pick the winners" rule reproduces **137%** of the spread. The +22%/yr is a number only hindsight can buy. |
| **A repeatable strategy?** | ![Busted](https://img.shields.io/badge/Repeatable_strategy%3F-Busted-8b949e?style=flat-square) | "Hold the seven that already won" is a label for the realised winners, not a rule. Run it *forward* and you get the random-basket distribution (median **+3%/yr**, a wide tail and a −45% drawdown), not +22%. |

> **In one sentence:** the Mag-7 outperformance is genuinely on the tape — and it is the tape that *named* the basket: strip the hindsight and a forward-pickable seven beats the index by ~4%/yr, not ~22%; the rest is selection.

## What we tested

We hold the **Mag 7** (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA) equal-weight, monthly-rebalanced, over **130 months (2015-08 → 2026-05)** of yfinance monthly total returns, and race them against the **S&P 500 (SPY)** and an **equal-weight basket of the same 40-name mega-cap field** (the weighting control). The spread is real and significant (`t = 3.4`) — so the question is *why*. We decompose it three ways: against the cap index, against the equal-weight field, and against the **ex-post "pick the 7 winners in hindsight"** rule — the Mag-7 selection rule made explicit. A deterministic synthetic null (no name has any edge) proves that this look-ahead selection manufactures a large spread *from nothing*, and that the famous seven sit at the 99.7th percentile of 2,000 random blind baskets. Survivorship and look-ahead are named on the **Signal** axis. (Same selection-artefact family as [Study 345](../../345-survivorship-bias/) and [Study 350](../../350-dartboard-portfolio/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the seven really did win, why you couldn't have named them in 2015, and why "pick the winners" is a trick — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC t-stat of the spread, the equal-weight-field decomposition, the ex-post-selection placebo, the random-basket distribution, and a synthetic null that manufactures the spread from zero edge |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`magnificent_seven/`](magnificent_seven/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
