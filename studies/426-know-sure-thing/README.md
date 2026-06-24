# Study 426 — Know Sure Thing 🤔

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the KST rule carry information? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The long/flat book clears HAC **t = 2.65** and beats a sign-flip coin with identical exposure (**p = 0.0044**) — a faint, *real* directional pulse. But it is **beta in disguise**: the value-add over buy-and-hold is significantly *negative* (KST−BAH spread **t = −3.31**). Genuine signal, no separable skill. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net excess Sharpe KST **0.445** < buy-and-hold **0.643** < one-line SMA-200 **0.727**. The rule sheds **~7%/yr** vs doing nothing — **even at zero cost** — and the long/short variant is Sharpe **−0.14** at a **−81%** drawdown. Its only virtue (a −26% vs −55% drawdown) is matched by SMA-200 with far more return. |
| **"A sure thing"?** | ![Busted](https://img.shields.io/badge/A_sure_thing%3F-Busted-8b949e?style=flat-square) | KST loses the race to **both** buy-and-hold and a one-line 200-day average, and only ties MACD. Pring's four-ROC machinery buys nothing the simplest trend filter doesn't do better. |

> **In one sentence:** Martin Pring's grandly-named "Know Sure Thing" has a faint, genuinely-real momentum pulse (it beats a coin at p = 0.004 and clears HAC t = 2.65 on its own returns), but that t-stat is just the equity risk premium it collects while invested — measured honestly as value-add it *underperforms* buy-and-hold by ~7%/yr (spread t = −3.31), loses outright to a one-line 200-day moving average, and blows up if you let it short, so it is a weak signal wearing a mirage of an edge.

## What we tested

We build Pring's daily KST (ROC 10/15/20/30, SMA 10/10/10/15, signal 9) on **33 years of SPY** (yfinance, total return, 1993→2026), turn the KST/signal-line crossover into a daily **long/flat** (and long/short) timing overlay, and run the one test the brochures skip: a **NET excess-of-cash Sharpe race** against the obvious simpler benchmarks — **buy-and-hold**, a **200-day SMA** (Faber), and **MACD** — with realistic one-way costs × NAV, a one-day execution lag, a HAC *t* on the book's excess returns, a 5,000-draw **sign-flip permutation** placebo, and a cost sweep. A deterministic synthetic tape with a **planted-trend** knob confirms the engine *can* reward trend (KST overtakes buy-and-hold under strong planted trend) — proving the real-tape miss is a true negative, not a broken harness.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what KST is, why a fancy indicator with a confident name still has to beat the boring tools, the Sharpe race, and why "beats a coin" ≠ "beats holding the market" — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | long/flat & long/short KST books, NET excess-vs-excess Sharpe vs buy-and-hold/SMA-200/MACD, HAC *t*, a sign-flip permutation, the decisive KST−BAH spread *t*, a cost sweep, and a planted-trend positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`know_sure_thing/`](know_sure_thing/). SPY total-return daily; one execution lag; one-way cost × NAV; excess-of-cash Sharpe both sides. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
