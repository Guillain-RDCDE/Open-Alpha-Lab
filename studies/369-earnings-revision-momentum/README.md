# Study 369 — Earnings-Revision-Momentum ✏️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do up-revised stocks keep beating the market? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | On a transparent revision **proxy** (realized surprise + q/q estimate change), the long-top / short-bottom tercile spread is **+0.21%** at the canonical quarterly horizon — **Welch *t* = 0.42**, placebo ***p* = 0.30** — and the long leg beats SPY a flat **50%** of the time (a coin flip). A whisper appears only at 6 months (*t* = **1.98**) and **just misses the *t* ≥ 2 bar**. Decades of literature, but a positive-and-insignificant tape ⇒ Weak, not Real. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of **10 bps/leg** one-way costs + **50 bps/yr** short borrow, the spread is **negative** at 1 month (−0.20%) and at the quarter (−0.32%); the one positive net horizon (6m, +0.80%) sits at *t* = **1.09**. The gross whisper is **exactly the size of the frictions** — no implementable book. |
| **"Free lunch"?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | "Buy the upgrades" is a real *gross* tendency that **evaporates the moment you price execution** — a textbook factor-zoo mirage. The synthetic control proves the engine *would* have flagged a real edge, so the flat *t* is an **absent** edge, not a missed one. |

> **In one sentence:** earnings-revision momentum has a four-decade pedigree, but on a public surprise-plus-estimate-change proxy the long-short spread is statistically indistinguishable from a coin-shuffle at the canonical quarterly horizon (*t* = 0.42), turns **negative net of costs and borrow**, and its long leg beats the market exactly half the time — real-as-lore, weak-as-edge, and a mirage as a strategy.

## What we tested

Live analyst-revision feeds (IBES) aren't free, so we **build a transparent proxy** for "estimates being revised up": from yfinance `get_earnings_dates` we take each quarter's **realized earnings surprise** (reported EPS beats the consensus estimate) and the **q/q change in the consensus estimate**, z-scored cross-sectionally — a name that keeps beating and whose estimate keeps rising is the public stand-in for an up-revision. Each earnings quarter we rank a fixed 40-name basket by this revision score, go **long the top tercile / short the bottom tercile**, enter one day after the release, hold ~a quarter, and measure each leg's return **in excess of SPY** — then confront the long-short spread with a Welch *t*, a within-quarter label-shuffle placebo null, one-way costs and short-leg borrow. A deterministic synthetic panel with a *planted* revision edge confirms the engine is faithful (edge 0 ⇒ no spread; a modest edge ⇒ *t* ≫ 2). Survivorship and the proxy's lag are named on the Signal axis — both can only *flatter* what we report.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an "up-revision" is, why "buy the upgrades" sounds like free money, and why the long leg beating the market half the time means there's no streak — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the revision score, the long-short tercile book excess-of-SPY, a Welch *t* + label-shuffle placebo null, costs + borrow, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`earnings_revision_momentum/`](earnings_revision_momentum/). The revision signal here is an explicit **proxy** (realized surprise + estimate change), not a live IBES revision feed. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
