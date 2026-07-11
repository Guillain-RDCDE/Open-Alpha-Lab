# Study 665 — Titanic-Syndrome 🚢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a near-high breadth collapse warn of a decline? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | 13 clusters in 18 years; forward SPY returns indistinguishable from a drift-matched random-entry baseline **and** the unconditional mean at every horizon (max \|Welch *t*\| = 1.24, 1/5/20/60 sessions). Survivorship (current Dow-30 membership) named. |
| **Tradability** — does a timer that exits on the signal pay? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The "sit in cash after a signal" timer nominally edges buy-and-hold (Sharpe 0.71 vs 0.66) but sits at only the 92nd percentile of a random-timing control (*p* = 0.08) — and shares buy-and-hold's identical worst drawdown, because that crash predates the earliest possible signal. |
| **False-alarm machine?** | ![Confirmed](https://img.shields.io/badge/False--alarm_machine%3F-Confirmed-8b949e?style=flat-square) | 21 raw signal days, 13 clusters, a 58.3% "hit" rate that is statistically identical to the market's own 54.9% base rate of a ≥5% drawdown somewhere in any 60-session window (*t* = 0.22). |

> **In one sentence:** Bill Ohama's 1965 rule — a fresh 52-week high within 7 sessions, undercut by more new lows than new highs — fires 13 times in 18 years of Dow-30 breadth, and every downstream test (forward returns, the false-alarm rate, an actual exit-on-signal timer) says the tape can't tell those days apart from an ordinary Tuesday: the Hindenburg Omen's older, cruder cousin, and just as much a false-alarm machine.

## What we tested

Bill Ohama's 1965 **Titanic Syndrome**: the market must print a fresh high within the
past **7 trading sessions**, and on that reading the count of stocks hitting fresh
**52-week lows** must exceed the count hitting fresh **52-week highs** — breadth failing
to confirm the new high, "the band playing while the ship lists." We build the breadth
proxy from the **30 current Dow Jones members** (yfinance, 2008-06 → 2026-06;
survivorship-biased current membership, named) and read "near a high" off ^GSPC's
252-session rolling maximum. Signal sessions within 21 calendar days are merged into
clusters (13 of them), and we test SPY forward returns at 1/5/20/60 sessions against a
drift-matched random-entry baseline **and** the unconditional mean, the false-alarm
rate (a ≥5% drawdown within 60 sessions vs the market's own base rate), and an actual
"exit on signal" timer graded against a random-timer control. **Dedup:**
[167-hindenburg-omen](../167-hindenburg-omen/) is the quantified, multi-condition
sibling (threshold + trend filter + oscillator); [493-new-highs-new-lows](../493-new-highs-new-lows/)
tests the mirror-image *bullish* breadth-thrust claim; [168-advance-decline](../168-advance-decline/)
uses a different breadth statistic (cumulative A/D) entirely. None of them test Ohama's
specific 7-session/52-week construction. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Titanic Syndrome is, why it sounds like the Hindenburg Omen's scarier cousin, and why the tape can't tell a signal day from a random one |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/HAC splits, the random-entry and random-timer controls, the false-alarm arithmetic, survivorship, and a 20-seed synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`titanic_syndrome/`](titanic_syndrome/). Breadth basket = current Dow-30
members (survivorship named); index context = ^GSPC; forward returns/timer = SPY.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
