# Study 652 — Index-Deletion-Bounce

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do deleted stocks get dumped, then rebound? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The dump [-5..0] is **-0.99%** (*t* = -1.41); the rebound [+1..+40] is **+0.05%** (*t* = +0.02, 95% bootstrap CI [-4.3%, +4.5%]) — both indistinguishable from zero on 48 real events with a full window. Splitting the 2012-2025 sample in half changes nothing: neither the early (*n*=11, *t*=+0.65) nor the late (*n*=37, *t*=-0.35) half shows a rebound. |
| **Tradability** — does it survive costs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | There is no edge to harvest: a "long the deleted stock, hold 40 sessions" timer nets **-0.05%** at 5 bps costs (*t* = -0.02), worst single event **-38%**. A third of the basket (22/70) later vanished from Yahoo's own archive entirely after an unrelated bankruptcy/take-private, which is itself a warning about what you'd actually be holding. |
| **Has the deletion bounce escaped inclusion's fate (CNS 2004's asymmetry claim)?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | CNS (2004) argued deletion, unlike inclusion, does **not** decay. On this fresh 2012-2025 tape it isn't alive to decay from: neither era shows a rebound, and sibling [249-index-inclusion](../../249-index-inclusion/) already found the ADD-side pop dead too (ex-outlier, *t* ≈ 0). Both halves of the classic asymmetry have gone quiet. |

> **In one sentence:** Chen, Noronha & Singal's 2004 finding that S&P 500 deletions get dumped and then rebound — unlike the inclusion pop, supposedly permanent no longer — does not survive a fresh, honestly-sampled 2012-2025 tape: the dump is a statistical shrug (*t* = -1.41), the rebound is exactly zero (*t* = +0.02), and there is nothing here to trade.

## What we tested

Index funds must sell a stock the moment S&P removes it from the S&P 500 — forced,
price-insensitive selling into a thin market with few natural buyers at that exact moment.
Chen, Noronha & Singal (2004) found this dump reverses once the flow ends — the opposite of
the inclusion pop, which they found had become permanent. We hardcode **70 real S&P 500
deletions (2012-12-11 → 2025-09-22)**, restricted to removals S&P itself coded "market
capitalization change" (distress deletions, not M&A/spin-offs — those tickers just vanish),
sourced from S&P Dow Jones Indices' own announcement PDFs via the Wikipedia change log. For
each we compute the market-adjusted (vs SPY) cumulative abnormal return over [-5..+40]
trading sessions around the effective date, a random-day placebo, an era split, and a
long-the-deleted timer with costs. **22 of the 70 (31%) have no usable tape left on Yahoo at
all** — a later, unrelated bankruptcy or take-private wiped their whole history — a real,
directional survivorship caveat named up front, not buried. Dedup:
[249-index-inclusion](../../249-index-inclusion/) (the ADD side of this exact mechanism),
[320-russell-reconstitution](../../320-russell-reconstitution/) (the whole-index Russell ETF,
not single S&P names) and [250-reverse-split](../../250-reverse-split/) (a related but
distinct distress signature) never test the S&P 500 deletion event on its own — this study
does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why deleted stocks are "supposed" to bounce, the missing-tape problem, why the bounce doesn't show up, and why the trade isn't there even on paper |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the CAR anatomy, the bootstrap and random-day placebo, the era split, the survivorship accounting, the cost sweep and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`index_deletion_bounce/`](index_deletion_bounce/). The deletion calendar is
hardcoded from S&P Dow Jones Indices' own announcement PDFs (via the Wikipedia change log);
survivorship (22/70 tickers with no tape left) is named on the Signal axis. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
