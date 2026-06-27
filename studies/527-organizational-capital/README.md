# Study 527 -- Organizational-Capital

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Do firms that have spent years building "organizational capital" -- the know-how baked in by past SG&A -- earn higher returns because they carry more risk?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- do high-org-capital firms out-earn low-OC firms? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On a 39-name large-cap **survivor** basket the high-minus-low long-short earns **-1.68%/yr**, HAC *t* = **-0.92** -- below the \|*t*\|>=2 bar and with the **wrong sign**: high-OC firms *underperformed*. The label-shuffle placebo can't tell it from noise (*p* = **0.33**). |
| **Tradability** -- does the spread pay after costs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The gross spread is already negative; netting 5 bps/leg + 50 bps/yr borrow on the short leg only deepens it to **-2.28%/yr**. Nothing here to trade. |
| **"Does the org-capital premium have the sign E-P predicted?"** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Eisfeldt-Papanikolaou (2013) predict high-OC > low-OC by ~4.6%/yr. On this survivor basket the sign **flips**: the high-OC half *trailed* the low-OC half. The synthetic control proves the engine recovers a planted premium with the right sign -- so the flip is the **tape**, not the method. |

> **In one sentence:** the organizational-capital premium is a genuine, well-cited risk story (high-OC firms bear talent-mobility risk), but replicated honestly on a small survivor basket of large-caps the high-minus-low spread shows up with the **wrong sign** and a HAC *t* the tape can't separate from zero -- None signal, Mirage tradability, premium-sign Busted.

## What we tested

Eisfeldt & Papanikolaou (2013): treat **SG&A as investment in organizational capital**, accumulate
it by perpetual inventory (depreciation `delta = 15%/yr`, steady-state seed), scale by total assets,
and sort the cross-section. High-OC firms are predicted to earn a *risk premium* because their key
input -- talent -- is mobile. We pull **~18-19 years of annual SG&A and total assets from EDGAR**
(data.sec.gov 10-K XBRL) for a fixed ~40-name large-cap basket, build the OC stock, rank annually,
go long the high-OC half / short the low-OC half, hold 12 months with **one execution lag**, and
measure the spread with a one-sample **Newey-West HAC** *t*, a **label-shuffle placebo**, costs +
borrow, a depreciation-rate robustness sweep, and a deterministic synthetic positive control. The
basket is **survivorship-biased** -- current large-caps still trading in 2026; the delisted high-risk
tail that an OC risk story would load onto the long leg is absent, so any positive result would be an
upper bound. *Distinct from [238 Betting-Against-Beta](../238-betting-against-beta/) (beta sort) and
[330 Low-Volatility-Anomaly](../330-low-volatility-anomaly/) (vol sort) -- here the sort variable is a
fundamentals-built intangible-capital stock.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "organizational capital" is, why the risk story is plausible, and why the obvious "buy the know-how" trade doesn't pay on large-cap survivors |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the perpetual-inventory build, the annual sort, the long-short with HAC *t* and placebo *p*, costs + borrow, the depreciation-rate sweep, and the 12-seed synthetic control |

The fingerprinted real-data run (EDGAR + yfinance) is in [docs/results.md](docs/results.md);
the offline machinery proof runs on the synthetic world in
[`organizational_capital/data.py`](organizational_capital/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`organizational_capital/`](organizational_capital/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
