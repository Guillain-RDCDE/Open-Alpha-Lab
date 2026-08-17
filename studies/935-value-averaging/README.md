# Study 935 — Value Averaging 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is value averaging's advantage real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On identical committed capital the rule ends **−1.37 cents per dollar contributed** behind plain DCA (HAC *t* = **−3.87**, bootstrap CI **[−2.06, −0.74]**, ahead in only **29.0%** of 193 programmes) — negative in both eras, on **IEF** and **QQQ**, and at every horizon from 24 to 120 months. The sign is set by the **value path's growth rate**, a non-tape ASSUMPTION that is really an equity-weight dial (63.4% invested at 0%/yr vs DCA's 69.9%; the gap crosses zero at ~8%/yr, exactly where the two weights meet). Exposure-matched on an *in-sample* λ, the residual **+0.70c** sits *inside* the spread twelve SPY-calibrated **random walks** produce with no predictability at all — z = **+0.66**, and **2 of the 12** beat it outright (one-sided *p* = 0.23). |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The advertised win survives only in the **equity-only IRR** (+0.92 pp/yr, 14.90% vs 13.97%) — a figure that barely moves across the entire 0–24-month buffer sweep (14.73% → 14.91%) while the whole-programme IRR it ignores *halves* over the same range (12.97% → 6.22%), so it cannot be measuring the programme; at the 6-month default the whole-programme IRR is **10.07% vs 10.64%**. Friction is not the obstacle (VA trades *less* notional than DCA, 34.7 vs 36.0, and the gap is unchanged at 25 bp). The obstacle is the funding: the cap binds in **3.1%** of programmes — all six starting mid-2007 — with a worst *single-month* call of **3.3** monthly contributions (**6.6** summed over one programme's binding months), and **86.5%** of programmes are unfundable with no buffer at all. |

> **In one sentence:** Edleson's rule really does buy low and sell high, and really does win on the return measure his book quotes — but that measure ignores the buffer that funds it, and once you count every dollar the saver had to commit, value averaging finishes **behind** dollar-cost averaging in seven programmes out of ten, for the mundane reason that a flat value path in a rising market is a **hidden instruction to hold less equity**.

## What we tested

$1 a month for **36 months** into **SPY**, two ways: fixed amount (DCA) versus trading
to Edleson's value path (VA). Both arms are handed **identical committed capital** — a
pre-funded **6-month buffer** plus the same contributions — idle money earns **BIL**'s
actual total return, and the accounts are compared on **whole-account terminal wealth**.
The buffer is finite: when VA demands more, the purchase is **capped** and the shortfall
recorded. Rolled over **every** start month of SPY∩BIL 2007-05-30 → 2026-06-30
(total-return closes, `auto_adjust=True`), one execution lag (sized at a month-end close,
filled the next day), 1 bp one-way, no shorting and so no borrow. Win rate with a Wilson
interval, HAC *t* and a 36-window block bootstrap, era cut, both IRR measures, four
sweeps (path growth / buffer / cost / horizon), an **exposure-matched** control and a
**calibrated random-walk placebo**, plus IEF and QQQ cross-checks. Non-tape ASSUMPTIONS —
path growth, buffer size, cost, horizon — are labelled and swept; the exposure-matching λ
is fitted **in-sample**, and so is the placebo's, which is what makes them comparable.
**Survivorship** is index-level, identical in both arms, and cancels in the difference.
**Dedup:** distinct
from **934-lump-sum-vs-dca** and **101-slow-and-steady**, which ask how fast to deploy a
windfall with the tranche size held fixed; here the schedule length is fixed and the
tranche size varies *by rule*, which is where the funding problem lives. Distinct from
**102-free-rebalance** (rebalancing between sibling assets, not against a growing target
funded from outside) and **596-bond-tent-glidepath** (a *declared* equity-weight
schedule, where 935's punchline is that VA is an undeclared one).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the two savers, the metric that flatters one of them, the hidden equity dial, the 2008 cash call |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | wealth gap and HAC *t*, both IRRs, cap-binding stats, four sweeps, the exposure-matched control, the random-walk placebo, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`value_avg/`](value_avg/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
