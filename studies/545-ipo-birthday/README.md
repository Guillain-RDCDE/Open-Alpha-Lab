# Study 545 — IPO-Birthday 🎂

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do stocks show abnormal returns around their IPO anniversary? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No. Across **312 firm-year events (29 names)** the mean `[-5,+5]` anniversary CAR is **+62.75 bps** at one-sample **_t_ = 1.36** — below the bar. Median **+12 bps**, only **50.6%** of events positive. The tight `[-1,+1]` window is **+3.7 bps (_t_ 0.15)**; the CAR only grows with window width (generic drift). No window clears *t* ≥ 2 that survives the placebo. |
| **Tradability** — does a birthday trade pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A survivor basket of high-flying growth IPOs, one round trip per event. The "effect" is not distinguishable from just holding these outperformers in *any* window: the **random-anchor placebo null mean is +86 bps** — *higher* than the observed anniversary CAR — placebo **_p_ = 0.648**. Gross +62.75 → net +52.75 bps, all noise. |

> **In one sentence:** stocks do **not** pop on their IPO birthday — the mean `[-5,+5]` anniversary abnormal return is +62.75 bps at *t* 1.36 (median +12 bps, a coin-flip 50.6% positive), and a random-anchor placebo whose null mean is *higher* (+86 bps, *p* 0.648) proves that small positive number is nothing but the generic outperformance of a survivor basket of growth IPOs showing up in any 11-day window.

## What we tested

The **IPO 'birthday effect'** (behavioural folklore, in the attention-drives-return tradition of
Barber & Odean 2008 and Da-Engelberg-Gao 2011): a listed company should show a small positive
abnormal return around the yearly anniversary of its IPO, from retrospective press and calendar
attention. We run a classic **event study** (Fama-Fisher-Jensen-Roll 1969; MacKinlay 1997) on a
curated 30-name US IPO basket: for each firm-**year**, take the IPO-date anniversary, cumulate
**market-adjusted** returns (name − SPY) over the `[-5,+5]` trading-day window into one CAR, and test
the mean CAR across all events with a one-sample *t*, a **random-anchor placebo** null (re-run the
same window on random non-anniversary dates), a multi-window robustness sweep, round-trip costs, and
a deterministic, seed-robust synthetic positive control that plants an anniversary bump and proves
the engine catches it. *Distinct from [Study 219 — IPO-Pop](../219-ipo-pop/) (the first-day launch
pop and long-run drift) and [Study 265 — IPO-Volume](../265-ipo-volume/) (issuance as a timing
signal): this study is the **recurring annual anniversary** return, as a CAR event study.*
Survivorship is named on the Signal axis — the basket is names still trading in 2026.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a 'birthday effect' would be, why a fixed calendar date shouldn't pay, and why a rising survivor basket makes every window look positive |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the CAR event study with a one-sample *t*, the random-anchor placebo, the window sweep (CAR grows with width = drift), costs, and the seed-robust synthetic positive control |

The fingerprinted real-data run (30-name basket, prices fp `19fee811b30d`, event CARs fp
`73d3051eb6f4`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery
proof runs on the deterministic synthetic world in [`ipo_birthday/data.py`](ipo_birthday/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`ipo_birthday/`](ipo_birthday/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
