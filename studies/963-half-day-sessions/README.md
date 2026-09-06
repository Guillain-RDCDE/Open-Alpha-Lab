# Study 963 — The Half Day 🕐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the half-day session's return different from a normal day's? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The volume collapse is unmistakable — a confirmed 1 p.m. session trades **42%** of a normal day's shares. The *return* is the fragile part: SPY's half day gaps **+14.7 bps** against an ordinary session (*t* = +1.77, CI includes zero), and of the **15** half-day cells (5 tickers x 3 families) **3** clear |*t*| = 2 against 0.75 expected by luck — all positive, none reaching |*t*| = 2.5, and the five tapes are not five independent tests. The sharper cells in the wider battery (9 of 45) sit at offset −1: that is the known pre-holiday effect, not the half day. |
| **Tradability** — can two sessions a year be turned into money? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | About **1.9 confirmed sessions a year**. The largest gap in the lot (GLD +37.1 bps/session, CI [-11, +85]) grosses +68.9 bps a year and nets **+65.2 bps** after a 1 bp round trip — a real number attached to an interval that contains zero, harvested twice a year. Break-even is **18.6 bps** one-way. |

> **In one sentence:** The early close genuinely empties the tape — **42%** of a normal day's volume — and the sessions around it do lean positive, but on the half day itself the lean is smaller than its own error bar, the sharper results belong to the day *before* (the long-documented pre-holiday effect), and 1.9 sessions a year is not a strategy.

## What we tested

Two or three times a year the NYSE rings the bell at 1 p.m.: the session before
Independence Day, the Friday after Thanksgiving, and Christmas Eve. The folk claim, repeated
every November on trading forums, is that these sessions are *different* — thin, drifting
gently upward, "free money" for anyone willing to show up while the desk is empty, or
alternatively dangerous, because a small order moves an empty book. We test the whole window
— the day before, the shortened session itself, the day after — on **SPY, QQQ, IWM, TLT and
GLD** with daily OHLCV bars, splitting each day into its overnight gap and its open-to-close
session so the "shortened session" is measured as the shortened session and not as a
close-to-close day that happens to contain one.

The early-close dates are **derived and then confirmed, never typed in**: a calendar rule
proposes candidates, and each one is kept only if the tape shows it trading a fraction of a
normal day's volume — with the reverse check (thin days the rule missed) published too.
Inference is Newey-West HAC plus a bootstrap on the event-minus-ordinary difference, and
because 5 tickers × 3 families × 3 windows is 45 tests on a sample of ~2 events a year, the
multiple-testing arithmetic is printed next to the results rather than buried.
**Dedup:** distinct from **95-holiday-cheer** and **79-sleigh-ride** (holiday *seasonality*
over whole weeks), **194-turkey** (the Thanksgiving *week*), **780-long-weekend-drift** (the
session before a full holiday, any holiday, at full length), **90-weekend** and
**609-vix-weekend-arithmetic** (the non-trading gap) and **116-power-hour** / **98-high-noon**
(time *within* a normal session) — this study is about the sessions the exchange itself cuts
in half.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why an empty exchange sounds like an opportunity, the volume collapse in one chart, what two sessions a year can and cannot buy, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | calendar derivation and volume confirmation, three return legs, HAC *t* and event bootstrap, family and era cuts, the 45-cell multiplicity arithmetic, threshold sweep, cost arithmetic and the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`half_day/`](half_day/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
