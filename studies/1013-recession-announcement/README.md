# Study 1013 — It's Official 📣

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the market lead the NBER, and is the announcement a buy signal? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) — ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) on the lead · ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) on the buy signal | The market bottomed before the official trough in **14 of 15** uncensored cycles since 1926, more often than the same turning-point rule manages on month-shifted NBER calendars (93% vs 71%, rotation p = **0.032**; 8 of 9 window choices clear 0.05), though the *size* of the lead does not stand out. At trough announcements the market was already a median **+63%** off its low. But the 12-month excess return after the 10 announcements on the total-return tape was **+3.1% vs +8.5%** after every circular shift of the same dates — rotation p = **0.82** (daily price tape p = 0.54). With 10 events the test needs a **17%** planted edge for 80% power. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | "Equity for a year after each announcement" (one lag, 10 bp one-way): net excess Sharpe **0.14 vs 0.52** for buy-and-hold, 1980–2018 (HAC t −3.17; no better than the same rule on shifted dates, p = 0.76); −0.27 on the daily tape. The reverse reflex — bills while "officially" in recession — cost −0.09 / −0.07 of Sharpe. It loses before costs, so no cost level matters. |

> **In one sentence:** The market really does turn before the NBER says so — a median 4-month lead at the trough, and a median +63% rebound already banked by the day the recovery is declared — but the announcement itself is no buy signal (rotation p = 0.82), and the claim's own trading rule trailed buy-and-hold.

## What we tested

The folk claim, steelmanned: *the NBER dates recessions so late that by the time it officially
declares one, stocks have already done their falling — the announcement is a buy signal; and by
the time it declares the recovery, the cheap prices are long gone* (the committee announced the
Great Recession's start on 1 December 2008, a year after it began). We hard-code the
[NBER chronology](https://www.nber.org/research/business-cycle-dating) — 16 cycles since 1926 and
all 12 announcements since 1980 — and measure the market's lead over a century of Fama-French
**total-return** data, plus the S&P 500 **price** tape (1990–2022) that alone reaches 2020. The
buy signal is tested by an exact rotation test against every circular shift of the announcement
dates, its power is priced on a synthetic world, and the believer's rules are booked with one
lag and costs. **Dedup:** **268-sahm-rule** tests a real-time unemployment trigger as a *sell*
button and **626-unemployment-trend-timing** / **881-jobless-claims-nowcast** use labour data as
signals; this study is about the official, after-the-fact NBER *dating announcement* itself, as
a *buy* signal, and about how far the market leads the reference dates.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how late "official" is, where the market really turned, what was already done on announcement day, and why buying the press release lost to holding |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | turning points against a calendar-rotation null (the estimator leans), the exact rotation test on 12 events, power on the synthetic world, and the rules with one lag, costs and shifted-date controls |

Reproducible headline run: [docs/results.md](docs/results.md) (as-of 2018-11-30 monthly / 2022-11-30 daily, fingerprinted).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/) + [`nberclock/`](nberclock/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
