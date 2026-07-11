# Study 654 — Quiet-Period-Expiry 🤐📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**The bank that just sold you the IPO is legally silenced for 25 days — does its "all clear" Buy note make the stock pop?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the stock pop when underwriters can finally speak? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Bradley, Jordan & Ritter (2003) found a real pop on 1996–2000 IPOs. On **60 real, underwritten US IPOs (2015–2025)**, CAR vs SPY over trading days **[20..30]** is **+3.35%** but one-sample **t = +1.56** — under the bar — and every robustness cut agrees: a paired within-IPO placebo is **t = +0.29**, a 20,000-draw random-window placebo gives **p = 0.26**, and no single day in [15..35] clears *t* = 2 on the claim's side. Literature says real; this tape can't certify it. |
| **Tradability** — can you bank the pop? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The literal "buy day 22, sell day 27" timer nets **+2.06%/trade** at 5 bps but at **t = 1.50** — uncertifiable — on a **+30.3% / −24.3%** best/worst spread across 60 trades. A handful of volatile small/mid-cap flyers carry the average; costs are not the problem, there's no statistically real numerator to charge them against. |
| **Faded since 2021?** | ![Mixed](https://img.shields.io/badge/Faded_since_2021%3F-Mixed-8b949e?style=flat-square) | The point estimate collapses from **+5.0%** (2015→2021-06, *t* = 1.72 — the closest this study gets to significance) to **+0.4%** (2021-06→2025, *t* = 0.15) — but the difference itself is **not certified** (Welch *t* = −1.11). Consistent with the folklore getting arbitraged away as it became common knowledge; not proven. |

> **In one sentence:** the "quiet-period pop" is real in the original 1996–2000 sample
> (Bradley-Jordan-Ritter 2003) but on 60 real 2015–2025 IPOs the same test gives *t* = 1.56
> — a positive lean that never clears the bar on any cut we tried, on a trade whose
> tails (+30%/−24%) dwarf its mean — **Weak, and a tradability Mirage**.

## What we tested

We hardcode **66 real, underwritten US IPOs (2015→2025)** — direct listings and SPAC
mergers excluded by construction, since FINRA Rule 2711's 25-calendar-day quiet period
binds the *managing underwriters of a firm-commitment offering*, not those structures —
and pull daily adjusted closes + SPY from yfinance. Abnormal return = stock − SPY on the
same calendar date; "day *t*" = *t* trading sessions after the first close (day 0). The
headline test is the per-IPO cumulative abnormal return over trading days **[20..30]**
(bracketing the trading-day-25 proxy for the 25-calendar-day quiet period), one-sample
*t* across 60 cross-sectionally independent IPOs, cross-checked by a paired within-IPO
placebo, a random-window placebo, and a day-by-day anatomy. The third axis is the literal
retail trade — buy day 22 close, sell day 27 close, net of costs — plus an era split
(2015→2021-06 vs 2021-06→2025) testing whether the effect faded as it became folklore. A
20-seed synthetic control (its false-positive rate matches the nominal 5% almost exactly)
proves the machinery is unbiased. **Dedup:** [219-ipo-pop](../219-ipo-pop/) is the
**day-1** underpricing pop; [319-lockup-expiry](../319-lockup-expiry/) is the
**180-calendar-day share lock-up** (a supply event, same panel design, completely
different mechanism); [623-ipo-long-run-underperformance](../623-ipo-long-run-underperformance/)
is the **3–5 year** drift; [636-exchange-listing-pop](../636-exchange-listing-pop/) is the
**crypto** listing pop on Coinbase. None of them test the **25-day analyst quiet period**
— this study is that axis. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why underwriters can't publish research for 25 days, why that silence should end in a coordinated wave of Buy notes, what the tape actually shows — and why the trade's tails are scarier than its average |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the CAR-window *t*, the paired and random-window placebos, the day-by-day anatomy, the timer-strategy cost sweep, the era contrast, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quiet_period_expiry/`](quiet_period_expiry/). Basket survivorship (data
availability only) is named on the Signal axis. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
