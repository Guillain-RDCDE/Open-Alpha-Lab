# Study 1015 — The 1929 Overlay 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do the past windows that best match today's path forecast what comes next? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No, on either tape. The pre-registered monthly forecaster (24-month match, 5 analogues, 12-month forecast, 1950-2018 out of sample) has a Newey-West **t of +1.77** and an out-of-sample R² of −21.6% against the plain historical drift; the daily S&P 500 version gives **t −1.63**. Across all **126 combinations** tried, 4% were raw-significant — chance gives 5% — and the best Holm-adjusted p is **1.00**. The one number that survives a two-sided correction points the *wrong* way (daily, 63-day match, t −4.31): recorded as a lead, not a finding. |
| **Tradability** — does a long/flat rule built on it beat holding the market? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Going to cash when the analogues forecast a loss (one period of lag, 10 bp one-way): net excess Sharpe **0.47 vs 0.52** for buy-and-hold monthly, **0.22 vs 0.34** daily; White's Reality Check over every rule, **p = 1.00** on both tapes. Months whose path matched the 1929 run-up at 0.9 or better fell 20% within a year **5.3%** of the time — against 4.3% for all months. |
| **Shape check** — does the past match itself better than a random walk does? | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | A shade: the median best 24-month match on the real tape is 0.946 against 0.903 for a random walk with the same drift and vol, and most of that gap is volatility clustering (0.919 with the real vol path, 0.931 for a 6-month block bootstrap). A random walk finds a past match above 0.9 about half the time. The shapes rhyme; what follows them does not. |

> **In one sentence:** A random walk with the market's drift and volatility finds a past match above 0.9 about half the time, and forecasting from such analogues has no skill over 69 years of monthly or 23 years of daily out-of-sample forecasts (headline t +1.77 and −1.63; best Holm p 1.00 across 126 combinations).

## What we tested

Every few years the same chart goes viral: today's index laid over the run-up to 1929 (or 1987,
or 2000), the lines tracking each other uncannily, a correlation above 0.9 in the corner, and the
crash that came next left for the reader to extrapolate. The steelman is a real forecasting
method: **analogue (nearest-neighbour) forecasting**. At every date, find the *k* past windows
(strictly earlier, never overlapping today's) whose last *L* periods best match today's — by
Pearson correlation of the log-price path, as the charts do, or of z-scored returns — and forecast
the next *h* periods as the average of what followed them. It is scored out of sample on the
Fama-French **monthly total return** (1926-2018, the tape that actually contains 1929) and the
**daily S&P 500 price index** (1990-2022), with Newey-West and block-bootstrap inference for the
overlapping horizons, a lagged and costed long/flat rule, and a 126-combination sweep reported in
full with Holm and Reality-Check corrections. The teaching core asks what the chart never asks:
how high a correlation would two **unrelated random walks** with the market's drift and vol
print, and how high the best match becomes once you search the whole past for it. A synthetic tape
in which a template genuinely recurs shows the detector fires when there is something to find.
**Dedup:** **1000-fourier-cycles** and **06-clockwork-vol** test cycles (spectral peaks) against
noise and **835-spurious-regression** tests a regression of one random walk on another; none
searches history for a matching *path*, forecasts from what followed it, or measures how the
search itself manufactures the 0.9.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why two unrelated price charts look alike, the 1929 overlay scored month by month since 1931, and what happened after the famous matches |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the analogue search, overlap-robust inference, the full 126-combination sweep with Holm and the Reality Check, richer nulls, the wrong-way lead, and a planted-template power curve |

Reproducible headline run on both real tapes, with provenance, SHA pins and every combination: [docs/results.md](docs/results.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/) + [`overlay/`](overlay/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
