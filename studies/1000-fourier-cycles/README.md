# Study 1000 — The Cycle Hunt 🌊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — are there real periodic cycles in equity returns? | ![Busted](https://img.shields.io/badge/Busted-c0392b?style=flat-square) | Across 7 assets the strongest spectral peak averaged **8.0× the average power**, which sounds like a finding until you compute what noise gives. With 3,882 frequency bins the periodogram ordinates are independent exponentials, so their maximum is expected at **8.8×** the mean — and simulating random walks of the same length put the 95th percentile at **10.3×**. Fisher's *g* test, which has had the correct answer since 1929, rejected the null for **0 of 7** assets against white noise and **1** against an AR(1) null — and the AR(1) column is the honest one, because returns are autocorrelated and a flat null tilts every test toward finding long cycles. The positive control worked: UNG, which has a genuine annual demand cycle, showed a peak at **246 sessions** (0.98 years), so the machinery detects cycles when they exist. On SPY the best period found in the first half of the sample was 14 sessions and in the second half 8, and the phase coherence was **0.49** against the 0.70 needed to call a cycle coherent — which is what no cycle looks like. |
| **Tradability** — can a spectral peak be traded? | ![Partial](https://img.shields.io/badge/Partial-dab617?style=flat-square) | Trading the best detected period out of sample — fitting the sinusoid on a rolling 1000-session window and taking the next step's sign — returned +2.47%/yr at a Sharpe of **0.22**, with a 52.2% hit rate against the 50% a coin gives, and 108 position changes a year to pay for. Buy-and-hold over the same window returned +9.65%. That is the expected outcome for a peak that is a peak because some bin has to be the largest. |

> **In one sentence:** The market's strongest spectral peak is 8.0× the average power, and a random walk of the same length gives 10.3× at its 95th percentile — so the honest reading of a cycle chart is that 1 of 7 assets have anything to explain.

## What we tested

Run a Fourier transform on the stock market and peaks appear. They always do: the
periodogram of white noise is **not flat** — each ordinate is an independent exponential draw, so
the largest of *m* bins sits around `log(m) + 0.577` times the mean. With 4,000 daily
observations that is roughly **8×**, at a frequency that will happily support a story.

This study measures the noise benchmark three ways — the closed form, simulated random walks of
the same length, and **Fisher's exact test**, which solved this problem in 1929 and has been
ignored by most of the cycle literature since. It then applies the correction that matters more:
returns are autocorrelated, an AR(1) spectrum tilts toward low frequencies, and testing a peak
against a *flat* null biases every result toward "discovering" long cycles that are nothing but
the autocorrelation showing through.

Three things keep it honest. A **positive control** — natural gas, which has a genuine annual
demand cycle — because a method that only ever finds nothing has not shown that nothing is there.
A **split-sample phase test**, because a real cycle keeps its period *and* where it is in that
cycle, and phase is by far the harder test to pass by luck. And a **power curve** saying what
amplitude the method could have detected, so "no cycle found" is reported as the bounded claim it
actually is rather than as proof of absence.
**Dedup:** distinct from **996-palindrome-dates** (searching across unrelated hypotheses rather
than across frequencies, and with no positive control), **067-monday-effect** and
**283-sell-in-may** (specific calendar seasonals with candidate mechanisms),
**999-cusum-change-points** (aperiodic regime shifts) and **992-vol-clustering-halflife**
(autocorrelation structure in the time domain rather than periodicity in the frequency one).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the cycles a Fourier transform finds in the stock market, and the identical cycles it finds in a random walk |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | periodograms with detrending and windowing made explicit, Fisher's exact g test, AR(1) nulls, a natural-gas positive control, split-sample phase agreement, an out-of-sample cycle trade, and a detection-power curve |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`cyclehunt/`](cyclehunt/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
