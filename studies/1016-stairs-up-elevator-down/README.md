# Study 1016 — Stairs Up, Elevator Down 🛗

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the index path lopsided in time, the way the proverb says? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | The S&P 500 path is **time-irreversible**: the Ramsey-Rothman statistic is negative at every lag, Newey-West t = **−3.39**, block-bootstrap t = **−4.04**, −5.19 with the crash tails clipped, replicated on the Nasdaq (−3.63). But the literal proverb runs **backwards**: losing 10% from a peak took a geometric-mean **18.5 sessions**, regaining 10% off a trough **13.4** — the first leg *up* is the fast one (sign-flip null p = 0.01 in that direction), on every tape and threshold bar one. Only *completed* legs fit the saying: whole declines are 1.76× steeper than whole advances, because bull legs run long. A GJR-GARCH fitted to the tape reproduces 72% of the irreversibility and the fast rebound too; symmetric GARCH and i.i.d. reproduce none. |
| **Tradability** — does knowing the shape pay an index holder? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The pre-registered fast-exit / slow-entry rule (out at −5%, back in at +10%, one lag, 5 bp) trails buy-and-hold by **−0.30** of net excess Sharpe on the S&P 500 and **−0.22** on the Nasdaq — on price-only tapes that flatter a rule sitting in cash. Its +0.14 on 92 years of monthly total returns shrinks to +0.08 (p = 0.23) once 1929-32 is out, and a *symmetric* 10%/10% rule matches it. The asymmetry is a description of risk — what put skew prices — not a timing edge. |

> **In one sentence:** The index is measurably time-irreversible (t = −3.4) because volatility jumps after falls — yet the same 10% is regained off a trough faster (13 sessions) than it is lost from a peak (18), and a fast-exit / slow-entry rule built on the proverb does not beat buy-and-hold on daily data after costs.

## What we tested

*"Stocks take the stairs up and the elevator down"* — a trading-floor staple, and in its strongest
form a claim about **time irreversibility**, not skewness: run the index tape backwards and it
should look different, with slow slides and sudden rockets. We time the first X% of every drawup
and drawdown (X = 5, 10, 20%) against sign-flip and shuffle nulls, run the Ramsey-Rothman (1996)
reversibility test with HAC and block-bootstrap errors, measure skewness by horizon, and ask how
much a GJR / EGARCH leverage model reproduces against a symmetric GARCH and an i.i.d. null — on
three frozen tapes: the S&P 500 price index 1990-2022, the Nasdaq Composite 1999-2018 and the
Fama-French market total return 1926-2018. Then a fast-exit / slow-entry rule, fixed before the
run, races buy-and-hold after costs and one lag.
**Dedup:** distinct from **993-leverage-effect-asymmetry** (the volatility *response* to returns —
here that is the mechanism, and the question is the path's shape), **991-aggregational-gaussianity**
(how kurtosis decays with horizon, not the direction of time), **816-drawdown-duration** (time
underwater as a cross-sectional predictor of returns) and **867-currency-crash-risk** (skewness of
carry currencies).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the tape played forwards and backwards, how long a 10% gain and a 10% loss really take, why the bounce off the bottom is the fastest move, and what happens if you trade the proverb |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | first-passage legs with sign-flip and permutation nulls, Ramsey-Rothman with NW and block-bootstrap errors, skew by horizon, GARCH / GJR / EGARCH mechanism shares, a planted-leverage control, and the pre-registered rule race with cost sweep |

Reproducible headline run (provenance, SHA pins, fingerprints): [docs/results.md](docs/results.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/) + [`stairs/`](stairs/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
