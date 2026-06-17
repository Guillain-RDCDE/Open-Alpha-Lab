# Study 299 — Keynote-Drift

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Pre-keynote abnormal CAR **−0.19%**, t = **−0.39**, permutation p = **0.75** (AAPL benchmarked against its own mean). Post-event droop (t = −1.73) is suggestive but short of the t ≥ 2 bar and non-significant on its subset. n = 48 too small to detect < 1.4%/event. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The "buy the rumor" leg nets ~0.4%/event — that's AAPL's ordinary 4-day drift, not a keynote edge — earned with concentrated single-name gap risk while out of the market most of the year. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The folklore confuses a 50-bagger's secular trend with a calendar effect; the abnormal (de-trended) keynote windows are flat. |

> **In one sentence:** "buy the rumor, sell the Apple keynote" is a story told about a stock that happened to go up — strip out AAPL's own trend and the keynote windows show no abnormal drift.

## What we tested

We hardcode **48 major Apple keynotes (2008–2025)** — WWDC, the September iPhone
event, and Spring specials — in `data.py`, then run a constant-mean event study on
AAPL daily adjusted closes (price-only). The key discipline: returns are measured
**in excess of AAPL's own unconditional daily mean** (abnormal returns), so the
stock's enormous 2008–2025 uptrend doesn't masquerade as a keynote effect. We
t-test the per-event pre- and post-keynote cumulative abnormal returns (CARs),
validate with a permutation null that re-anchors the windows on random trading
days, slice by event type and sub-period, and run a tradable "buy the rumor"
backtest gross **and** net of costs (one trading-day execution lag, 10 bps round
trip). A synthetic positive control confirms the machinery detects a keynote
drift when one is planted; the real tape confirms there is none.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the benchmark trap (raw vs abnormal run-up), the post-event whisper, plain-English verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-event CARs, one-sample t-tests, the permutation null, event-type/sub-period slicing, the n=48 power calc, gross/net backtest |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`keynote_drift/`](keynote_drift/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
