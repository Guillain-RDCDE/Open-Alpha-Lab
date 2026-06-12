# Study 02 — Falling-Knife 🔪

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md). Companion to [01 — Overnight Anomaly](../01-overnight-anomaly/).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) at −3% · ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) in deep panic | At −3% the excess over a random day is zero-to-negative (p≈0.8 on ^NDX, p≈0.99 on QQQ); a real, significant bounce only appears at −5%/−7%. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The deep-dip excess straddles zero under a clustering-aware bootstrap; capacity ~3 events/decade dominated by 2–3 crashes; a fixed rule flips +1.30 → −1.35 Sharpe out-of-sample. |

> **In one sentence:** the famous −3% dip is folklore — indistinguishable from buying a random day — and even the genuine panic-bounce at −5%/−7% fails the tests that matter for trading it (clustering, capacity, out-of-sample).

## What we tested

*"Buy when there's blood in the streets."* The version people quote on the Nasdaq-100 is the **−3% day**: the index closes down 3%, you buy, you wait for the bounce. Stated testably: *buying after a −3% close earns more than buying on an ordinary day.* We test the whole **family** of four defensible "falling-knife" definitions (close-to-close, intraday, drawdown-from-high, cumulative bleed) on two faces of the market — `^NDX`/`^GSPC` spot (deep sample) and `QQQ`/`SPY` (actually tradeable) — always measuring **excess over a random-day null**, never absolute return.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the one trap (random-day baseline), the −3% non-result, and the deep-panic twist, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | event study, permutation benchmark, block bootstrap, threshold sweep, family scan + deflated Sharpe, regime split, capacity, and the in/out-of-sample collapse |

Prefer scripts? [`examples/`](examples/) has `run_synthetic_demo.py` (offline — watch the toolchain tell a real edge from a fake one), `verify_ndx.py`, `sweep_thresholds.py`, `panic_zoom.py`, `compare_indices.py`. Every headline number is fingerprinted in [docs/results.md](docs/results.md) (as-of date + content hash of the exact price inputs).

---

*Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
