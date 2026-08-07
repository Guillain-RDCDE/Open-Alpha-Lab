# Study 837 — Look-Ahead Standardization 🔮

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real edge to find? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On a **random-walk feature** with a forward return that is *provably unpredictable* from the past (iid increments), **full-sample** z-scoring manufactures a cross-sectional rank IC of **−0.138** (Newey-West *t* = **−12.1**) and a fake long-short Sharpe of **15.8**, significant on **20/20** seeds — out of pure noise. The honest **expanding / point-in-time** z-score on the *same panel* reads IC **−0.000** (*t* −0.01), Sharpe **1.0**, significant on **0/20**. A synthetic-only method demo — no real tape, so it can never earn `REAL`. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The fake book needs the **full-sample** mean/std — data from *after* every trade — so it is **unimplementable**; friction is beside the point (a Sharpe-16 look-ahead shrugs off 10 bps). Re-standardise point-in-time and the Sharpe collapses ~16 → ~1 (statistically 0). By construction there is nothing to harvest. |
| **Does full-sample standardisation genuinely leak the future?** | ![Confirmed](https://img.shields.io/badge/Full--sample_standardisation_leaks%3F-Confirmed-8b949e?style=flat-square) | *Yes — severely.* The leak scales exactly like a finite-sample artefact (**grows** with the forward horizon: IC −0.03→−0.28 for H 1→40; **dilutes** with sample length: −0.28→−0.10 for T 250→2000), vanishes on a **stationary** feature (full IC −0.000 — the contrast that pins it to non-stationarity), and the honest expanding method both kills it *and* still recovers a **planted real edge** (*t* +21.6) — so it is unbiased, not always-zero. |

> **In one sentence:** the one-line habit `z = (x - x.mean()) / x.std()` run over the *whole* sample
> z-scores every feature with statistics that include the future, and on a non-stationary feature that
> smuggles a Sharpe-16 "edge" into a backtest of pure noise — while the free, non-negotiable fix
> (compute the mean & std on an **expanding, past-only** window) reads exactly zero.

## What we tested

A **specific, ubiquitous** look-ahead leak: normalising a predictive feature with the **full-sample**
mean & standard deviation instead of an **expanding / point-in-time** window. We build three
deterministic synthetic feature+return panels and score each two ways — cross-sectional rank IC (with
a Newey-West *t* on the daily IC series) and a long-short fractile Sharpe. On the **non-stationary
null** (a random-walk feature, forward return = its own future 10-day change, so genuinely
unpredictable) the full-sample z-score prints a large fake IC/Sharpe on every seed while the expanding
one reads ~0; on the **stationary null** neither leaks (pinning the pitfall to non-stationarity); on a
**planted real edge** the expanding method recovers it (*t* +21.6) — the machinery is unbiased, not
blind. Horizon and sample-length sweeps show the leak scaling like the finite-sample artefact it is.
**Dedup:** [347-look-ahead-bias](../347-look-ahead-bias/) is the **generic** case (mis-*timing* a
signal); 837 is the **normalisation-leakage** case (the signal is correctly timed, but its
**standardisation statistic** is fit on the full sample). [344-backtest-overfitting](../344-backtest-overfitting/)
inflates a Sharpe by **searching** many rules; [831-gold-real-yield-timing](../831-gold-real-yield-timing/)
is a nearby timing study — 837 needs **no search and no market data**, just one mis-specified
preprocessing line on one feature. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "just z-score the feature" quietly cheats, how a Sharpe-16 strategy appears out of noise, and the one-word fix — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | full-sample vs expanding standardisation, the cross-sectional rank IC + Newey-West *t*, the random-walk mean-reversion mechanism, the horizon/length scaling laws, the costed timer, and the planted-edge positive control |

The fingerprinted headline run (sim config fp `5f6dcb4c991c`, as-of 2026-06-30) is in
[docs/results.md](docs/results.md); the whole machinery runs offline and deterministic on the
synthetic worlds in [`lookahead_standardization/data.py`](lookahead_standardization/data.py).

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [`examples/verify.py`](examples/verify.py).

---

*Engine: [`lookahead_standardization/`](lookahead_standardization/). Synthetic-only method demo — no
real tape, so capped at `NONE` (a `REAL` stamp requires a robust *t* ≥ 2 on a real tape). **Not
investment advice** — research & education. See [LICENSE](../../LICENSE).*
