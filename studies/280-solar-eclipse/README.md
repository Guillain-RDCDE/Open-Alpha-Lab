# Study 280 — Solar-Eclipse

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Do solar eclipses spook the market?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | All-eclipse abnormal return **+13 bps** on the eclipse trading day (the *wrong sign* for "spook"), HAC t = **+1.12**, perm p = **0.31**; the lone "significant" slice (annular, raw p = 0.02) dies under Bonferroni (4 slices → 0.08). n ≈ 80 is too small to detect anything below ~25 bps. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No tradable vehicle — a once-per-eclipse, single-day overlay with one-way costs (shorts pay borrow) is dominated by passive buy-and-hold; there is no edge to monetise. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Eclipses have been "bad omens" since antiquity, but on 98 years of ^GSPC the eclipse day is, if anything, mildly *up* — indistinguishable from any other day. |

> **In one sentence:** measured honestly against the market's ~3 bps/day drift, the S&P 500 does nothing unusual on solar-eclipse days — the omen is folklore, not a factor.

## What we tested

We hardcode every **central** solar eclipse (total / annular / hybrid) from 1930 to
2026 in `data.py`, each tagged with its type and whether the path of totality crossed
the contiguous US. We align ^GSPC daily returns (Shiller-era 1928–2026 cache) into a
symmetric event window, define **event day 0** as the first trading day on/after the
eclipse (one built-in execution-lag day), and test the mean **abnormal** event-day
return (raw minus the full-sample daily drift) with a Newey-West (HAC) t-stat and a
10,000-draw permutation test. We slice by eclipse type and US-path and Bonferroni-correct
for the four simultaneous tests. A synthetic positive control confirms the engine finds a
planted eclipse drag; the real tape has none.

Survivorship: ^GSPC is the index itself (no constituent survivorship), but **price-only**
(no dividends) — the daily drift is the honest baseline.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the omen, the base-rate trap, the eclipse-day chart, the verdict in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | event-study CAR, HAC t-stat, permutation distribution, the annular multiple-comparisons mirage, the tiny-n power calc, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`solar_eclipse/`](solar_eclipse/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
