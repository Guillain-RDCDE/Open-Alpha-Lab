# Study 533 — Analyst-Dispersion 🗣️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Do stocks that analysts **disagree** about most go on to earn **lower** returns?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do high-disagreement names earn less? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The DMS direction shows up at **exactly one** horizon — trailing **1-month**, low-minus-high **+10.46%** at one-sample **t = 3.21** (placebo *p* = 0.003, Spearman **−0.435**) — but it's a **single cross-section on a single date** that can't be averaged over independent draws, and it **reverses at every longer horizon** (3 / 6 / 12-mo all wrong-sign). A lone window's *t* > 3 is **Weak, not Real**. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | There is **no forward trade**: yfinance gives only a *current* dispersion snapshot, so the sort is against **trailing** returns — a contemporaneous association, not a strategy. The "net +9.98%" is bookkeeping on already-realised returns, not P&L. 40 survivor names, no dispersion history. |
| **"Does high dispersion really earn LESS?"** | ![Mixed](https://img.shields.io/badge/Does_dispersion_lose%3F-Mixed-8b949e?style=flat-square) | Direction- and horizon-dependent: **yes at 1 month**, but the 12-month tercile sort runs cleanly **monotone the WRONG way** (low **+16%**, high **+56%**) — the surviving high-dispersion winners (NVDA, TSLA, ORCL) dominate. The academic puzzle is real in the literature; it does not cleanly replicate on a free, single-snapshot, survivor basket. |

> **In one sentence:** the Diether-Malloy-Scherbina puzzle (high forecast dispersion → low returns) is a real, replicated *academic* effect, but on the only thing free data lets us build — a single current dispersion snapshot sorted against trailing returns on 40 survivor large-caps — it shows up at just one horizon (1-month, *t* = 3.21) and reverses everywhere else (12-month tercile sort runs monotone the wrong way), so it lands **Weak signal, Mirage tradability** with the verdict swamped by survivorship and the absence of a dispersion history.

## What we tested

Diether, Malloy & Scherbina (2002): stocks with **high dispersion in analysts' earnings
forecasts** earn **lower** future returns (a puzzle vs the risk intuition; mechanism is Miller
1977's short-sale-constrained over-pricing). We proxy forecast dispersion from yfinance's
per-name analyst EPS-estimate spread — `dispersion = (high − low) / |mean|` on the
current-fiscal-year consensus — for a fixed **40-name large-cap** basket, sort into dispersion
terciles, and test the **low-minus-high** long-short against the trailing 1 / 3 / 6 / 12-month
return with a one-sample *t*, a 20,000-draw label-shuffle placebo, a Spearman rank check, and
costs. **The honest data caveat** (named on the Signal axis): yfinance exposes only a *current*
snapshot — no historical dispersion panel — so this is a *contemporaneous* sort on one
cross-section, not the tradable forward panel the academic study uses. A deterministic synthetic
control with a planted DMS drag (seed-robust over 25 seeds) proves the sort engine is faithful.
*Nearest method neighbour: [532 Firm-Age-Anomaly](../532-firm-age-anomaly/) (a survivor-basket
sort whose sign is also flipped by survivorship); sibling earnings plumbing in
[363 PEAD-Drift](../363-pead-drift/).*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "analyst disagreement" means, why the textbook says it *loses*, and why our survivor basket says the opposite over a year — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the dispersion proxy, the low-minus-high tercile long-short across horizons, the one-sample *t* + label-shuffle placebo + Spearman, the 12-month wrong-way monotonicity, costs, and the seed-robust synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run:
[docs/results.md](docs/results.md).

---

*Engine: [`analyst_dispersion/`](analyst_dispersion/). Dispersion proxy is the current-fiscal-year
analyst EPS-estimate spread `(high − low) / |mean|`; the sort is against **trailing** returns (no
dispersion history is available free — named on the Signal axis). Basket is **survivors** — named
on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
