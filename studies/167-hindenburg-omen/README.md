# Study 167 — Hindenburg-Omen

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | 31 clusters in 20 years; forward SPY returns indistinguishable from the base at every horizon (Welch-t from −0.66 to +0.14, Bonferroni-corrected p = 1.00). Survivorship bias named. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Signal returns at or below the unconditional rate; crash rate after the signal (78.6%) is *lower* than on a random day (82.5%); nothing to capture. |
| **False-alarm machine?** | ![Confirmed](https://img.shields.io/badge/False--alarm_machine%3F-Confirmed-8b949e?style=flat-square) | 84 raw signal days, 31 clusters, zero statistically distinguishable crashes — the famous critique confirmed quantitatively. |

> **In one sentence:** the Hindenburg Omen fires often enough to feel spooky and rarely enough to seem exclusive — but its forward returns are indistinguishable from a random day, its crash rate is lower than the unconditional base, and with only ~31 independent events in 20 years the small-sample arithmetic makes a verdict impossible even if the direction were right.

## What we tested

A stock-market crash predictor from the 1990s that fires when, on the same day, *both* the fraction of S&P 500 issues making new 52-week highs *and* the fraction making new 52-week lows exceed 2.2% of all issues — with the index above its 50-day MA and advances below declines (a McClellan-oscillator proxy). The logic: a market split between strong and weak stocks is internally incoherent and historically precedes corrections. We built breadth from the S&P 500 constituent panel (yfinance, 2005–2026, current membership — survivorship bias named), collapsed consecutive signals into 31 clusters, and tested SPY forward returns at 30/60/90/120 days against a monthly-sampled unconditional base, with Bonferroni correction across 12 hypotheses (4 horizons × 3 thresholds). The crash-rate comparison (≥5% drawdown within 120 days) rounds out the analysis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the omen is, why it sounds compelling, the spooky-sounding numbers debunked, and what "false-alarm machine" means concretely |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, Welch t, Bonferroni correction, threshold sensitivity, crash-rate comparison, survivorship discussion, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`hindenburg_omen/`](hindenburg_omen/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
