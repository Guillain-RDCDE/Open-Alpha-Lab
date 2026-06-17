# Study 275 — Whisky-Cask

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Whisky **does not beat** the S&P (price-only): ~12.2% vs ~12.4%/yr, mean annual excess **−0.7pp**, HAC t = **−0.15**. The seductive Sharpe **1.04** is an appraisal-smoothing artifact — it collapses to **0.25** once unsmoothed; n = 16 has no power regardless. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No cask ETF, no daily NAV, reported in arrears. A ~**6.2pp/yr** cost wedge (markup, storage, insurance, evaporation, exit commission) cuts ~12% gross to ~**6% net** — the index is unreachable. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The three-part pitch — uncorrelated, low-vol, equity-beating — fails on all three: the low vol is measurement smoothing, the return ties (and loses net), the negative correlation is sign noise at n = 16. |

> **In one sentence:** rare-whisky casks are sold on a flattering appraisal index whose low volatility is a measurement illusion (lag-1 autocorrelation 0.78 → true vol ~29%, above equities) and whose gross return is roughly halved by costs the index never charges — an alternative-asset mirage in a bottle.

## What we tested

The cask-investment pitch makes three claims: rare whisky **beats equities**, at **lower
volatility**, while being **uncorrelated** to stocks. We hardcode a desk-reconstructed
annual rare-whisky / cask index (appraisal-based, the same numbers cask brokers quote)
in `data.py`, pair it with the S&P 500 (^GSPC) calendar-year **price** returns for
2009–2024, and take the pitch apart on three axes: (1) raw performance and correlation;
(2) **Geltner (1991) appraisal-unsmoothing**, which exposes the measured volatility as
~3× too low and the Sharpe as manufactured; (3) the **cost wedge** — dealer markup,
storage, insurance, the Angels' Share, and exit commission — that a real owner pays but
the index never sees. We close with a Newey-West HAC t-test on the annual excess return
and a synthetic positive control that recovers a known true Sharpe from a smoothed series.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the pitch, the boom-and-hangover chart, the smoothness illusion and the cost wedge in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | raw performance, Geltner unsmoothing, the HAC excess-return test, the cost wedge, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`whisky_cask/`](whisky_cask/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
