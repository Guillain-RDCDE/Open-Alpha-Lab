# Study 112 -- Move-Index

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Q5-Q1 spread: **-0.7 bps** (1d, t=-0.76), **+0.1 bps** (5d, t=-0.68), **-28.1 bps** (21d, t=-1.63). No horizon clears \|t\|>=2. MOVE adds nothing over VIX alone (t_MOVE=-0.80). |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No significant signal at any horizon; no edge to trade or cost-adjust. |
| **Beats VIX?** | ![No](https://img.shields.io/badge/No-8b949e?style=flat-square) | MOVE-VIX correlation = 0.59; joint regression shows neither index predicts forward SPY returns (t<2 for both at all horizons). |

> **In one sentence:** the MOVE index (ICE BofA bond implied vol) is a coincident barometer of cross-asset stress, not a forward-looking equity timer -- high MOVE tells you markets are already stressed, not where SPY will be next week.

## What we tested

The claim: *"MOVE is the VIX for bonds. When bond-market fear (MOVE) rises faster than equity fear (VIX), sophisticated fixed-income players are pricing in macro risk that equities haven't yet priced -- watch MOVE to front-run equity drawdowns."* We tested this steelmanned version on 5,796 daily observations (2003-2026) by sorting days into MOVE quintiles (strict out-of-sample 252-day rank) and measuring forward SPY log-returns at 1-, 5-, and 21-day horizons; running OLS of forward returns on standardised MOVE and VIX jointly (Newey-West HAC standard errors); and testing the MOVE/VIX ratio as a cross-asset divergence signal. A synthetic tape with a tunable `move_signal` knob confirms the engine detects real predictability when planted -- the real tape is consistent with `move_signal ~= 0`.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the MOVE concept, the quintile test in plain language, why "coincident" is not "leading" |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats on quintile spreads, joint MOVE-VIX regression, ratio signal, synthetic positive control |

Sources and literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`move_index/`](move_index/). **Not investment advice** -- research and education. See [LICENSE](../../LICENSE).*
