# Study 40 — Paper-Tiger 🐯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the dual-momentum *timing* real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The book makes money (excess-of-cash HAC t **2.9**) — but so does plain SPY on the same footing (Lo t **3.1**). The test that isolates the switching skill — the **spanning alpha** of GEM on its own ingredients (SPY/AGG) — is **+10 bp/mo with a HAC t of 0.49**: statistically zero. The significance belongs to the assets, not the rule. |
| **Tradability** — does it beat just owning a 60/40? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | No. Its one real gift is **crash protection** (max-DD **−23%** vs SPY's **−51%**), but that leans on essentially **one** crisis (2008), has **decayed** since publication (excess Sharpe 0.74 → 0.52), and a 30% GEM sleeve leaves a 60/40's Sharpe essentially flat (0.68 → 0.70 excess), only trimming its drawdown. |
| **"Beats the market"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The claim that sells it. GEM trails plain SPY in **both** Sharpe conventions (raw 0.76 vs 0.79, excess **0.62 vs 0.67**) and is **beaten by a naïve 60/40 (0.87 raw / 0.68 excess)**. No version of the headline survives the right benchmark. |

> **In one sentence:** dual momentum is a sensible, low-turnover crash-defensive rule — but its statistics belong to the assets it holds, not the switching (spanning alpha t ≈ 0.5), the backtest that *sells* it ("beats the market") fails in both Sharpe conventions, and its one genuine gift (shallower drawdowns) rests on a single crisis and has faded since it was published.

## What we tested

A vendor ([paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading), *"Momentum Asset Allocation"*, SSRN [1585517](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1585517) = Gary Antonacci's **dual momentum**) ships a headline backtest and a Sharpe, with runnable code — the pitch is "harvest the equity premium, dodge the crashes, beat the market". We rebuild a faithful **Global Equities Momentum** book — monthly, relative momentum between US (SPY) and foreign (EFA) equity, gated by an absolute-momentum T-bill filter into bonds (AGG) — on **real ETFs back to 2003**, charge a real 20 bp per switch, and ask the two questions that decide it: *does the headline survive the benchmarks a sceptic would actually use* (SPY and a 60/40, raw **and** excess-of-cash), and *does the switching itself carry any alpha once you regress the book on its own ingredients?* The offline control is a seeded synthetic world (a momentum premium + shared crashes, and a no-premium null) that exercises the machinery without the network.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a backtest that "beats the market" can still be a paper tiger, the crash-protection that's actually real, and why a boring 60/40 wins on the numbers that count |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | GEM vs SPY vs 60/40 in both Sharpe conventions, the spanning-alpha test that separates the asset premium from the timing skill, the post-publication decay, the crisis-month decomposition, and the portfolio-sleeve test |

The fingerprinted real-data run (SPY/EFA/AGG, 2003–2026, fp `c055e32230e2`) is in [docs/results.md](docs/results.md). Reproduce it via [examples/verify.py](examples/verify.py) (`--fetch` to download the tape); the offline machinery proof runs on the synthetic world in [paper_tiger/data.py](paper_tiger/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
