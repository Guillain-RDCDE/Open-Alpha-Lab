# Study 40 — Paper-Tiger 🐯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the dual-momentum return real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes. Net Sharpe **0.74** with a Lo (2002) t of **3.4** and a Newey-West HAC t of **3.6** (mean +77 bp/mo, 261 months) — it makes money, and the risk-adjusted return is statistically distinguishable from zero. |
| **Tradability** — does it beat just owning a 60/40? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | No. Its one real gift is **crash protection** (max-DD **−23%** vs SPY's **−51%**), but that leans on essentially **one** crisis (2008), has **decayed** since publication (Sharpe 0.87 → 0.65), and a 30% GEM sleeve **doesn't raise** a 60/40's Sharpe (0.86 → 0.86), only trims its drawdown. |
| **"Beats the market"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The claim that sells it. GEM's **0.74** ties plain SPY (**0.77**) and is **beaten by a naïve 60/40 (0.86)**. No version of the headline survives the right benchmark. |

> **In one sentence:** dual momentum is a *real*, sensible, low-turnover crash-defensive rule — but the backtest that *sells* it ("beats the market") doesn't survive a fair benchmark: it ties buy-and-hold, loses to 60/40 on Sharpe, and its one genuine gift (shallower drawdowns) rests on a single crisis and has faded since it was published.

## What we tested

A vendor ([paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading), *"Momentum Asset Allocation"*, SSRN [1585517](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1585517) = Gary Antonacci's **dual momentum**) ships a headline backtest and a Sharpe, with runnable code — the pitch is "harvest the equity premium, dodge the crashes, beat the market". We rebuild a faithful **Global Equities Momentum** book — monthly, relative momentum between US (SPY) and foreign (EFA) equity, gated by an absolute-momentum T-bill filter into bonds (AGG) — on **real ETFs back to 2003**, charge a real 20 bp per switch, and ask the only question that decides it: *does the headline survive contact with the benchmarks a sceptic would actually use?* The offline control is a seeded synthetic world (a momentum premium + shared crashes, and a no-premium null) that exercises the machinery without the network.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a backtest that "beats the market" can still be a paper tiger, the crash-protection that's actually real, and why a boring 60/40 wins on the numbers that count |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | GEM vs SPY vs 60/40, the Lo/HAC inference, the post-publication decay, the crisis-month decomposition, and the portfolio-sleeve test that fails to lift Sharpe |

The fingerprinted real-data run (SPY/EFA/AGG, 2003–2026, fp `e09091438e42`) is in [docs/results.md](docs/results.md). Reproduce it via [examples/verify.py](examples/verify.py) (`--fetch` to download the tape); the offline machinery proof runs on the synthetic world in [paper_tiger/data.py](paper_tiger/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
