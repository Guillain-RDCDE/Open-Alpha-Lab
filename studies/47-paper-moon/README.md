# Study 47 — Paper-Moon 🌙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the Fed Model time the market? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Barely, and not from the model. Over 125 years the timing earns **Sharpe 0.72** vs buy-and-hold's **0.73** — same risk-adjusted return, a point *less* CAGR. The mild forecasting power is plain valuation (E/P), not the model. |
| **Tradability** — worth the complexity? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. It matches buy-and-hold's Sharpe and worst drawdown (−81%) while compounding less — buy-and-hold with extra steps. |
| **"The model's logic holds"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Its defining move — compare E/P to the bond yield — is **inert**: the Fed signal forecasts at +0.12, **E/P alone at +0.16**. And E/P tracks inflation (+0.39) more than the yield it's meant to mirror (+0.26): a real-vs-nominal money illusion (Asness 2003). |

> **In one sentence:** the Fed Model is a famous market-timing rule whose signature ingredient — pitting the earnings yield against the bond yield — adds *nothing*: E/P alone forecasts returns better, the timing matches buy-and-hold while losing return, and the whole construct rests on confusing a real earnings yield with a nominal bond yield.

## What we tested

The **Fed Model** ([paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.369`): stocks are cheap when the S&P's earnings yield (E/P) exceeds the 10-year Treasury yield, expensive when below — so time the market by holding equities when E/P > yield, bonds otherwise. We run it on **125 years** of Robert Shiller's data (1900–2026), race the timing against buy-and-hold, and then ask the question that decides it: **does the bond-yield comparison add anything, or does E/P alone forecast just as well?** We close with the Asness (2003) money-illusion check (does E/P track the bond yield, or inflation?). The offline control is a synthetic world where only E/P is informative and the bond yield is independent — so "the bond term is inert" is provable without the network.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a famous, sensible-sounding rule is buy-and-hold with a logical bug |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | timing vs buy-and-hold, the Fed-signal-vs-E/P forecasting horse race, the inflation-illusion correlations |

The fingerprinted real-data run (Shiller 1900–2026, fp `e94500e7e09d`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [paper_moon/data.py](paper_moon/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
