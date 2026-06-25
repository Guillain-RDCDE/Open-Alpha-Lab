# Study 473 — Balance of Power ⚖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does smoothed BOP lead price? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the BOP up-cross" rule does **not** beat a drift-matched **random-entry** baseline: cross − random = **+3.9 / +3.2 / −7.9 / −32.6 bps** at 5/10/20/60 days, and the cross-vs-random Welch *t* **never clears 2** (max **+0.42** at 5d, *p* = 0.678). The big one-sample *t*'s (20d **+4.08**, 60d **+6.13**) are **pure beta** — the upward drift every long-only entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does buyer/seller balance forecast?"** | ![Busted](https://img.shields.io/badge/Does_buyer%2Fseller_balance_forecast%3F-Busted-8b949e?style=flat-square) | Scramble the **time order** of the BOP readings into nonsense (sign-scramble placebo) and the result *improves*: **99%** of shuffled-ordering BOP series match or beat the real one (*p* = **0.994**). The buyer/seller sequence carries no information. |

> **In one sentence:** Balance of Power looks sober because it's a clean formula — buyers' share of each bar's range, smoothed — but encode the "buy when buyers take over (smoothed BOP crosses up through zero)" rule mechanically and fire it **1,631 times** across 5 indices over 21 years, and it **ties or loses to buying on random days** (and a time-scramble of the readings *beats* it, *p* = 0.99): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. For every bar, **BOP = (close − open) / (high − low)** — the share of the day's range the buyers captured — clipped to [−1, +1] and smoothed with a causal **14-day** trailing average (no look-ahead). A long fires when smoothed BOP **crosses up through zero** (negative on *t−1*, ≥ 0 on *t*: buyers have just taken control), entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **cross vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **sign-scramble placebo** that permutes the BOP readings in time, destroying the sequence the cross depends on while keeping the marginal. Tradability charges costs on every cross. A deterministic synthetic control with a *planted* BOP-leads-price structure proves the detector is live (edge 0 → *t* = +0.14; planted lead → *t* = +3.19), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Balance of Power is, why a "buy when buyers win" rule on a rising market always looks good, the cross-vs-random race, and the time-scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the BOP formula and zero-cross, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the sign-scramble placebo, per-ticker deltas, costs, and a synthetic planted-lead control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`balance_of_power/`](balance_of_power/). BOP = (close−open)/(high−low), smoothed with a causal 14-day MA; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
