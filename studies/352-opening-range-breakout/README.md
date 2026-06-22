# Study 352 — Opening-Range Breakout 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does breaking the first 5-min range predict the day? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On 60 days of real QQQ 5-minute bars the ORB daily return is indistinguishable from zero (t vs 0 = **−0.20**) and **loses** to a same-exposure *random* entry by ~**12 bps/day** (paired t = −1.78, permutation p ≈ 0.996). SPY agrees. The break carries no info beyond the day's drift. |
| **Tradability** — is there an edge to harvest after costs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The break-even one-way cost vs the null is **negative (−6.1 bps)** — no realistic spread/slippage makes ORB win, because it is already behind a random entry **gross**. Net of any cost it's a small loser. |
| **Beats a coin?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The "ORB makes money" headline in a bull sample is the *drift*, not the breakout: a random same-exposure entry makes **+9.7 bps** vs the rule's **−2.5**. |

> **In one sentence:** the viral 5-minute opening-range breakout doesn't beat a coin — on a real intraday tape it earns roughly nothing and actually *underperforms* a random entry of the same exposure, so every dollar the backtest shows is the day's drift, not the morning range, and no cost level can rescue it.

## What we tested

Day-trading TikTok's favourite rule: mark the high/low of the first **5 (or 15) minutes** after the open, go **long** on a break above the range / **short** below, and hold to the close. There's a serious steelman — Zarattini & Grewal (2023), *"Can Day Trading Really Be Profitable?"*, a 5-minute ORB on QQQ — so we test it honestly: not *"does it make money"* (in a 60-day bull tape, any net-long rule does) but *"does breaking the range beat a **same-exposure random entry** on the same days, after realistic intraday costs?"* We fold **60 days** of real QQQ 5-minute bars (yfinance — Yahoo caps 5m history at ~60 days, named on every axis) into sessions, run the breakout with one execution lag, and race it against random entries that hold the same fraction of the day. Because the real sample is short, a deterministic **synthetic positive control** plants a known intraday trend-persistence to prove the engine *would* catch a real ORB edge — it does, t = 3.0–3.6 — while the real tape shows none.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "the range broke up, so buy" feels right, why a random entry beats it, and why the bull-market backtest is just drift — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | ORB vs a same-exposure random null (paired t + permutation p), the three null modes, the one-way cost sweep & negative break-even, and the planted-persistence positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`opening_range_breakout/`](opening_range_breakout/). **Not investment advice** — research & education; a 60-day bull sample can't disprove a multi-year leveraged ORB, but it shows the *bare* rule adds nothing over a random entry here. See [LICENSE](../../LICENSE).*
