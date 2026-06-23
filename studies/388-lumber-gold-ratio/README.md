# Study 388 — Lumber-Gold-Ratio 🪵

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the ratio time stocks vs bonds? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Over 18 years the rotation switch the folklore implies is **beaten by a static 60/40** (Sharpe **0.32** vs **0.71**, HAC *t* vs 60/40 = **−1.06**) and **ties a same-exposure coin** (*t* vs random = **+0.25**). The synthetic control recovers a planted edge at **t = 41**, so the flat real-tape result is *no signal*, not low power. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | **~609 whole-book rotations** in 18 years for a book that under-performs the passive mix and carries a **−59.5%** drawdown (twice the 60/40's). A "defensive" switch *more* drawdown-prone than buy-and-hold is the opposite of the pitch. |
| **Beats 60/40?** | ![Busted](https://img.shields.io/badge/Beats_60%2F40%3F-Busted-8b949e?style=flat-square) | The whole point of an intermarket switch is that the *timing* earns its keep over the obvious passive blend. The head-to-head *t* is **negative**, the bootstrap CI straddles zero, and **no lookback/threshold flips it positive**. |

> **In one sentence:** the lumber/gold ratio is a vivid intermarket *story* — but as a tradable stock/bond rotation switch on 18 years of real data it **loses to a static 60/40** (Sharpe 0.32 vs 0.71, *t* = −1.06) and **ties a same-exposure random timer** (*t* = +0.25), with twice the drawdown and ~600 costly rotations, so there is no timing edge to deploy.

## What we tested

True lumber futures (`LBS=F`) were **discontinued in May 2023**, so for a current 2008→2026 window we use the **WOOD** timber-equity ETF as a transparent **lumber proxy** (named on the Signal axis). The signal is the standardised deviation of the *log* **WOOD/GLD** ratio from its 60-day trailing mean; the switch holds **SPY** when the ratio is stretched-high (risk-on, z>0) and **TLT** otherwise (risk-off), with a **1-day execution lag** and a **5 bps one-way cost** on every rotation. Over **18.0 years** (2008-06-25 → 2026-06-22, **4,525** days) we **race** it, on an **excess-of-cash** basis, against buy-and-hold SPY, a **static 60/40** SPY/TLT blend, and a **same-exposure random-timing** control, and read the HAC *t* (and block-bootstrap CI) that the switch's net excess return beats each. A deterministic synthetic tape with a planted-edge knob `pred_r` confirms the engine has power (it recovers a real edge) and does not manufacture significance from the null. (Same cyclical-over-defensive commodity-ratio genre as [Study 305 — Gold-Oil-Ratio](../../305-gold-oil-ratio/) and [Study 85 — Dr Copper](../../85-dr-copper/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the lumber/gold story claims, why "beats the market" means *beats 60/40 not a coin*, and why this switch loses both races — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the rotation switch, the excess-of-cash race vs 60/40 / buy-and-hold / random rotation, HAC *t* + block-bootstrap CIs, costs, and a synthetic planted-edge power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`lumber_gold_ratio/`](lumber_gold_ratio/). Lumber here is an explicit **proxy** (the WOOD timber-equity ETF, since `LBS=F` futures ended May 2023), not the cash lumber price. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
