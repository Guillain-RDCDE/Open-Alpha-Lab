# Study 615 — Yen Safe Haven 🇯🇵

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the yen rally when stocks fall? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | Over the **full** 1996–2026 tape, decisively: downside beta **−0.113** (**HAC t = −4.62**, n = 7,418), worst-quintile SPY days pay **+12.17 bps/day** of yen (Welch *t* = **+5.81**), and it is *not* a lucky-episode artefact — it survives dropping every crash year (all six events out: HAC *t* = **−4.27**), dropping the 100 worst SPY days (*t* = **−3.02**), and a 20-seed stationary bootstrap (Welch *t* mean **+5.72**, 100% > 2). **But it is strongly non-stationary and effectively decayed.** Split the sample in half: 1996–2010 carries the whole thing (HAC *t* = **−6.18**, Q1 **+24.04 bps/day**), while **2011–2026 is a flat zero** (HAC *t* = **−0.89**, Q1 **−0.83 bps/day**, Welch *t* = **+1.03**) and **2016–2026** is dead (HAC *t* = **−0.38**, Welch *t* = **+0.16**). And even historically it was **fast-twitch**: at the *monthly* horizon it is absent in every era (down- vs up-months Welch *t* = **0.26**; worst-decile months up only **57%** of the time). Real in the tape, gone from the last decade. |
| **Tradability** — can you own the hedge? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The always-on JPY sleeve bled **−4.34%/yr excess** (−4.74%/yr net of FXY-style expense) for **29.4 years** — an equity-premium-sized insurance bill — while failing outright in the 2022 bear and the Mar-2020 dollar scramble. The negative carry *is* the position; you can't trade your way around it. |
| **"2022 — the hedge that failed exactly when needed?"** | ![Confirmed](https://img.shields.io/badge/2022_failure%3F-Confirmed-8b949e?style=flat-square) | SPY **−24.50%** top→trough while JPY fell **−21.07%** — 86% as much as the thing it was hedging. Mechanical, not bad luck: among SPY down-months the yen makes **+112.5 bps/mo** when yields *fall* (growth scares) and **−150.1 bps/mo** when yields *rise* (rate shocks, Welch *t* of the gap = **+5.14**). 2022 was a rate-shock bear. |

> **In one sentence:** the flight-to-yen was real on crash *days* over the full tape (HAC *t* = −4.62, robust to dropping every episode and to bootstrap) — but it is a **fading** signal, carried by 1996–2010 (*t* = −6.18) and statistically absent since 2011 (*t* = −0.89, dead flat 2016–2026), and even at its peak it never worked at the monthly horizon; as an ownable hedge it is a mirage: ~4.7%/yr of negative carry for three decades, zero monthly hedge value, and a mechanical no-show in rate-driven bears, which is exactly what 2022 was.

## What we tested

The carry-unwind legend: yen-funded carry trades unwind in risk-off, so JPY is the "free" crash hedge. Real tape is daily yfinance `JPY=X` + `SPY` (+ `^IRX`/`^TNX` for regimes), 1996–2026, **7,418 days / 356 complete months**. We estimate the piecewise **downside beta** (jpy ~ min(spy,0) + max(spy,0), HAC/Newey-West t), a **quintile ladder** of yen returns across SPY days (Welch t), the **monthly** down- vs up-month split, six **event studies** (2008, Aug-2015, COVID's two legs, the 2022 bear, Aug-2024) with dates hardcoded and returns computed from the tape, and two **regime splits** — by the carry-stock proxy (US bill ≥ 2% vs ZIRP; the hedge survives *both*, so it's a general risk-off reflex, not purely carry positioning) and by yield direction among down-months (the variable that actually turns it off). Tradability charges the honest bill: yen spot (price-only, labeled) minus the US bill given up, excess-vs-excess; the sleeve is static — held in advance, no timing signal, no lag (documented). A seeded synthetic world with a **tunable planted crash-hedge** proves the machinery (null stays flat, plant lights up) and is never cited for a stamp. Sibling: [69-safe-haven](../69-safe-haven/) asks the same question of **gold** — different asset, different mechanism (store-of-value vs positioning), and a different failure mode (gold: coin-flip payoff, no bill; yen: decisive daily payoff, heavy bill).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the yen jumps on scary days, the four crashes it hedged and the two it didn't, and why "free crash insurance" cost 4.7% a year — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | piecewise HAC betas, the quintile ladder, monthly-horizon evaporation, event windows, carry-regime and yield-direction splits, the carry bill, and the planted-hedge synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md) (as-of **2026-06-30**, fingerprint `6d34d723338d`).

---

*Engine: [`yen_safe_haven/`](yen_safe_haven/). JPY return = **minus** the `JPY=X` (USDJPY) change; price-only spot vs total-return SPY, labeled everywhere; excess-vs-excess against the US bill. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
