# Study 612 — EM-Debt-Carry 🌍

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — a fat spread you actually collect? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the coupon · None on the collected spread.* The **promised** pickup is mechanically real: EMB paid **+2.46%/yr** more coupon than IEF at **HAC *t* = +20.1**, positive in every rolling year. The **collected** total-return spread — what the claim actually asserts — is **+2.16%/yr at HAC *t* = +0.99** over 18.5 years (skew **−1.72**), and no lag choice or subperiod pushes it near *t* = 2. Index funds carry their own defaults, so survivorship is not the out — the crash months are. |
| **Tradability** — is it carry worth owning? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The package is **hidden equity beta plus duration**: b_SPY = +0.46 (*t* = +7.85), R² = 0.57, and the NW-HAC alpha vs a DIY IEF+SPY mix is **negative** (−2.30%/yr, *t* = −1.44). The spread pays +70 bps/mo in calm months and hands back **−438 bps/mo** in the worst SPY decile (Welch *t* = **−4.84**, corr +0.69) — insurance sold on the equity market you already own. The local-currency version (EMLC) *lost* to Treasuries outright (−0.62%/yr; $118 vs $140). |
| **"Survives the crises it insures"?** | ![Mixed](https://img.shields.io/badge/Survives_the_crises%3F-Mixed-8b949e?style=flat-square) | Arithmetically yes: **+39.6%** cumulative spread remains after 2008/2013/2018/2020/2022. Statistically no: **all five windows are negative**, 33 crisis months hand back **58%** of the 189 calm months' carry (−164 vs +49.6 bps/mo, Welch *t* = −2.15) and drag the whole-sample HAC *t* from **+3.38** (calm) to **+0.99** (all-in). |

> **In one sentence:** EM sovereigns really do pay a fat, never-absent coupon pickup (+2.5%/yr, HAC *t* ≈ 20) — but eighteen and a half years of collecting it netted a total-return spread statistically indistinguishable from zero (*t* = 0.99), because the package is equity beta in a bond wrapper that surrenders half a decade of carry in every risk-off window, and the local-currency version lost to Treasuries outright.

## What we tested

The pitch, steelmanned: *"EM sovereign bonds yield 300-450 bp over Treasuries — fat carry you collect."* We run the packaged versions on a double yfinance tape — total-return *and* price-only — so the monthly difference isolates the **coupon stream** (the promised carry) from the price leg: **EMB** (hard-currency EMBI, 2008-01 → 2026-06, GFC included) against the duration-matched **IEF**, plus the **EMLC** local-currency contrast (2010-08+). The Signal axis tests the *collected* TR spread with a Newey-West HAC *t* (lags 3/6/12, ex-GFC and last-decade subperiods) against the *promised* coupon pickup (rolling-12m path shown). Tradability regresses EMB's excess return on IEF + SPY excess (NW-HAC): the intercept is what's left after duration and the hidden equity beta, and a worst-decile-SPY conditional split (Welch *t*) draws the carry-crash profile. The third axis runs a fixed five-window crisis ledger (2008, 2013, 2018, 2020, 2022): cumulative spread surrendered inside vs collected outside. A deterministic synthetic world with planted spread/coupon/risk-off-beta knobs proves the machinery faithful. **Sibling of [610 — Fallen-Angels-Premium](../610-fallen-angels-premium/) and [611 — mREIT-Carry](../611-mreit-carry/)** — same packaged-carry family, new asset: here the coupon is *sovereign credit* and the killer is the equity-beta-plus-crisis left tail; the currency leg of the story is [364 — FX-Carry-Trade](../364-fx-carry-trade/). Index funds, so the defaults (Venezuela, Lebanon, Russia…) are *on* the tape — survivorship named and dismissed on the Signal axis. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | where the "fat spread" goes, the $233-vs-$170 wealth race that still isn't statistically real, why the crash months always coincide with your stocks, and why the local-currency version lost outright |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | promised-vs-collected decomposition (HAC *t* 20 vs 1), the NW-HAC equity-beta regression, the worst-decile Welch split, the five-window crisis ledger, lag/subperiod robustness, synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (2008-01 → 2026-06, fp `1bff8a946cdf`): [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py).

---

*Engine: [`em_debt_carry/`](em_debt_carry/). The signal is the EMB−IEF total-return spread (HAC *t*) against the TR-minus-price-only coupon pickup; the myth-check is the five-crisis ledger. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
