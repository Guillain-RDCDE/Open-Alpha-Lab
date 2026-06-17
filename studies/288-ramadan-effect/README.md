# Study 288 — Ramadan-Effect

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Does Ramadan lift MENA (or global) equities?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | MENA Ramadan-vs-rest gap **+239 bps/mo** (~+40%/yr vs +6%/yr) with the right sign, lower vol, a clean S&P placebo and academic support — but HAC dummy t = **+1.55**, Welch t = **+1.51**, perm p = **0.15** on just **11** Ramadan months. Below the t ≥ 2 bar: literature support without real-tape significance is Weak, not Real. Proxy is `KSA` from 2015 (survivorship/representativeness named). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A "long MENA in Ramadan, flat otherwise" clock nets **+2.2%/yr** vs **+6.5%/yr** buy-and-hold; sitting out 11 months a year plus ~60 bps/yr round-trip cost dominates the single-month edge. |
| **Busted?** | ![Mostly](https://img.shields.io/badge/Mostly-8b949e?style=flat-square) | On the buyable post-2015 proxy the effect is inside the noise band; the S&P placebo gap is negative (clean). The original 1989–2007 local-index result may still hold. |

> **In one sentence:** the Ramadan effect has genuine behavioural pedigree and the right sign on a buyable MENA proxy, but with only 11 Ramadan months since 2015 the +2.4pp/mo gap is statistically silent (HAC t = 1.55) and there is no way to trade one month a year without losing to buy-and-hold.

## What we tested

The Ramadan effect (Bialkowski, Etebari & Wisniewski 2012): equity returns in
Muslim-majority markets are claimed to be higher and less volatile during the holy
month. We hardcode every Ramadan window (Umm al-Qura calendar, 2000–2026) in
`data.py`, label each month as Ramadan or not (≥ 50% calendar overlap, known years
in advance — no look-ahead), and join to monthly returns of a buyable MENA proxy
(iShares MSCI Saudi Arabia, `KSA`) and an S&P 500 **placebo**. We report a HAC
(Newey-West) t on the Ramadan-dummy regression, a Welch t-test, a label-permutation
test, the volatility comparison, an n = 11 power calculation, and a costed
Ramadan-timing backtest. The synthetic positive control confirms the engine fires
when a premium is planted; the real tape confirms the gap is inside the noise band.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the big-looking gap, the permutation null, the placebo, why you can't trade it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC dummy regression, Welch t, permutation distribution, S&P placebo, volatility test, the n=11 power floor, costed backtest, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`ramadan_effect/`](ramadan_effect/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
