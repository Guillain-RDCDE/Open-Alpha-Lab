# Study 660 — Carry-Everywhere 🌍💰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a diversified carry basket earn a real premium? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Equal-weight combo of FX / bond / equity / commodity carry: **−0.07%/yr, Sharpe −0.02, HAC *t* = −0.08** — statistically zero, block-bootstrap 95% Sharpe CI **[−0.45, +0.43]** straddles it. No single leg clears the bar either (best: commodity roll-yield *t* = 1.39, bond term-spread *t* = 1.33); the equity-dividend leg is outright negative (*t* = −1.44, swamped by the growth/value divergence). Diversifying four legs that individually miss *t* ≥ 2 does not manufacture one. |
| **Tradability** — can you harvest it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Already ~zero gross; net of rebalance costs and short-leg borrow it is **−0.26%/yr (5 bps) to −0.30%/yr (10 bps)**, Sharpe **−0.07 / −0.08**. There is no edge to size, let alone one that survives costs. |
| **Crashes everywhere at once?** | ![Busted](https://img.shields.io/badge/Crashes_everywhere_at_once%3F-Busted-8b949e?style=flat-square) | The folklore "carry crash" is a *synchronized* one — every leg unwinds together. It didn't happen: in the **2008 GFC** window the combo was flat (**−0.96%** cumulative) because FX carry cratered **−27.1%** while bond, equity and commodity carry were solidly positive (+4.7% / +13.7% / +8.4%); in **COVID 2020** the combo was **+2.3%** because equity carry fell **−10.7%** while commodity carry spiked **+20.9%**. The legs crash — just not together, in this sample. |

> **In one sentence:** Koijen-Moskowitz-Pedersen-Vrugt's "carry pays everywhere" rebuilds on 19 years of free FX/Treasury/dividend/commodity-roll data (2007-07 → 2026-06) into four legs that individually miss *t* ≥ 2 (best *t* = 1.39) and an equal-weight combo that is statistically indistinguishable from zero (*t* = −0.08, bootstrap CI straddling zero) — not because the legs crash together (they don't, in 2008 or 2020), but because there is no premium there to diversify in the first place on this tape.

## What we tested

The claim, steelmanned: Koijen, Moskowitz, Pedersen & Vrugt (2018, *Carry*, JFE) show
that "the return you'd earn if prices never moved" — carry — predicts returns not just
in FX but in **every** major asset class, and a diversified cross-asset carry basket
is a *more robust* premium than any single sleeve because the legs are only loosely
correlated. We build four **static, ex-ante-fixed** long/short sleeves on free
yfinance total-return data (zero look-ahead — the basket composition is fixed from
each asset class's textbook carry classification, never fit to the sample): **FX**
(long AUD/NZD, short JPY/CHF — the classic carry pairs), **bond** (long IEF 7-10y,
short SHY 1-3y — the term-spread trade), **equity** (long VYM high-dividend-yield,
short VUG growth/low-yield) and **commodity** (long DBC's carry-optimised roll,
short GSG's naive front-month roll — isolating roll yield). Each sleeve's monthly
return gets a Newey-West HAC *t*; the equal-weight combo adds a circular
block-bootstrap Sharpe CI, one-way rebalance costs × turnover plus ETF short-leg
borrow, and the brief's own crash test: cumulative combo (and per-leg) return inside
the hardcoded **2008 GFC** and **2020 COVID** windows. A deterministic synthetic
control (tunable planted carry mean, tunable synchronized-crash factor) proves the
harness is unbiased and shows exactly how a real premium can still fail to certify
once a common tail factor is added. **Dedup:** [364-fx-carry-trade](../364-fx-carry-trade/)
is this study's FX leg alone, done properly (monthly HAC, skew, drawdown) — we reuse
its verdict directionally but don't re-derive it; [147-fx-momentum](../147-fx-momentum/)
is a different FX signal (trend, not carry); [612-em-debt-carry](../612-em-debt-carry/)
is a single packaged-carry sleeve (EM sovereign coupon) with its own promised-vs-collected
decomposition; [638-value-momentum-everywhere](../638-value-momentum-everywhere/) is
the sibling "everywhere" combo study for a *different* signal pair (value+momentum,
not carry) — same multi-sleeve combo architecture, different ingredients, and it also
lands on a statistical-zero combo. This study is the "carry" cell of that same
question, asked on its own terms. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what carry means in four different markets, why "borrow low, lend high" was supposed to travel everywhere, and why blending four so-so ingredients doesn't bake a cake |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the four HAC-t sleeve legs, the correlation matrix, the block-bootstrap Sharpe CI, inverse-vol and ex-equity combo variants, the cost/turnover sweep, the 2008/2020 crisis ledger, and a synthetic control that plants both a carry premium and a synchronized crash |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`carry_everywhere/`](carry_everywhere/). All four sleeves are transparent,
static **proxies** (fixed ex-ante classification, not live deposit rates, futures
curves or a monthly-re-ranked cross-section) — labelled throughout. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
