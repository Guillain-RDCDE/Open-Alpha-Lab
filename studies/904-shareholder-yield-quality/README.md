# Study 904 — Shareholder-Yield + Quality 💰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does quality-screened shareholder yield beat the market? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The quality overlay genuinely **cleans up raw buybacks**: a shallower crash (**−25.1%** vs **−29.3%**) and a **+0.106** excess-Sharpe gap, positive in *both* eras (+0.163 / +0.073), bootstrap **96% positive** (CI [−0.013, +0.241]). A real "real-buybacks-not-theatre" cleanup in the right direction — but it never clears the bar (NW *t* **+0.57**, no era significant). Short single-regime tape (QUAL-bound to 2013); BUYB too young to test. |
| **Tradability** — can you bank it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Trivially buyable — cheap, liquid, long-only, monthly rebalance drag **0.3 bps/yr**, so nothing erases the QSY-over-RAW edge (not a Mirage). But there is no significant premium to bank: you buy a shallower-drawdown *cleanup* of raw buybacks, not a certified edge — and the whole complex trails SPY. |
| **"Beat the market?"** | ![Busted](https://img.shields.io/badge/Beat_market%3F-Busted-8b949e?style=flat-square) | **No.** Both buyback sleeves trailed plain SPY: quality-screened **−0.91 pp/yr** (*t* = −0.90, bootstrap P(behind)=0.96); raw buyback **−1.51 pp/yr**. Owning SPY beat owning either wrapper. |

> **In one sentence:** overlaying a quality screen on a buyback / shareholder-yield sleeve
> does genuinely clean up raw buybacks — a ~4-point-shallower crash and a **+0.11** excess
> Sharpe over raw PKW, positive in both eras — but the improvement never clears *t* = 2
> (NW *t* 0.57), and **neither** the quality-screened nor the raw sleeve beat plain SPY, so
> it's a real-but-thin tilt cleanup, not the market-beating "real buybacks, not dilution
> theatre" premium the pitch promises.

## What we tested

Two long-only, equal-weight, **monthly-rebalanced** sleeves from live ETFs —
**QSY** = PKW + QUAL (buyback screen + quality overlay, the "real buybacks" blend) and
**RAW** = PKW (unscreened buyback) — raced against each other and **SPY** on monthly
**total returns**, all **excess of cash** (minus BIL, whose monthly return *is* the
realized cash return), over the common window 2013-08 → 2026-06 (155 months, QUAL-bound),
as-of 2026-06-30. We report the excess-of-cash Sharpe race, the QSY-minus-RAW and
QSY-minus-SPY HAC *t*, a paired moving-block bootstrap CI on each Sharpe gap, max drawdown,
a calendar-year table, an era cut (2020-01), and a costed (turnover × spread) net version.
A deterministic synthetic world with a planted quality-over-raw edge proves the machinery.
Short-history / survivor selection is named on the Signal axis; **BUYB** lists 2026-05 and
is too young to race. **Dedup:** siblings [368-buyback-drift](../368-buyback-drift/) (the
event-study drift after buyback *authorizations*), [233-shareholder-yield](../233-shareholder-yield/)
(the cross-sectional dividends-plus-buybacks *factor*), and [519-net-share-issuance](../519-net-share-issuance/)
(the net-issuance *factor*) grade the *signals*; [900-quality-income](../900-quality-income/)
is the sibling *dividend* product race — this study is the *buyback* product race with vs
without a quality overlay, asking the beat-the-market question, on its template.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "dilution theatre" means, why the quality overlay rode +12.8% through 2020 while raw buyback made +8.4%, why a shallower crash ≠ a market-beating edge, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-of-cash Sharpe race, the QSY-minus-RAW / QSY-minus-SPY NW *t*, the block-bootstrap Sharpe-gap CIs, the era cut, the vs-SPY races, the costed sleeves, and the planted-edge synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sy_quality/`](sy_quality/). The audited unit is the LIVE product sleeve net of
its own fee; the only trading is a monthly rebalance to equal weight (one clean
within-month drift, no look-ahead), costed at the ETF spread. **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
