# Study 921 — Bill Ladder vs ETF 🪜

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does rolling your own T-bill ladder out-earn the cash ETF that does it for you? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | **Read the leg first: the ladder is MODELLED, not traded** — it is simulated off **^IRX**, a secondary-market *discount* quote, not auction stop-outs, with no execution in it. On that construction it out-accrues **BIL** by **+12.83 bps/yr**. The significance does not rest on a tuning knob: the daily difference is bounce-negative (lag-1 ACF **−0.37**, Roll 1984), so the naive *t* = +1.15 is too *small*, and **non-overlapping** weekly/monthly/quarterly sums — no bandwidth, no block length, nothing to choose — give *t* = **+2.18 / +3.27 / +3.54**, agreeing with HAC (+2.75) and the bootstrap (CI [+5.3, +20.5]). The mechanism is then nailed shut: adding BIL's expense ratio back to its net return leaves a residual of **−0.4 bps/yr** — the gap *is* the fee — and it is flat in the level of rates (+12.6 at a 0.15% quote vs +13.1 at 3.23%). *Named: era 1 (2007-15) is positive but not individually significant (monthly *t* = +1.3); on the conservative raw-quote convention it is +9.44 (monthly *t* = +2.41) — still clear, but only just.* |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The headline charges **no friction to either leg**, and the edge is capped by the fee it *is*: it dies at **3 bps** per auction (*t* = +0.18) or **3 idle days** per roll (*t* = +1.71). Against **SGOV** it is already only **+2.76 bps** (*t* = +0.62) — the one-click substitute collects most of it without 52 auctions a year. The ladder's 0.15% volatility vs BIL's 0.49% is amortised-cost accounting, not risk reduction: you bought illiquidity, not safety. |

> **In one sentence:** Running your own 3-month T-bill ladder really does beat the cash ETF — by **exactly its expense ratio and not a basis point more** — which makes this a mechanically-attributed 13 bps a year rather than a discovery, and three basis points of friction, three idle days a roll, or simply owning the cheaper fund will each erase it.

## What we tested

A **simulated** rolling **13-rung, 91-day held-to-maturity bill ladder** — one rung bought
every seven days, priced off **^IRX** (the 13-week bill **discount** quote, a PROXY: it is a
secondary-market quote, not the auction stop-out a real ladder would receive). Converted to
bond-equivalent via `P = 1 − d·91/360`, `BEY = (1−P)/P · 365/91` — worth ~9 bps at current
rates, the same order as the effect, so the race is **also run on the raw quote** as a floor.
One execution lag: yesterday's close prices today's purchase. Raced against **BIL**, **SGOV**
and **SHV** total return (`auto_adjust=True`), ^IRX∩BIL 2007-05-31 → 2026-06-30. Cash is the
numeraire, so nothing is excess of anything and **no Sharpe is quoted** — the ladder's
near-zero vol is an accounting artefact, not less risk. Gross-of-fee attribution, a
**bandwidth-free non-overlapping-sums** inference audit (plus a disclosed HAC-bandwidth and
bootstrap-block scan, because HAC *helps* us here), a three-era cut, a cut by *rate level*, a
rung-schedule check, and two swept **PROXY** frictions (per-auction cost, idle reinvestment
days; expense ratios are sponsor stickers, never inputs to a return). **Survivorship:** no
cross-section, but the three funds raced are three that *survived* — closed cash ETFs skew
expensive, so their absence biases *toward* the ladder. **Dedup:** distinct from
**892-corporate-bond-ladder** (ladder-vs-fund in *credit*, where the two hold different
bonds), **885-ultra-short-credit-pickup** (credit *over* bills, which takes this benchmark on
trust), **380-curve-roll-down** (deliberately taking duration — SHV is here as that control),
and **603-treasury-auction-concession** (trading *around* auctions, not rolling through them).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the arithmetic should be boring, the fee recovered to half a basis point, the level-invariance tell, and the three ways to lose it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | ladder construction and the one lag, the discount→BEY conversion, the inference audit (bounce, HAC bandwidth scan, non-overlapping sums), the bootstrap, the gross-of-fee attribution, era and rate-level cuts, both friction sweeps, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`bill_ladder/`](bill_ladder/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
