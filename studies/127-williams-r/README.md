# Study 127 — Williams-R

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Zone-entry gross **−9.5 bps/trade**, HAC *t* = **−0.88**; does not beat a random-direction control (Δ = −8.3 bps); no instrument \|*t*\| ≥ 1.5. Cross-back framing is actively negative (*t* = −2.14). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Gross already negative at all hold horizons tested (1, 3, 5, 10 days); no positive break-even cost exists. |
| **Beats a coin?** | ![No](https://img.shields.io/badge/Beats_a_coin%3F-No-8b949e?style=flat-square) | Both the zone-entry (enter on onset of oversold/overbought) and the cross-back (enter on the confirmed reversal) are outperformed by a random-direction control on identical entry dates. |

> **In one sentence:** Larry Williams' %R(14) signals oversold and overbought levels on the daily chart but carries no directional information beyond a coin flip — the "exhausted move" bounce does not reliably materialise, and the cross-back confirmation enters after the move has already occurred.

## What we tested

A classic of the retail charting toolkit: Williams %R normalises the close within the trailing 14-bar high-low range to [−100, 0] — near −100 means the close is at the bottom of its recent range ("oversold"), near 0 means it is at the top ("overbought"). The folk rule buys when %R dips below −80 and sells/shorts when it rises above −20, betting on a mean-reversion bounce. We steelman this as: *the extreme position of the close within its range carries enough directional information over the next 1–10 days to beat a random-direction entry on the same bars.* We test both the *zone-entry* framing (enter at the onset of the extreme condition) and the *cross-back* framing (enter when %R exits the zone, a "confirmed reversal"), across six liquid daily tapes (SPY, QQQ, IWM, AAPL, TSLA, NVDA) over 10 years, vs a random-direction control. A deterministic synthetic tape with tunable mean-reversion serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what %R is, the oversold bounce story in plain language, the fair bet vs a coin, why both timing choices fail |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, hold-period sweep, zone vs cross-back comparison, the cost arithmetic, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`williams_r/`](williams_r/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
