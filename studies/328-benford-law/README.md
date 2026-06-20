# Study 328 — Benford-Law 🔢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | A long-conforming / short-deviant cross-sectional sort clears the bar (HAC *t* ≈ **+3.4**, robust to costs) — **but only on a current-membership, survivorship-biased panel**, and its premise (single-name prices "obey Benford") is false. Cannot be certified on a point-in-time universe. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The score needs a year of trailing prices *and* a universe whose membership you couldn't have known in real time. The "deviation" is a proxy for a name's price-range and survival — not a tradable, point-in-time edge. |
| **A forensic red-flag for trouble?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | A single ETF's price **fails** Benford (SPY MAD 0.042; TLT 0.134) — that's its narrow price *range*, not its integrity. The thing that *does* conform is **returns** (SPY abs-return MAD 0.009). Benford here measures range, not manipulation. |

> **In one sentence:** stock *prices* don't obey Benford's Law the way the folklore claims (a single name's "deviation" is just how wide its price range is), the one quantity that *does* conform is returns, and the only place a deviation-sort looks tradable is a survivorship-biased panel you couldn't have traded — so it's a forensics lesson, not a signal.

## What we tested

Benford's Law — leading digit `d` appears with probability `log10(1 + 1/d)` — is a genuine forensic-accounting tool: auditors flag fabricated figures by their first-digit fingerprint (Nigrini 1996, 2012), and the index-level digits of the S&P 500 broadly conform (Ley 1996). The viral trading leap is the claim that **a stock whose prices deviate from Benford is "off," and a deviation score should predict lower forward returns.** We take that literally: we measure first-digit conformity (Pearson χ² and Nigrini's MAD) of real split-adjusted prices and returns, then run the forensic claim as a cross-sectional backtest — every month, short the most-deviant names and long the most-conforming, with one execution lag, costs one-way × NAV, an HAC *t* and a block-bootstrap CI. A deterministic Benford-conforming geometric walk is the positive control; a range-bound, non-Benford price is the anti-control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Benford's Law is, why a single price *fails* it (and returns pass), and why a "deviation = trouble" trade is a survivorship trap |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | χ²/MAD conformity, the cross-sectional sort with HAC *t*, the survivorship + level-base confounds, costs, the synthetic controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`benford_law/`](benford_law/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
