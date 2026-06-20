# Study 314 — Jackson-Hole 🏔️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a drift around the symposium? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No session offset (−5…+5) around the Friday keynote clears the bar; the strongest statistic anywhere is **HAC *t* = +1.75** — and that's the *max over 11 correlated offsets*. The speech day's 95% CI is **[−18, +61] bps**, across zero. |
| **Tradability** — could you harvest it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The literal "react to the speech" trade is **net negative** at a one-session hold (−19.3 bps, *t* = −0.78). One event a year caps a career at ~31 round trips — there is no edge to scale. |
| **A "Jackson Hole drift"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | A couple of famous keynotes (Bernanke 2010, Powell 2022) seeded a folk belief the 33-year record doesn't carry. Salient anecdotes, not a signal. |

> **In one sentence:** the Fed's late-August Jackson Hole symposium produces unforgettable headlines but no measurable, tradable drift — across 33 annual keynotes the S&P's behaviour around the speech is statistical noise, and the one-event-a-year cadence means there'd be nothing to trade even if it weren't.

## What we tested

Market folklore holds that the S&P "drifts" around the Kansas City Fed's annual
**Jackson Hole Economic Symposium** — anticipating a dovish keynote in the run-up, or
trending after the Fed Chair's Friday-morning speech. The belief is fed by a few
market-moving addresses (Bernanke's 2010 QE2 hint, Powell's hawkish 2022 eight minutes).
We take it literally as an **event study**: tag the trading session at each fixed offset
(−5 … +5) from each of **33 Friday keynotes** (1993–2025) on shared SPY daily total
returns, measure the abnormal return with a HAC *t* and a block-bootstrap CI, and run the
one trade you could actually place — long the session after the speech. A deterministic
synthetic tape with a tunable speech-day bump (and a null) is the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the famous speeches feel like a pattern, and what 33 years actually show |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the offset sweep with HAC *t*, the thin-sample bootstrap CI, the traded arm, the multiple-testing trap, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`jackson_hole/`](jackson_hole/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
