# Study 312 — Debt-Ceiling 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The VIX *fell* 5.2% into the deadline on average (HAC *t* = −0.55, only 2 of 7 episodes positive); both legs are indistinguishable from random dates (permutation *p* ≈ 0.5). Seven events anyway. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The long-vol-into / short-vol-out round-trip loses **−10.4% per event** *gross* — dragged by the long-vol carry tax before a cent of transaction cost. No abnormal vol to harvest. |
| **Is brinkmanship a vol trade?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The one "vol spike" (2011) came *after* the deal on the S&P downgrade — the opposite of the trade. One coincidence wearing a strategy's clothes. |

> **In one sentence:** implied vol does **not** reliably ramp into a debt-ceiling deadline or collapse on resolution — the famous 2011 spike came *after* the deal on a credit downgrade, and waiting long-vol for the X-date just bleeds you the carry tax.

## What we tested

Every time Washington nears its debt limit, the derivatives desks circulate the same trade:
*"buy protection into the X-date — implied vol always ramps as the default tail builds, then
collapses the moment Congress blinks, so go long vol into the deadline and short vol on
resolution."* We take it at full strength as a clean **volatility event study** — distinct
from the directional "buy the dip" sibling [Study 311](../../311-government-shutdown/): line
up every brink-going US debt-ceiling deadline since the VIX's 1990 inception (a hardcoded,
Treasury/CRS-sourced table of seven), measure the **VIX** log-change over the 20 sessions
*into* and *out of* each deadline, race both against 3,000 random dates, and score the
canonical vol round-trip **net of the long-vol carry tax**. Real CBOE VIX daily levels back
to 1990; the offline core and tests run on a deterministic mean-reverting synthetic tape
that plants (or withholds) a vol hump.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes in plain language, the random-date comparison, the 2011-carries-it tell |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the VIX hump path, naive *t* vs permutation *p*, the carry-tax round-trip, the synthetic positive control |

The fingerprinted real run lives in [docs/results.md](docs/results.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
