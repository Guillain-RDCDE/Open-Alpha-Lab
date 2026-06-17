# Study 229 — Beneish M-score

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Hedge (low-M minus high-M) = **+0.87%/yr**, HAC *t* = **+0.65**; firm-level corr(M, return) = **+0.051** (wrong sign). Survivorship-biased upper bound. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Tiny gross spread wiped by short-selling costs, transaction costs, and survivorship haircut; genuine manipulators are absent from survivor panels. |
| **Can the M-score sniff out earnings manipulators before they blow up?** | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Yes in Beneish's original sample of known SEC violators. No as a live equity signal on the S&P 500 survivor universe — the manipulators who blew up aren't in the data. |

> **In one sentence:** the Beneish M-score is a legitimate fraud-detection tool — but in a survivorship-biased S&P 500 panel the manipulators who would have made the short leg profitable simply aren't there, and the HAC t-stat of +0.65 is noise.

## What we tested

The Beneish (1999) 8-variable composite score was designed to detect earnings manipulation
from annual 10-K filings:

M = −4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI
         + 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI

where DSRI = days-sales receivables growth, GMI = gross margin change, AQI = asset quality
change, SGI = sales growth, DEPI = depreciation intensity change, SGAI = SGA ratio change,
TATA = total accruals / total assets, LVGI = leverage change. M > −1.78 flags a likely
manipulator.

We build M-scores from the desk's shared EDGAR caches plus two study-local caches (AR, PPE)
for 161 S&P 500 tickers (2009–2023, one-year lag, 1,089 firm-year pairs), sort into terciles,
and measure the annual low-M minus high-M hedge against a HAC t-stat inference bar.
Survivorship bias is explicit: the panel excludes firms that exited — including many genuine
manipulators — so results are *biased against* finding the short-M edge.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how the M-score works, the manipulation-detection vs. return-prediction distinction, why the signal isn't there in survivor data |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | annual bucket table, HAC t-stat, bootstrap Sharpe CI, firm-level cross-section, survivorship-bias anatomy, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`beneish_m_score/`](beneish_m_score/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
