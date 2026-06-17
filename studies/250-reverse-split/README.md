# Study 250 — Reverse-Split

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Post-RS returns are significantly negative at 1m (−14.2%, HAC *t* = −3.49), 6m (−21.6%, *t* = −3.01), and 12m (−31.5%, *t* = −2.70), but the 3m horizon fails (−2.4%, *t* = −0.23). Three of four horizons clear \|*t*\| ≥ 2, yet the distress confound is severe: the same names show poor returns in the random baseline, and the negative drift likely reflects **ongoing distress continuation**, not a clean RS signal. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The implied short-RS trade is not executable: distressed small-caps are often un-borrowable; when borrowable, borrow costs reach 50–200%/yr. n = 17 over 15 years (~1 event/yr) is far too thin for live sizing. The large recovery cases (AIG, Citigroup, GE) cut against the pure "kiss of death" story. |
| **Kiss of death?** | ![Overstated](https://img.shields.io/badge/Overstated-8b949e?style=flat-square) | The negative drift is real *in the data*, but inseparable from the pre-existing distress that caused the reverse split. Three large-company exceptions (AIG, Citigroup, GE) illustrate that the narrative fails for deliberate restructurings. The reverse split is a *symptom*, not the *cause*. |

> **In one sentence:** reverse splits do predict poor subsequent returns — but almost entirely because the companies doing them are already distressed, not because the reverse split itself is a tradable signal, and the short trade is practically un-executable.

## What we tested

The "kiss of death" narrative holds that a reverse split flags a company in such dire straits that further decline is all but guaranteed.  We test this with a hardcoded table of 17 confirmed US reverse-split effective dates (2009–2024), ranging from distressed micro-caps (Exela 1-for-200, Sharps Technology 1-for-100) to large-company restructurings (GE 1-for-8, Citigroup 1-for-10).  Forward returns at 1/3/6/12-month horizons are computed from the day after the effective date using `yfinance` split-adjusted daily closes.

**The key honest caveat:** reverse splits cluster in distressed names.  The negative post-event returns may reflect **distress continuation** rather than any causal effect of the RS event itself.  Without a distress-matched control group (same Altman-Z, leverage, coverage ratio — but no reverse split), the two effects cannot be separated.  We name this confound loudly rather than claim a "pure RS signal".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the kiss-of-death narrative, the horizon sweep, the distress confound, the three exceptions, why shorting RS names is impractical |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-horizon HAC t-stats, RS vs baseline comparison table, positive control, borrow cost analysis, survivorship accounting |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`reverse_split/`](reverse_split/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
