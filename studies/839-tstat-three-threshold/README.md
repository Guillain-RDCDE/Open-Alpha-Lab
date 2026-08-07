# Study 839 — The t > 3 Threshold 🚩

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real edge to find? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On a factor zoo we *built* to be pure noise, **44 of 1,000** candidate factors clear `\|t\| > 2` (4.40%) and only **3** clear `\|t\| > 3` (0.30%) — dead-on the Gaussian tails (4.55% / 0.27%). Nothing is real; every "significant" factor is a false discovery. A synthetic-only method demo — no real tape, so it can never earn `REAL`. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A factor that exists only because the bar was set at t>2 has, by construction, no out-of-sample paycheck — that is the whole point of the haircut. There is nothing to harvest. |
| **Does the t>2 bar inflate false discoveries?** | ![Confirmed](https://img.shields.io/badge/Does_t%3E2_inflate_false_discoveries%3F-Confirmed-8b949e?style=flat-square) | *Yes.* The corrected hurdle rises with the test count (Bonferroni **1.96 → 3.78** for N = 1 → 316, BHY ~3.7); moving from t>2 to t>3 collapses the realized FDR on a planted mixture from **47.6% → 6.4%** (a 7.4× cut); the corrections reject **0** on the pure null and keep a real subset in the control. |

> **In one sentence:** a single-test *t* of 2 means one thing when you tested one hypothesis
> and something else entirely when the profession quietly tested three hundred — a few
> hundred data-mined factors manufacture a paper's worth of "significant" results from noise
> alone, so the honest, multiple-testing-adjusted hurdle climbs to about **t ≈ 3.0**, which
> is exactly what Harvey-Liu-Zhu (2016) recommend.

## What we tested

Harvey, Liu & Zhu (2016), **"… and the Cross-Section of Expected Returns"**: the published
factor zoo is the survivor set of an enormous, largely-unreported search, so the conventional
**t > 2** bar is far too lax and a haircut is required — a new factor should clear a *t* of
about **3.0**. We make the arithmetic undeniable on a **synthetic factor zoo built to contain
nothing**: 1,000 candidate long-short factors × 240 monthly periods, each a zero-mean noise
stream. We count the fraction clearing t>2 vs t>3, compute the family-wise (Bonferroni /
Holm) and false-discovery-rate (Benjamini-Hochberg / Benjamini-Yekutieli) cutoffs as an
implied `|t|`, and quantify the **publication haircut** a claimed *t* suffers once the search
size is disclosed. A **positive control** buries 50 genuinely-priced factors (expected
`|t| = 4`) in the noise to show the corrections *keep the real ones while purging the fakes*;
every synthetic claim is averaged over ≥ 20 seeds. Framed specifically around the **3.0
hurdle** and the **publication haircut**, not the generic Bonferroni lesson. **Dedup:**
[346-multiple-testing](../346-multiple-testing/) is the *generic* family-wise-error demo (any
domain); [536-anomaly-decay-post-publication](../536-anomaly-decay-post-publication/) studies
what happens to a factor *after* it clears the bar; [343-data-mining-roulette](../343-data-mining-roulette/)
mines a *single* series for a lucky rule rather than a cross-section of factors. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "t-stat above 2" proves nothing after you've tried hundreds of factors, in plain language — a paper's worth of "discoveries" conjured from pure noise, and the honest bar |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the per-factor t-stats, the Gaussian-tail clearing fractions, the Bonferroni/Holm/BH/BHY hurdles as an implied `\|t\|`, the FDR collapse on a planted mixture, the publication haircut, and the seed-robust controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`tstat_threshold/`](tstat_threshold/). A synthetic-only research-method demo — no
real market data, capped at `NONE` on the Signal axis by design. **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
