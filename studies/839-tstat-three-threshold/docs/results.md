# Results — Study 839 (The t > 3 Threshold): how high must the bar really be?

*Generated from [`tstat_threshold/`](../tstat_threshold/) on a **deterministic, offline
synthetic factor zoo** (seed 839). This is a research-method demo, so the tape is built on
purpose: the **null** zoo is pure noise — every candidate factor is a zero-mean return
stream, so any "significant" factor is a false discovery — and the **positive control**
plants a known subset of genuinely-priced factors. Real free factor data can never certify
"zero edge", so there is no real-tape stamp; the data-availability limitation is named on
the SIGNAL axis and the study is capped at `NONE`. Null-zoo fingerprint `2a151cf27292`
(1,000 candidate factors × 240 monthly periods); planted-mixture fingerprint
`7ade07aad75d`.*

**Fingerprint `839:N1000:T240:true50:et4.0:a0.05`** (sim config: seed 839, 1,000 factors,
240 periods, 50 planted true at expected |t|=4, α=0.05) · data as-of 2026-06-30.

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Does the t>2 bar inflate false discoveries?" `CONFIRMED`

Harvey, Liu & Zhu (2016) argue the published cross-sectional "factor zoo" is the survivor
set of an enormous, largely-unreported multiple-testing exercise, and that the conventional
single-test **t > 2** hurdle is far too lax: a haircut is required, and a newly claimed
factor should clear a *t* of about **3.0**, not 2.0. We make the case undeniable on a zoo
we *built* to have nothing in it.

- **The lax bar manufactures discoveries.** On a pure-noise zoo of 1,000 candidate factors,
  **44 (4.40%)** clear `|t| > 2` and only **3 (0.30%)** clear `|t| > 3` — matching the
  Gaussian tails `2·Φ(−2) = 4.55%` and `2·Φ(−3) = 0.27%` almost exactly (a **~17×** gap).
  A few hundred data-mined noise factors therefore *guarantee* a paper's worth of
  "significant" results at t>2.
- **The corrected hurdle rises toward and past 3.0.** For HLZ's tally of **316** factors,
  the Bonferroni family-wise `|t|` cutoff is **3.78** at α = 0.05 (and it climbs with the
  test count: **1.96 → 2.81 → 3.48 → 3.78 → 4.06** for N = 1 → 10 → 100 → 316 → 1,000). The
  FDR-controlling Benjamini-Yekutieli hurdle lands around **3.7–3.8** on this zoo. HLZ round
  the recommendation to the memorable **t ≈ 3.0** — comfortably above the naive bar.
- **The publication haircut is brutal.** Once a 316-test search is disclosed, a factor
  reported at **|t| = 2.0** has a Bonferroni-adjusted p of **1.00** — an effective `|t|` of
  **0.00**, a **100%** haircut, dead on arrival. Even `|t| = 3.0` is haircut **94%** (does
  not survive α = 0.05); only `|t| = 4.0` survives (haircut still **42%**).

So `NONE` on the signal axis (a synthetic-only method demo — the zoo has no real edge, so
there is nothing to detect), `MIRAGE` on tradability (a factor that exists only because the
bar was set at t>2 has, by construction, no out-of-sample paycheck), and `CONFIRMED` on the
myth-check (yes — the conventional t>2 threshold genuinely inflates false discoveries in a
data-mined zoo, and the multiple-testing-adjusted hurdle really does rise to ~3).

## Data stamp

- **Null zoo** (`n_true = 0`, pure noise): 1,000 candidate factors × 240 monthly periods,
  fingerprint `2a151cf27292`, seed 839.
- **Positive-control mixture** (`n_true = 50`, each sized to an expected single-test
  `|t| = 4.0`): same shape, fingerprint `7ade07aad75d`, seed 839.
- **Seed-robust controls**: the null fractions and the mixture FDR/power are each averaged
  over **20 seeds** (839–858), the house rule for any synthetic-dependent claim.

## The headline — how many pure-noise factors clear each bar (null zoo, N = 1,000)

| Threshold | Cleared | Fraction | Null-expected fraction | Null-expected count |
|---|--:|--:|--:|--:|
| `\|t\| > 2` | **44** | **4.40%** | 4.55% | 45.5 |
| `\|t\| > 3` | **3** | **0.30%** | 0.27% | 2.7 |

Observed ≈ expected: the zoo is exactly as noisy as theory says. The max `|t|` across all
1,000 pure-noise factors is **3.23**. The lax bar admits **~17×** more noise than the t>3
bar.

## The hurdle rises with the number of tests (Bonferroni `|t|` cutoff, α = 0.05)

| # tests N | Bonferroni `\|t\|` cutoff |
|---|--:|
| 1 | 1.96 |
| 10 | 2.81 |
| 100 | 3.48 |
| **316** (HLZ's zoo) | **3.78** |
| 1,000 | 4.06 |

The single-hypothesis 1.96 is the number everyone quotes; the moment you admit to searching
hundreds of factors, the honest bar is **~3.8**. HLZ's practical **t ≈ 3.0** recommendation
sits between the naive 2.0 and the strict Bonferroni 3.8.

## The corrections on the pure-null zoo — a paper of nothing, purged

| Rule | Implied `\|t\|` cutoff | Discoveries |
|---|--:|--:|
| naive t>2 | 2.00 | **44** |
| naive t>3 | 3.00 | 3 |
| Bonferroni | 4.06 | **0** |
| Holm | — | **0** |
| BH (FDR, indep.) | — | **0** |
| BHY (FDR, dep.) | — | **0** |

Every multiple-testing correction rejects **zero** on the pure-noise zoo — exactly right:
the corrected bar does not manufacture discoveries where there are none. Only the naive t>2
bar keeps a paper's worth (44) of pure false positives.

## The positive control — a planted mixture (50 true in 1,000, expected `|t| = 4`)

| Threshold | Discoveries | True | False | Realized FDR | Power |
|---|--:|--:|--:|--:|--:|
| `\|t\| > 2` | 93 | 50 | **43** | **46.2%** | 100.0% |
| `\|t\| > 3` | 44 | 41 | 3 | **6.8%** | 82.0% |

Raising the bar from 2 to 3 collapses the realized false-discovery rate from **~half** of
all "discoveries" to **~7%**, at the cost of a modest drop in power (100% → 82%). The
corrections on the same mixture:

| Rule | Implied `\|t\|` cutoff | Discoveries |
|---|--:|--:|
| naive t>2 | 2.00 | 93 |
| naive t>3 | 3.00 | 44 |
| Bonferroni | 4.06 | 19 |
| Holm | 4.28 | 19 |
| BH | 3.18 | 43 |
| BHY | 3.73 | 31 |

## Seed-robust controls (20 seeds) — the machinery is unbiased and keeps the real ones

**Null control** — the pure-noise clearing fractions are unbiased estimators of the tails:

| Quantity | 20-seed mean | Theory |
|---|--:|--:|
| Fraction clearing t>2 | **4.62%** | 4.55% |
| Fraction clearing t>3 | **0.310%** | 0.270% |
| Count clearing t>2 | 46.2 | 45.5 |
| Count clearing t>3 | 3.1 | 2.7 |
| Ratio (t>2 : t>3) | **14.9×** | 16.9× |
| Mean max `\|t\|` (1,000 noise factors) | 3.42 | — |

**Mixture control** — the FDR collapse and BHY's retained power are stable across seeds:

| Quantity | t>2 | t>3 |
|---|--:|--:|
| Mean realized FDR | **47.6%** | **6.4%** (a **7.4×** collapse) |
| Mean power | 97.2% | 85.0% |

BHY on the mixture keeps **30.8** factors on average (implied `|t|` cutoff **3.76**) with a
realized false-discovery rate of just **0.3%** — it is neither the trivial "reject nothing"
nor "reject everything": it purges the noise while retaining genuinely strong factors.

## The publication haircut — a claimed `|t|` once a 316-test search is disclosed (Bonferroni)

| Reported `\|t\|` | Naive p | Adjusted p | Effective `\|t\|` | Haircut | Survives α = 0.05? |
|---|--:|--:|--:|--:|:--:|
| 2.0 | 4.55e-02 | 1.000 | 0.00 | **100%** | No |
| 2.5 | 1.24e-02 | 1.000 | 0.00 | 100% | No |
| 3.0 | 2.70e-03 | 0.853 | 0.19 | 94% | No |
| 3.5 | 4.65e-04 | 0.147 | 1.45 | 59% | No |
| 4.0 | 6.33e-05 | 0.020 | 2.33 | **42%** | **Yes** |

This is the haircut peculiar to the factor zoo (distinct from the generic family-wise
problem in [346](../../346-multiple-testing/)): a factor is only "discovered" against the
backdrop of every other factor that was tried. Disclose the search size and the reported
significance evaporates — a t=2.0 factor is *entirely* an artefact of the search.

## Why the verdict is what it is

1. **Nothing real to detect.** The zoo is a synthetic-only construction — the null carries
   zero edge by design, and no synthetic control can ever certify a real edge (that needs a
   robust *t* ≥ 2 on a real tape). **Signal `NONE`.**
2. **Nothing to trade.** A factor that exists only because the bar was set at t>2 has, by
   construction, no out-of-sample return — the whole point of the haircut. **Tradability
   `MIRAGE`.**
3. **The pitfall is real.** On a zoo we *built* to be empty, the conventional t>2 bar keeps
   44 pure false positives (4.4%), the corrected hurdle rises to ~3.8, and moving to t~3
   collapses the false-discovery rate ~7×. The machinery is unbiased (matches the Gaussian
   tails), fires on nothing in the null, and keeps a planted true subset in the control.
   **`CONFIRMED`.**

## The honest takeaway

A single-test *t* of 2 means one thing when you tested one hypothesis and something entirely
different when you (or the profession) tested three hundred. Harvey-Liu-Zhu's arithmetic is
un-arguable: a few hundred data-mined factors manufacture a paper's worth of "significant"
results from noise alone, the multiple-testing-adjusted hurdle climbs to ~3.8 (Bonferroni)
or ~3.7 (BHY), and the memorable **t ≈ 3.0** rule of thumb is the honest floor. `NONE` ×
`MIRAGE`, myth `CONFIRMED`. This is a method demo on a synthetic world by design — it can
never earn `REAL`, which requires a robust *t* ≥ 2 on a real tape.
