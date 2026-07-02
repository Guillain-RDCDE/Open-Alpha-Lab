# Results — Study 565 (Filing-Readability): the Loughran-McDonald readability anomaly

*Generated from [`filing_readability/`](../filing_readability/) on this study's **synthetic**
readability panel (there is no free real 10-K-text tape — see the SIGNAL axis). Headline world:
`synthetic_panel(n_stocks=400, obf_alpha=-0.10, seed=565)`, panel fingerprint `53136605f951`.
The real-tape stub `fetch_panel()` is **empty by construction** on a free retail stack, so
`HAVE_REAL` is `False` and no `REAL` stamp can be earned. As-of **2026-06-30**.*

## The verdict, earned — Signal `WEAK` · Tradability `FRAGILE`

Loughran & McDonald (2014) document the **readability anomaly**: 10-K filings that are *harder to
read* — a higher Gunning **fog** index, longer documents, a bigger **file size** (their preferred
robust proxy) — precede **lower** future returns and **higher** post-filing volatility, consistent
with managers obfuscating bad news and the market underreacting. We build a standardised readability
(obfuscation) score from the three legs, sort a synthetic cross-section into terciles, and test
whether the *readable* filers beat the *murky* ones.

There is **no free real tape** for this claim: a point-in-time 10-K text panel (EDGAR full-text
parsed to fog/length/file-size, lined up by filing date with a survivorship-free CRSP-style forward
return) is not a no-key retail artifact. So this study is **synthetic-only**, and the desk rule is
explicit — `REAL` needs a robust *t* ≥ 2 on a **real** tape, which we cannot reach. The literature
support is strong (a *Journal of Finance* anomaly, widely replicated), so the honest stamp is
**`WEAK`** on the signal axis: *"the literature says real; this stack has no tape to certify it."*
Tradability is **`FRAGILE`** — in the plausibility model the long-readable/short-murky book is
net-positive, but the murky short leg is exactly the small/illiquid, hard-to-borrow tail, and with
no real tape there is no capacity or turnover estimate to stand on.

## The synthetic headline run — the anomaly, planted and recovered

The headline world plants the LM anomaly at `obf_alpha = -0.10` (murkier filings earn lower returns)
across 400 firms. The engine recovers it cleanly:

| | value |
|---|---|
| Information coefficient (Spearman, obfuscation vs forward return) | **−0.343** |
| IC *t* (Fisher-z) | **−7.11** |
| Firm-level slope (forward_ret on obfuscation) | **−2.18%** per unit |
| Slope *t* | **−8.11** |
| corr(obfuscation, forward return) | **−0.38** |

A *negative* IC/slope is the anomaly: less-readable filings earn lower returns. (This is a
**machinery proof on planted data**, never market evidence — see the inference bar.)

## The long-short — long readable, short murky (terciles, 120 names/leg)

| Tercile (120 names) | Forward return |
|---|---|
| **Readable** (lowest obfuscation) | **+13.6%** |
| **Murky** (highest obfuscation) | **+1.5%** |
| **Spread (readable − murky)** | **+12.1%** (two-sample *t* **+6.44**) |

The label-shuffle placebo *p* = **0.0005** — the spread sits deep in the tail of the shuffled null,
so on the planted world this is not an artifact of the sort.

## The risk leg — murkier filings, higher post-filing volatility

| | value |
|---|---|
| Slope (post-filing vol on obfuscation) | **+0.030** per unit |
| Slope *t* | **+36.25** |

A *positive* slope is the LM "higher risk" leg: murkier filings carry more post-filing uncertainty.

## Per-leg IC sweep — the sign is stable across legs (LM: file size is cleanest)

| Leg | IC | IC *t* |
|---|---|---|
| Fog index | **−0.285** | −5.85 |
| Length (thousands of words) | **−0.300** | −6.16 |
| File size (MB) | **−0.285** | −5.83 |
| **Composite** | **−0.343** | **−7.11** |

Each individual leg carries the same negative sign; the composite is the strongest — consistent with
LM's point that no single readability proxy is perfect (fog is noisy; file size is a clean, robust
proxy) and a blend is stronger than any leg alone.

## Costs

| | value |
|---|---|
| Gross spread (readable − murky) | **+12.1%** |
| Net (5 bps/leg round-trip + 150 bps/yr borrow, 1y hold) | **+10.4%** |

In the plausibility model the book survives frictions — but the murky short leg is the small,
illiquid, expensive-to-borrow tail, so the 150 bps borrow is deliberately punitive and still
understates the real-world drag a genuine tape would impose.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `obf_alpha` | Mean IC (25 seeds) | Mean IC *t* | |
|---|---|---|---|
| 0.00 (null) | **−0.008** | **−0.17** | flat — no false signal |
| −0.04 | −0.125 | −2.52 | anomaly emerging |
| −0.10 (headline) | −0.284 | −5.82 | anomaly visible |
| −0.16 | −0.410 | −8.69 | strong |
| −0.24 | −0.535 | −11.91 | very strong |

At the null the IC and its *t* are ≈ 0 (and the null long-short *t* is +1.05, placebo *p* 0.30 —
noise); planting a genuine anomaly (`obf_alpha < 0`) drives the IC negative and its *t* past −2 from
`obf_alpha = -0.04` upward. The detector works and does **not** manufacture significance at the null
— so the machinery is sound. (Control only; never cited for a Signal stamp.)

## Why the anomaly can't certify here

1. **No real tape.** The decisive limitation: a point-in-time 10-K readability panel (EDGAR text +
   survivorship-free forward returns via CRSP/Compustat identifiers) is not reachable on a free,
   no-key retail stack. yfinance carries no filing text. Every number above is from the **synthetic**
   world, which is calibrated to the LM literature but is **not market evidence**.
2. **Survivorship, the wrong way.** The murkiest filers are disproportionately the firms that later
   blow up. A survivor return panel would strip those out — biasing a real replication *against* the
   anomaly, exactly as in the desk's distress and accounting-anomaly studies.
3. **Point-in-time alignment & the execution lag.** A real test must line up the *filing-date*
   readability with the return measured strictly *after* the filing is public (a ≥1-day lag). The
   synthetic panel bakes this in (`forward_ret` is post-filing by construction), but a live pipeline
   has to enforce it or it look-aheads the result.

## The honest takeaway

The Loughran-McDonald readability anomaly is a well-documented *Journal of Finance* result, and our
engine recovers it cleanly on a calibrated synthetic world (IC −0.34, *t* −7.1; long-short +12.1%,
*t* +6.4; risk leg positive) while staying flat at the null (IC-*t* −0.17). But the desk certifies
`REAL` only from a **real tape**, and no free real 10-K-text panel exists — so the signal stamp is
**`WEAK`** (literature-real, tape-uncertifiable here) and tradability is **`FRAGILE`** (net-positive
in the model, but a small/illiquid short leg and no capacity estimate). The synthetic control proves
the harness would bank a real anomaly if it could ever see the tape.
