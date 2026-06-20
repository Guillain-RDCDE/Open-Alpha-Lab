# References & literature map — Study 312 (Debt-Ceiling), the volatility angle

## The claim under test

- **"Debt-ceiling brinkmanship is a volatility trade."** The recurring market take every
  time the US approaches its debt limit: implied vol rises into the Treasury "X-date" as
  default-tail hedging demand builds, then collapses the instant Washington blinks — so the
  trade is *long vol into the deadline, short vol on resolution* (long S&P 500 puts /
  straddles / VIX calls before, sell after). Voiced in countless brokerage and financial-
  media wrap-ups (Goldman Sachs, JPMorgan and BofA derivatives desks publish "debt-ceiling
  hedge" notes; CNBC/Bloomberg "how to trade the debt ceiling" pieces). This is a testable
  hypothesis about the **VIX level path** around the binding deadline, and we test it
  directly as an event study, racing it against random dates.

## Why this is the *volatility* sibling of Study 311, not a duplicate

- **[Study 311 — Government-Shutdown](../../311-government-shutdown/)** tests the
  *directional* "buy the dip" forward **return** on SPY total-return around funding-gap
  *shutdowns*. This study tests a *different event table* (binding debt-ceiling X-dates,
  not shutdowns), on a *different instrument* (the **VIX** level, plus SPY realized vol),
  on a *different signal axis* (**volatility** level/change, not directional return), with
  a *different trade* (a long-vol/short-vol round-trip with a carry tax, not a buy-and-hold
  dip). The two are deliberately orthogonal; reading them together separates the
  "stocks bounce" folklore from the "vol spikes" folklore.

## The actual debt-ceiling episodes and the "X-date"

- **US Treasury, "Debt Limit" press releases** — the Secretary's letters to Congress
  projecting the "X-date" (the date extraordinary measures are exhausted), the contemporary
  number markets priced.
- **Congressional Research Service**, *The Debt Limit Since 2011* (R43389) and *The Debt
  Limit* primers — the authoritative chronology of brink-going episodes (2011, 2013, 2021,
  2023) and the routine raises we exclude.
- **Bipartisan Policy Center**, debt-limit analyses — independent X-date estimates used by
  markets, and the post-mortems on each standoff.
- **Standard & Poor's (2011)** — the August 5, 2011 downgrade of US sovereign debt from
  AAA to AA+, days *after* the deal was signed: the single event that produced a genuine
  vol spike in this sample, and it spiked on resolution, not into the deadline.

## What actually drives the VIX, and why an event window is hard to read

- **VIX construction.** Whaley (1993, 2009), *Derivatives on Market Volatility* and *The
  Investor Fear Gauge* — the VIX is the 30-day model-free implied vol of S&P 500 options
  (CBOE white paper, 2003/2014). It is an *index level*, not a tradable total-return
  series; you trade VIX futures/options or variance swaps, all of which carry roll.
- **The long-vol carry tax.** VIX futures are usually in **contango** (upward-sloping term
  structure), so a long-vol position rolls down the curve and bleeds value while waiting —
  documented in Alexander & Korovilas (2012), *Diversification with Volatility Products*,
  and in the chronic decay of long-VIX ETPs (VXX). Our `daily_carry_bps` haircut makes
  this the dominant cost of "buy protection into the event."
- **Mean reversion in implied vol.** The VIX is sharply mean-reverting; any conditional
  "spike" must be read against its strong tendency to revert regardless of the event —
  which is exactly why a random-date placebo is essential here.

## Method lineage (the desk's shared engine)

- **Event study around announced dates.** MacKinlay (1997), *Event Studies in Economics and
  Finance* (Journal of Economic Literature) — the cumulative-abnormal-path framework; here
  applied to the VIX level path rather than to returns.
- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.hac_tstat`](../debt_ceiling/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992/1994) — the block-bootstrap CI that
  respects the local dependence the inference must preserve
  ([`strategy.block_bootstrap_ci`](../debt_ceiling/strategy.py)).
- **Permutation / placebo test.** The event mean is compared to a random-date null
  distribution ([`strategy.permutation_pvalue`](../debt_ceiling/strategy.py)) — the small-n
  honest test of "is this window unusual at all?".
- **Low-n caveat.** Seven brink-going episodes is a small-sample event study by
  construction; the wide bootstrap CIs are reported in the open and carry the verdict,
  rather than a point estimate dressed up as significant.

## Data sources used here

- **CBOE VIX** daily index level (cached `^VIX_raw.parquet`, raw — the VIX has no
  dividends/splits so total-return adjustment is meaningless), and **SPY total-return**
  daily bars for the realized-vol cross-check. All headline numbers are pinned with an
  as-of date and content fingerprint (see [`docs/results.md`](results.md)). The offline
  reproducible core and test-suite run on the deterministic
  [`data.synthetic_vix`](../debt_ceiling/data.py) generator, never the network.

## Related desk studies

- **[Study 311 — Government-Shutdown](../../311-government-shutdown/)**: the directional
  sibling (does SPY bounce after a shutdown). Same low-n event-study machinery, opposite
  signal axis. Read together they cover both halves of "how to trade Washington's fiscal
  cliffs."
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: the desk's reference for the
  package + honest-inference shape.
