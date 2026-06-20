# References & literature map — Study 315 (Sovereign-Downgrade)

## The claim under test

- **"A US downgrade tanks stocks."** The folk version, repeated every time the debt-ceiling
  fight nears a deal: *when a rating agency strips the United States of its top rating, risk
  assets sell off — the downgrade is a flashing-red macro signal, so sell the announcement.*
  The canonical anecdote is the week of **S&P's August 5, 2011 cut to AA+**, when the S&P 500
  fell ~6.7% on the following Monday (Aug 8, 2011). We take the claim literally as a
  directional abnormal-return event study on SPY total return and ask whether the three
  actual US sovereign downgrades carry a statistically distinguishable equity reaction.

## The three downgrade events (the hardcoded table)

- **S&P Global Ratings**, *United States of America Long-Term Rating Lowered To 'AA+' On
  Political Risks And Rising Debt Burden; Outlook Negative* — August 5, 2011 (after the US
  close). The first-ever cut of the US AAA by a major agency.
- **Fitch Ratings**, *Fitch Downgrades the United States' Long-Term Ratings to 'AA+' from
  'AAA'; Outlook Stable* — August 1, 2023 (after the close).
- **Moody's Ratings**, *Moody's Ratings downgrades United States ratings to Aa1 from Aaa,
  changes outlook to stable* — May 16, 2025 (after the close). The last AAA the US held.

  All three announcements landed after the US cash close, so the first *tradeable* session is
  the next business day — the engine's canonical `lag=1` event bar. The dates and post-close
  timing are corroborated by contemporaneous Reuters, Wall Street Journal and Financial Times
  wrap-ups.

## The academic event-study literature on sovereign downgrades

- **Afonso, Furceri & Gomes (2012)**, *Sovereign credit ratings and financial markets
  linkages* (Journal of International Money and Finance) — downgrades have significant, mostly
  spillover, effects on sovereign bond yields and CDS; equity effects are weaker and noisier.
- **Brooks, Faff, Hillier & Hillier (2004)**, *The national market impact of sovereign rating
  changes* (Journal of Banking & Finance) — sovereign downgrades move the *local* equity
  index, but the effect is concentrated in the announcement window and varies widely by
  episode.
- **Gande & Parsley (2005)**, *News spillovers in the sovereign debt market* (Journal of
  Financial Economics) — asymmetry: downgrades carry information, upgrades much less, but the
  cross-border equity reaction is small.
- **Kaminsky & Schmukler (2002)**, *Emerging market instability: do sovereign ratings affect
  country risk and stock returns?* (World Bank Economic Review) — rating events move emerging
  equity markets; the developed-market (and especially US) reaction is far more muted, since
  US Treasuries remain the global risk-free benchmark regardless of the notch.
  The takeaway for the US specifically: the literature predicts a *modest, episode-dependent*
  reaction, not a reliable crash — consistent with WEAK, not REAL, on a three-event sample.

## Why n = 3 cannot certify a signal

- **Small-sample event studies.** With only three events, a per-event t-stat is dominated by
  a single observation and an unreliably small standard error; the honest inference is a
  **permutation / placebo test** against a synthetic control of matched non-event windows
  (the bootstrap event-study tradition; see MacKinlay 1997 below). The desk's inference bar
  is explicit that literature support plus a sub-bar tape reads **WEAK**.
- **MacKinlay (1997)**, *Event studies in economics and finance* (Journal of Economic
  Literature) — the canonical reference for abnormal-return windows, the constant-mean-return
  market model used here, and the cross-sectional inference that a three-event sample cannot
  support.

## The timing trap (why the trade is a mirage)

- **Post-close announcements and the un-tradeable gap.** All three US downgrades were
  announced after the cash close, so the entire reaction arrives as an opening gap on the next
  session — the −6.8% of August 8, 2011 was a single Monday move. A trader acting on the news
  can only enter at or after that bar's close, by which point the move is already in the tape.
  Shorting after the gap then fights the market's long-run upward drift (~+10.8%/yr total
  return) plus short borrow — the structural reason "sell the downgrade" loses.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  reported for completeness in [`strategy.summarize`](../sovereign_downgrade/strategy.py),
  but explicitly *not* load-bearing at n = 3.
- **Circular block bootstrap / permutation inference.** Politis & Romano (1994), *The
  stationary bootstrap* (JASA); the placebo/synthetic-control logic mirrors the desk's other
  low-n event studies.

## Data sources used here

- **SPY total-return daily bars** (dividends reinvested), via the shared
  `_cache/SPY_total_return.parquet` panel with a `quantlab.data` fallback. As-of 2026-06-12;
  all headline numbers pinned with a content fingerprint (see [`docs/results.md`](results.md)).
  The offline reproducible core and test-suite run on the deterministic
  [`data.synthetic_spy`](../sovereign_downgrade/data.py) generator, never the network.

## Related desk studies

- **[Study 312 — Debt-Ceiling](../../312-debt-ceiling/)**: the *volatility* sibling — the
  long-vol-into / short-vol-out VIX round-trip around debt-ceiling X-dates. That study notes
  in passing that the one real 2011 vol spike came *after* the deal, on the S&P downgrade;
  this study is the directional-equity teardown of that same downgrade family (announcement
  abnormal returns + synthetic control), and reaches the same broad conclusion from the other
  side. Distinct apparatus, distinct events (rating actions, not X-dates), distinct tape (SPY
  total return, not the VIX).
- **[Study 311 — Government-Shutdown](../../311-government-shutdown/)**: the directional
  "buy/sell the Washington headline" cousin — same "macro scare = tradeable equity move?"
  question, same WEAK/MIRAGE shape.
- **[Study 118 — Fed-Model](../../118-fed-model/)** and **[Study 115 — Credit-Spreads](../../115-credit-spreads/)**:
  the macro-valuation-timing family this study joins.
