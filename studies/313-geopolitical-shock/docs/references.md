# References & literature map — Study 313 (Geopolitical-Shock)

## The claim under test

- **"Markets shrug off wars and attacks within days."** A perennial of financial
  journalism and advisor commentary after every shock: the market dips on the news, then
  recovers within a session or two, so geopolitical events are noise for a long-term
  investor and — its tradable corollary — *"buy the geopolitical dip."* We steelman the
  strongest version: a curated table of the 28 events any reasonable observer would call a
  major geopolitical shock (invasions, wars, terror attacks, assassinations, missile
  crises), each mapped to the first NYSE session on which it was tradable, and we ask both
  halves of the claim — is there a measurable drift, and is the "shrug it off in days"
  story true?

## The event-study method

- **Constant-mean / market-model event studies.** Brown & Warner (1985), *Using Daily
  Stock Returns: The Case of Event Studies* (Journal of Financial Economics) — the
  reference for daily-return event studies, abnormal returns, and the cumulative abnormal
  return (CAR). We use the simplest benchmark (constant mean) since the "market" here *is*
  the index. MacKinlay (1997), *Event Studies in Economics and Finance* (Journal of
  Economic Literature) — the canonical survey of the CAR apparatus and its test statistics.
- **The placebo / randomization control.** A synthetic control built by recomputing the
  CAR on thousands of random non-event dates is a non-parametric falsification test in the
  spirit of Fisher's randomization inference; it tells you what a CAR of the observed size
  looks like by pure chance. It is a *machinery* check (it can refute, never certify) — see
  the desk's METHODOLOGY on why a synthetic control may not back a Signal stamp.

## What the literature actually finds about geopolitics and stocks

- **Caldara & Iacoviello (2022), *Measuring Geopolitical Risk* (American Economic Review).**
  The GPR index — the data-driven way to date and weight geopolitical shocks (network-
  blocked here, hence our curated table). They find GPR spikes are associated with lower
  stock returns and higher volatility *contemporaneously*, but the effects are modest and
  short-lived for broad equities.
- **Rigobon & Sack (2005), *The Effects of War Risk on US Financial Markets* (Journal of
  Banking & Finance).** Higher war risk pushed equities down and Treasuries up *ahead of*
  the 2003 Iraq invasion — i.e. the move is largely in the anticipation, not the event.
- **Schneider & Troeger (2006), *War and the World Economy* (Journal of Conflict
  Resolution).** Major-conflict events move equity indices, but the reactions are small and
  quickly absorbed.
- **"Geopolitical sell-offs are short-lived" (practitioner literature).** LPL Research,
  Vanguard and others have repeatedly tabulated that the S&P 500's average drawdown around
  geopolitical shocks is shallow and recovered within weeks — the very "shrug it off"
  pattern this study tests directly.
- **Efficient-markets baseline.** Fama (1970), *Efficient Capital Markets* (Journal of
  Finance). If the news is public and unsurprising in direction, prices adjust in the
  session it breaks and there is no exploitable drift afterward — exactly what an
  insignificant post-event CAR would show.

## Method lineage (the desk's shared engine)

- **Block / event bootstrap CIs.** Politis & Romano (1994), *The Stationary Bootstrap*
  (JASA) — the resampling discipline behind the desk's CIs; here the resampling unit is the
  event (events are far apart, so each is summarised to one CAR).
- **Cross-event t-statistic.** Standard parametric test of the mean CAR across independent
  events (MacKinlay 1997).

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`, `auto_adjust=True`), SPY 1993–2026; the
  shared `quantlab.data` cache is a fallback source. The shock dates are a hand-curated,
  hardcoded table ([`data.py`](../geopolitical_shock/data.py)) because the GPR index feed is
  network-blocked. All headline numbers are pinned with an as-of date and content
  fingerprint (see [`docs/results.md`](results.md)). The offline reproducible core and the
  test-suite run on the deterministic [`data.synthetic_daily`](../geopolitical_shock/data.py)
  generator, never the network.

## Related desk studies

- **Macro & valuation-timing family** — this study sits with the desk's other "does macro
  news predict the index" teardowns (Fed-model, ECY, credit spreads, MOVE), almost all of
  which land Mirage. Geopolitics is the most viscerally compelling of the family and,
  fittingly, among the emptiest as a tradable signal.
- **Calendar / event-table studies** (Super-Bowl, FOMC-cycle, October-effect) — same
  hardcoded-event-table apparatus and the same lesson: a vivid story, a flat abnormal-
  return path.
