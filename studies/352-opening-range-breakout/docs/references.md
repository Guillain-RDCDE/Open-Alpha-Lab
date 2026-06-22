# References & literature map — Study 352 (Opening-Range Breakout)

## The claim under test

- **The viral pitch.** A day-trading staple all over TikTok / YouTube / "prop-firm" Twitter:
  *"Mark the high and low of the first 5 minutes after the open. When price breaks **above** the
  range, go **long**; when it breaks **below**, go **short**. Hold to the close. The first break
  of the morning range tells you the day's direction."* Often dressed up with a fixed
  risk/reward, a 15-minute variant, or an ATR stop. The testable core: (H₁) the first
  opening-range break carries *directional information beyond the day's drift*; (H₂) that
  information survives realistic intraday costs and beats a same-exposure random entry.
- **The academic steelman.** Zarattini & Grewal (2023), *Can Day Trading Really Be Profitable?*
  (SSRN 4416622) — a **5-minute opening-range breakout on QQQ**, 2016–2023, with a leverage
  overlay and an ATR-based stop/target, reporting a large reported alpha vs buy-and-hold. We
  steelman this version explicitly, then ask the question their headline does not isolate:
  **how much of it is the opening range, and how much is intraday drift + leverage during a
  bull regime?** Our same-exposure random-entry null is the control that separates them.

## Market mechanics & the test object

- **The opening range.** US equity regular session opens 09:30 ET. The "opening range" is the
  high/low of the first *R* minutes (R = 5 → one 5-minute bar; R = 15 → three bars). A break is
  the first subsequent bar whose **close** exits the range; the trade is entered **one bar
  later** (one execution lag, applied once) at that bar's open, in the break's direction, and
  exits at the session close.
- **The same-exposure random null.** The decisive control. On each day the ORB fires, a random
  entry draws a uniform entry bar in the *same eligible window* and holds to the same close,
  with (a) the **same direction** as the ORB trade (isolates *timing* — does breaking the range
  beat entering at a random time?), (b) a **coin** direction (isolates timing **and** sign), or
  (c) **always long** (the day's drift benchmark — the hardest, most honest control in a bull
  tape). If breaking the range carries information, ORB must beat these. The paired, day-by-day
  (ORB − random) difference is the Signal-axis statistic.
- **Costs.** Charged one-way in basis points on NAV, twice per round trip (entry + exit). The
  break-even one-way cost is the level at which the net ORB edge over the null crosses zero.

## Why this is the right test — the relevant finance

- **Intraday momentum is real, but tiny and crowded.** Gao, Han, Li & Zhou (2018), *Market
  Intraday Momentum* (Journal of Financial Economics) — the **first half-hour** return predicts
  the **last half-hour** return on the S&P 500. That is a genuine, published intraday effect —
  but it is small, it is about the *first-30-min vs last-30-min* relationship, and it is not the
  same object as "buy whichever way the 5-minute range breaks." A real adjacent effect is
  exactly the kind of thing that makes a folk rule *look* plausible.
- **Drift dominates in short bull samples.** Over a 60-day yfinance window (the most 5-minute
  history Yahoo serves) the index simply drifts up. Any rule that is net-long most of the day
  will print a positive number; the random-entry null exists precisely to strip that drift out.
  Cooper, Gutierrez & Marcum and the broader "always-in-the-market" literature warn that
  *exposure*, not *timing*, explains most such backtests.
- **Selection & multiplicity in folk rules.** The popular ORB has many free knobs (range length,
  stop, target, time-of-day filter, leverage). Each knob is a fork; reporting the best one is
  significant by construction. We pre-register R ∈ {5, 15} and a fixed hold-to-close, take no
  stop/target tuning, and let the null and the cost sweep decide — no search over the headline.
- **Data-snooping / Reality Check.** White (2000), *A Reality Check for Data Snooping*; Bailey,
  Borwein, López de Prado & Zhu (2014), *Pseudo-Mathematics and Financial Charlatanism* — a
  backtest with enough tuning will always find a winner. The honest defence is an *announced*
  null and an *announced* cost, both fixed before the run.

## Method lineage (the desk's shared engine)

- **Same-exposure random-entry null.** The Signal axis is the *paired* t-stat of
  (ORB − matched-random) day returns, plus a permutation p-value of the ORB mean against the
  distribution of random-entry means ([`strategy.random_null`](../opening_range_breakout/strategy.py)).
  A directional rule must beat its own exposure, drawn at random, to earn `REAL`.
- **Deterministic synthetic positive control.** A fixed-seed per-bar AR(1) day generator
  ([`data.synthetic_days`](../opening_range_breakout/data.py)) plants a *known* intraday
  trend-persistence: the harness must recover a positive, significant ORB-minus-random edge that
  rises with the persistence, and ~0 on a pure random walk. A machinery proof, never market
  evidence (METHODOLOGY → *the inference bar*).
- **One execution lag, costs one-way × NAV.** Enter the bar *after* the breakout bar (one
  `shift`, applied once); costs charged twice per round trip on NAV; break-even cost reported
  next to the edge.

## Data sources used here

- **yfinance** (Yahoo public endpoint, no key): **QQQ** and **SPY** 5-minute bars, ~60 calendar
  days to the cache date, regular session only, cached under `_cache/`. Yahoo caps intraday
  history at ~60 days, so the real sample is **short by construction** — the limitation is named
  on every axis and the synthetic control carries the machinery proof. All headline numbers are
  pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- **[Study 351 — BTC 5-minute Polymarket momentum](../../351-btc-5m-polymarket-momentum/)**: a
  real intraday momentum signal that is entirely priced in — the companion case where the
  *signal* survives but the *edge* does not. Here the signal itself fails to clear a random
  null.
- **[Study 301 — Triple-RSI](../../301-triple-rsi/)**: another viral "it just works" trading
  rule whose headline is an artefact of exposure/exit shape, not a real edge.
