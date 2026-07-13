# References & literature map — Study 744 (Tetraphobia)

## The claim under test

- **The superstition.** *Tetraphobia* is the East-Asian avoidance of the number 4,
  whose pronunciation (Mandarin *sì*, Cantonese *sei*, Japanese *shi*, Korean *sa*) is a
  near-homophone of the word for **death**. Its mirror is the auspicious **8** (*bā* /
  *baat*), a homophone of *prosperity/wealth*. The aversion is documented well outside
  finance: buildings omit 4th/14th/24th floors, hospital rooms and flight rows skip 4,
  and car licence plates or phone numbers ending in 4 sell at a **discount** while those
  ending in 8 sell at a **premium** (Woo & Kwok 1994; Bourassa & Peng 1999 for housing).
- **The financial folklore, two halves.** (1) *Prices avoid 4 / prefer 8* — the market
  microstructure version, which **is** an established academic finding. (2) *The unlucky
  date 4/4 underperforms* — the tradable-calendar version, pure media/social folklore,
  with no academic support, tested here for the first time we are aware of.

## The academic anchor — price clustering (a real, replicated effect)

- **Brown & Mitchell (2008), "Culture and stock price clustering: Evidence from The
  People's Republic of China", *Pacific-Basin Finance Journal* 16(1-2), 95-120.** The
  canonical result: the **trailing digit** of Chinese stock prices clusters *away from 4*
  and *toward 8*, a cultural overlay on top of the universal round-number (0/5)
  clustering. This study's real-tape clustering test is a direct, live-data replication
  of that finding (Taiwan and mainland A-shares reproduce it; internationalised Hong Kong
  does not).
- **Brown, Chua & Mitchell (2002), "The influence of cultural factors on price
  clustering: Evidence from Asia-Pacific stock markets", *Pacific-Basin Finance Journal*
  10(3), 307-332** — the broader Asia-Pacific version, establishing that digit preference
  varies with the local culture's lucky/unlucky numbers.
- **Bhattacharya, Kuo, Lin & Zhao (2018), "Do Superstitious Traders Lose Money?",
  *Management Science* 64(8), 3772-3791** — superstition-driven limit-order placement
  (avoiding 4, favouring 8) in Taiwan; superstitious traders earn *worse* execution, a
  direct link from the digit preference to the order book. Motivates why the clustering
  is a real behavioural footprint, not a data artefact.
- **Round-number clustering (the culture-neutral baseline we subtract):** Osborne (1962),
  Harris (1991, "Stock Price Clustering and Discreteness", *RFS*) — prices cluster on
  round increments (…0, …5) in every market. This study excludes digits 0 and 5 from the
  tetraphobia test precisely because that effect is universal and is *not* the claim.

## Sentiment / superstition-and-returns (the calendar half's cousins)

- **Kolb & Rodriguez (1987), "Friday the Thirteenth: Part VII — A Note", *Journal of
  Finance*** and **Dyl & Maberly (1988)** — the Western "unlucky date" return studies;
  the 4/4 test is the East-Asian analogue (an unlucky *calendar* date, not an unlucky
  *day-of-week*).
- **Hirshleifer, Jiang & Meng / Hirshleifer & Shumway (2003), "Good Day Sunshine"** —
  mood-and-returns evidence; the mechanism the calendar folklore would need (a shared
  superstitious mood depressing a specific date) and does not find.
- **Edmans, García & Norli (2007), "Sports Sentiment and Stock Returns", *JF*** — the
  desk's canonical "a shared national mood moves a whole market on a specific date"
  result, the bar a genuine 4/4 effect would have to clear. It does not.

## What we measure, and the honesty rails

- **Two tapes, two adjustment modes, stated as decisions.** The clustering test needs the
  **raw, un-adjusted, local-currency** close (`auto_adjust=False`) — the actual traded
  price, whose last digit is the object of the superstition; a split/dividend adjustment
  produces a back-computed number no one ever traded, and its trailing digit is
  meaningless. It is therefore **price-only** by construction. The calendar-returns test
  is an ordinary return study, so it uses **total-return** closes (`auto_adjust=True`).
- **The US basket is the placebo.** Every market shows round-number (0/5) clustering, so a
  4-deficit *in Asia alone* proves nothing without a control from a culture with no such
  homophone. The US basket, run through the identical statistic, is that control — and it
  is flat (*z* = −0.84), so the Asian asymmetry is cultural, not generic microstructure.
- **One documented calendar convention, no look-ahead.** 4 April is a fixed date known
  years ahead, so the returns test has no execution lag to get wrong; the only rule is
  snapping a weekend/holiday 4/4 forward to the next session.
- **Inference unit.** Each year's 4/4 is one independent, non-overlapping event → a
  **one-sample t** across years (like the sibling event studies), not a daily panel; a
  random-calendar placebo checks the observed mean against same-size draws of random days.
- **Survivorship** is not a concern for the clustering test (it counts digits of prices
  that traded, not a cross-sectional return panel), and the calendar ETFs are broad,
  continuously-listed index vehicles.

## Data sources

- **Trailing-digit baskets** (raw local-currency closes) and **calendar ETFs**
  (total-return closes) — yfinance (no key), cached under `_cache/`. Tickers listed in
  [`data.py`](../tetraphobia/data.py).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [708-eurovision-effect](../../708-eurovision-effect/) and
  [707-plane-crash-effect](../../707-plane-crash-effect/) — same event-study machinery
  (hardcoded calendar, one-sample *t* across independent events, random-window/placebo,
  costed trade), different trigger. Neither has a *price-clustering* half.
- [158-super-bowl](../../158-super-bowl/), [234-olympic-year](../../234-olympic-year/) —
  folklore *calendar* signals on a single market, the same honest treatment; none test a
  microstructure digit-preference.
- This study's own contribution: pairing a **real, confirmed microstructure footprint**
  of a superstition (the digit clustering) with the **absent return footprint** of the
  same superstition (the 4/4 date) — the cleanest illustration on the bench that a belief
  can be measurably real in *where prices settle* and pure fiction in *when returns
  happen*.
