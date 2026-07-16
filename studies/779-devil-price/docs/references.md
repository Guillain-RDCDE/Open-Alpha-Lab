# References & literature map — Study 779 (Devil-Price)

## The claim under test

- **The folklore.** *666 is the number of the beast* (Revelation 13:18) — the most famous
  unlucky number in the Western world — and markets are riddled with number superstition.
  The trader's version: a stock whose price sits *on a 666 handle* (6.66 / 66.6 / 666 …)
  carries a curse and goes on to lag. The most-cited real-world echo is the **S&P 500's
  12 March 2009 intraday bottom near 666**, endlessly annotated as a "devilish low" — which
  in fact marked the *start* of a decade-long bull, the opposite of a curse.
- **Why it's a clean cross-sectional test.** A price mantissa is observable in real time and
  known before the forward window, so ranking the S&P 100 by log-price distance to the
  nearest 666 level and sorting into quintiles is a zero-look-ahead characteristic sort. The
  characteristic is **decade-invariant** ($6.66, $66.6, $666 all score identically), so the
  "devil" quintile is populated every week — see [`data.py`](../devil_price/data.py).
- **The efficient-markets prior.** A stock's *nominal* share price is an artefact of split
  history, not fundamentals (Modigliani-Miller: splits create no value). A sort on an
  arbitrary price digit should therefore be indistinguishable from a random sort — the
  desk's prior is a flat zero; see Fama (1970, *Efficient Capital Markets*, JF).

## What the literature actually says about numbers, prices, and returns

- **Price clustering & psychological barriers** — Osborne (1962, *Operations Research*) and
  Harris (1991, *RFS*) document that prices cluster at round numbers; Donaldson & Kim (1993,
  *JFQA*) find weak "support/resistance" at Dow round-hundred levels. This is a *round-number*
  effect (…00), not a 666 effect, and it concerns micro-level order placement, not a
  cross-sectional return premium.
- **Nominal price level & returns** — Baker, Greenwood & Wurgler (2009, *JFE*, "Catering
  through nominal share prices") and the low-priced-stock literature show nominal price
  correlates with clientele (retail ownership, idiosyncratic vol). Crucially, our
  characteristic is **not** the price level — it is the decade-invariant mantissa, so a $6.66
  and a $666 name score the same; the devil sort deliberately strips the level effect out.
- **Superstition & numerology in markets** — Kolb & Rodriguez (1987) and the "Friday the
  13th" / lunar-calendar anomaly literature (see sibling studies 785-788) test whether dated
  superstitions move prices; the near-universal finding is None. Hirshleifer & Shumway
  (2003, *JF*, weather/mood) is the canonical "sentiment-but-not-tradable" cousin.
- **Numerological price preferences** — Bhattacharya et al. (2012, *RFS*) on the "4/8" digit
  superstition in Chinese and Taiwanese markets shows number taboos *do* shift where prices
  cluster — but as an execution/clustering phenomenon, again not a forward-return edge.

## Data & method

- **Real tape:** ~99 `S&P 100` (CBOE OEX) constituents + `SPY` daily adjusted (total-return)
  closes via [yfinance](https://github.com/ranaroussi/yfinance), one combined panel. The sort
  legs are cross-sectionally demeaned, so the market drops out; SPY is only the "vs the
  market" cross-check.
- **Statistics:** one-sample *t* of the per-rebalance clean−devil spread; Wilson hit-rate
  interval; a 20-seed × 100-draw random-**bucket** placebo (does the *devil* characteristic,
  vs a random quintile split, carry the spread?); a leave-one-year-out jackknife; and a
  **non-overlapping monthly** re-run that is the decisive de-overlapped test.
- **Synthetic positive control:** a seeded panel of correlated names spread across the
  log-price circle with a *planted* devil-quintile underperformance — the detector must
  recover a planted drag monotonically and stay quiet on the null. See
  [`strategy.py`](../devil_price/strategy.py).

*Fama, E. (1970). Efficient Capital Markets. **Journal of Finance**. · Osborne, M. (1962).
**Operations Research**. · Harris, L. (1991). **Review of Financial Studies**. · Donaldson,
R. & Kim, H. (1993). **JFQA**. · Baker, M., Greenwood, R. & Wurgler, J. (2009). **JFE**. ·
Hirshleifer, D. & Shumway, T. (2003). **JF**. · Bhattacharya, U. et al. (2012). **RFS**.*
