# References & literature map — Study 743 (Lucky-Number-8)

## The claim under test

- **The superstition.** In Chinese (and much of East-Asian) culture the digit **8** (八,
  *bā*) is near-homophonous with **發** (*fā*, "to prosper / get rich") and is considered
  intensely lucky, while **4** (四, *sì*) sounds like **死** (*sǐ*, "death") and is
  avoided. The folklore, and a real academic literature, is that this shows up in markets
  two ways: prices/limit-orders **cluster** on 8-ending values and shun 4, and
  auspicious dates (above all **8/8**, and the famous 08/08/08 8:08 pm Beijing Olympics
  opening) carry a buying/feel-good **premium**.
- **This desk starts skeptical for the tradable (US-listed) version.** The clustering
  evidence is strongest in *mainland A-share* order books and IPO offer prices — venues
  full of retail investors who literally choose the digits. Whether that fingerprint
  survives onto US ADRs and a US-listed China ETF, priced by global market-makers, is the
  open question this study answers on the real tape.

## The academic anchor — real work, mostly on mainland venues

- **Brown, Chua & Mitchell (2002), *The influence of cultural factors on price clustering:
  Evidence from Asia-Pacific stock markets*, Pacific-Basin Finance Journal 10(3):307–332.**
  Finds trailing-digit clustering that tracks cultural lucky/unlucky numbers across
  Asia-Pacific markets — the seminal "culture moves the last digit" result.
- **Brown & Mitchell (2008), *Culture and stock price clustering: Evidence from The
  People's Republic of China*, Pacific-Basin Finance Journal 16(1–2):95–120.** Documents a
  strong 8-preference / 4-avoidance in Chinese A-share transaction and limit-order prices.
  The closest direct anchor for Part B — note it is measured on mainland order flow, not
  US ADR closes.
- **Bhattacharya, Kuo, Lin & Zhao (2018), *Do Superstitious Traders Lose Money?*,
  Management Science 64(8):3772–3791.** Taiwanese account-level data: retail traders
  submit disproportionately many limit orders at 8-ending prices and *lose* money doing
  so — superstition as a measurable, costly behavioral bias.
- **Hirshleifer, Jian & Zhang (2018), *Superstition and Financial Decision Making*,
  Management Science 64(1):235–252.** Chinese IPOs with lucky (8) listing-code digits are
  overpriced and subsequently underperform — the "superstition premium" and its reversal.
- **Fortin, Hu & Wang (Kramer) — number preference in prices / IPO offer prices.** The
  general finding across this literature: auspicious digits earn a premium at the point of
  human price-setting (IPO offer prices, limit orders) that fades in secondary trading.
- **Harris (1991), *Stock Price Clustering and Discreteness*, Review of Financial Studies
  4(3):389–415.** The universal baseline: prices cluster on **round** numbers (0 and 5) in
  *every* market, superstition or not. This is the confound Part B must difference out —
  and the effect that actually dominates our χ² (China ADRs cluster on 0, not 8).

## What we measure, and the honesty rails

- **Two hardcoded, cited constructs.** (i) The lucky-date calendar (`data.LUCKY_DATES`):
  the 8th of August, one event per year 2005→2025 (the first Aug 8 after FXI's 2004-10
  inception through the last complete one). (ii) The clustering baskets
  (`data.CHINA_ADRS`, `data.CONTROL_US`): 15 US-listed China ADRs and 15 matched
  US-domestic large-caps.
- **Zero look-ahead, by construction.** 8/8 is a *calendar-known* date — unlike an
  earnings print or a plane crash, everyone knows it is coming, so any trade around it is
  fully executable with no information advantage. The single documented convention is the
  snap (day(-1) = last session before Aug 8; day(0) = first on/after).
- **Abnormal, not raw.** The event return is `FXI` minus the `EEM` emerging-markets
  benchmark (both total-return), so a common EM move is differenced out and only the
  China-specific piece is tested — the correct counterfactual for a *China*-culture claim.
- **The right inference unit.** Each 8/8 is one independent, non-overlapping event → a
  **one-sample t across years** (n = 21), never a daily panel. A 20-seed random-window
  placebo and a leave-one-out jackknife guard the one cut that clears the bar.
- **Adjustment mode is a decision.** The trailing-digit test runs on the **raw** Close
  (`auto_adjust=False`) — a split/dividend-*adjusted* close is multiplied by an arbitrary
  factor and its last digit is meaningless. The event returns run on **Adj Close**
  (total-return). Both are labelled everywhere.
- **Round-number confound, differenced out.** Both baskets share the universal 0/5
  round-number preference and a uniform $0.01 tick, so the decisive statistic is the
  China−control two-proportion *z* on digit 8 (and 4), which cancels the common structure.
  The pooled χ² is reported for context with its serial-dependence caveat stated out loud.
- **Survivorship (Signal axis).** The clustering baskets are *current* US-listed names;
  a China ADR that delisted (e.g. under the HFCAA threat) drops out of the pool. The
  contrast is a China-vs-control *digit* comparison, not a return panel, so the bias is
  second-order — but it is named, and both baskets are frozen in `data.py`.

## Data sources

- **Daily raw + adjusted closes** for FXI, EEM, the 15 China ADRs and the 15 US controls
  — yfinance (no key), cached under `_cache/`.
- **The 8/8 calendar and the culture facts** (八≈發, 四≈死, the 08/08/08 8:08 pm Olympics
  opening) — hardcoded in [`data.py`](../lucky_number_8/data.py); the Olympics timing is
  widely documented (IOC / Beijing 2008 official records).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [708-eurovision-effect](../../708-eurovision-effect/) — the shared engine shape (a
  hardcoded annual event calendar, abnormal return vs a regional benchmark, one-sample
  *t*, random-window placebo, jackknife, calendar-known execution). Different trigger
  (a song contest vs a lucky date) and no digit-clustering head.
- [707-plane-crash-effect](../../707-plane-crash-effect/) — an event study on a
  hardcoded disaster calendar with a random-date placebo and a costed overlay. A
  *news-shock* (execution lag matters) vs this study's *calendar-known* date.
- [158-super-bowl](../../158-super-bowl/) / [234-olympic-year](../../234-olympic-year/) —
  folklore calendar signals on a single index; neither tests a per-culture abnormal
  return *or* a price-digit distribution.

No sibling tests **trailing-digit clustering** or a **numerology calendar date** — the
"does the lucky 8 survive onto the US tape, in the last digit *and* around 8/8" angle is
this study's own contribution.
