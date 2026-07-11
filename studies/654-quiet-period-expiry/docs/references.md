# References & literature map — Study 654 (Quiet-Period-Expiry)

## The claim under test

- **The folklore.** "When the SEC/FINRA 25-day analyst quiet period ends, the underwriters who
  just took the company public finally let their own analysts speak — and they initiate with
  Buy. The stock pops." IPO trading desks and retail "IPO calendar" sites have repeated this for
  two decades: the conflict of interest (the bank that priced the deal is not about to publish a
  Sell on it three weeks later) is real and well documented; the question is whether it shows up
  as a *tradable* price move.

- **The academic anchor.** **Bradley, D. J., Jordan, B. D., & Ritter, J. R. (2003).** *"The
  Quiet Period Goes Out with a Bang."* Journal of Finance 58(1), 1–36. The canonical paper:
  studies 1996–2000 IPOs and finds a statistically significant **positive abnormal return of
  roughly 2–4%** around the initiation date, concentrated in cases where the coverage is
  Buy-rated (nearly all of it), with a **long-run reversal** afterward — consistent with
  temporary price pressure from a coordinated, predictable burst of one-sided "buy" opinions
  rather than new information.

- **The regulatory object.** FINRA Rule 2711 (successor to NASD Rule 2711, itself tightened after
  the 2003 **Global Settlement** between the SEC/NASD/NYSE and ten investment banks over
  research-analyst conflicts of interest) bars a managing or co-managing underwriter from
  publishing research on the issuer for **25 calendar days** after an IPO (and additional windows
  around lock-up expirations and secondary offerings). Michaely, R., & Womack, K. L. (1999,
  *Review of Financial Studies*) is the pre-Global-Settlement evidence that underwriter analyst
  recommendations are systematically more optimistic than unaffiliated analysts' — the structural
  reason the initiation burst is expected to be one-sided.

## What we measure, and the honesty rails

- **Day-count convention (stated as a decision).** "Day *t*" = *t* trading sessions after the
  stock's first trading day (day 0 = the first close — the same day-0 convention as siblings
  219/319/623, never the offer price). The 25-*calendar*-day quiet period lands near trading day
  ~18, but initiation reports cluster in the days *after* expiration (compliance sign-off,
  scheduling), so — per the brief — we test the trading-day window **[20..30]**, bracketing the
  literal trading-day-25 proxy, plus a "buy day 22, sell day 27" timer as the retail-facing
  version of the trade.
- **Market-adjusted (beta = 1) abnormal returns**, stock return − SPY return on the same
  calendar date — the same convention as [319-lockup-expiry](../319-lockup-expiry/), chosen for
  the same reason: fitting a per-IPO beta on a few weeks of post-listing history is noisier than
  assuming beta ≈ 1.
- **Cross-sectional independence.** Unlike the FOMC/lockup studies where events can share a
  calendar date, each IPO here has its own listing date — the events are (largely) independent,
  so the primary test is a plain one-sample *t* across IPOs, cross-checked by a **paired
  within-IPO placebo** ([3..13] vs [20..30], same ticker) and a **random-window placebo**
  (20 seeds × 1,000 draws of same-width random windows) for a distribution-free *p*.
- **Direct listings and SPAC mergers excluded by construction.** FINRA 2711's 25-day quiet
  period restricts the *managing underwriters of a firm-commitment offering* — it does not bind
  the same way on a direct listing (no underwriter) or a de-SPAC (no traditional book-build), so
  including them would test a different mechanism under the same label.
- **Survivorship on data availability, named.** The basket only keeps tickers with a full
  post-listing window still queryable on yfinance — this can only *thin* the basket (fewer
  events), never manufacture a pop that isn't there.

## Data sources

- **Daily adjusted closes** for the hardcoded IPO basket and **SPY** — yfinance (no key),
  cached under `_cache/` (`prices_654_<TICKER>_1d.parquet`).
- **The IPO basket, hardcoded** in [`data.py`](../quiet_period_expiry/data.py): 66 real,
  underwritten US IPOs, 2015 → 2025 (first trading day). Sources: company IPO prospectuses,
  exchange listing newsroom notices, and public IPO trackers (Renaissance Capital, NASDAQ/NYSE).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [219-ipo-pop](../219-ipo-pop/) — the **day-1** pop (offer price → first close). This study
  starts measuring abnormal returns from that same day-0 close and looks **three-to-six weeks
  later**, at a completely different mechanism (analyst opinions, not underpricing).
- [319-lockup-expiry](../319-lockup-expiry/) — the **180-calendar-day share lock-up** expiry
  (insiders' shares become sellable — a supply-side event). This study is the **25-calendar-day
  quiet period on underwriter *research*** — a much earlier, opinion-side event. Same event-time
  panel design, completely different calendar and completely different economic mechanism (an
  opinion burst, not a share unlock).
- [623-ipo-long-run-underperformance](../623-ipo-long-run-underperformance/) — the **3–5 year**
  drift documented by Ritter (1991). This study's entire window closes by trading day 30 — six
  weeks, not years.
- [636-exchange-listing-pop](../636-exchange-listing-pop/) — the **crypto** listing-day pop and
  fade on Coinbase. Different asset class, different day-0 event (a listing announcement, not an
  IPO), no quiet period involved.

None of the siblings test what happens when the **underwriters' own analysts are finally allowed
to speak** — the quiet-period-expiry claim is this study's own axis.
