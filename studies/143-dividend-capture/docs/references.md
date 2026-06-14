# References & literature map — Study 143 (Dividend-Capture)

## The claim under test

The "dividend-capture" strategy is a folk recipe: buy a stock a few days before its
ex-dividend date, collect the dividend when it goes ex, and sell immediately after —
supposedly pocketing the dividend as a free gain. The steelmanned version is that stock
prices systematically *under-adjust* on the ex-date (the price drops by less than the
dividend), leaving a systematic gross profit net of the two-way price move. We test this
directly: measure the ex-date price drop vs the dividend across a 25-year panel of 8 US
dividend-paying stocks and ETFs, run the capture trade, and compare to an efficient-markets
baseline.

## The efficient-markets null — why the price should drop by the full dividend

- **Miller & Modigliani (1961).** *Dividend policy, growth, and the valuation of shares.*
  Journal of Business 34(4): 411–433. The foundational irrelevance proposition: in a
  frictionless market, dividends do not create wealth — the stock price falls by exactly the
  dividend on the ex-date, leaving the total portfolio value (price + cash) unchanged. Any
  capture strategy should earn zero gross.

- **Elton & Gruber (1970).** *Marginal stockholder tax rates and the clientele effect.*
  Review of Economics and Statistics 52(1): 68–74. Classic study of ex-date price drops.
  Documents that the drop is less than the dividend for high-dividend stocks (consistent with
  a tax-induced clientele: ordinary-income tax on dividends exceeds capital-gains tax, so the
  market adjusts to the *after-tax* equivalence). Ratio found to be ~0.78 for top-bracket
  investors, meaning some gross capture existed in the 1960s–70s — before dividend tax
  equalisation and the rise of institutional holders.

- **Frank & Jagannathan (1998).** *Why do stock prices drop by less than the value of the
  dividend? Evidence from a country without taxes.* Journal of Financial Economics 47(2):
  161–188. In Hong Kong (no dividend tax), the ex-date drop still fell short of the dividend
  due to the mechanics of market making (bid-ask bounce), not taxes. This suggests that even
  without a tax wedge, pure capture is not free.

## The institutional-arbitrage argument — why it was once bigger and is now gone

- **Kalay (1982).** *The ex-dividend day behavior of stock prices: a re-examination of the
  clientele effect.* Journal of Finance 37(4): 1059–1070. Showed that a dividend-capture
  strategy was bounded in profitability by transactions costs; high-frequency traders could
  push the ratio toward 1 minus costs. As costs fell and institutional arbitrage grew, the
  gap closed.

- **Boyd & Jagannathan (1994).** *Ex-dividend price behavior of common stocks.* Review of
  Financial Studies 7(4): 711–741. Documents that ex-day price drops are close to the
  dividend amount and show no systematic exploitable deviation — consistent with the
  prediction of strong-form efficiency among professional arbitrageurs.

- **Bali & Hite (1998).** *Ex-dividend day stock price behavior: discreteness or tax-induced
  clienteles?* Journal of Financial Economics 47(2): 127–159. Confirms that since the Tax
  Reform Act of 1986 equalized the marginal tax on dividends and capital gains for many
  investors, the ex-date price drop is indistinguishable from the dividend — consistent with
  our finding of a drop ratio ~1.0.

## Why real-world capture strategies fail

- **Graham & Dodd (1934) through standard fixed-income texts.** The simplest argument: you
  don't get the dividend for free. A stock trading at $100 with a $2 dividend goes ex at
  $98. You paid $100 for something worth $98 the next day plus $2 cash — net zero, before
  costs. The "trick" only works if the drop is systematically less than $2.

- **Tax wedge for retail investors.** Qualified US dividends are taxed at 15–20% (plus
  state), while short-term capital gains are taxed at ordinary income rates up to 37%. A
  round-trip capture trade within the 60-day holding period receives *ordinary income* tax
  treatment on the dividend (not the lower qualified rate), making the effective tax cost
  larger than for a long-term holder. The net-of-tax capture return is further reduced.

- **Bid-ask spread and slippage.** The Frank & Jagannathan (1998) Hong Kong study found that
  even with zero tax, bid-ask mechanics absorbed most of the potential gain at the ex-date.
  At 5 bps one-way cost (a round-trip of 10 bps) our study finds net mean *t* = −3.34.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  used in ``strategy.drop_ratio_stats`` and ``strategy.capture_trade_stats`` for all
  inference. Analogous to [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).

- **Unadjusted prices for ex-date analysis.** We use ``auto_adjust=False`` from yfinance:
  backward-adjusted prices distribute the dividend adjustment across all historical bars
  and do not reproduce the raw ex-date drop mechanics. This is the methodological standard
  in the academic literature (Elton-Gruber, Frank-Jagannathan, Boyd-Jagannathan all use
  unadjusted prices).

## Related desk studies

- **[Study 55 — Summer-Lull](../../55-summer-lull/)** and **[Study 67 — Fed-Drift](../../67-fed-drift/):**
  other calendar/seasonal strategies tested with the same disciplined baseline framework.
- **[Study 48 — Groundhog](../../48-groundhog/):** another "event before a known date" study —
  same discipline of asking "does the event window have special positive drift?"
- **[Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/):** the CAPE-based equity
  risk premium, which includes the dividend yield component — the long-run companion to this
  short-run capture study.
