"""Study 12 — Paper-Prophet: is an ARIMA+GARCH stack forecasting the SPY, or vol-targeting?

The viral thread (@RohOnChain, "How To Build A Time Series Model To Win Every Single Trade")
sells a complete stack: ARIMA(1,0,1) for the *direction*, GARCH(1,1) for the *position size*,
rolling 252-day walk-forward for honesty. The author concedes in his own Part 5 that "the
GARCH-based position sizing is doing more work than the ARIMA forecast direction" and that
directional accuracy is "52 to 55 percent". This package operationalises that concession:

    * :mod:`paper_prophet.data`      — the SPY tape (cached, offline) + the ADF discipline the
      article gets right (prices have a unit root, returns don't).
    * :mod:`paper_prophet.stack`     — the article's ``TimeSeriesTradingSystem`` ported faithfully,
      walk-forward, with a switch to replace ``sign(forecast)`` by a constant-long signal so the
      *forecast* can be isolated from the *sizing*.
    * :mod:`paper_prophet.decompose` — the teardown: directional hit-rate vs 50%, the Sharpe
      decomposition (stack vs vol-targeting control vs flat-sized forecast), alpha-vs-managed-beta,
      the cost sweep, and the in-sample-vs-walk-forward inflation control.
"""
