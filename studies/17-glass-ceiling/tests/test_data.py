"""The synthetic tape bakes in exactly what the study needs: a post-breakout drift that is zero on the
null tape (coin-flip bracket) and non-zero on the steelman tapes, on a clean OHLCV frame."""

import numpy as np

from glass_ceiling import data


def test_deterministic_given_seed():
    a, _ = data.synthetic_intraday(n_bars=5_000, seed=17)
    b, _ = data.synthetic_intraday(n_bars=5_000, seed=17)
    assert (a.to_numpy() == b.to_numpy()).all()


def test_ohlc_is_well_formed(null_tape):
    bars, truth = null_tape
    assert list(bars.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert (bars["High"] >= bars[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (bars["Low"] <= bars[["Open", "Close"]].min(axis=1) + 1e-9).all()
    assert (bars["Close"] > 0).all()
    assert (bars["Volume"] > 0).all()
    assert bars.index.name == "bar"
    assert len(bars) == truth.n_bars


def test_groundtruth_flags():
    _, null = data.synthetic_intraday(n_bars=2_000, cont_drift=0.0, seed=17)
    _, up = data.synthetic_intraday(n_bars=2_000, cont_drift=0.001, seed=17)
    _, dn = data.synthetic_intraday(n_bars=2_000, cont_drift=-0.001, seed=17)
    assert null.is_null and null.edge_sign == 0 and null.fair_win_rate == 0.5
    assert (not up.is_null) and up.edge_sign == 1
    assert (not dn.is_null) and dn.edge_sign == -1


def test_continuation_drift_lifts_post_breakout_return(cont_tape, null_tape):
    """On the continuation tape the average return in the bars right after a fresh high is positive;
    on the null tape it is ~zero. This is the mechanism the bracket later monetizes."""
    def post_breakout_mean(bars, lookback=30, horizon=10):
        close = bars["Close"].to_numpy()
        rets = np.diff(close) / close[:-1]
        fired = []
        for t in range(lookback + 1, len(close) - horizon):
            if close[t] > close[t - lookback:t].max():
                fired.append(rets[t:t + horizon].mean())
        return float(np.mean(fired))
    assert post_breakout_mean(cont_tape[0]) > 2e-4
    assert abs(post_breakout_mean(null_tape[0])) < 1e-4


def test_fetch_bars_cache_only_is_offline():
    """Cache miss with fetch=False returns an empty frame and never imports yfinance."""
    out = data.fetch_bars("DOES_NOT_EXIST", interval="5m", cache_dir="/tmp/nope_glassceiling", fetch=False)
    assert out.empty
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
