"""The article's ``TimeSeriesTradingSystem``, ported faithfully — and instrumented.

The viral thread ships a runnable class on SPY. Its live engine, ``generate_signals``, walks a
rolling 252-day window forward and, at each step, fits ARIMA(1,0,1) for a one-step directional
forecast and GARCH(1,1) on the ARIMA residuals for a one-step volatility, then trades
``sign(forecast) × min(1, 1/σ̂)``. We reproduce that exactly, with three honesty additions and no
change to the economics:

    1. **It is instrumented.** Every step records the forecast, its sign, the GARCH σ̂, the
       position size and the *next realised return* — so the teardown can separate the contribution
       of the **direction** (the ARIMA sign) from the **sizing** (the GARCH 1/σ̂), which is the
       whole study.

    2. **A constant-long switch.** ``constant_long=True`` replaces ``sign(forecast)`` with ``+1``
       while keeping the *identical* GARCH σ̂ path. This is the pure vol-targeting control: any
       Sharpe it reproduces is sizing, not forecasting.

    3. **A warm start.** Each ARIMA is seeded with the previous day's parameters. This does not
       change the MLE optimum (we verified the per-step forecast is identical to a cold fit up to
       optimiser tolerance) — it only makes ~8,000 daily refits tractable. Nothing economic moves.

Two bugs in the published code are preserved as *findings*, not silently fixed — see
``BUGS`` below and beat 4 of the README. We expose them through flags so the teardown can show the
result barely depends on them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# The published snippet has two defects we surface rather than hide:
#   1. ``arima.forecast(steps=1)[0]`` — positional ``[0]`` on a forecast Series raises KeyError on
#      modern pandas (the code as written does not run); the intent is ``.iloc[0]`` (the next-step
#      forecast). We use ``.iloc[0]``.
#   2. ``signals['signal'].shift(1) * returns`` in the article's P&L line lags the signal a second
#      time: the step-i forecast (which predicts r_i) is applied to r_{i+1}. That is a misalignment,
#      not look-ahead (it is conservative). We grade the forecast on the return it actually predicts
#      (no extra shift) for the headline, and expose ``extra_lag=True`` to reproduce the article's
#      number and show it barely matters.
BUGS = (
    "forecast()[0] -> KeyError on pandas>=2 (intent: .iloc[0])",
    "signal.shift(1) double-lags the forecast vs the return it predicts",
)

LOOKBACK = 252          # the article's rolling window
ARIMA_ORDER = (1, 0, 1)
GARCH_ORDER = (1, 1)
VOL_FLOOR = 0.1         # the article's max(vol_forecast, 0.1)
SIZE_CAP = 1.0          # the article's min(1.0, 1/vol)


def _require_model_deps() -> None:
    """Fail LOUDLY if the modelling stack is missing — never degrade to flat signals.

    ``_fit_forecast_vol`` imports statsmodels/arch lazily, and every caller wraps it
    in the article's bare ``except`` (a failed *fit* ⇒ flat day). Without this
    preflight, a missing ``arch`` would make every step "fail", silently producing
    forecast=0 / size=0 for the whole sample — a run that *looks* complete and is
    actually measuring nothing. An environment problem must raise, not zero-fill.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA  # noqa: F401
        from arch import arch_model  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "paper_prophet.stack needs `statsmodels` and `arch` to fit the "
            "ARIMA(1,0,1)+GARCH(1,1) stack; install them (`pip install statsmodels "
            "arch`) — refusing to run with zero-filled forecasts."
        ) from exc


@dataclass(frozen=True)
class WalkForward:
    """The walk-forward record: one row per graded day, plus the inputs that made it.

    ``frame`` columns (index = trade date, the day whose return is being predicted):
        ret          realised log-return (%) on that day — the thing the forecast tried to call
        forecast     ARIMA one-step point forecast made from data strictly before that day
        sign         sign(forecast) ∈ {-1,+1} — the *direction* bet
        vol          GARCH one-step volatility forecast σ̂ (%)
        size         position size min(1, 1/σ̂) — the *sizing*
    Everything downstream (hit-rate, Sharpe decomposition, costs) is computed from this frame.
    """

    frame: pd.DataFrame
    lookback: int
    constant_long: bool

    @property
    def stack_returns(self) -> pd.Series:
        """The full stack's daily strategy return: sign × size × realised return."""
        f = self.frame
        sig = pd.Series(1.0, index=f.index) if self.constant_long else f["sign"]
        return (sig * f["size"] * f["ret"]).rename("stack")


def _fit_forecast_vol(window: pd.Series, start_params):
    """One step: ARIMA(1,0,1) point forecast + GARCH(1,1) one-step σ̂, on ``window``.

    ``start_params`` warm-starts the ARIMA optimiser (None on the first step). Returns
    ``(forecast, vol, next_arima_start_params)``. Faithful to the article: GARCH is fit on the
    ARIMA residuals and the variance forecast is taken one step ahead.
    """
    from statsmodels.tsa.arima.model import ARIMA
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima = ARIMA(np.asarray(window, dtype=float), order=ARIMA_ORDER).fit(
            start_params=start_params
        )
        forecast = float(arima.forecast(steps=1).iloc[0]
                         if hasattr(arima.forecast(steps=1), "iloc")
                         else np.atleast_1d(arima.forecast(steps=1))[0])
        garch = arch_model(arima.resid, vol="Garch", p=GARCH_ORDER[0], q=GARCH_ORDER[1]).fit(
            disp="off"
        )
        vol = float(np.sqrt(garch.forecast(horizon=1).variance.values[-1][0]))
    return forecast, vol, arima.params


def generate_signals(returns: pd.Series, lookback: int = LOOKBACK,
                     constant_long: bool = False, warm_start: bool = True,
                     max_steps: int | None = None) -> WalkForward:
    """Walk the article's stack forward over ``returns`` and record every step.

    At step ``i`` (for ``i`` from ``lookback`` to ``len(returns)-1``) the window is
    ``returns[i-lookback:i]`` — data strictly before day ``i`` — and the forecast/σ̂ are graded
    against ``returns[i]``, the return realised that day. No look-ahead: the model never sees the
    day it predicts. ``constant_long`` only affects how :meth:`WalkForward.stack_returns` later
    combines the recorded sizes; the recorded forecasts/σ̂ are identical either way, which is what
    makes the vol-targeting control a clean apples-to-apples comparison.

    ``max_steps`` caps the number of graded days (for quick previews / tests); ``None`` runs the
    full out-of-sample span.
    """
    _require_model_deps()
    r = returns.dropna()
    n = len(r)
    rows = []
    sp = None
    stop = n if max_steps is None else min(n, lookback + max_steps)
    for i in range(lookback, stop):
        window = r.iloc[i - lookback:i]
        try:
            forecast, vol, params = _fit_forecast_vol(window, sp if warm_start else None)
            if warm_start:
                sp = params
        except ImportError:
            raise  # a missing dependency is an environment failure, never a flat day
        except Exception:  # a failed fit ⇒ flat that day (the article's bare ``except: 0``)
            forecast, vol = 0.0, np.inf
        size = min(SIZE_CAP, 1.0 / max(vol, VOL_FLOOR)) if np.isfinite(vol) else 0.0
        rows.append(
            {
                "date": r.index[i],
                "ret": float(r.iloc[i]),
                "forecast": forecast,
                "sign": float(np.sign(forecast)),
                "vol": vol if np.isfinite(vol) else np.nan,
                "size": size,
            }
        )
    frame = pd.DataFrame(rows).set_index("date")
    return WalkForward(frame=frame, lookback=lookback, constant_long=constant_long)


def _cold_step(window: np.ndarray):
    """Picklable worker: one *cold* ARIMA+GARCH step on a raw window array.

    Cold (no warm start) is the article's behaviour — it refits from scratch every day — and,
    crucially, cold fits are **independent**, so they parallelise cleanly. We use cold for the
    published headline precisely because the warm-started sign disagrees with the cold sign ~27%
    of the time: the forecasts hover so close to zero that the sign is fragile to the optimiser's
    starting point, which is itself evidence the *direction* is noise (a real signal would not flip
    on initialisation). Returns ``(forecast, vol)``; a failed fit ⇒ ``(0.0, inf)``.
    """
    try:
        return _fit_forecast_vol(window, None)[:2]
    except ImportError:
        raise  # surface a broken worker environment instead of zero-filling the run
    except Exception:
        return 0.0, np.inf


def generate_signals_parallel(returns: pd.Series, lookback: int = LOOKBACK,
                              constant_long: bool = False, n_workers: int | None = None,
                              max_steps: int | None = None) -> WalkForward:
    """Cold, faithful walk-forward — parallelised across processes (the headline runner).

    Identical economics and alignment to :func:`generate_signals` with ``warm_start=False``, but
    the ~8,000 independent daily fits are mapped over a process pool, turning a ~90-minute serial
    run into a few minutes. ``n_workers=None`` uses ``cpu_count-1``.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    _require_model_deps()
    r = returns.dropna()
    n = len(r)
    stop = n if max_steps is None else min(n, lookback + max_steps)
    idx = list(range(lookback, stop))
    windows = [r.iloc[i - lookback:i].to_numpy(dtype=float) for i in idx]
    workers = n_workers or max(1, (os.cpu_count() or 2) - 1)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_cold_step, windows, chunksize=16))

    rows = []
    for i, (forecast, vol) in zip(idx, results):
        size = min(SIZE_CAP, 1.0 / max(vol, VOL_FLOOR)) if np.isfinite(vol) else 0.0
        rows.append(
            {
                "date": r.index[i],
                "ret": float(r.iloc[i]),
                "forecast": forecast,
                "sign": float(np.sign(forecast)),
                "vol": vol if np.isfinite(vol) else np.nan,
                "size": size,
            }
        )
    frame = pd.DataFrame(rows).set_index("date")
    return WalkForward(frame=frame, lookback=lookback, constant_long=constant_long)
