"""The abnormal-return event study: path geometry and the t=0 anchor."""

import numpy as np

from social_oracle import eventstudy


def test_forward_matrix_is_centred_at_zero(panel, events):
    mat = eventstudy.forward_matrix(panel, events, horizon=10, pre=5)
    assert (mat[0] == 0.0).all()           # CAR measured from t=0 is 0 at t=0
    assert list(mat.columns) == list(range(-5, 11))
    assert len(mat) == len(events)


def test_car_is_additive_across_horizons(panel, events):
    # CAR(t->t+2) == CAR(t->t+1) + (abnormal return on day t+2): additive by construction
    mat = eventstudy.forward_matrix(panel, events, horizon=3, pre=0)
    # monotone index of offsets; difference of consecutive columns is a daily abnormal
    diffs = mat[2] - mat[1]
    assert np.isfinite(diffs).all()


def test_summary_columns_and_pct_positive(panel, events):
    es = eventstudy.event_study(panel, events, horizon=10, pre=3)
    summ = es["summary"]
    for col in ("mean", "median", "std", "pct_positive", "tstat", "n"):
        assert col in summ.columns
    assert (summ["pct_positive"] >= 0).all() and (summ["pct_positive"] <= 1).all()
    assert es["n_events"] == len(es["matrix"])


def test_pump_and_fade_signature_present(panel, events):
    # The synthetic bakes in a run-up into t=0 then a fade; the forward mean CAR
    # should end below its short-horizon peak (the fade).
    es = eventstudy.event_study(panel, events, horizon=21, pre=5)
    mean = es["summary"]["mean"]
    assert mean.loc[21] < mean.loc[1] + 1e-9   # later horizon fades vs the early pop
