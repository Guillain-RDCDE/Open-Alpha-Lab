"""The synthetic world is deterministic; the pre-FOMC drift is detected only when present; the null
shows nothing; the mask tags exactly one session per announcement; pre-FOMC days are rarer than the rest."""
import numpy as np
from fed_drift import data, strategy as st


def test_world_deterministic(drift_world):
    ret, fomc, truth = drift_world
    ret2, fomc2, _ = data.synthetic_world(drift=0.005, seed=67)
    assert np.allclose(ret.to_numpy(), ret2.to_numpy())
    assert list(fomc) == list(fomc2)
    assert truth.has_drift


def test_drift_detected(drift_world):
    ret, fomc, _ = drift_world
    t = st.drift_table(ret, fomc, lead=1)
    assert t["pre_mean"] > t["rest_mean"]
    assert t["tstat"] > 2.0
    assert t["pre_share"] > 0.05


def test_null_world_no_drift(null_world):
    ret, fomc, _ = null_world
    t = st.drift_table(ret, fomc, lead=1)
    assert abs(t["tstat"]) < 2.0


def test_mask_one_session_per_meeting(drift_world):
    ret, fomc, _ = drift_world
    mask = st.pre_fomc_mask(ret.index, fomc, lead=1)
    assert int(mask.sum()) == len(fomc)               # exactly one pre-FOMC day per announcement


def test_pre_fomc_is_rare(drift_world):
    ret, fomc, _ = drift_world
    t = st.drift_table(ret, fomc, lead=1)
    assert t["n_pre"] < t["n_total"] * 0.1            # a handful of days a year
