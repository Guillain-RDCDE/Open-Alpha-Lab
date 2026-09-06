"""Strategy tests for Study 1001 — validation schemes graded against a known truth."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from leakyfold import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Features and labels
# --------------------------------------------------------------------------- #
def test_features_are_strictly_backward_looking():
    w = st.synthetic_panel(n=2000)
    px = w["prices"]
    bad = px.copy()
    bad.iloc[1500:] *= 10
    a = st.make_features(px).iloc[:1300]
    b = st.make_features(bad).iloc[:1300]
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_labels_are_forward_returns():
    idx = pd.bdate_range("2020-01-01", periods=100)
    px = pd.Series(np.linspace(100, 200, 100), index=idx)
    L = st.make_labels(px, horizon=10)
    assert L["label"].iloc[0] == pytest.approx(px.iloc[10] / px.iloc[0] - 1)


def test_labels_carry_their_window():
    idx = pd.bdate_range("2020-01-01", periods=200)
    px = pd.Series(np.linspace(100, 200, 200), index=idx)
    L = st.make_labels(px, horizon=20)
    assert (L["label_start"] == L.index).all()
    assert L["label_end"].iloc[0] == idx[20]
    assert (L["label_end"] > L["label_start"]).all()


def test_overlap_is_complete_for_adjacent_long_labels():
    assert st.overlap_fraction(20, 1) == pytest.approx(19 / 20)
    assert st.overlap_fraction(20, 20) == 0.0
    assert st.overlap_fraction(1, 1) == 0.0
    assert st.overlap_fraction(120, 1) > 0.99


# --------------------------------------------------------------------------- #
# The fold schemes
# --------------------------------------------------------------------------- #
def test_every_scheme_partitions_the_test_sets():
    n = 1000
    for folds in (st.kfold_shuffled(n, 5), st.kfold_sequential(n, 5)):
        seen = np.concatenate([t for _, t in folds])
        assert len(seen) == n
        assert len(np.unique(seen)) == n


def test_train_and_test_never_intersect():
    n = 1000
    w = st.synthetic_panel(n=n)
    for folds in (st.kfold_shuffled(n, 5), st.kfold_sequential(n, 5),
                  st.purged_kfold(w["labels"], 5, 0.02),
                  st.walk_forward(n, 5)):
        for train, test in folds:
            assert len(np.intersect1d(train, test)) == 0


def test_sequential_folds_are_contiguous():
    for train, test in st.kfold_sequential(1000, 5):
        assert np.all(np.diff(test) == 1)


def test_walk_forward_never_trains_on_the_future():
    """The defining property, checked rather than assumed."""
    for train, test in st.walk_forward(1000, 5):
        assert len(train) == 0 or train.max() < test.min()


def test_shuffled_folds_do_train_on_the_future():
    """The failure mode, also checked — a test that cannot fail proves nothing."""
    violations = 0
    for train, test in st.kfold_shuffled(1000, 5):
        violations += int((train > test.min()).sum() > 0)
    assert violations == 5


def test_purging_removes_overlapping_training_labels():
    w = st.synthetic_panel(n=2000, horizon=20)
    L = w["labels"]
    plain = st.kfold_sequential(len(L), 5)
    purged = st.purged_kfold(L, 5, embargo=0.0)
    for (tp, _), (tq, _) in zip(plain, purged):
        assert len(tq) <= len(tp)
    assert sum(len(t) for t, _ in purged) < sum(len(t) for t, _ in plain)


def test_no_purged_training_label_touches_its_test_block():
    """The property purging is supposed to guarantee, verified directly."""
    w = st.synthetic_panel(n=2000, horizon=20)
    L = w["labels"]
    starts = L["label_start"].to_numpy()
    ends = L["label_end"].to_numpy()
    for train, test in st.purged_kfold(L, 5, embargo=0.0):
        t0, t1 = starts[test[0]], ends[test[-1]]
        assert not ((ends[train] >= t0) & (starts[train] <= t1)).any()


def test_a_longer_horizon_purges_more():
    w20 = st.synthetic_panel(n=3000, horizon=20)
    w120 = st.synthetic_panel(n=3000, horizon=120)
    kept20 = sum(len(t) for t, _ in st.purged_kfold(w20["labels"], 5))
    kept120 = sum(len(t) for t, _ in st.purged_kfold(w120["labels"], 5))
    assert kept120 < kept20


def test_the_embargo_removes_additional_training_points():
    w = st.synthetic_panel(n=3000, horizon=20)
    none = sum(len(t) for t, _ in st.purged_kfold(w["labels"], 5, embargo=0.0))
    some = sum(len(t) for t, _ in st.purged_kfold(w["labels"], 5, embargo=0.05))
    assert some < none


def test_walk_forward_uses_less_data_than_kfold():
    """The genuine cost of the honest scheme, which is why people avoid it."""
    n = 2000
    wf = np.mean([len(t) for t, _ in st.walk_forward(n, 5)])
    kf = np.mean([len(t) for t, _ in st.kfold_sequential(n, 5)])
    assert wf < kf


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
def test_ridge_recovers_a_planted_linear_relationship():
    rng = np.random.default_rng(1001)
    n = 3000
    X = rng.normal(0, 1, (n, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(0, 0.5, n)
    train = np.arange(0, 2000)
    test = np.arange(2000, n)
    p = st.fit_predict(X, y, train, test, ridge=1e-6)
    assert np.corrcoef(p, y[test])[0, 1] > 0.9


def test_standardisation_uses_only_the_training_fold():
    """A small leak, but the same mistake in miniature — so it is checked."""
    rng = np.random.default_rng(1001)
    n = 1000
    X = rng.normal(0, 1, (n, 2))
    y = X[:, 0] + rng.normal(0, 0.5, n)
    train, test = np.arange(0, 700), np.arange(700, n)
    a = st.fit_predict(X, y, train, test)
    X2 = X.copy()
    X2[test] *= 100          # tampering with the test features must not change the fit
    b = st.fit_predict(X2, y, train, test)
    # predictions change (inputs changed) but the coefficients must not
    assert not np.allclose(a, b)
    c = st.fit_predict(X, y, train, test)
    assert np.allclose(a, c)


def test_fit_predict_declines_on_a_tiny_training_set():
    X = np.random.default_rng(1).normal(0, 1, (100, 3))
    y = np.random.default_rng(2).normal(0, 1, 100)
    out = st.fit_predict(X, y, np.arange(5), np.arange(50, 60))
    assert np.isnan(out).all()


# --------------------------------------------------------------------------- #
# The leak, measured against a known truth
# --------------------------------------------------------------------------- #
def test_shuffled_cv_invents_skill_where_there_is_none():
    """The headline. Predictability is exactly zero and shuffled CV still scores."""
    shuf, wf = [], []
    for k in range(6):
        w = st.synthetic_panel(n=4000, predictability=0.0, horizon=20, seed=1001 + k)
        c = st.compare_schemes(w["features"], w["labels"], k=5)
        shuf.append(c.loc["k-fold, shuffled", "ic"])
        wf.append(c.loc["walk-forward", "ic"])
    assert np.mean(shuf) > 0.02
    assert np.mean(shuf) > np.mean(wf) + 0.02


def test_honest_schemes_report_nothing_when_there_is_nothing():
    ics = []
    for k in range(5):
        w = st.synthetic_panel(n=4000, predictability=0.0, horizon=20, seed=1001 + k)
        c = st.compare_schemes(w["features"], w["labels"], k=5)
        ics.append(c.loc["walk-forward", "ic"])
    assert abs(np.mean(ics)) < 0.06


def test_purging_removes_most_of_the_illusion():
    """Averaged over seeds: purged sits closer to the honest answer than shuffled does.

    A single draw is far too noisy for this — the per-fold IC on 4,000 observations has a
    standard deviation of several hundredths — so the claim is made where it belongs, over
    repeated worlds.
    """
    gaps_shuffled, gaps_purged = [], []
    for k in range(6):
        w = st.synthetic_panel(n=4000, predictability=0.0, horizon=20, seed=1001 + k)
        d = st.leakage_decomposition(w["features"], w["labels"], k=5)
        gaps_shuffled.append(abs(d["shuffled_ic"] - d["walk_forward_ic"]))
        gaps_purged.append(abs(d["purged_ic"] - d["walk_forward_ic"]))
    assert np.mean(gaps_purged) < np.mean(gaps_shuffled)


def test_pooling_predictions_across_folds_invents_a_negative_ic():
    """The trap that this module's scoring avoids, pinned so it cannot creep back.

    Concatenating every fold's predictions and correlating once measures the arrangement of
    the per-fold prediction clouds, not the skill within them. On signal-free data with
    persistent features it reports a large NEGATIVE information coefficient — an artefact
    bigger than the leak the study is about.
    """
    pooled, perfold = [], []
    for k in range(6):
        w = st.synthetic_panel(n=4000, predictability=0.0, horizon=20, seed=1001 + k)
        F, L = w["features"], w["labels"]
        common = F.index.intersection(L.index)
        X = F.loc[common].to_numpy(dtype=float)
        y = L.loc[common, "label"].to_numpy(dtype=float)
        s = st.score_scheme(X, y, st.purged_kfold(L.loc[common], 5, 0.01))
        pooled.append(s["ic_pooled"])
        perfold.append(s["ic"])
    assert np.mean(pooled) < -0.03            # the artefact is real and it is large
    assert abs(np.mean(perfold)) < abs(np.mean(pooled))


def test_label_overlap_is_a_separate_leak_from_shuffling():
    """The point most treatments miss: a sequential split is still not enough."""
    leaks = []
    for k in range(6):
        w = st.synthetic_panel(n=5000, predictability=0.0, horizon=60, seed=1001 + k)
        leaks.append(st.leakage_decomposition(w["features"], w["labels"], k=5)["overlap_leak"])
    assert np.mean(leaks) > 0.0


def test_the_overlap_leak_grows_with_the_label_horizon():
    def leak(hz):
        out = []
        for k in range(4):
            w = st.synthetic_panel(n=5000, predictability=0.0, horizon=hz, seed=1001 + k)
            out.append(st.leakage_decomposition(w["features"], w["labels"], k=5)["overlap_leak"])
        return float(np.mean(out))
    assert leak(120) > leak(1)


def test_a_one_day_label_has_no_overlap_to_purge():
    w = st.synthetic_panel(n=3000, horizon=1)
    plain = sum(len(t) for t, _ in st.kfold_sequential(len(w["labels"]), 5))
    purged = sum(len(t) for t, _ in st.purged_kfold(w["labels"], 5, embargo=0.0))
    assert purged >= plain * 0.97      # almost nothing to remove


# --------------------------------------------------------------------------- #
# Does the fix keep real signal?
# --------------------------------------------------------------------------- #
def test_purged_cv_still_finds_a_planted_signal():
    """A fix that removed the signal along with the leak would be worthless."""
    w = st.synthetic_panel(n=6000, predictability=1.5, horizon=20)
    c = st.compare_schemes(w["features"], w["labels"], k=5, embargo=0.01)
    assert c.loc["purged + embargo", "ic"] > 0.05


def test_the_purged_estimate_tracks_the_planted_size():
    ics = []
    for pred in (0.0, 1.0, 2.5):
        w = st.synthetic_panel(n=6000, predictability=pred, horizon=20)
        c = st.compare_schemes(w["features"], w["labels"], k=5, embargo=0.01)
        ics.append(c.loc["purged + embargo", "ic"])
    assert ics[0] < ics[1] < ics[2]


def test_purged_and_walk_forward_broadly_agree():
    """Both are honest, so they should give similar answers — that is the validation."""
    w = st.synthetic_panel(n=6000, predictability=1.5, horizon=20)
    c = st.compare_schemes(w["features"], w["labels"], k=5, embargo=0.01)
    assert abs(c.loc["purged + embargo", "ic"] - c.loc["walk-forward", "ic"]) < 0.12


def test_compare_schemes_runs_every_scheme():
    w = st.synthetic_panel(n=3000)
    c = st.compare_schemes(w["features"], w["labels"], k=5)
    for name in ("k-fold, shuffled", "k-fold, sequential", "purged k-fold",
                 "purged + embargo", "walk-forward"):
        assert name in c.index


def test_leakage_decomposition_adds_up():
    w = st.synthetic_panel(n=4000, predictability=0.0, horizon=20)
    d = st.leakage_decomposition(w["features"], w["labels"], k=5, embargo=0.01)
    rebuilt = (d["temporal_leak"] + d["overlap_leak"] + d["embargo_effect"]
               + d["embargo_ic"] - d["walk_forward_ic"])
    assert rebuilt == pytest.approx(d["total_illusion"], abs=1e-9)


def test_leakage_decomposition_handles_a_short_series():
    w = st.synthetic_panel(n=200)
    assert st.leakage_decomposition(w["features"], w["labels"]) == {}


def test_the_horizon_sweep_covers_every_horizon():
    w = st.synthetic_panel(n=5000)
    sw = st.horizon_sweep(w["prices"], horizons=(1, 20, 60))
    assert len(sw) == 3
    assert sw["overlap_with_neighbour"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"asset": "SPY", "horizon": 20, "overlap_pct": 0.95, "embargo": 0.01,
         "shuffled_ic": 0.121, "shuffled_r2": 0.013, "sequential_ic": 0.084,
         "purged_ic": 0.021, "embargo_ic": 0.014, "honest_ic": 0.011,
         "temporal_leak": 0.037, "overlap_leak": 0.063, "total_illusion": 0.110,
         "overlap_share": 0.57, "leak_at_short": 0.004, "leak_at_long": 0.142,
         "longest_horizon": 120, "planted_ic": 1.5, "planted_recovered": 0.19,
         "planted_detected": True, "purged_data_loss": 0.18}
    h.update(over)
    return h


def test_verdict_signal_needs_a_material_illusion():
    assert st.verdict(_headline())["signal"] == "Confirmed"
    assert st.verdict(_headline(total_illusion=0.005))["signal"] == "Partial"
    assert st.verdict(_headline(total_illusion=-0.01))["signal"] == "Busted"


def test_verdict_tradability_needs_the_fix_to_work_and_not_overcorrect():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(planted_detected=False))["trad"] == "Partial"
    assert st.verdict(_headline(planted_detected=False,
                                embargo_ic=0.09))["trad"] == "Mirage"


def test_verdict_prose_separates_the_two_leaks():
    v = st.verdict(_headline())
    assert "Label overlap" in v["signal_why"]
    assert "no date out of order" in v["signal_why"]
    assert "overlapping labels" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
