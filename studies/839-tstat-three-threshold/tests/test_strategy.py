"""The engine and the study's spine: (1) the conventional t>2 bar inflates false
discoveries on a data-mined zoo (a pure-noise zoo manufactures a paper's worth of
'significant' factors); (2) the multiple-testing corrections raise the hurdle toward /
beyond ~3.0 and purge that noise (FWER/FDR controlled ~ 0 on the null); (3) the
corrections still KEEP a genuinely planted subset (the positive control), and the FDR
collapses ~8x going from t>2 to t>3. Plus the inference primitives and the publication
haircut."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tstat_threshold import data  # noqa: E402
from tstat_threshold import strategy as st  # noqa: E402


# ---- inference primitives --------------------------------------------------
def test_inference_primitives_sanity():
    rng = np.random.default_rng(0)
    x = 0.5 + rng.standard_normal(500)
    assert st.one_sample_t(x) > 5          # a real mean is detected
    assert abs(st.one_sample_t(rng.standard_normal(500))) < 3
    a = 1.0 + rng.standard_normal(400); b = rng.standard_normal(400)
    assert st.welch_t(a, b) > 5
    assert np.isfinite(st.newey_west_t(x, lags=6))
    lo, hi = st.wilson_interval(5, 100)
    assert 0.0 < lo < 0.05 < hi < 0.15


def test_factor_tstats_matches_one_sample_t(null_zoo):
    """The vectorised per-factor t equals the one-sample t computed column by column."""
    R, _, _ = null_zoo
    t = st.factor_tstats(R)
    for j in (0, 17, 512, 999):
        assert abs(t[j] - st.one_sample_t(R[:, j])) < 1e-9


# ---- the pitfall: t>2 inflates false discoveries on the null ---------------
def test_prob_exceed_matches_normal_tail():
    assert abs(st.prob_exceed(2.0) - 0.0455) < 1e-3
    assert abs(st.prob_exceed(3.0) - 0.0027) < 1e-3
    assert st.prob_exceed(2.0) / st.prob_exceed(3.0) > 15  # ~17x rarer at t>3


def test_null_zoo_manufactures_false_discoveries(null_zoo):
    """On a zoo we BUILT to be pure noise, the naive t>2 bar still 'finds' factors — a
    whole paper's worth — while t>3 lets only a handful through."""
    R, _, _ = null_zoo
    z = st.zoo_stats(R)
    assert z["n_gt2"] > 25            # dozens of false 'discoveries' from nothing
    assert z["n_gt2"] > 5 * max(1, z["n_gt3"])   # far fewer clear the t>3 bar


def test_threshold_summary_expected_false_counts(null_zoo):
    R, _, _ = null_zoo
    ts = st.threshold_summary(st.factor_tstats(R), thresholds=(2.0, 3.0), n_factors=1000)
    assert abs(ts.loc[2.0, "exp_false_frac"] - 0.0455) < 1e-3
    assert abs(ts.loc[2.0, "exp_false_count"] - 45.5) < 0.5
    assert abs(ts.loc[3.0, "exp_false_count"] - 2.7) < 0.2


# ---- the correction: the hurdle rises toward ~3.0 and purges the noise ------
def test_bonferroni_hurdle_rises_with_n():
    cuts = [st.bonferroni_t(n) for n in (1, 10, 100, 316, 1000)]
    assert all(cuts[i] < cuts[i + 1] for i in range(len(cuts) - 1))  # monotone up
    assert abs(st.bonferroni_t(316) - 3.78) < 0.03   # HLZ's zoo -> ~3.78
    assert st.bonferroni_t(316) > 3.0                # well above the naive t>2


def test_corrections_control_error_on_pure_null(null_zoo):
    """Bonferroni / Holm / BH / BHY reject ~ nothing on the pure-noise zoo — exactly the
    point: the corrected hurdle does not manufacture discoveries where there are none."""
    R, _, _ = null_zoo
    t = st.factor_tstats(R)
    assert st.holm_reject(t)["n_rej"] == 0
    assert st.benjamini_hochberg(t, dependency=False)["n_rej"] == 0
    assert st.benjamini_hochberg(t, dependency=True)["n_rej"] == 0
    mt = st.multiple_testing_table(t)
    assert mt.loc["naive t>2", "n_discoveries"] > 25    # the naive bar still 'finds' many
    assert mt.loc["Bonferroni", "n_discoveries"] == 0   # the corrected bar finds none


# ---- the positive control: FDR collapses t>2 -> t>3, and BHY keeps the real ones ----
def test_fdr_collapses_from_t2_to_t3(mixture_zoo):
    R, is_true, _ = mixture_zoo
    t = st.factor_tstats(R)
    d2 = st.detection(t, is_true, 2.0)
    d3 = st.detection(t, is_true, 3.0)
    assert d2["fdr"] > 0.35                 # ~half of t>2 'discoveries' are false
    assert d3["fdr"] < 0.15                 # t>3 collapses the false-discovery rate
    assert d2["fdr"] > 3 * d3["fdr"]        # a big collapse
    assert d3["power"] > 0.6                 # true factors are still mostly kept


def test_bhy_keeps_real_ones_low_realized_fdr(mixture_zoo):
    """BHY (the FDR control HLZ lean on) recovers a chunk of the true factors while its
    realized false-discovery rate stays low — it is not the trivial 'reject everything'
    (null) or 'reject nothing' (over-strict) degenerate."""
    R, is_true, _ = mixture_zoo
    t = st.factor_tstats(R)
    bhy = st.benjamini_hochberg(t, alpha=0.05, dependency=True)
    assert bhy["n_rej"] > 10                          # it does keep real factors
    assert bhy["t_cutoff"] > 3.0                       # implied hurdle above the t>3 bar
    det = st.detection(t, is_true, bhy["t_cutoff"] - 1e-9)
    assert det["fdr"] < 0.10                           # almost no false discoveries


def test_seed_robust_mixture_fdr_collapse():
    sm = st.seed_robust_mixture(data, n_factors=1000, n_true=50, expected_t=4.0, n_seeds=20)
    assert sm["mean_fdr_t2"] > 0.35
    assert sm["mean_fdr_t3"] < 0.12
    assert sm["fdr_collapse"] > 3.0
    assert sm["mean_bhy_fdr"] < 0.05                   # BHY realized FDR near zero
    assert sm["mean_bhy_cutoff"] > 3.0


# ---- the publication haircut -----------------------------------------------
def test_publication_haircut_kills_t2_spares_strong():
    h2 = st.publication_haircut(2.0, 316)
    h4 = st.publication_haircut(4.0, 316)
    assert not h2["survives_005"]            # a t=2 factor from a 316-test search is dead
    assert h2["haircut"] > 0.9               # ~fully haircut
    assert h4["survives_005"]                # a genuinely strong factor survives
    assert h4["haircut"] < h2["haircut"]     # the stronger the claim, the smaller the cut


def test_haircut_grows_as_reported_t_shrinks():
    hair = [st.publication_haircut(t, 316)["haircut"] for t in (4.0, 3.5, 3.0, 2.5)]
    assert all(hair[i] <= hair[i + 1] + 1e-9 for i in range(len(hair) - 1))
