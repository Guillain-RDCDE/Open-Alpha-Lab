"""Fully offline, deterministic tests for Study 721 — Most-Admired.

No network: everything runs on the fixed-seed synthetic control and pure-function checks.
Guards the two claims the study leans on — (1) the HAC inference is unbiased (no false
positive at edge=0, a clear hit at a large planted edge), and (2) the Newey-West mean-t
reduces to the plain iid t when there is zero autocorrelation.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from most_admired import data, strategy as st


def test_newey_west_matches_iid_t_at_zero_lags():
    """With lags=0 the HAC mean-t is exactly the ordinary sqrt(n)*mean/std (ddof=0)."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.01, 0.05, 240)
    nw = st.newey_west_t(x, lags=0)
    iid = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert abs(nw["t"] - iid) < 1e-9


def test_synthetic_control_no_false_positive_at_zero_edge():
    """edge=0 must NOT manufacture a significant admiration premium."""
    syn = data.synthetic_admired(n_names=15, edge_ann=0.0, seed=721)
    s = st.summarize(syn, lagged=False)
    assert abs(s["hac_t"]) < 2.0
    assert abs(s["alpha_t"]) < 2.0


def test_synthetic_control_detects_large_planted_edge():
    """A large planted premium must light up on both the excess-t and the alpha-t."""
    syn = data.synthetic_admired(n_names=15, edge_ann=0.06, seed=721)
    s = st.summarize(syn, lagged=False)
    assert s["hac_t"] > 4.0
    assert s["alpha_t"] > 4.0
    # planted 6%/yr should be recovered to within a couple pp on the alpha
    assert 0.04 < s["alpha_ann"] < 0.08


def test_market_model_beta_near_one_for_market_plus_noise():
    """The synthetic book is market + idiosyncratic noise, so beta ~ 1 and R^2 is positive."""
    syn = data.synthetic_admired(n_names=15, edge_ann=0.0, seed=721)
    book = st.admired_book(syn["prices"], syn["admired"], lagged=False)
    mm = st.market_model_alpha(book, syn["prices"])
    assert 0.85 < mm["beta"] < 1.15
    assert mm["r2"] > 0.3


def test_lagged_book_excludes_names_before_entry():
    """A name must contribute nothing before its publication-lagged entry date."""
    syn = data.synthetic_admired(n_names=5, edge_ann=0.0, seed=1)
    # force one name to enter only in the second half
    adm = list(syn["admired"])
    entry = {t: syn["prices"].index[0] for t, *_ in adm}
    late = adm[0][0]
    entry[late] = syn["prices"].index[len(syn["prices"]) // 2]
    book_all = st.admired_book(syn["prices"], adm, entry={k: syn["prices"].index[0] for k in entry},
                               lagged=True)
    book_late = st.admired_book(syn["prices"], adm, entry=entry, lagged=True)
    # the two books must differ in the first half (the late name is absent there)
    first_half = slice(1, len(syn["prices"]) // 2)
    assert not np.allclose(book_all.iloc[first_half].to_numpy(),
                           book_late.iloc[first_half].to_numpy(), equal_nan=True)
