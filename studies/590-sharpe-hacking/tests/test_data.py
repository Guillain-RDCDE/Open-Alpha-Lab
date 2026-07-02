"""The synthetic tape is deterministic, offline, and hits its honest-Sharpe target with zero alpha
serial correlation in the mean (so the raw series is honestly annualisable — only the transforms can
distort it)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sharpe_hacking import data  # noqa: E402


def test_series_deterministic():
    a, _ = data.synthetic_series(seed=590)
    b, _ = data.synthetic_series(seed=590)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_null_has_no_planted_alpha():
    _, truth = data.synthetic_series(alpha=0.0, seed=590)
    assert not truth.has_edge
    _, truth2 = data.synthetic_series(alpha=0.2, seed=590)
    assert truth2.has_edge


def test_null_honest_sharpe_near_base():
    """The raw null series annualises honestly to ~base_sharpe *on average* (no runaway drift).

    A single finite path's realised Sharpe is a noisy estimate (SE ~ 1/sqrt(years)), so we average
    the realised annual Sharpe over many seeds — the expectation must sit near ``base_sharpe``, not at
    an absurd 5+ (the bug this guards against was a drift 10x too large)."""
    srs = []
    vols = []
    for s in range(590, 620):
        ret, _ = data.synthetic_series(n_days=3000, alpha=0.0, base_sharpe=0.45, seed=s)
        r = ret.to_numpy()
        srs.append(r.mean() / r.std(ddof=1) * np.sqrt(252))
        vols.append(r.std(ddof=1) * np.sqrt(252))
    assert 0.25 < np.mean(srs) < 0.65          # mean realised Sharpe ~ base_sharpe (not 5+)
    assert 0.10 < np.mean(vols) < 0.30         # realistic annual vol


def test_raw_series_has_no_mean_serial_correlation():
    """Raw returns are iid-in-the-mean: lag-1 autocorrelation is tiny, so the naive Sharpe is honest
    BEFORE any transform (the transforms are the only source of inflation)."""
    ret, _ = data.synthetic_series(n_days=4000, alpha=0.0, seed=590)
    r = ret.to_numpy() - ret.to_numpy().mean()
    ac1 = np.sum(r[:-1] * r[1:]) / np.sum(r * r)
    assert abs(ac1) < 0.06


def test_higher_alpha_lifts_realized_return():
    lo, _ = data.synthetic_series(n_days=4000, alpha=0.0, seed=590)
    hi, _ = data.synthetic_series(n_days=4000, alpha=0.30, seed=590)
    assert hi.mean() > lo.mean()
