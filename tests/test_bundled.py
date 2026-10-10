"""quantlab.bundled — the frozen real tapes that ship inside packages.

The arch and statsmodels tapes are always present (both are desk dependencies), so their
tests run everywhere. The skfolio tapes need one fetch and are cache-gated like every other
real-tape test on the desk.
"""

import gzip

import numpy as np
import pandas as pd
import pytest

from quantlab import bundled as b

needs_arch = pytest.mark.skipif(not b.have_arch(), reason="arch not installed")
needs_skfolio = pytest.mark.skipif(not b.have_skfolio(),
                                   reason="skfolio tapes not cached (run fetch_skfolio)")

EXPECTED = {  # name: (first date, last date, rows)
    "sp500": ("1999-01-04", "2018-12-31", 5031),
    "nasdaq": ("1999-01-04", "2018-12-31", 5031),
    "vix": ("2014-01-03", "2019-01-03", 1259),
    "wti": ("1986-01-02", "2019-01-03", 8321),
    "crude": ("1987-05-31", "2020-01-31", 393),
    "default": ("1919-01-31", "2018-12-31", 1200),
    "frenchdata": ("1926-07-31", "2018-11-30", 1109),
    "core_cpi": ("1957-01-31", "2018-11-30", 743),
}


@needs_arch
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_arch_tapes_load_pinned_and_clean(name):
    d = b.load_arch(name)
    first, last, rows = EXPECTED[name]
    assert d.index[0] == pd.Timestamp(first) and d.index[-1] == pd.Timestamp(last)
    assert len(d) == rows
    assert d.index.is_monotonic_increasing and not d.index.has_duplicates
    assert d.notna().all().all()


@needs_arch
def test_ff_total_return_is_excess_plus_rf():
    ff = b.ff_monthly_total_return()
    assert np.allclose(ff["mkt"], ff["mkt_rf"] + ff["rf"])
    assert ff["mkt"].abs().max() < 0.5          # decimal, not percent
    assert ff.loc["1929-10-31", "mkt"] < -0.15  # the crash month is where it should be


@needs_arch
def test_pin_rejects_altered_bytes():
    raw = open(b.arch_path("vix"), "rb").read()
    tampered = gzip.compress(gzip.decompress(raw).replace(b"13.76", b"13.77"))
    with pytest.raises(b.TapeMismatch):
        b._read_pinned(tampered, b.ARCH_FILES["vix"][2], "vix")


def test_unknown_names_raise():
    with pytest.raises(KeyError):
        b.load_arch("nope")
    with pytest.raises(KeyError):
        b.load_skfolio("nope")


def test_statsmodels_tapes():
    m = b.load_macrodata()
    assert m.index[0] == pd.Timestamp("1959-03-31") and len(m) == 203
    assert {"realgdp", "cpi", "tbilrate", "unemp"} <= set(m.columns)
    g = b.load_interest_inflation()
    assert g.index[0] == pd.Timestamp("1972-06-30") and len(g) == 107


def test_skfolio_absent_cache_raises(tmp_path):
    assert b.have_skfolio(cache_dir=str(tmp_path)) is False
    with pytest.raises(FileNotFoundError):
        b.load_skfolio("sp500_index", cache_dir=str(tmp_path))


def test_fingerprint_stable_and_sensitive():
    a = pd.DataFrame({"x": [1.0, 2.0, np.nan]})
    assert b.fingerprint(a) == b.fingerprint(a.copy()) and len(b.fingerprint(a)) == 12
    assert b.fingerprint(a) != b.fingerprint(a * 2)
    assert b.fingerprint(a["x"]) == b.fingerprint(a)


@needs_skfolio
def test_skfolio_tapes():
    idx = b.load_skfolio("sp500_index")
    stocks = b.load_skfolio("sp500_dataset")
    assert idx.index[0] == pd.Timestamp("1990-01-02") and len(idx) == 8313
    assert stocks.shape == (8313, 20) and stocks.index.equals(idx.index)
    assert b.load_skfolio("factors_dataset").shape[1] == 5


@pytest.mark.parametrize("name", sorted(b.STATSMODELS_FILES))
def test_statsmodels_pins_hold(name):
    b._check_statsmodels_pin(name)  # raises TapeMismatch if the shipped bytes moved
