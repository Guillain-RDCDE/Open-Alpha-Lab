"""Real tapes that ship inside Python packages — frozen, fingerprinted, offline.

Most studies on this desk read Yahoo! Finance through a git-ignored parquet cache. That
works on a workstation and fails in any sandbox that cannot reach Yahoo, and a cache
refreshed months later is a different tape. A handful of well-known packages carry real
market and macro series *inside the wheel*, frozen at the package's release:

``arch`` (already a desk dependency, ``[studies]`` extra)
    ``sp500`` / ``nasdaq``   daily OHLC, adjusted close and volume, 1999-01-04 → 2018-12-31
    ``vix``                  daily VIX close, 2014-01-03 → 2019-01-03
    ``wti``                  daily WTI spot (FRED ``DCOILWTICO``), 1986-01-02 → 2019-01-03
    ``crude``                monthly Brent and WTI spot, 1987-05 → 2020-01
    ``default``              monthly Moody's AAA / BAA yields, 1919-01 → 2018-12
    ``frenchdata``           monthly Fama-French Mkt-RF / SMB / HML / RF (percent), 1926-07 → 2018-11
    ``core_cpi``             monthly US core CPI (``CPILFESL``), 1957-01 → 2018-11

``statsmodels`` (already a desk dependency; files pinned the same way)
    ``macrodata``            quarterly US macro (real GDP, CPI, T-bill, unemployment…), 1959Q1 → 2009Q3
    ``interest_inflation``   quarterly German ``Dp`` (quarterly change in the log GDP deflator)
                             and ``R`` (nominal long-term rate, annual decimal), 1972Q2 → 1998Q4

``skfolio`` 1.8.1 (NOT a dependency — the pinned wheel is fetched once and cached)
    ``sp500_index``          daily S&P 500 price index, 1990-01-02 → 2022-12-28
    ``sp500_dataset``        daily adjusted closes of 20 large US stocks, same window
    ``factors_dataset``      daily MTUM / QUAL / SIZE / USMV / VLUE ETF closes, 2014 → 2022

Every file is pinned by SHA-256. A loader refuses a file whose bytes differ from the pin,
so a package upgrade that silently swaps a tape cannot move a published number: it fails
loudly instead. ``fetch_skfolio`` is the only function here that touches the network, and
it downloads one wheel from PyPI by exact URL and verifies its hash before reading it.

Two honesty notes every study using this module must carry forward:

* ``sp500`` / ``nasdaq`` / ``sp500_index`` are **price indices** — dividends are not in
  them. ``frenchdata`` ``Mkt-RF + RF`` is a **total return**. Label which one a number uses.
* ``sp500_dataset`` is 20 companies that were large and alive in 2022 — a **survivor**
  sample by construction. Usable for correlation and risk measurement; never for a claim
  about average returns.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "studies", "_cache", "bundled"))

# SHA-256 of each gzip file as shipped. arch has carried these bytes unchanged for years;
# the pin turns "unchanged" from an assumption into a check.
ARCH_FILES = {
    "sp500": ("sp500", "sp500.csv.gz",
              "1e028cbb9c400cc018c816ccc439b33c919387e726c3ed5ca2c05c82746059de"),
    "nasdaq": ("nasdaq", "nasdaq.csv.gz",
               "b2009535343ebb14e0aa11ed086788d2997a416ffe4f9450eed9a561ae30ac59"),
    "vix": ("vix", "vix.csv.gz",
            "410bd07360dc798f7206d681b8347c6150e7fbd8b05fb1ae29bbe2c9985d66ef"),
    "wti": ("wti", "wti.csv.gz",
            "b83498894138a65cdc14a67fc1716aa90b3cfda274f976bd3692cc3f1500084d"),
    "crude": ("crude", "crude.csv.gz",
              "4190b34f6130069a5b91ec03e8ea1a4dfffba8d379033f9cf3934fa695da7ef7"),
    "default": ("default", "default.csv.gz",
                "cc5453068cedc7ef4836956358dce1d20eaf1e639b69c3107a7531a8e6bb982d"),
    "frenchdata": ("frenchdata", "frenchdata.csv.gz",
                   "f23436727b01d879e1372e3ee277514b0962a23d2b158f1aee6a402a90994f52"),
    "core_cpi": ("core_cpi", "core-cpi.csv.gz",
                 "81fc912d5967ad23f4b3722a10f76867034405354b8f03f3951ebce31e5bd412"),
}

SKFOLIO_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/9c/75/"
    "eac312c38935bba39f4aaedd0d557f9802b8b8611223b86646ca913979d8/"
    "skfolio-1.8.1-py3-none-any.whl")
SKFOLIO_WHEEL_SHA256 = "5ad89bcd52513b1e400ed1fe840db99d0a86af0ca9842c0e790e3618fe234852"
SKFOLIO_FILES = {
    "sp500_index": ("sp500_index.csv.gz",
                    "894c34431c284f86aa64a467e176e2a3c9b697c857eb9ab35006c2dd55d3fdf2"),
    "sp500_dataset": ("sp500_dataset.csv.gz",
                      "ee21cac28befb1d0a739a9ceb22184f995394726aa0cfde9a21941d1ac04ac0d"),
    "factors_dataset": ("factors_dataset.csv.gz",
                        "6eb089953db90c9a38d3283582751701a90db6f59965847fd64b0578de497161"),
}


class TapeMismatch(RuntimeError):
    """A bundled file's bytes differ from the pinned SHA-256."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_pinned(raw: bytes, expected: str, label: str) -> pd.DataFrame:
    got = _sha256(raw)
    if got != expected:
        raise TapeMismatch(
            f"{label}: SHA-256 {got[:12]} does not match the pinned {expected[:12]}. "
            f"The package shipped a different tape; published numbers would move.")
    return pd.read_csv(io.BytesIO(gzip.decompress(raw)))


def fingerprint(obj) -> str:
    """Short content fingerprint of a frame or series (12 hex chars), NaN-stable."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    arr = np.nan_to_num(np.ascontiguousarray(obj.to_numpy(dtype=float)), nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# arch
# --------------------------------------------------------------------------- #
def arch_path(name: str) -> str:
    """Absolute path of one arch data file inside the installed package."""
    import arch  # lazy: arch is in the [studies] extra
    sub, fname, _ = ARCH_FILES[name]
    return os.path.join(os.path.dirname(arch.__file__), "data", sub, fname)


def have_arch(name: str = "sp500") -> bool:
    try:
        return os.path.exists(arch_path(name))
    except ImportError:
        return False


def _arch_raw(name: str) -> pd.DataFrame:
    _, _, sha = ARCH_FILES[name]
    with open(arch_path(name), "rb") as fh:
        return _read_pinned(fh.read(), sha, f"arch/{name}")


def load_arch(name: str) -> pd.DataFrame:
    """One arch dataset as a clean, date-indexed frame (see the module docstring).

    Daily sets are indexed by trading date; monthly sets by month-end. ``frenchdata``
    stays in **percent**, exactly as Ken French publishes it. Rows with no data (FRED
    writes a missing WTI print as ``.``) are dropped, never filled.
    """
    if name not in ARCH_FILES:
        raise KeyError(f"unknown arch dataset {name!r}; one of {sorted(ARCH_FILES)}")
    df = _arch_raw(name)
    if name == "frenchdata":
        idx = pd.to_datetime(df["Date"].astype(int).astype(str), format="%Y%m")
        df = df.drop(columns="Date")
        df.index = idx + pd.offsets.MonthEnd(0)
    elif name in ("default", "core_cpi", "crude"):
        idx = pd.to_datetime(df["Date"])
        df = df.drop(columns="Date")
        df.index = idx + pd.offsets.MonthEnd(0)
    else:
        df.index = pd.to_datetime(df["Date"], format="%m/%d/%Y") if "/" in str(
            df["Date"].iloc[0]) else pd.to_datetime(df["Date"])
        df = df.drop(columns="Date")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if name == "wti":
        df = df.rename(columns={"DCOILWTICO": "wti"}).dropna()
    df.index.name = "date"
    return df.sort_index()


# --------------------------------------------------------------------------- #
# statsmodels
# --------------------------------------------------------------------------- #
STATSMODELS_FILES = {
    "macrodata": ("macrodata", "macrodata.csv",
                  "d93c0d3a7a77ef83c3af14e46032bb1d02ae3a512b22ab94159a8ca226fcf708"),
    "interest_inflation": ("interest_inflation", "E6.csv",
                           "458edec09a9f6b91fcc96a1b31f0b889f5f152089c1393d146b82e0ec0b41f3f"),
}


def _check_statsmodels_pin(name: str) -> None:
    """Refuse a statsmodels tape whose bytes differ from the pin (same rule as arch)."""
    import statsmodels.datasets as smd
    sub, fname, sha = STATSMODELS_FILES[name]
    path = os.path.join(os.path.dirname(smd.__file__), sub, fname)
    with open(path, "rb") as fh:
        got = _sha256(fh.read())
    if got != sha:
        raise TapeMismatch(
            f"statsmodels/{name}: SHA-256 {got[:12]} does not match the pinned {sha[:12]}.")


def load_macrodata() -> pd.DataFrame:
    """US quarterly macro 1959Q1-2009Q3, indexed by quarter-end date.

    These are **revised** (final-vintage) figures, not what was known at the time — a
    study that times anything on them must say so, and lag every release.
    """
    from statsmodels.datasets import macrodata
    _check_statsmodels_pin("macrodata")
    df = macrodata.load_pandas().data.copy()
    q = pd.PeriodIndex.from_fields(year=df["year"].astype(int),
                                   quarter=df["quarter"].astype(int), freq="Q")
    df.index = q.to_timestamp(how="end").normalize()
    df.index.name = "date"
    return df.drop(columns=["year", "quarter"])


def load_interest_inflation() -> pd.DataFrame:
    """German quarterly ``Dp`` (quarterly change in the log GDP deflator — multiply by 4
    for an annual rate) and ``R`` (nominal long-term rate, annual decimal), 1972Q2-1998Q4."""
    from statsmodels.datasets import interest_inflation
    _check_statsmodels_pin("interest_inflation")
    df = interest_inflation.load_pandas().data.copy()
    q = pd.PeriodIndex.from_fields(year=df["year"].astype(int),
                                   quarter=df["quarter"].astype(int), freq="Q")
    df.index = q.to_timestamp(how="end").normalize()
    df.index.name = "date"
    return df.drop(columns=["year", "quarter"])


# --------------------------------------------------------------------------- #
# skfolio (fetched once, cached, verified)
# --------------------------------------------------------------------------- #
def _skfolio_cache(name: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, SKFOLIO_FILES[name][0])


def have_skfolio(cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every skfolio tape is already in the cache (offline-testable)."""
    return all(os.path.exists(_skfolio_cache(n, cache_dir)) for n in SKFOLIO_FILES)


def fetch_skfolio(cache_dir: str = DEFAULT_CACHE, url: str = SKFOLIO_WHEEL_URL) -> list[str]:
    """Download the pinned skfolio wheel, verify it, cache its three data files.

    The only network call in this module. The wheel is checked against its PyPI SHA-256
    and each extracted file against its own pin before anything is written.
    """
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (pinned https URL)
        blob = resp.read()
    if _sha256(blob) != SKFOLIO_WHEEL_SHA256:
        raise TapeMismatch("skfolio wheel SHA-256 does not match the PyPI pin")
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name, (fname, sha) in SKFOLIO_FILES.items():
            raw = zf.read(f"skfolio/datasets/data/{fname}")
            if _sha256(raw) != sha:
                raise TapeMismatch(f"skfolio/{name}: SHA-256 mismatch inside the wheel")
            path = _skfolio_cache(name, cache_dir)
            with open(path, "wb") as fh:
                fh.write(raw)
            out.append(path)
    return out


def load_skfolio(name: str, cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """One skfolio tape from the cache, date-indexed. Raises if absent (never fetches)."""
    if name not in SKFOLIO_FILES:
        raise KeyError(f"unknown skfolio dataset {name!r}; one of {sorted(SKFOLIO_FILES)}")
    path = _skfolio_cache(name, cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached skfolio tape at {path}. Run quantlab.bundled.fetch_skfolio() once.")
    with open(path, "rb") as fh:
        df = _read_pinned(fh.read(), SKFOLIO_FILES[name][1], f"skfolio/{name}")
    df.index = pd.to_datetime(df.pop("Date"))
    df.index.name = "date"
    return df.sort_index().apply(pd.to_numeric, errors="coerce")


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def ff_monthly_total_return() -> pd.DataFrame:
    """Fama-French monthly as **decimal** returns: ``mkt`` (total return), ``rf``,
    ``mkt_rf``, ``smb``, ``hml``. 1926-07 → 2018-11."""
    ff = load_arch("frenchdata") / 100.0
    out = pd.DataFrame({"mkt_rf": ff["Mkt-RF"], "smb": ff["SMB"], "hml": ff["HML"],
                        "rf": ff["RF"]})
    out["mkt"] = out["mkt_rf"] + out["rf"]
    return out


def describe() -> pd.DataFrame:
    """One row per locally available tape: rows, first and last date, fingerprint."""
    rows = []
    for name in ARCH_FILES:
        if have_arch(name):
            d = load_arch(name)
            rows.append({"source": "arch", "tape": name, "rows": len(d),
                         "first": d.index[0].date(), "last": d.index[-1].date(),
                         "fingerprint": fingerprint(d)})
    for name, loader in (("macrodata", load_macrodata),
                         ("interest_inflation", load_interest_inflation)):
        try:
            d = loader()
        except ImportError:
            continue
        rows.append({"source": "statsmodels", "tape": name, "rows": len(d),
                     "first": d.index[0].date(), "last": d.index[-1].date(),
                     "fingerprint": fingerprint(d)})
    if have_skfolio():
        for name in SKFOLIO_FILES:
            d = load_skfolio(name)
            rows.append({"source": "skfolio", "tape": name, "rows": len(d),
                         "first": d.index[0].date(), "last": d.index[-1].date(),
                         "fingerprint": fingerprint(d)})
    return pd.DataFrame(rows)
