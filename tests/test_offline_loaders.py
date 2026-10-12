"""fetch=False means cache-only: no loader reaches for the network unless asked to.

These loaders used to download on a cache miss even with ``fetch=False`` (or had no way
to say no), so a test run could fetch SPY mid-run and a cache-gated test elsewhere would
run or skip depending on which xdist worker got there first. Each case calls the loader
the way an offline test does and asserts, through the conftest network guard, that
nothing was attempted: the loader either reads its cache or fails cleanly
(``FileNotFoundError``, or the empty frame its docstring promises).
"""

from __future__ import annotations

import importlib
import os
import socket

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (study dir, module, loader, kwargs). kwargs of "<tmp>" point a cache argument at an
# empty directory, so the miss path runs even where a local cache exists.
CASES = [
    # the quantlab fallback on the fetch=False branch now passes download=False
    ("311-government-shutdown", "government_shutdown.data", "load_real", {"fetch": False}),
    ("312-debt-ceiling", "debt_ceiling.data", "load_real", {"fetch": False}),
    ("315-sovereign-downgrade", "sovereign_downgrade.data", "load_real", {"fetch": False}),
    # the S&P 500 list is no longer fetched before the panel cache is checked
    ("18-dull-roar", "dull_roar.data", "fetch_panel", {"fetch": False}),
    ("24-stampede", "stampede.data", "fetch_panel", {"fetch": False}),
    ("25-clean-slate", "clean_slate.data", "fetch_panel", {"fetch": False}),
    ("26-sand-castle", "sand_castle.data", "fetch_panel", {"fetch": False}),
    ("38-chorus", "chorus.data", "fetch_panel", {"fetch": False}),
    # loaders that had no fetch switch at all
    ("97-balancing-act", "balancing_act.data", "load_real", {"fetch": False}),
    ("100-melting-ice", "melting_ice.data", "load_pair", {"fetch": False}),
    ("238-betting-against-beta", "betting_against_beta.data", "fetch_panel",
     {"fetch": False, "cache_dir": "<tmp>"}),
    ("247-bond-seasonality", "bond_seasonality.data", "load_real", {"fetch": False}),
    ("304-carbon-credits", "carbon_credits.data", "load_real",
     {"fetch": False, "cache_dir": "<tmp>"}),
    ("339-convertible-bonds", "convertible_bonds.data", "load_real", {"fetch": False}),
    # "cache-first, network on a miss": a miss without fetch=True now raises
    ("360-naaim-exposure", "naaim_exposure.data", "fetch_spy",
     {"fetch": False, "path": "<tmp>/spy_daily.csv"}),
    ("403-hammer-hanging-man", "hammer_hanging_man.data", "load_real",
     {"fetch": False, "cache_dir": "<tmp>"}),
    ("404-shooting-star", "shooting_star.data", "load_real",
     {"fetch": False, "cache_dir": "<tmp>"}),
    ("420-awesome-oscillator", "awesome_oscillator.data", "load_real",
     {"fetch": False, "cache_dir": "<tmp>"}),
    ("422-elder-ray", "elder_ray.data", "load_real", {"fetch": False, "cache_dir": "<tmp>"}),
    ("423-force-index", "force_index.data", "load_real",
     {"ticker": "SPY", "fetch": False, "cache_dir": "<tmp>"}),
    ("424-ease-of-movement", "ease_of_movement.data", "load_real",
     {"fetch": False, "cache_dir": "<tmp>"}),
    ("429-mass-index", "mass_index.data", "fetch_panel", {"fetch": False, "cache_dir": "<tmp>"}),
    ("431-schaff-trend-cycle", "schaff_trend_cycle.data", "load_real", {"fetch": False}),
    ("432-hull-moving-average", "hull_moving_average.data", "load_real",
     {"fetch": False, "cache_dir": "<tmp>"}),
    ("433-kama-adaptive", "kama_adaptive.data", "load_real",
     {"fetch": False, "cache_dir": "<tmp>"}),
    ("441-camarilla-pivots", "camarilla_pivots.data", "fetch_intraday",
     {"ticker": "SPY", "fetch": False, "cache_dir": "<tmp>"}),
    ("445-elliott-wave", "elliott_wave.data", "load_real", {"fetch": False}),
    ("446-wyckoff-method", "wyckoff_method.data", "load_real", {"fetch": False}),
    ("541-fibonacci-retracement", "fibonacci_retracement.data", "load_real", {"fetch": False}),
    ("672-mcginley-dynamic", "mcginley_dynamic.data", "load_real", {"fetch": False}),
    ("673-t3-tillson", "t3_tillson.data", "load_real", {"fetch": False}),
    ("674-vidya", "vidya.data", "load_real", {"fetch": False}),
    ("684-inverted-hammer", "inverted_hammer.data", "load_real",
     {"fetch": False, "cache_dir": "<tmp>"}),
    ("691-homing-pigeon", "homing_pigeon.data", "load_real",
     {"fetch": False, "cache_dir": "<tmp>"}),
    ("697-wolfe-waves", "wolfe_waves.data", "load_real", {"fetch": False}),
    ("704-three-drives", "three_drives.data", "load_real", {"fetch": False}),
]


def _resolve(kwargs: dict, tmp_path) -> dict:
    out = {}
    for k, v in kwargs.items():
        out[k] = v.replace("<tmp>", str(tmp_path)) if isinstance(v, str) else v
    return out


def _is_empty(result) -> bool:
    items = result if isinstance(result, tuple) else (result,)
    return all(isinstance(x, (pd.DataFrame, pd.Series)) and x.empty for x in items)


@pytest.mark.parametrize(
    "study,module,loader,kwargs", CASES, ids=[c[0].split("-")[0] for c in CASES]
)
def test_loader_with_fetch_false_never_downloads(
    study, module, loader, kwargs, tmp_path, monkeypatch, network_guard
):
    monkeypatch.syspath_prepend(os.path.join(ROOT, "studies", study))
    func = getattr(importlib.import_module(module), loader)
    with network_guard.capture() as attempts:
        try:
            result = func(**_resolve(kwargs, tmp_path))
        except FileNotFoundError:
            result = None
    assert attempts == [], f"{study}: {loader}(fetch=False) reached for the network"
    if result is not None and "<tmp>" in str(kwargs.values()):
        # a cache argument pointed at an empty dir: only an empty result is honest
        assert _is_empty(result), f"{study}: {loader} returned data from an empty cache"


# --- the engine's cache-only switches ------------------------------------------------
def test_quantlab_fetch_download_false_is_cache_only(tmp_path, monkeypatch, network_guard):
    from quantlab import data as qdata

    monkeypatch.setattr(qdata, "CACHE_DIR", tmp_path)
    with network_guard.capture() as attempts, pytest.raises(FileNotFoundError):
        qdata.fetch("SPY", mode="total_return", download=False)
    assert attempts == []


def test_hf_load_symbol_download_false_is_cache_only(tmp_path, monkeypatch, network_guard):
    from quantlab import hf_data

    monkeypatch.setattr(hf_data, "_CACHE", tmp_path)
    with network_guard.capture() as attempts, pytest.raises(FileNotFoundError):
        hf_data.load_symbol("AAPL", download=False)
    assert attempts == []


def test_universe_download_false_is_cache_only(tmp_path, monkeypatch, network_guard):
    from quantlab import universe

    monkeypatch.setattr(universe, "_CACHE", tmp_path)
    with network_guard.capture() as attempts:
        with pytest.raises(FileNotFoundError):
            universe.sp500_symbols(allow_survivorship_bias=True, download=False)
        with pytest.raises(FileNotFoundError):
            universe.download_panel(["SPY"], allow_survivorship_bias=True, download=False)
    assert attempts == []


# --- the guard itself -----------------------------------------------------------------
def test_guard_blocks_and_records_outbound_attempts(network_guard):
    """A socket connection, a DNS lookup and a Yahoo download are all caught."""
    if network_guard.mode != "block":
        pytest.skip("only meaningful when the guard blocks (OAL_NETWORK_GUARD=block)")
    import yfinance

    with network_guard.capture() as attempts:
        with pytest.raises(network_guard.blocked):
            socket.create_connection(("192.0.2.1", 80), timeout=1)  # TEST-NET-1
        with pytest.raises(network_guard.blocked):
            socket.getaddrinfo("example.com", 443)
        with pytest.raises(network_guard.blocked):
            yfinance.download("SPY", period="5d", progress=False)
    assert len(attempts) == 3
    assert any("yfinance.download" in a for a in attempts)


def test_guard_lets_loopback_through(network_guard):
    with network_guard.capture() as attempts:
        socket.getaddrinfo("localhost", 0)
        server = socket.socket()
        try:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            client = socket.create_connection(server.getsockname(), timeout=2)
            client.close()
        finally:
            server.close()
    assert attempts == []
