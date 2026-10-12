"""The test run is offline by contract: a network guard for every suite (tests/ and studies/).

Real-data tests read caches and skip without them; nothing in the suite may download. A
loader that reached Yahoo on a cache miss used to make CI fetch SPY mid-run, and whether a
cache-gated test then ran or skipped depended on which xdist worker got there first. This
guard makes such an attempt a failure instead of a race:

- every outbound attempt is recorded -- a socket connection or DNS lookup to a non-loopback
  host, ``yfinance`` (which talks through curl_cffi, below the socket module), curl_cffi
  itself, and the engine's own download entry points in ``quantlab``;
- ``OAL_NETWORK_GUARD=block`` refuses the attempt and fails the test, or the collection of
  the module, that made it -- even when the code swallowed the error and the test went on
  to skip;
- ``OAL_NETWORK_GUARD=record`` (the default until every loader is cache-only) lets the
  attempt through and reports it as a warning; ``off`` disables the guard;
- tests marked ``@pytest.mark.network`` are exempt: they exist to touch the network.
"""

from __future__ import annotations

import os
import socket
import traceback
import warnings

import pytest

MODE = os.environ.get("OAL_NETWORK_GUARD", "record").strip().lower()
_LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


class NetworkAccessBlocked(RuntimeError):
    """Raised (in ``block`` mode) when code under test reaches for the network."""


class NetworkAccessWarning(UserWarning):
    """Emitted (in ``record`` mode) for each network attempt made under test."""


class _Guard:
    def __init__(self) -> None:
        self.where = "<collection>"
        self.exempt = False
        self.depth = 0
        self.attempts: list[tuple[str, str]] = []

    def hit(self, what: str) -> None:
        if self.exempt:
            return
        frames = [
            f"{_short(f.filename)}:{f.lineno}"
            for f in traceback.extract_stack()[:-3]
            if "site-packages" not in f.filename and not f.filename.endswith("conftest.py")
        ]
        detail = f"{what} from {' <- '.join(reversed(frames[-3:]))}"
        self.attempts.append((self.where, detail))
        if MODE == "record":
            warnings.warn(f"network attempt in {self.where}: {detail}", NetworkAccessWarning, stacklevel=3)
            return
        raise NetworkAccessBlocked(f"the test suite is offline; blocked {detail}")


GUARD = _Guard()


def _short(path: str) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:  # another drive on Windows
        return path


def _is_loopback(host: object) -> bool:
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    return host is None or str(host) in _LOOPBACK or str(host).startswith("127.")


def _wrap(original, label: str, check=None):
    def wrapper(*args, **kwargs):
        if GUARD.depth:  # inside an attempt already recorded (e.g. yfinance under _download)
            return original(*args, **kwargs)
        if check is None or check(*args, **kwargs):
            GUARD.hit(label if check is not None else f"{label}{args[:1]!r}")
        GUARD.depth += 1
        try:
            return original(*args, **kwargs)
        finally:
            GUARD.depth -= 1

    wrapper.__wrapped__ = original
    return wrapper


def _install() -> None:
    socket.getaddrinfo = _wrap(
        socket.getaddrinfo, "DNS lookup", lambda host, *a, **k: not _is_loopback(host)
    )
    socket.socket.connect = _wrap(
        socket.socket.connect,
        "socket connect",
        lambda self, address: isinstance(address, tuple) and not _is_loopback(address[0]),
    )
    socket.socket.connect_ex = _wrap(
        socket.socket.connect_ex,
        "socket connect",
        lambda self, address: isinstance(address, tuple) and not _is_loopback(address[0]),
    )
    try:  # yfinance resolves and connects inside libcurl, out of the socket module's sight
        import yfinance
        from curl_cffi import requests as curl_requests

        yfinance.download = _wrap(yfinance.download, "yfinance.download")
        curl_requests.Session.request = _wrap(curl_requests.Session.request, "curl_cffi request")
    except Exception:  # absent or reshaped: the socket layer still guards
        pass
    from quantlab import data as qdata
    from quantlab import hf_data

    qdata._download = _wrap(qdata._download, "quantlab.data._download")
    hf_data._fetch_symbol_raw = _wrap(hf_data._fetch_symbol_raw, "quantlab.hf_data._fetch_symbol_raw")


def pytest_configure(config: pytest.Config) -> None:
    if MODE in ("block", "record"):
        _install()


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):
    before = len(GUARD.attempts)
    GUARD.where = f"collection of {collector.nodeid or '<root>'}"
    outcome = yield
    GUARD.where = "<collection>"
    new = GUARD.attempts[before:]
    if new and MODE == "block":
        report = outcome.get_result()
        report.outcome = "failed"
        report.longrepr = "network access during collection (the suite is offline):\n" + "\n".join(
            d for _, d in new
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem):
    GUARD.where = item.nodeid
    GUARD.exempt = item.get_closest_marker("network") is not None
    item.stash[_START] = len(GUARD.attempts)
    yield
    GUARD.where = "<between tests>"
    GUARD.exempt = False


_START = pytest.StashKey[int]()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call):
    outcome = yield
    if MODE != "block":
        return
    start = item.stash.get(_START, len(GUARD.attempts))
    new = GUARD.attempts[start:]
    if new:
        item.stash[_START] = len(GUARD.attempts)  # report each attempt once
        report = outcome.get_result()
        report.outcome = "failed"
        report.longrepr = f"network access during {call.when} (the suite is offline):\n" + "\n".join(
            d for _, d in new
        )
