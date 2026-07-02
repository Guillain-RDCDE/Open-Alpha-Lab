"""Data layer for Study 570 (Goodwill-Impairment) — the overpaid-acquisition panel.

The claim, at full strength: a firm whose **goodwill / total assets** ratio is high has paid up
for acquisitions; the excess purchase price sits on the balance sheet as goodwill until an
impairment test forces a write-down. High-goodwill firms therefore (a) **impair** at higher rates
in subsequent years and (b) earn **lower forward returns**, with a sharp negative jolt around the
write-down announcement the market was slow to anticipate.

This study is **synthetic-only** (see ``__init__``): the point-in-time goodwill panel + tagged
impairment events the real test needs is not reachable from a no-key retail stack. So the core is
a deterministic, seeded generator whose single knob plants (or removes) the effect:

- ``synthetic_panel`` — one row per firm. A latent *overpayment* score ``o`` drives the observable
  ``goodwill_ta`` (goodwill / total assets), the **probability of a future impairment**, and,
  through ``ret_alpha`` (≤ 0 = the claim), the firm's forward return. With ``ret_alpha < 0`` the
  most-goodwill-bloated firms both impair more *and* underperform (the puzzle); ``ret_alpha = 0``
  is the null (goodwill unrelated to returns). The positive control and the null in one bottle.
  Fully offline; tests never touch the network.

There is a deliberate ``fetch_panel`` stub that raises: the real point-in-time data is out of
reach, and rather than fake a real tape we fail loudly and keep the study honestly synthetic.

No look-ahead by construction: ``goodwill_ta`` is the *year-t* balance (public at the scoring
date); ``impaired`` and ``forward_ret`` are *year t+1..t+H* outcomes. The signal is built only
from the year-t column; the outcomes are strictly forward.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

SEED = 570  # study number = base seed (house convention)


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    ret_alpha: float   # forward-return alpha per unit of (centred) overpayment (<0 = the claim)
    imp_beta: float    # log-odds of impairment per unit overpayment (>0 = high goodwill impairs)

    @property
    def has_effect(self) -> bool:
        # The claim is the *return* drag; impairment linkage alone is not the tradable puzzle.
        return self.ret_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 400,
    ret_alpha: float = -0.14,
    imp_beta: float = 2.2,
    idio_vol: float = 0.09,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible cross-section with a tunable goodwill-impairment effect.

    Each firm draws a latent **overpayment** score ``o`` (a right-skewed Beta: most firms bought
    cheap, a tail of serial over-payers). From ``o`` we derive the observables the strategy sees:

    - ``goodwill_ta``  — goodwill / total assets, monotone in ``o`` (with noise), in [0, ~0.7].
    - ``impaired``     — a 0/1 forward impairment event, Bernoulli with log-odds
      ``base + imp_beta * (o - o_bar)`` (higher overpayment → higher impairment probability).
    - ``forward_ret``  — the forward holding-period return:

          forward_ret = drift + ret_alpha * (o - o_bar) + imp_hit * impaired + idio

      where ``imp_hit = ret_alpha`` (the announcement jolt scales with the planted effect, so a
      firm that actually impairs takes an *extra* drop on top of the overpayment drag). With
      ``ret_alpha < 0`` the bloated-goodwill firms earn *less* both directly and via the write-down
      jolt. ``ret_alpha = 0`` switches **both** return channels off — a genuinely flat return-null
      — while the *impairment-event link* (high goodwill still impairs more, via ``imp_beta``)
      persists, exactly as in the real accounting story.

    Returns ``(panel, truth)``. The schema — ``goodwill_ta``, ``impaired``, ``forward_ret`` — is
    the one the strategy consumes.
    """
    rng = np.random.default_rng(seed)
    o = rng.beta(1.5, 5.0, size=n_firms)          # overpayment score, right-skewed
    o_bar = float(o.mean())
    oc = o - o_bar                                 # centred

    # Observable: goodwill / total assets, monotone in o (rank imperfect thanks to noise).
    goodwill_ta = np.clip(0.04 + 0.75 * o + 0.05 * rng.standard_normal(n_firms), 0.0, 0.80)

    # Forward impairment event (Bernoulli); base rate ~ 12% at the mean firm.
    base_logit = -2.0
    p_imp = 1.0 / (1.0 + np.exp(-(base_logit + imp_beta * oc)))
    impaired = (rng.random(n_firms) < p_imp).astype(float)

    # Forward return: drift + overpayment drag + write-down jolt (scales with the effect) + idio.
    # imp_hit is tied to ret_alpha so the null (ret_alpha=0) is a genuinely flat return-null while
    # the impairment-event link (imp_beta) still fires — the tradable RETURN puzzle is the knob.
    drift = 0.08
    imp_hit = 0.9 * ret_alpha                       # extra jolt on firms that actually impair
    idio = idio_vol * rng.standard_normal(n_firms)
    forward_ret = drift + ret_alpha * oc + imp_hit * impaired + idio

    tickers = [f"F{j:04d}" for j in range(n_firms)]
    panel = pd.DataFrame(
        {
            "goodwill_ta": goodwill_ta,
            "impaired": impaired,
            "forward_ret": forward_ret,
            "true_o": o,
        },
        index=pd.Index(tickers, name="ticker"),
    )
    return panel, WorldTruth(ret_alpha=ret_alpha, imp_beta=imp_beta)


# ---------------------------------------------------------------------------
# Real tape — deliberately unavailable (honest synthetic-only study)
# ---------------------------------------------------------------------------
def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Real point-in-time goodwill panel — **not available** to a no-key retail stack.

    The overpaid-acquisition test needs, per firm-year: point-in-time goodwill / total assets,
    tagged goodwill-impairment events, and forward returns. yfinance exposes neither a clean
    point-in-time goodwill history nor impairment tags, and there is no free survivorship-free
    source for it. Rather than fabricate a real tape, this stub returns an empty frame with
    ``fetch=False`` (the offline default) and raises on ``fetch=True`` — the data-availability
    limitation is named openly on the SIGNAL axis, and the study is capped at ``WEAK``/``NONE``.
    """
    if not fetch:
        return pd.DataFrame()
    raise NotImplementedError(
        "Study 570 is synthetic-only: a point-in-time goodwill / impairment / forward-return "
        "panel is not reachable from a no-key retail stack (yfinance exposes neither a clean "
        "point-in-time goodwill series nor tagged impairment events). See docs/results.md."
    )


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
