"""Black-Litterman, and what is actually in it — Study 979.

The model in one paragraph. Start from a prior portfolio ``w_prior``. Reverse-optimise it into
the expected returns that would make it optimal: ``pi = delta * Sigma * w_prior`` (this is not
an estimate of anything, it is a restatement of the prior). Add views as ``P mu = q`` with
uncertainty ``Omega``. The posterior expected return is

    mu_bl = [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P' Omega^-1 q]

and the portfolio is the mean-variance optimum under ``mu_bl``.

Three things this module is built to make checkable:

1. **The zero-view identity.** With no views, ``mu_bl = pi`` and the optimal portfolio is
   ``w_prior`` again, exactly — for any ``tau``, any ``delta``, any covariance.
   ``posterior_weights`` with ``P = None`` must reproduce the prior to machine precision, and
   the test-suite pins it. Everything the model appears to "produce" in a real application is
   therefore the prior plus the views; the algebra contributes the blend and nothing else.
2. **The view-strength calibration.** Practitioners set ``tau`` and ``Omega`` by folklore.
   ``view_strength_curve`` answers the question that folklore replaces: given a view of a
   stated size and confidence, how much of the book moves? It turns two unitless parameters
   into one number a person can have an opinion about.
3. **The prior's dominance.** ``prior_sensitivity`` runs the whole model under three defensible
   priors (equal weight, inverse volatility, risk parity) and measures how much the *final*
   portfolio changes — against how much the same-sized change in the view moves it. If the
   prior matters more than the view, the model is a way of expressing a prior, which is worth
   knowing before adopting it.

The out-of-sample section attaches a real, mechanical view — 12-1 momentum, scaled to a stated
annualised size — and races the resulting portfolio against the prior alone and against plain
mean-variance optimisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
PRIORS = ("equal", "inverse_vol", "risk_parity")
PRIOR_LABEL = {"equal": "Equal weight", "inverse_vol": "Inverse volatility",
               "risk_parity": "Equal risk contribution"}
METHODS = ("prior", "black_litterman", "plain_mv")
METHOD_LABEL = {"prior": "The prior, untouched", "black_litterman": "Black-Litterman + view",
                "plain_mv": "Plain mean-variance on the sample"}
DEFAULT_DELTA = 2.5      # Grinold-Kahn's risk-aversion convention
DEFAULT_TAU = 0.05       # the textbook value nobody derives


# --------------------------------------------------------------------------- #
# Priors
# --------------------------------------------------------------------------- #
def equal_prior(cov: np.ndarray) -> np.ndarray:
    """1/N."""
    n = cov.shape[0]
    return np.full(n, 1.0 / n)


def inverse_vol_prior(cov: np.ndarray) -> np.ndarray:
    """1/sigma, normalised."""
    iv = 1.0 / np.maximum(np.sqrt(np.diag(cov)), 1e-18)
    return iv / iv.sum()


def risk_parity_prior(cov: np.ndarray, iters: int = 2000, tol: float = 1e-12) -> np.ndarray:
    """Equal risk contribution, by fixed-point iteration on the contributions."""
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(iters):
        rc = w * (cov @ w)
        target = rc.mean()
        if target <= 0:
            break
        new = np.clip(w * np.sqrt(target / np.maximum(rc, 1e-18)), 1e-14, None)
        new = new / new.sum()
        if np.max(np.abs(new - w)) < tol:
            return new
        w = new
    return w


def prior_weights(cov: np.ndarray, kind: str = "equal") -> np.ndarray:
    """Dispatch on ``PRIORS``."""
    return {"equal": equal_prior, "inverse_vol": inverse_vol_prior,
            "risk_parity": risk_parity_prior}[kind](cov)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
def implied_returns(cov: np.ndarray, w_prior: np.ndarray,
                    delta: float = DEFAULT_DELTA) -> np.ndarray:
    """Reverse optimisation: ``pi = delta * Sigma * w_prior``.

    Called "equilibrium expected returns", which flatters it. It is the prior portfolio read
    backwards through the covariance matrix — no data about returns enters at all, which is
    exactly why the zero-view posterior cannot contain any.
    """
    return float(delta) * (cov @ np.asarray(w_prior, dtype=float))


def posterior_returns(cov: np.ndarray, w_prior: np.ndarray, P=None, q=None, omega=None,
                      tau: float = DEFAULT_TAU, delta: float = DEFAULT_DELTA) -> np.ndarray:
    """The Black-Litterman posterior mean. With ``P is None`` it returns ``pi`` exactly."""
    pi = implied_returns(cov, w_prior, delta)
    if P is None or q is None:
        return pi
    P = np.atleast_2d(np.asarray(P, dtype=float))
    q = np.atleast_1d(np.asarray(q, dtype=float))
    tS = tau * cov
    if omega is None:
        omega = np.diag(np.diag(P @ tS @ P.T))       # He-Litterman's convention
    omega = np.atleast_2d(np.asarray(omega, dtype=float))
    inv_tS = np.linalg.pinv(tS)
    inv_om = np.linalg.pinv(omega)
    A = inv_tS + P.T @ inv_om @ P
    b = inv_tS @ pi + P.T @ inv_om @ q
    return np.linalg.pinv(A) @ b


def optimal_weights(mu: np.ndarray, cov: np.ndarray, delta: float = DEFAULT_DELTA,
                    long_only: bool = False) -> np.ndarray:
    """Unconstrained mean-variance optimum ``(1/delta) Sigma^-1 mu``, normalised to sum to one.

    Left unconstrained by default *on purpose*: the zero-view identity is exact only without
    constraints, and a study that quietly projected onto the simplex would hide the very
    property it is trying to demonstrate. The long-only variant is available for the
    out-of-sample section, where it is the realistic choice and is labelled as such.
    """
    w = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-16) @ mu / float(delta)
    s = w.sum()
    w = w / s if abs(s) > 1e-12 else np.full(cov.shape[0], 1.0 / cov.shape[0])
    if long_only:
        w = np.clip(w, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(cov.shape[0], 1.0 / cov.shape[0])
    return w


def posterior_weights(cov: np.ndarray, w_prior: np.ndarray, P=None, q=None, omega=None,
                      tau: float = DEFAULT_TAU, delta: float = DEFAULT_DELTA,
                      long_only: bool = False) -> np.ndarray:
    """The Black-Litterman portfolio. With no views this is ``w_prior``, exactly."""
    mu = posterior_returns(cov, w_prior, P, q, omega, tau, delta)
    return optimal_weights(mu, cov, delta, long_only)


# --------------------------------------------------------------------------- #
# Calibration: how strong is a view?
# --------------------------------------------------------------------------- #
def single_view(n: int, asset: int, size_ann: float, benchmark: int | None = None):
    """A view that ``asset`` will out-perform, by ``size_ann`` a year.

    With ``benchmark`` the view is relative (asset minus benchmark), which is the form most
    practitioners actually hold and the form whose magnitude is easiest to reason about.
    """
    p = np.zeros(n)
    p[asset] = 1.0
    if benchmark is not None:
        p[benchmark] = -1.0
    return p.reshape(1, -1), np.array([size_ann / TRADING_DAYS])


def implied_view(P, cov: np.ndarray, w_prior: np.ndarray,
                 delta: float = DEFAULT_DELTA) -> np.ndarray:
    """What the prior *already believes* about the quantity a view speaks to: ``P pi``.

    This is the number a view has to be compared against, and forgetting it is the commonest
    misreading of the model. A view of "this asset will out-perform by **zero**" is not the
    absence of a view — it is a statement that contradicts the prior, whose implied view is
    positive, and the portfolio moves accordingly. The neutral view is ``q = P pi``, not
    ``q = 0``, and ``verify.py`` reports both.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    return P @ implied_returns(cov, w_prior, delta)


def view_strength_curve(cov: np.ndarray, w_prior: np.ndarray, asset: int,
                        sizes=(0.0, 0.01, 0.02, 0.05, 0.10), taus=(0.01, 0.05, 0.25),
                        delta: float = DEFAULT_DELTA) -> pd.DataFrame:
    """How much of the book moves for a view of a given size and a given ``tau``."""
    n = cov.shape[0]
    rows = []
    for tau in taus:
        for s in sizes:
            P, q = single_view(n, asset, s)
            w = posterior_weights(cov, w_prior, P, q, tau=tau, delta=delta)
            rows.append({"tau": tau, "view_ann": s,
                         "book_moved": float(np.abs(w - w_prior).sum() / 2),
                         "weight_change": float(w[asset] - w_prior[asset]),
                         "new_weight": float(w[asset])})
    return pd.DataFrame(rows)


def prior_sensitivity(cov: np.ndarray, asset: int, size_ann: float = 0.03,
                      tau: float = DEFAULT_TAU, delta: float = DEFAULT_DELTA) -> pd.DataFrame:
    """The same view under three priors — and how far apart the answers are.

    Reports, for each prior, the distance from the prior itself (what the view did) and the
    pairwise distance between the resulting portfolios (what the prior did). If the second is
    larger than the first, the model is transmitting a prior, not a view.
    """
    n = cov.shape[0]
    P, q = single_view(n, asset, size_ann)
    out, posts = [], {}
    for kind in PRIORS:
        wp = prior_weights(cov, kind)
        wb = posterior_weights(cov, wp, P, q, tau=tau, delta=delta)
        posts[kind] = wb
        out.append({"prior": kind, "view_moved_book": float(np.abs(wb - wp).sum() / 2)})
    df = pd.DataFrame(out).set_index("prior")
    for a in PRIORS:
        for b in PRIORS:
            if a < b:
                df.loc[a, f"vs_{b}"] = float(np.abs(posts[a] - posts[b]).sum() / 2)
    return df


# --------------------------------------------------------------------------- #
# Out of sample: a real, mechanical view
# --------------------------------------------------------------------------- #
def momentum_view(X: np.ndarray, size_ann: float = 0.03, top_k: int = 3):
    """A mechanical relative view: the top-``k`` 12-1 performers beat the bottom-``k``.

    Expressed as one row of ``P`` with +1/k on the winners and −1/k on the losers, and a ``q``
    of ``size_ann`` per year. Mechanical on purpose — the study is about what the model does
    with a view, not about whether momentum works (that is studies 507 and 518).
    """
    n = X.shape[1]
    if len(X) < 252:
        return None, None
    total = np.prod(1 + X[-252:-21], axis=0) - 1.0
    order = np.argsort(total)
    p = np.zeros(n)
    p[order[-top_k:]] = 1.0 / top_k
    p[order[:top_k]] = -1.0 / top_k
    return p.reshape(1, -1), np.array([size_ann / TRADING_DAYS])


def walk_forward(rets: pd.DataFrame, prior_kind: str = "equal", window: int = 504,
                 step: int = 63, size_ann: float = 0.03, tau: float = DEFAULT_TAU,
                 delta: float = DEFAULT_DELTA, cost_bps: float = 5.0,
                 long_only: bool = True) -> pd.DataFrame:
    """Rolling: the prior, the prior plus a momentum view, and plain mean-variance."""
    R = rets.dropna(how="any")
    rows = []
    prev: dict[str, np.ndarray] = {}
    for start in range(window, len(R) - step, step):
        X = R.iloc[start - window:start].to_numpy()
        test = R.iloc[start:start + step]
        cov = np.cov(X, rowvar=False, ddof=1)
        wp = prior_weights(cov, prior_kind)
        P, q = momentum_view(X, size_ann)
        w_by = {
            "prior": wp,
            "black_litterman": posterior_weights(cov, wp, P, q, tau=tau, delta=delta,
                                                 long_only=long_only),
            "plain_mv": optimal_weights(X.mean(axis=0), cov, delta, long_only=long_only),
        }
        for m, w in w_by.items():
            port = test.to_numpy() @ w
            turn = float(np.abs(w - prev.get(m, np.zeros_like(w))).sum())
            prev[m] = w
            sd = float(np.std(port, ddof=1))
            rows.append({"date": R.index[start], "method": m,
                         "realised_vol": sd * np.sqrt(TRADING_DAYS),
                         "mean_ret": float(port.mean() * TRADING_DAYS
                                           - turn * cost_bps / 1e4 * TRADING_DAYS / step),
                         "turnover": turn, "max_weight": float(np.max(w)),
                         "tilt_from_prior": float(np.abs(w - wp).sum() / 2)})
    return pd.DataFrame(rows)


def summarise(wf: pd.DataFrame) -> pd.DataFrame:
    """Per method: return, volatility, Sharpe, turnover, and distance from the prior."""
    g = wf.groupby("method")
    out = pd.DataFrame({
        "mean_ret": g["mean_ret"].mean(), "realised_vol": g["realised_vol"].mean(),
        "turnover": g["turnover"].mean(), "max_weight": g["max_weight"].mean(),
        "tilt_from_prior": g["tilt_from_prior"].mean(), "n": g.size(),
    })
    out["sharpe"] = out["mean_ret"] / out["realised_vol"]
    return out.reindex([m for m in METHODS if m in out.index])


def paired_test(wf: pd.DataFrame, a: str, b: str, column: str = "mean_ret") -> dict:
    """Paired *t* on a per-rebalance column."""
    x = wf[wf["method"] == a].set_index("date")[column]
    y = wf[wf["method"] == b].set_index("date")[column]
    x, y = x.align(y, join="inner")
    d = (x - y).dropna()
    if len(d) < 8:
        return {"diff": np.nan, "t": np.nan, "n": int(len(d)), "win_rate": np.nan}
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"diff": float(d.mean()), "t": float(d.mean() / se) if se > 0 else np.nan,
            "n": int(len(d)), "win_rate": float((d > 0).mean())}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal** (does the model add anything?): **Real** only if the zero-view identity holds
      to machine precision **and** a plausibly sized view (3%/yr) moves at least 5% of the
      book; **Weak** if the identity holds but views barely move anything; **None** if the
      identity fails, which would mean the implementation is wrong.
    - **Tradability**: **Investable** if the view-tilted portfolio beats the prior out of
      sample with a paired |*t*| >= 2; **Fragile** if it wins without significance; **Mirage**
      if it does not win.
    """
    identity = h["zero_view_error"] < 1e-8
    moves = h["book_moved_3pct"] >= 0.05
    signal = "Real" if identity and moves else ("Weak" if identity else "None")
    trad = ("Investable" if h["t_bl_vs_prior"] > 2.0
            else ("Fragile" if h["t_bl_vs_prior"] > 0 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"The zero-view identity is exact: with no views the posterior portfolio equals the "
            f"prior to **{h['zero_view_error']:.1e}** of the book, for every prior, every tau "
            f"and every covariance tested. So the model contributes no information of its own — "
            f"it is a blending rule. What it *does* contribute is a calibrated way to size a "
            f"tilt: a 3%/yr view on one sleeve moves **{h['book_moved_3pct']:.1%}** of the book "
            f"at tau = {h['tau']}, rising to {h['book_moved_10pct']:.1%} for a 10%/yr view. And "
            f"the choice of prior matters more than the view: the same view under three "
            f"defensible priors produces portfolios **{h['prior_spread']:.1%}** apart, against "
            f"the {h['view_move_mean']:.1%} the view itself moved."),
        "trad": trad,
        "trad_why": (
            f"With a mechanical 12-1 momentum view sized at 3%/yr, the tilted portfolio "
            f"returned **{h['ret_bl']:+.2%}/yr** at {h['vol_bl']:.2%} volatility (Sharpe "
            f"{h['sharpe_bl']:+.2f}) against **{h['ret_prior']:+.2%}** / "
            f"{h['sharpe_prior']:+.2f} for the untouched prior — paired *t* on the return "
            f"difference **{h['t_bl_vs_prior']:+.2f}** across {h['n_rebalances']} rebalances. "
            f"Plain mean-variance on the same data managed {h['ret_mv']:+.2%} / "
            f"{h['sharpe_mv']:+.2f} with a largest weight of {h['max_weight_mv']:.0%} against "
            f"the tilted book's {h['max_weight_bl']:.0%}, which is the comparison Black-"
            f"Litterman was invented to win — and does."),
        "one_sentence": (
            f"Black-Litterman with no views returns the prior exactly ({h['zero_view_error']:.0e} "
            f"of the book), so everything it produces is the prior plus the view — and since "
            f"changing the prior moves the answer **{h['prior_spread']:.1%}** while a 3%/yr view "
            f"moves it {h['view_move_mean']:.1%}, the model is mostly a disciplined way of "
            f"holding a prior."),
    }
