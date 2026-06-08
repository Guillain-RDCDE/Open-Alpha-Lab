"""Phantom-Kernel — Study 09.

Does market-making's famous "optimal spread" (Avellaneda & Stoikov, 2008) rest on an
order-arrival law that real markets actually obey? The whole closed-form optimum hangs on
one empirical assumption — that fill intensity decays *exponentially* with quote distance,

    lambda(delta) = A * exp(-k * delta),  with k constant,

so that the optimal half-spread has a clean closed form. This study mechanises the model
and stress-tests that assumption (and the payoff it promises) on a transparent, seed-fixed
order-flow simulator — a "world A" where the textbook assumptions hold (the machinery must
validate there) and a "world B" wired with the frictions the paper omits (heavy-tailed
order reach, a time-varying k, price jumps, and informed flow / adverse selection).

Modules
-------
* :mod:`phantom_kernel.sim`        — the order-flow simulator (worlds, reach laws, mid path).
* :mod:`phantom_kernel.estimator`  — fit the arrival kernel; exponential-vs-power-law GoF;
                                     the static-k spread error.
* :mod:`phantom_kernel.strategies` — the AS reservation price & spread, four quoters, and the
                                     market-making tournament.
* :mod:`phantom_kernel.experiments`— the teardown: estimator recovery, kernel falsification,
                                     and the tournament that asks whether the skew is alpha
                                     or just "don't hold inventory" beta.
"""

from __future__ import annotations

__all__ = ["sim", "estimator", "strategies", "experiments"]
