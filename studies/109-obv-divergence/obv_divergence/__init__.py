"""Study 109 — OBV-Divergence: does volume precede price?

Granville (1963) coined the rule: "volume precedes price." We test two
versions — (a) the OBV trend/cross as a directional signal and (b) OBV-vs-price
divergence as a reversal predictor — on SPY and a basket of liquid names (daily
bars), versus an unconditional baseline and a random-entry control.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
