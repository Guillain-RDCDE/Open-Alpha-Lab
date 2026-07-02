"""Study 588 — LLM-Headline-Sentiment: an LLM reads the day's headlines and scores the mood.

Method demo + overfitting teardown. Synthetic-only (no free tape of dated LLM headline
scores exists), so the study is capped at WEAK on the SIGNAL axis by construction — see
``data.py`` and ``strategy.py``.
"""
