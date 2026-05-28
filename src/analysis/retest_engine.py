"""
5-Tier Retest Engine — Phase 3B
Classifies the entry context relative to nearby structural levels.
Returns a RetestResult with a score-modifier delta and entry_type for VTM.

Priority cascade (evaluated top-to-bottom; first match wins):
  1. CLEAN      (−0.20)  — price at a defended level; textbook pullback entry
  2. BREAKOUT   (+0.10 / +0.20 / +0.40) — fresh Livermore state (age ≤ 5 bars)
  3. WICK       (  0.00) — sweep + close recovery through level (spring entry)
  4. CHASE_HARD (+1.50)  — price too extended; entry_type = REJECT
  5. CHASE_SOFT (+0.75)  — moderately extended; elevated threshold
  6. NO_LEVEL_NEARBY (+0.35 / +0.40) — fallback when no structural reference

All numeric thresholds are loaded from config/aggregator_presets.json
RETEST_ENGINE section — zero magic numbers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── entry_type string constants ────────────────────────────────────────────────
ET_MR_PULLBACK     = "MR_PULLBACK"
ET_TREND_FOLLOWING = "TREND_FOLLOWING"
ET_SPRING_ENTRY    = "SPRING_ENTRY"
ET_RANGE_BOUNDARY  = "RANGE_BOUNDARY"
ET_REJECT          = "REJECT"

# ── retest_type string constants ───────────────────────────────────────────────
RT_CLEAN           = "CLEAN"
RT_BREAKOUT        = "BREAKOUT"
RT_WICK            = "WICK"
RT_CHASE_SOFT      = "CHASE_SOFT"
RT_CHASE_HARD      = "CHASE_HARD"
RT_NO_LEVEL_NEARBY = "NO_LEVEL_NEARBY"

# Livermore states that confirm a LONG direction on 4H
_LONG_CONFIRMING_4H_STATES  = frozenset({
    "MAIN_UP", "NATURAL_RETRACEMENT", "SECONDARY_RETRACEMENT"
})
# Livermore states that confirm a SHORT direction on 4H
_SHORT_CONFIRMING_4H_STATES = frozenset({
    "MAIN_DOWN", "NATURAL_REBOUND", "SECONDARY_REBOUND"
})

# Symbols treated as BTC-class (crypto volatility profile)
_BTC_SYMBOLS = frozenset({"BTC", "BTCUSDT", "BTC/USDT", "BTCUSD"})

# Symbols with FX volatility profile
_FX_PREFIXES = ("EUR", "GBP", "AUD", "JPY", "USD", "CAD", "CHF", "NZD")


@dataclass
class RetestResult:
    """Output of RetestEngine.classify()."""
    retest_type: str            # one of the RT_* constants above
    modifier: float             # score threshold delta (negative = easier, positive = harder)
    entry_type: Optional[str]   # ET_* constant; None only on error paths
    direction: int              # +1 LONG / -1 SHORT (pass-through from caller)
    level: Optional[float]      # reference level used; None when NO_LEVEL_NEARBY


class RetestEngine:
    """
    Classifies each candidate trade entry into one of 5 tiers based on
    proximity to structural levels, Livermore state age, and sweep detection.

    Instantiate once and call classify() per candle.
    Thread-safe (no mutable state after __init__).
    """

    def __init__(self, cfg: dict) -> None:
        """
        Parameters
        ----------
        cfg : dict
            The RETEST_ENGINE sub-dict from aggregator_presets.json.
        """
        self._cfg = cfg

        # ── top-level scalar thresholds ────────────────────────────────────
        self._clean_atr_mult        = float(cfg.get("clean_proximity_atr_mult", 0.5))
        self._breakout_age_max      = int(cfg.get("breakout_age_max_bars", 5))

        # ── fixed modifiers ────────────────────────────────────────────────
        self._mod_clean             = float(cfg.get("modifier_clean",          -0.20))
        self._mod_wick              = float(cfg.get("modifier_wick",            0.00))
        self._mod_chase_soft        = float(cfg.get("modifier_chase_soft",      0.75))
        self._mod_chase_hard        = float(cfg.get("modifier_chase_hard",      1.50))
        self._mod_no_level_default  = float(cfg.get("modifier_no_level_default", 0.35))

        # ── breakout alignment modifiers ────────────────────────────────────
        self._mod_breakout_aligned_btc  = float(cfg.get("modifier_breakout_aligned_btc",  0.10))
        self._mod_breakout_aligned_fx   = float(cfg.get("modifier_breakout_aligned_fx",   0.20))
        self._mod_breakout_misaligned   = float(cfg.get("modifier_breakout_misaligned",   0.40))

        # ── per-asset override sub-dicts ────────────────────────────────────
        self._assets = cfg.get("assets", {})

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def classify(
        self,
        df: pd.DataFrame,
        state,          # CompositeState — typed loosely to avoid circular import
        symbol: str,
        direction: int, # +1 LONG / -1 SHORT
    ) -> RetestResult:
        """
        Run the priority cascade and return a RetestResult.

        Minimum requirements on `df`:
          - At least 2 rows of 1H OHLCV with an 'atr' column populated.
          - df.iloc[-1] is the last *closed* candle.

        Minimum requirements on `state`:
          - nearby_4h_level, level_defended
          - livermore_state_age_1h, livermore_state_1h, livermore_state_4h
          - livermore_anchor_* (used for BREAKOUT level lookup)
          - sweep_detected, sweep_level
        """
        if len(df) < 2:
            logger.debug("retest_engine: df too short (%d rows) — NO_LEVEL_NEARBY", len(df))
            return self._no_level(symbol, direction)

        atr = self._get_atr(df)
        if atr <= 0:
            logger.debug("retest_engine: ATR=0 — NO_LEVEL_NEARBY")
            return self._no_level(symbol, direction)

        close     = float(df["close"].iloc[-1])
        level     = getattr(state, "nearby_4h_level", None)
        asset_cfg = self._assets.get(symbol, self._assets.get("DEFAULT", {}))

        # ── 1. CLEAN ─────────────────────────────────────────────────────────
        # Price within clean_proximity_atr_mult × ATR of the nearby 4H level,
        # AND the level has been actively defended (level_defended = True).
        if level is not None:
            dist_atr = abs(close - level) / atr
            level_defended = getattr(state, "level_defended", False)
            if dist_atr <= self._clean_atr_mult and level_defended:
                logger.debug("retest_engine: CLEAN @ %.5f (dist=%.2f ATR)", level, dist_atr)
                return RetestResult(
                    retest_type=RT_CLEAN,
                    modifier=self._mod_clean,
                    entry_type=ET_RANGE_BOUNDARY,
                    direction=direction,
                    level=level,
                )

        # ── 2. BREAKOUT ───────────────────────────────────────────────────────
        # Livermore 1H state flipped within the last breakout_age_max_bars bars.
        # Price must still be within breakout_proximity_atr_mult × ATR of the
        # Livermore anchor that was just broken.
        age_1h = int(getattr(state, "livermore_state_age_1h", 999))
        if age_1h <= self._breakout_age_max:
            bo_level = self._get_breakout_level(state, direction)
            if bo_level is not None:
                dist_atr = abs(close - bo_level) / atr
                prox_mult = float(asset_cfg.get(
                    "breakout_proximity_atr_mult",
                    2.0 if self._is_btc(symbol) else 1.25,
                ))
                if dist_atr <= prox_mult:
                    mod = self._breakout_modifier(symbol, state, direction)
                    logger.debug(
                        "retest_engine: BREAKOUT @ %.5f (dist=%.2f ATR, mod=%.2f)",
                        bo_level, dist_atr, mod,
                    )
                    return RetestResult(
                        retest_type=RT_BREAKOUT,
                        modifier=mod,
                        entry_type=ET_TREND_FOLLOWING,
                        direction=direction,
                        level=bo_level,
                    )

        # ── 3. WICK ───────────────────────────────────────────────────────────
        # A liquidity sweep (sweep_detected) where price has already closed back
        # through the swept level in the trade direction.  Classic spring/upthrust.
        sweep_detected = getattr(state, "sweep_detected", False)
        if sweep_detected and self._wick_recovered(df, state, direction):
            sweep_level = getattr(state, "sweep_level", None)
            logger.debug("retest_engine: WICK @ sweep_level=%s", sweep_level)
            return RetestResult(
                retest_type=RT_WICK,
                modifier=self._mod_wick,
                entry_type=ET_SPRING_ENTRY,
                direction=direction,
                level=sweep_level,
            )

        # ── 4 & 5. CHASE (requires a nearby level to measure distance from) ──
        if level is not None:
            dist_atr = abs(close - level) / atr
            chase_hard = float(asset_cfg.get(
                "chase_hard_atr_mult",
                2.0 if self._is_btc(symbol) else 2.5,
            ))
            chase_soft = float(asset_cfg.get(
                "chase_soft_atr_mult",
                1.2 if self._is_btc(symbol) else 1.5,
            ))
            # CHASE_HARD takes precedence over CHASE_SOFT
            if dist_atr >= chase_hard:
                logger.debug(
                    "retest_engine: CHASE_HARD @ %.5f (dist=%.2f ATR)", level, dist_atr
                )
                return RetestResult(
                    retest_type=RT_CHASE_HARD,
                    modifier=self._mod_chase_hard,
                    entry_type=ET_REJECT,
                    direction=direction,
                    level=level,
                )
            if dist_atr >= chase_soft:
                logger.debug(
                    "retest_engine: CHASE_SOFT @ %.5f (dist=%.2f ATR)", level, dist_atr
                )
                return RetestResult(
                    retest_type=RT_CHASE_SOFT,
                    modifier=self._mod_chase_soft,
                    entry_type=ET_MR_PULLBACK,
                    direction=direction,
                    level=level,
                )

        # ── 6. NO_LEVEL_NEARBY (fallback) ─────────────────────────────────────
        return self._no_level(symbol, direction)

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _no_level(self, symbol: str, direction: int) -> RetestResult:
        asset_cfg = self._assets.get(symbol, self._assets.get("DEFAULT", {}))
        mod = float(asset_cfg.get("modifier_no_level", self._mod_no_level_default))
        return RetestResult(
            retest_type=RT_NO_LEVEL_NEARBY,
            modifier=mod,
            entry_type=ET_TREND_FOLLOWING,
            direction=direction,
            level=None,
        )

    def _get_atr(self, df: pd.DataFrame) -> float:
        """
        Return ATR of the last candle.
        Prefers the 'atr' column populated by DataManager; falls back to a
        14-bar simplified TR mean if that column is absent or NaN.
        """
        if "atr" in df.columns:
            v = float(df["atr"].iloc[-1])
            if not np.isnan(v) and v > 0:
                return v
        # Simplified ATR fallback
        n = min(14, len(df) - 1)
        if n >= 1:
            highs  = df["high"].iloc[-(n + 1):-1].reset_index(drop=True)
            lows   = df["low"].iloc[-(n + 1):-1].reset_index(drop=True)
            closes = df["close"].iloc[-(n + 2):-2].reset_index(drop=True)
            tr = pd.concat([
                highs - lows,
                (highs - closes).abs(),
                (lows  - closes).abs(),
            ], axis=1).max(axis=1)
            v = float(tr.mean())
            if v > 0:
                return v
        # Last-resort: single-bar range
        return max(float(df["high"].iloc[-1] - df["low"].iloc[-1]), 1e-10)

    @staticmethod
    def _is_btc(symbol: str) -> bool:
        return symbol.upper() in _BTC_SYMBOLS

    @staticmethod
    def _is_fx(symbol: str) -> bool:
        s = symbol.upper()
        return any(s.startswith(p) for p in _FX_PREFIXES)

    def _get_breakout_level(self, state, direction: int) -> Optional[float]:
        """
        Return the most relevant Livermore anchor level for a fresh-breakout entry.
        For a LONG breakout: the level that was just cleared to the upside.
        For a SHORT breakout: the level that was just broken to the downside.
        Falls back to nearby_4h_level if no anchor is available.
        """
        ls = getattr(state, "livermore_state_1h", None)
        if direction == 1:
            if ls in ("MAIN_UP", "NATURAL_RETRACEMENT"):
                anchor = getattr(state, "livermore_anchor_main_up_max", None)
                if anchor is not None:
                    return anchor
            if ls == "SECONDARY_RETRACEMENT":
                anchor = getattr(state, "livermore_anchor_natural_low", None)
                if anchor is not None:
                    return anchor
        elif direction == -1:
            if ls in ("MAIN_DOWN", "NATURAL_REBOUND"):
                anchor = getattr(state, "livermore_anchor_main_down_min", None)
                if anchor is not None:
                    return anchor
            if ls == "SECONDARY_REBOUND":
                anchor = getattr(state, "livermore_anchor_natural_high", None)
                if anchor is not None:
                    return anchor
        # Final fallback
        return getattr(state, "nearby_4h_level", None)

    def _breakout_modifier(self, symbol: str, state, direction: int) -> float:
        """
        Determine the breakout threshold modifier based on 4H alignment.
          +0.10 — BTC, 4H-aligned (smallest raise)
          +0.20 — FX / GOLD / USTEC, 4H-aligned
          +0.40 — any symbol, 4H-misaligned (largest raise: counter-trend breakout)
        """
        ls4h = getattr(state, "livermore_state_4h", None)
        aligned = (
            (direction == 1  and ls4h in _LONG_CONFIRMING_4H_STATES) or
            (direction == -1 and ls4h in _SHORT_CONFIRMING_4H_STATES)
        )
        if not aligned:
            return self._mod_breakout_misaligned
        return (
            self._mod_breakout_aligned_btc
            if self._is_btc(symbol)
            else self._mod_breakout_aligned_fx
        )

    def _wick_recovered(self, df: pd.DataFrame, state, direction: int) -> bool:
        """
        Return True if the close has recovered back through the swept level.
          LONG setup: sweep went below level, close is now *above* level.
          SHORT setup: sweep went above level, close is now *below* level.
        """
        sweep_level = getattr(state, "sweep_level", None)
        if sweep_level is None:
            return False
        close = float(df["close"].iloc[-1])
        if direction == 1:
            return close > sweep_level
        if direction == -1:
            return close < sweep_level
        return False
