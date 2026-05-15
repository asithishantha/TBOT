"""
Shadow Trading Engine (T3.1)

Tracks every signal that was BLOCKED by any gate (gatekeeper, governor, sniper,
trap filter, AI validation, etc.) as a virtual position and records its outcome.

Purpose
-------
Every gate evaluation and ML retraining decision depends on labelled outcomes.
Currently blocked signals vanish — we have no idea if a blocked signal would
have been profitable. This module captures that missing ground truth.

Architecture
------------
Two-tier design for performance:
  Tick tier  (every ~5s): Pure price-vs-stop/target arithmetic. No TA-Lib.
                           ~0.05ms for 20 open positions.
  Candle tier (every 5min): Bar counter increment, MFE/MAE tracking.
                             Full ATR/ADX recalc only if needed.

Key fields per position
-----------------------
  strategy_source   : Which strategy (TF / MR / EMA / consensus) sourced the signal
  peak_profit_bar   : Bar number when Maximum Favourable Excursion occurred
  friction_penalty  : Asset-specific round-trip slippage
  net_pnl_pct       : Gross P&L minus friction (used for ML labels)
  regime_score      : Regime at entry (for regime-intensity analysis)
  gate_blocked_by   : Which gate killed the real signal (for gate scorecard)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Asset-specific round-trip friction penalties (slippage + commission).
# Applied before ML labelling so models learn from net, not gross, P&L.
# ─────────────────────────────────────────────────────────────────────────────
FRICTION_PENALTIES: Dict[str, float] = {
    "BTC":    0.0003,   # 0.03% round-trip
    "BTCUSDT": 0.0003,
    "GOLD":   0.0008,   # 0.08%
    "XAUUSD": 0.0008,
    "XAUUSDm": 0.0008,
    "USTEC":  0.0005,   # 0.05%
    "USTECm": 0.0005,
    "EURJPY": 0.0004,   # 0.04%
    "EURJPYm": 0.0004,
    "EURUSD": 0.0003,
    "EURUSDm": 0.0003,
    "GBPUSD": 0.0003,   # 0.03% — tight spread major pair
    "GBPUSDm": 0.0003,
    "USDJPY": 0.0002,   # 0.02% — tightest spread major
    "USDJPYm": 0.0002,
    "USOIL":  0.0006,   # 0.06% round-trip (oil has wider spreads)
    "USOILm": 0.0006,
    "GBPAUD": 0.0005,   # 0.05% round-trip
    "GBPAUDm": 0.0005,
}
_DEFAULT_FRICTION = 0.0005


@dataclass
class ShadowPosition:
    """A single virtual (shadow) trade tracking a blocked signal's outcome."""

    # Identity
    asset: str
    side: str               # "long" | "short"
    strategy_source: str    # "TF" | "MR" | "EMA" | "consensus"
    gate_blocked_by: str    # e.g. "blocked_by_governor", "no_sniper_confirmation"

    # Entry
    entry_price: float
    entry_time: datetime
    regime_score: float = 0.0
    regime_name: str = "UNKNOWN"

    # Stop & target (rough estimates — we use ATR-based defaults)
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Live tracking
    current_price: float = 0.0
    bars_open: int = 0
    peak_profit_bar: int = 0

    # Extremes
    mfe_pct: float = 0.0    # Maximum Favourable Excursion
    mae_pct: float = 0.0    # Maximum Adverse Excursion

    # Outcome
    closed: bool = False
    close_price: float = 0.0
    close_time: Optional[datetime] = None
    close_reason: str = ""

    # P&L
    gross_pnl_pct: float = 0.0
    friction_pct: float = 0.0
    net_pnl_pct: float = 0.0

    # Strategy vote snapshot at entry (for ML feature construction)
    strategy_votes: Dict = field(default_factory=dict)

    # J2.1: CompositeState snapshot at entry
    composite_state: Dict = field(default_factory=dict)

    # J2.2: VTM-lite trailing stop (standardized — same for every shadow trade)
    trailing_active: bool = False
    trailing_distance: float = 0.0        # Set at open from ATR × 1.5
    trailing_activation_pct: float = 0.0  # Set at open from ATR × 1.0 / entry
    highest_price: float = 0.0            # For longs
    lowest_price: float = 0.0             # For shorts

    # J2.3: Breakeven after TP1
    tp1_reached: bool = False
    tp1_price: float = 0.0               # First partial target: entry ± 1.5 × ATR

    def _profit_pct(self, price: float) -> float:
        """Current unrealised P&L as a fraction of entry price."""
        if self.entry_price == 0:
            return 0.0
        if self.side == "long":
            return (price - self.entry_price) / self.entry_price
        return (self.entry_price - price) / self.entry_price

    def tick_update(self, current_price: float) -> bool:
        """
        Tick-tier update. Pure arithmetic — no TA-Lib calls.
        Returns True if the position closed on this tick.
        """
        if self.closed:
            return False

        self.current_price = current_price
        pnl = self._profit_pct(current_price)

        # Track MFE
        if pnl > self.mfe_pct:
            self.mfe_pct = pnl

        # Track MAE
        if pnl < self.mae_pct:
            self.mae_pct = pnl

        # J2.3: Breakeven after TP1 — simulates VTM's partial exit + BE lock
        if not self.tp1_reached and self.tp1_price > 0:
            if (self.side == "long" and current_price >= self.tp1_price) or \
               (self.side == "short" and current_price <= self.tp1_price):
                self.tp1_reached = True
                # Move SL to breakeven (side-aware, matching T1.4 fix)
                if self.side == "long" and self.stop_loss < self.entry_price:
                    self.stop_loss = self.entry_price
                elif self.side == "short" and self.stop_loss > self.entry_price:
                    self.stop_loss = self.entry_price

        # J2.2: VTM-lite trailing stop
        # Activate trailing after 1.0× ATR favorable move
        if not self.trailing_active and self.trailing_activation_pct > 0:
            if pnl > self.trailing_activation_pct:
                self.trailing_active = True

        if self.trailing_active and self.trailing_distance > 0:
            if self.side == "long":
                self.highest_price = max(self.highest_price, current_price)
                _trail_sl = self.highest_price - self.trailing_distance
                if _trail_sl > self.stop_loss:
                    self.stop_loss = _trail_sl
            else:
                self.lowest_price = min(self.lowest_price, current_price)
                _trail_sl = self.lowest_price + self.trailing_distance
                if _trail_sl < self.stop_loss:
                    self.stop_loss = _trail_sl

        # Check stop loss hit
        if self.stop_loss > 0:
            if self.side == "long" and current_price <= self.stop_loss:
                return self._close(current_price, "stop_loss")
            elif self.side == "short" and current_price >= self.stop_loss:
                return self._close(current_price, "stop_loss")

        # Check take profit hit
        if self.take_profit > 0:
            if self.side == "long" and current_price >= self.take_profit:
                return self._close(current_price, "take_profit")
            elif self.side == "short" and current_price <= self.take_profit:
                return self._close(current_price, "take_profit")

        return False

    def candle_update(self) -> None:
        """
        Candle-tier update — called every 5 minutes.
        Increments bar counter and records peak_profit_bar.
        """
        if self.closed:
            return
        self.bars_open += 1
        if self._profit_pct(self.current_price) >= self.mfe_pct:
            self.peak_profit_bar = self.bars_open

        # Time-based exit: close after 72 wall-clock hours (3 days) if still open.
        # Wall-clock comparison is used instead of bar count because candle_update()
        # is called every 5-min bot loop — 72 bars would only be 6 hours, not 3 days.
        elapsed_hours = (datetime.now(timezone.utc) - self.entry_time).total_seconds() / 3600.0
        if elapsed_hours >= 72.0:
            self._close(self.current_price, "time_stop_72h")

    def _close(self, price: float, reason: str) -> bool:
        """Record the final outcome including friction-adjusted net P&L."""
        self.closed = True
        self.close_price = price
        self.close_time = datetime.now(timezone.utc)
        self.close_reason = reason
        self.gross_pnl_pct = self._profit_pct(price) * 100  # in percent
        self.friction_pct = FRICTION_PENALTIES.get(
            self.asset.upper(), _DEFAULT_FRICTION
        ) * 100
        self.net_pnl_pct = self.gross_pnl_pct - self.friction_pct
        logger.debug(
            f"[SHADOW] {self.asset} {self.side} closed: "
            f"reason={reason}, gross={self.gross_pnl_pct:.3f}%, "
            f"net={self.net_pnl_pct:.3f}%, bars={self.bars_open}"
        )
        return True

    def to_dict(self) -> dict:
        """Serialise to a flat dict suitable for DataFrame construction."""
        return {
            "asset":            self.asset,
            "side":             self.side,
            "strategy_source":  self.strategy_source,
            "gate_blocked_by":  self.gate_blocked_by,
            "entry_price":      self.entry_price,
            "stop_loss":        self.stop_loss,
            "take_profit":      self.take_profit,
            "entry_time":       self.entry_time.isoformat() if self.entry_time else None,
            "close_price":      self.close_price,
            "close_time":       self.close_time.isoformat() if self.close_time else None,
            "close_reason":     self.close_reason,
            "regime_score":     self.regime_score,
            "regime_name":      self.regime_name,
            "bars_open":        self.bars_open,
            "peak_profit_bar":  self.peak_profit_bar,
            "mfe_pct":          round(self.mfe_pct * 100, 4),
            "mae_pct":          round(self.mae_pct * 100, 4),
            "gross_pnl_pct":    round(self.gross_pnl_pct, 4),
            "friction_pct":     round(self.friction_pct, 4),
            "net_pnl_pct":      round(self.net_pnl_pct, 4),
            "strategy_votes":   self.strategy_votes,
            "composite_state":  self.composite_state,
        }


class ShadowTradingEngine:
    """
    Manages all open shadow positions and exposes the closed-trade results
    for ML labelling and gate scorecard analysis.

    Usage in main.py
    ----------------
    Initialise once after exchange handlers are ready:
        self.shadow_trader = ShadowTradingEngine()

    Open a shadow position when a signal is blocked:
        self.shadow_trader.open_position(asset, side, entry_price,
            strategy_source, gate_blocked_by, details)

    Call every 5 seconds (tick tier):
        self.shadow_trader.tick_update_all(price_map)

    Call every 5 minutes (candle tier):
        self.shadow_trader.candle_update_all(price_map)
    """

    def __init__(
        self,
        max_positions: int = 500,
        max_closed: int = 10000,
    ):
        self.open_positions: List[ShadowPosition] = []
        self.closed_results: List[dict] = []
        self._max_positions = max_positions
        self._max_closed    = max_closed

        logger.info(
            f"[SHADOW] ShadowTradingEngine initialised "
            f"(max_open={max_positions}, max_closed={max_closed}, "
            f"no cooldown, no per-asset cap)"
        )

    def open_position(
        self,
        asset: str,
        side: str,
        entry_price: float,
        strategy_source: str,
        gate_blocked_by: str,
        signal_details: dict,
        atr: Optional[float] = None,
        atr_multiplier: float = 1.8,
        tp_multiples: list = None,
        composite_state: dict = None,   # J2.1 — from CompositeState.to_dict()
    ) -> Optional[ShadowPosition]:
        """
        Open a new shadow position for a blocked signal.

        Parameters
        ----------
        asset            : Asset name, e.g. "BTC"
        side             : "long" or "short"
        entry_price      : Price at signal time
        strategy_source  : "TF", "MR", "EMA", or "consensus"
        gate_blocked_by  : The reasoning string from signal_details
        signal_details   : Full details dict from get_aggregated_signal
        atr              : Regime-adaptive ATR value (VTM-style ATR7/14/28)
        atr_multiplier   : SL distance = atr × multiplier (from asset risk_config)
        tp_multiples     : TP ATR multiples [tp1, tp2, tp3] — first entry used for TP1
        """
        if len(self.open_positions) >= self._max_positions:
            logger.debug("[SHADOW] Max positions reached, skipping")
            return None

        if entry_price <= 0:
            return None

        asset_key = asset.upper()
        now = datetime.now(timezone.utc)

        # Compute SL/TP using VTM's formula:
        #   SL distance = atr × atr_multiplier  (clamped: min 0.5×atr, max 5.0×atr)
        #   TP1          = entry ± atr × first partial_target multiple
        _stop_loss = 0.0
        _take_profit = 0.0
        if atr and atr > 0:
            _tp_mults = tp_multiples if tp_multiples else [2.5]
            _first_tp = float(_tp_mults[0]) if _tp_mults else 2.5

            # Match VTM clamp: min 0.5×atr, max 5.0×atr
            sl_dist = max(0.5 * atr, min(5.0 * atr, atr_multiplier * atr))
            tp_dist = _first_tp * atr

            if side == "long":
                _stop_loss   = entry_price - sl_dist
                _take_profit = entry_price + tp_dist
            else:
                _stop_loss   = entry_price + sl_dist
                _take_profit = entry_price - tp_dist

        # J2.2 + J2.3: Compute standardized trailing and TP1 params at entry time
        _trailing_distance = 0.0
        _trailing_activation_pct = 0.0
        _tp1_price = 0.0
        if atr and atr > 0 and entry_price > 0:
            _trailing_distance = atr * 1.5
            _trailing_activation_pct = atr / entry_price * 1.0  # 1.0× ATR
            _tp1_dist = 1.5 * atr
            if side == "long":
                _tp1_price = entry_price + _tp1_dist
            else:
                _tp1_price = entry_price - _tp1_dist

        pos = ShadowPosition(
            asset=asset,
            side=side,
            strategy_source=strategy_source,
            gate_blocked_by=gate_blocked_by,
            entry_price=entry_price,
            current_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            regime_score=signal_details.get("regime_score",
                signal_details.get("governor_data", {}).get("regime_score", 0.0)
                if isinstance(signal_details.get("governor_data"), dict) else 0.0
            ),
            regime_name=signal_details.get("regime", "UNKNOWN"),
            stop_loss=_stop_loss,
            take_profit=_take_profit,
            strategy_votes={
                "mr_signal":    signal_details.get("mr_signal", 0),
                "mr_conf":      signal_details.get("mr_confidence", 0.0),
                "tf_signal":    signal_details.get("tf_signal", 0),
                "tf_conf":      signal_details.get("tf_confidence", 0.0),
                "ema_signal":   signal_details.get("ema_signal", 0),
                "ema_conf":     signal_details.get("ema_confidence", 0.0),
                "signal_quality": signal_details.get("signal_quality", 0.0),
            },
            # J2.1: CompositeState snapshot
            composite_state=composite_state or {},
            # J2.2: Standardized trailing stop (same for every shadow trade)
            trailing_active=False,
            trailing_distance=_trailing_distance,
            trailing_activation_pct=_trailing_activation_pct,
            highest_price=entry_price,
            lowest_price=entry_price,
            # J2.3: Breakeven after TP1
            tp1_price=_tp1_price,
        )

        self.open_positions.append(pos)
        logger.info(
            f"[SHADOW] Opened {side.upper()} {asset} @ {entry_price:.5f} "
            f"(src={strategy_source}, gate={gate_blocked_by})"
        )
        return pos

    def tick_update_all(self, price_map: Dict[str, float]) -> int:
        """
        Tick-tier update — call every ~5 seconds.
        price_map: {"BTC": 94250.0, "GOLD": 2850.0, ...}
        Returns number of positions closed this tick.
        """
        closed_count = 0
        still_open = []
        for pos in self.open_positions:
            price = price_map.get(pos.asset)
            if price is None or price <= 0:
                still_open.append(pos)
                continue
            if pos.tick_update(price):
                self._archive(pos)
                closed_count += 1
            else:
                still_open.append(pos)
        self.open_positions = still_open
        return closed_count

    def candle_update_all(self, price_map: Dict[str, float]) -> None:
        """
        Candle-tier update — call every ~5 minutes.
        Increments bar counters and applies time stops.
        """
        still_open = []
        for pos in self.open_positions:
            price = price_map.get(pos.asset)
            if price and price > 0:
                pos.current_price = price
            pos.candle_update()
            if pos.closed:
                self._archive(pos)
            else:
                still_open.append(pos)
        self.open_positions = still_open

    def _archive(self, pos: ShadowPosition) -> None:
        """Move a closed position to results store."""
        self.closed_results.append(pos.to_dict())
        # Keep results bounded
        if len(self.closed_results) > self._max_closed:
            self.closed_results = self.closed_results[-self._max_closed:]

    def get_gate_scorecard(self) -> Dict[str, dict]:
        """
        Summarise performance by blocking gate — uses net_pnl_pct (after friction).
        Useful for identifying gates that are blocking profitable signals.

        Returns dict keyed by gate name:
            {"count": int, "win_rate": float, "avg_net_pnl": float}
        """
        from collections import defaultdict
        buckets: Dict[str, list] = defaultdict(list)
        for r in self.closed_results:
            buckets[r["gate_blocked_by"]].append(r["net_pnl_pct"])

        scorecard = {}
        for gate, pnls in buckets.items():
            wins = sum(1 for p in pnls if p > 0)
            scorecard[gate] = {
                "count":       len(pnls),
                "win_rate":    round(wins / len(pnls) * 100, 1) if pnls else 0.0,
                "avg_net_pnl": round(sum(pnls) / len(pnls), 3) if pnls else 0.0,
                "total_pnl":   round(sum(pnls), 3),
            }
        return dict(sorted(scorecard.items(), key=lambda x: x[1]["total_pnl"]))

    def get_strategy_scorecard(self) -> Dict[str, dict]:
        """Summarise performance by strategy source."""
        from collections import defaultdict
        buckets: Dict[str, list] = defaultdict(list)
        for r in self.closed_results:
            buckets[r["strategy_source"]].append(r["net_pnl_pct"])

        scorecard = {}
        for src, pnls in buckets.items():
            wins = sum(1 for p in pnls if p > 0)
            scorecard[src] = {
                "count":       len(pnls),
                "win_rate":    round(wins / len(pnls) * 100, 1) if pnls else 0.0,
                "avg_net_pnl": round(sum(pnls) / len(pnls), 3) if pnls else 0.0,
            }
        return scorecard

    def dump_state(self, path: str) -> None:
        """
        Write a JSON snapshot of the shadow engine's current state to *path*.
        Called periodically by the bot (e.g. every candle) so the dashboard
        process can read it without needing direct in-process access.
        """
        import json
        import os

        open_list = []
        for pos in self.open_positions:
            d = pos.to_dict()
            d["current_price"] = pos.current_price
            d["bars_open"] = pos.bars_open
            d["mfe_pct"] = round(pos.mfe_pct * 100, 4)
            d["mae_pct"] = round(pos.mae_pct * 100, 4)
            # live unrealised P&L
            if pos.entry_price > 0:
                raw = pos._profit_pct(pos.current_price)
                d["live_pnl_pct"] = round(raw * 100, 4)
            else:
                d["live_pnl_pct"] = 0.0
            open_list.append(d)

        state = {
            "open_positions": open_list,
            "closed_results": self.closed_results[-200:],   # last 200 for dashboard
            "gate_scorecard": self.get_gate_scorecard(),
            "strategy_scorecard": self.get_strategy_scorecard(),
            "summary": {
                "open_count": len(self.open_positions),
                "closed_count": len(self.closed_results),
            },
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, default=str)
            os.replace(tmp, path)          # atomic write
        except Exception as exc:
            logger.warning(f"[SHADOW] dump_state failed: {exc}")

    @property
    def summary(self) -> str:
        return (
            f"ShadowTrader: {len(self.open_positions)} open, "
            f"{len(self.closed_results)} closed"
        )
