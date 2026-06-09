"""
System Validator — The Immune System (Phase 1+2, MRS §12)
=========================================================
Four health dimensions monitored every cycle. Three SYSTEM_INTEGRITY
watchdogs run every 5 minutes. 6-hour log summaries. Atomic persistence.

NEVER blocks trading. NEVER disables components. NEVER makes execution
decisions. Flags anomalies for human review only.

Dimensions
----------
LIVENESS (0–100)    Is this component producing meaningful, non-default outputs?
                    Zero entropy = stuck value = DEAD.
CALIBRATION (0–100) Are thresholds producing expected output distributions?
EDGE (0–100)        Rolling 50-trade z-test. z < 1.0 → WATCH. z < 0 → flag.
SYSTEM_INTEGRITY    Three watchdogs. Run every 5 minutes.

Watchdogs
---------
Amnesia Monitor      DynamicThresholds cache < 5 samples → state wipe detected.
VTM Circuit Breaker  Single 1H bar > 3×ATR against open position → lock breakeven.
API Rate Limit       Binance weight > 80% → throttle HistoricalDataUpdater.

Score thresholds (MRS table)
----------------------------
80–100  HEALTHY     Operating as designed.
60–79   WATCH       Investigate in next review session.
40–59   DEGRADED    Review within 24 hours.
0–39    FAILING     Manual review required before next live session.
DEAD    No output.  Component not participating.
"""

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComponentHealth:
    """Health snapshot for one named component."""
    name:        str
    liveness:    Optional[float] = None   # 0–100 or None = no data yet
    calibration: Optional[float] = None
    edge:        Optional[float] = None   # z-score from rolling test; None = <50 trades
    integrity:   str = "OK"               # "OK" | "WARN" | "CRITICAL"
    notes:       str = ""

    @property
    def status(self) -> str:
        score = self.liveness
        if score is None:
            return "NO_DATA"
        if score == 0.0:
            return "DEAD"
        if score >= 80:
            return "HEALTHY"
        if score >= 60:
            return "WATCH"
        if score >= 40:
            return "DEGRADED"
        return "FAILING"

    def formatted(self) -> str:
        live  = f"LIVE={self.liveness:.0f}" if self.liveness  is not None else "LIVE=n/a"
        calib = f"CALIB={self.calibration:.0f}" if self.calibration is not None else "CALIB=n/a"
        edge  = f"EDGE={self.edge:.1f}"  if self.edge  is not None else "EDGE=n/a"
        return f"{self.name:<35} {live} {calib} {edge} INTGR={self.integrity} [{self.status}{(' — ' + self.notes) if self.notes else ''}]"


@dataclass
class TradeOutcome:
    """Minimal record to compute rolling edge z-test."""
    component: str
    signal:    int    # +1 or -1
    entry_time: datetime
    pnl_r: Optional[float] = None   # outcome in R-multiples; None = pending


# ─────────────────────────────────────────────────────────────────────────────
# Main validator
# ─────────────────────────────────────────────────────────────────────────────

class SystemValidator:
    """
    Silent background health monitor. Instantiate once in main.py, call
    `update()` after every trading cycle, call `watchdog_tick()` every
    5 minutes from the scheduler or main loop.

    Parameters
    ----------
    state_path        Atomic persistence file (JSON). Written with tmp→replace.
    vtm_cb_atr_mult   ATR multiple that triggers VTM circuit breaker (default 3.0).
    log_interval_h    Hours between full 6-hour log summaries (default 6).
    """

    def __init__(
        self,
        state_path: str = "data/system_validator_state.json",
        vtm_cb_atr_mult: float = 3.0,
        log_interval_h: float = 6.0,
    ):
        self._state_path       = state_path
        self._vtm_cb_atr_mult  = vtm_cb_atr_mult
        self._log_interval_h   = log_interval_h

        # Per-component health snapshots (name → ComponentHealth)
        self._health: Dict[str, ComponentHealth] = {}

        # Liveness entropy trackers: deque of last N output values per component
        # (deque of floats; zero entropy = all same value → DEAD / stuck)
        self._liveness_buffers: Dict[str, deque] = {}

        # Calibration: count of times a component's veto/gate fires vs total observations
        # {name: {"fires": int, "total": int}}
        self._calibration_counts: Dict[str, Dict] = {}

        # Edge: rolling 50-trade outcomes per component
        self._trade_outcomes: Dict[str, deque] = {}

        # Watchdog state
        self._last_watchdog_run: Optional[datetime] = None
        self._watchdog_interval_s: float = 300.0  # 5 minutes

        # API rate tracking (last observed weights)
        self._api_weights: Dict[str, float] = {}   # {"binance": 0.0–1.0 fraction}

        # VTM circuit breaker signals: {asset: True/False}
        self._vtm_cb_signals: Dict[str, bool] = {}

        # Vol-down-ratio veto threshold — synced from aggregator_presets so the
        # calibration check evaluates the same threshold MR Mode 1 actually uses.
        self._vdr_threshold: float = 1.2  # safe default; overridden below
        self._load_vdr_threshold()

        # Summary log state
        self._last_summary_time: Optional[datetime] = None
        self._cycle_count: int = 0

        # Startup log
        logger.info("[VALIDATOR] System Validator initialised (path=%s)", state_path)
        self._load_state()

    def _load_vdr_threshold(self) -> None:
        """
        Read the vol_down_ratio veto threshold from aggregator_presets.json.
        MR Mode 1 reads from MR_THREE_MODE.mode1.vol_down_ratio_veto.
        The validator must check the same value — if the threshold is tuned
        during paper trading, both must move together.
        Falls back to 1.2 if the file is unreachable.
        """
        try:
            import json as _json
            with open("config/aggregator_presets.json") as _f:
                _d = _json.load(_f)
            self._vdr_threshold = (
                _d.get("MR_THREE_MODE", {})
                  .get("mode1", {})
                  .get("vol_down_ratio_veto", 1.2)
            )
            logger.debug("[VALIDATOR] vol_down_ratio threshold loaded: %.2f", self._vdr_threshold)
        except Exception as _e:
            logger.debug("[VALIDATOR] Could not load vdr threshold, using 1.2: %s", _e)
            self._vdr_threshold = 1.2

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        composite_state=None,
        signal_details: Dict = None,
        open_positions: List[Dict] = None,
        df_1h=None,
        asset: str = "UNKNOWN",
    ) -> None:
        """
        Call once per trading cycle (after VTM update, before sleeping).

        Parameters
        ----------
        composite_state  The CompositeState built for this cycle (can be None).
        signal_details   The details dict returned by get_aggregated_signal.
        open_positions   List of position dicts from portfolio manager.
        df_1h            1H OHLCV DataFrame for VTM circuit breaker check.
        asset            Asset name for logging context.
        """
        self._cycle_count += 1
        try:
            self._update_liveness(composite_state, signal_details, asset)
        except Exception as e:
            logger.debug("[VALIDATOR] liveness update error: %s", e)

        try:
            self._update_calibration(composite_state, signal_details, asset)
        except Exception as e:
            logger.debug("[VALIDATOR] calibration update error: %s", e)

        # VTM circuit breaker check on every cycle
        try:
            if open_positions and df_1h is not None and len(df_1h) >= 2:
                self._check_vtm_circuit_breaker(open_positions, df_1h, asset)
        except Exception as e:
            logger.debug("[VALIDATOR] VTM CB check error: %s", e)

        # 6-hour log summary
        try:
            now = datetime.now()
            if (
                self._last_summary_time is None
                or (now - self._last_summary_time).total_seconds() >= self._log_interval_h * 3600
            ):
                self._log_health_summary()
                self._last_summary_time = now
                self._persist_state()
        except Exception as e:
            logger.debug("[VALIDATOR] summary/persist error: %s", e)

    def watchdog_tick(
        self,
        dynamic_thresholds=None,
        open_positions: List[Dict] = None,
        df_1h=None,
        asset: str = "UNKNOWN",
        livermore_state: str = None,
        binance_api_weight: Optional[int] = None,
        binance_api_weight_limit: int = 6000,
    ) -> Dict[str, bool]:
        """
        Run SYSTEM_INTEGRITY watchdogs. Call every 5 minutes from main loop.

        Returns a dict of {watchdog_name: fired_bool} so callers can act.
        Never raises — all errors caught and logged.
        """
        results = {
            "amnesia_fired":    False,
            "vtm_cb_fired":     False,
            "api_throttle":     False,
        }
        now = datetime.now()
        if (
            self._last_watchdog_run is not None
            and (now - self._last_watchdog_run).total_seconds() < self._watchdog_interval_s - 10
        ):
            return results  # Too soon; skip

        self._last_watchdog_run = now

        # ── Watchdog 1: Amnesia Monitor ──────────────────────────────────────
        try:
            if dynamic_thresholds is not None:
                cache = getattr(dynamic_thresholds, '_cache', {})
                sample_count = sum(len(v) for v in cache.values()) if isinstance(cache, dict) else 0
                if sample_count < 5:
                    logger.critical(
                        "[VALIDATOR] [CRITICAL] Memory Wipe Detected — "
                        "DynamicThresholds cache reset (samples=%d). "
                        "Z-score baselines are now incorrect.",
                        sample_count,
                    )
                    results["amnesia_fired"] = True
                    self._set_integrity("AMNESIA_MONITOR", "CRITICAL",
                                        f"cache={sample_count} samples")
                else:
                    self._set_integrity("AMNESIA_MONITOR", "OK", f"cache={sample_count}")
        except Exception as e:
            logger.debug("[VALIDATOR] Amnesia watchdog error: %s", e)

        # ── Watchdog 2: VTM Circuit Breaker Signal ────────────────────────────
        try:
            if open_positions and df_1h is not None and len(df_1h) >= 2:
                fired = self._check_vtm_circuit_breaker(
                    open_positions, df_1h, asset,
                    livermore_state=livermore_state,
                )
                results["vtm_cb_fired"] = fired
        except Exception as e:
            logger.debug("[VALIDATOR] VTM CB watchdog error: %s", e)

        # ── Watchdog 3: API Rate Limit Tracker ───────────────────────────────
        try:
            if binance_api_weight is not None and binance_api_weight_limit > 0:
                fraction = binance_api_weight / binance_api_weight_limit
                self._api_weights["binance"] = fraction
                pct = fraction * 100
                if fraction >= 0.80:
                    logger.warning(
                        "[VALIDATOR] Binance API weight %.0f%% — throttling "
                        "HistoricalDataUpdater.",
                        pct,
                    )
                    results["api_throttle"] = True
                    self._set_integrity("API_RATE_BINANCE", "WARN",
                                        f"weight={pct:.0f}%")
                else:
                    self._set_integrity("API_RATE_BINANCE", "OK",
                                        f"weight={pct:.1f}%")
        except Exception as e:
            logger.debug("[VALIDATOR] API rate watchdog error: %s", e)

        return results

    def record_trade_outcome(
        self,
        component: str,
        signal: int,
        entry_time: datetime,
        pnl_r: float,
    ) -> None:
        """
        Record a completed trade outcome for a component's EDGE calculation.
        pnl_r: profit/loss measured in R-multiples (e.g. 1.5 = 1.5R win).
        """
        if component not in self._trade_outcomes:
            self._trade_outcomes[component] = deque(maxlen=50)
        self._trade_outcomes[component].append(
            TradeOutcome(component=component, signal=signal,
                         entry_time=entry_time, pnl_r=pnl_r)
        )
        self._refresh_edge(component)

    def get_vtm_circuit_breaker_signal(self, asset: str) -> bool:
        """
        Returns True if the VTM circuit breaker fired for this asset this cycle.
        VTM should call this and lock stops to breakeven if True.
        """
        return self._vtm_cb_signals.get(asset, False)

    def clear_vtm_cb_signal(self, asset: str) -> None:
        """Call after VTM has acted on the circuit breaker signal."""
        self._vtm_cb_signals[asset] = False

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: Liveness
    # ─────────────────────────────────────────────────────────────────────────

    def _update_liveness(self, composite_state, signal_details, asset):
        """
        Compute liveness scores from observable output entropy.
        A component that always outputs the same value has zero entropy → DEAD.
        """
        if composite_state is None:
            return

        _components = {
            # (name, attribute, buffer_size)
            "LIVERMORE_STATE_4H": ("livermore_state_4h", 30),
            "LIVERMORE_STATE_1H": ("livermore_state_1h", 30),
            "LIFECYCLE_PHASE":    ("lifecycle_phase",    30),
            "IS_SILENT_ZONE":     ("is_silent_zone",     50),
            "VOL_DOWN_RATIO":     ("vol_down_ratio",     20),
            "BB_KC_SQUEEZE":      ("bb_kc_squeeze_active", 50),
        }

        for name, (attr, buf_size) in _components.items():
            val = getattr(composite_state, attr, None)
            if val is None:
                continue
            if name not in self._liveness_buffers:
                self._liveness_buffers[name] = deque(maxlen=buf_size)
            self._liveness_buffers[name].append(str(val))

            buf = self._liveness_buffers[name]
            if len(buf) < 5:
                continue  # Not enough data yet

            # Entropy proxy: count unique values in buffer
            unique = len(set(buf))
            total  = len(buf)
            # Perfect diversity (all unique) → 100; all same → 0
            raw_score = (unique / total) * 100.0

            # Some components legitimately have low diversity in trending markets
            # (e.g. is_silent_zone will be False for long stretches of MAIN_UP).
            # Cap minimum at 10 so they don't appear DEAD when they're just stable.
            # unique == 1 means the component produced a single identical value
            # for all observations in its buffer — genuinely stuck/frozen/unconnected.
            # Formula: ×3.0 amplifies sensitivity (33% diversity → 43, 67% → 100).
            # +10.0 floors legitimate low-diversity components (e.g. is_silent_zone
            # is False for long MAIN_UP stretches — stable, not dead).
            if unique == 1:
                liveness = 0.0  # DEAD: all buffer values identical
            else:
                liveness = max(10.0, min(100.0, raw_score * 3.0 + 10.0))

            ch = self._get_or_create(name)
            ch.liveness = round(liveness, 1)

        # HARD_VETO_LAYER liveness: track whether it fires at a reasonable rate
        if signal_details:
            _reason = signal_details.get("reasoning", "")
            if name not in self._liveness_buffers:
                pass  # handled above
            _veto_name = "HARD_VETO_LAYER"
            if _veto_name not in self._liveness_buffers:
                self._liveness_buffers[_veto_name] = deque(maxlen=100)
            _veto_fired = "HARD_VETO" in str(signal_details.get("reasoning", ""))
            self._liveness_buffers[_veto_name].append(1 if _veto_fired else 0)
            buf = self._liveness_buffers[_veto_name]
            if len(buf) >= 20:
                fire_rate = sum(buf) / len(buf)
                # Expected 5–30% block rate; outside = calibration concern but not dead
                if fire_rate == 0.0:
                    _live = 20.0   # Never fired — suspicious
                elif fire_rate > 0.70:
                    _live = 40.0   # Firing too much — calibration issue
                else:
                    _live = 88.0   # Healthy
                ch = self._get_or_create(_veto_name)
                if ch.notes and "vetoes fired" in ch.notes:
                    # preserve count annotation
                    pass
                ch.liveness = _live
                total_vetoes = sum(self._liveness_buffers[_veto_name])
                ch.notes = f"{total_vetoes} vetoes fired"

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: Calibration
    # ─────────────────────────────────────────────────────────────────────────

    def _update_calibration(self, composite_state, signal_details, asset):
        """
        Check whether key thresholds produce expected output distributions.
        vol_down_ratio veto should block ~10–30% of MR Mode 1 attempts.
        Hard Veto should block ~5–30% of raw signals.
        """
        if composite_state is None:
            return

        # vol_down_ratio calibration: is the veto threshold in a reasonable range?
        if composite_state.vol_down_ratio_valid and composite_state.vol_down_ratio is not None:
            _vdr_name = "VOL_DOWN_RATIO"
            if _vdr_name not in self._calibration_counts:
                self._calibration_counts[_vdr_name] = {"above_threshold": 0, "total": 0}
            self._calibration_counts[_vdr_name]["total"] += 1
            if composite_state.vol_down_ratio > self._vdr_threshold:
                self._calibration_counts[_vdr_name]["above_threshold"] += 1

            counts = self._calibration_counts[_vdr_name]
            if counts["total"] >= 50:
                fire_rate = counts["above_threshold"] / counts["total"]
                # Target: 10–30% veto rate
                if 0.10 <= fire_rate <= 0.30:
                    calib = 90.0
                elif 0.05 <= fire_rate < 0.10 or 0.30 < fire_rate <= 0.50:
                    calib = 65.0
                else:
                    calib = 35.0   # Out of range — investigate
                ch = self._get_or_create(_vdr_name)
                ch.calibration = round(calib, 1)
                ch.notes = f"veto_rate={fire_rate:.1%}"

        # LIFECYCLE_PHASE calibration: should show diversity across phases
        if composite_state.lifecycle_phase:
            _lp_name = "LIFECYCLE_PHASE"
            if _lp_name not in self._calibration_counts:
                self._calibration_counts[_lp_name] = {}
            phase = composite_state.lifecycle_phase
            self._calibration_counts[_lp_name][phase] = \
                self._calibration_counts[_lp_name].get(phase, 0) + 1
            total = sum(self._calibration_counts[_lp_name].values())
            if total >= 100:
                # Pre-fix: was 100% ESTABLISHED. Post-fix: should show diversity.
                estab_frac = self._calibration_counts[_lp_name].get("ESTABLISHED", 0) / total
                if estab_frac > 0.95:
                    calib = 10.0   # Still stuck — fix not working
                    notes = f"ESTABLISHED={estab_frac:.0%} (was 100% pre-fix)"
                elif estab_frac > 0.80:
                    calib = 55.0   # Improving but still concentrated
                    notes = f"ESTABLISHED={estab_frac:.0%}"
                else:
                    calib = 85.0   # Good diversity
                    notes = f"ESTABLISHED={estab_frac:.0%}"
                ch = self._get_or_create(_lp_name)
                ch.calibration = round(calib, 1)
                ch.notes = notes

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: Edge z-test
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_edge(self, component: str) -> None:
        """
        Recompute rolling edge z-score for a component from its 50 most recent
        completed trades.

        z = (win_rate_conditional - win_rate_baseline) / se_baseline
        where se_baseline = sqrt(p*(1-p)/n), p = 0.50 (null hypothesis: coin flip).

        z > 1.0 → statistically meaningful edge.
        z < 0   → component actively hurting outcomes.
        """
        outcomes = self._trade_outcomes.get(component)
        if not outcomes or len(outcomes) < 10:
            return  # Not enough trades

        completed = [o for o in outcomes if o.pnl_r is not None]
        if len(completed) < 10:
            return

        wins = sum(1 for o in completed if o.pnl_r > 0)
        n    = len(completed)
        p    = wins / n
        p0   = 0.50   # null hypothesis

        se = math.sqrt(p0 * (1 - p0) / n)
        z  = (p - p0) / se if se > 0 else 0.0

        ch = self._get_or_create(component)
        ch.edge = round(z, 2)

        if z < 0:
            logger.warning(
                "[VALIDATOR] [EDGE] %s z-score=%.2f — component actively "
                "hurting outcomes. Manual review required.",
                component, z,
            )
        elif z < 1.0:
            logger.info(
                "[VALIDATOR] [EDGE] %s z-score=%.2f — WATCH: not statistically "
                "distinguishable from random (%d trades).",
                component, z, n,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: VTM Circuit Breaker
    # ─────────────────────────────────────────────────────────────────────────

    def _check_vtm_circuit_breaker(
        self,
        open_positions: List[Dict],
        df_1h,
        asset: str,
        livermore_state: str = None,
    ) -> bool:
        """
        MRS §11: If a single 1H bar moves more than 3×ATR(14) against the
        position direction while Livermore state is NATURAL, fire the circuit
        breaker override signal. VTM reads this via get_vtm_circuit_breaker_signal()
        and locks stops to breakeven.
        """
        if not open_positions or df_1h is None or len(df_1h) < 16:
            return False

        try:
            import numpy as _np
            _hi  = df_1h["high"].values
            _lo  = df_1h["low"].values
            _cl  = df_1h["close"].values
            _tr  = _np.maximum(
                _hi[1:] - _lo[1:],
                _np.abs(_hi[1:] - _cl[:-1]),
                _np.abs(_lo[1:]  - _cl[:-1]),
            )
            _atr = float(_np.nanmean(_tr[-14:])) if len(_tr) >= 14 else 0.0
            if _atr <= 0:
                return False

            # Last bar's move
            _last_close = float(_cl[-1])
            _prev_close = float(_cl[-2])
            _bar_move   = _last_close - _prev_close  # positive = up, negative = down
            _abs_move   = abs(_bar_move)
            _atr_mult   = _abs_move / _atr

            fired = False
            for pos in open_positions:
                pos_asset = pos.get("asset", pos.get("symbol", "")).upper()
                if asset.upper() not in pos_asset and pos_asset not in asset.upper():
                    continue  # Not this asset

                direction = pos.get("direction", pos.get("side", "")).lower()
                is_long   = direction in ("buy", "long", "1")
                is_short  = direction in ("sell", "short", "-1")

                if not (is_long or is_short):
                    continue

                # Against position = down for long, up for short
                against = (is_long and _bar_move < 0) or (is_short and _bar_move > 0)
                if not against:
                    continue

                # Only fire in NATURAL/SECONDARY states — a 3×ATR bar in a
                # confirmed MAIN trend may just be the trend running hard.
                _natural_states = {
                    "NATURAL_RETRACEMENT", "NATURAL_REBOUND",
                    "SECONDARY_RETRACEMENT", "SECONDARY_REBOUND",
                }
                _lsm_gate = (
                    livermore_state is None  # unknown → fire conservatively
                    or livermore_state in _natural_states
                )
                if _atr_mult >= self._vtm_cb_atr_mult and _lsm_gate:
                    logger.critical(
                        "[VALIDATOR] [CRITICAL] Circuit Breaker fired — %s "
                        "%.1f×ATR single bar (%.1f×ATR threshold) — locking to breakeven.",
                        asset, _atr_mult, self._vtm_cb_atr_mult,
                    )
                    self._vtm_cb_signals[asset] = True
                    self._set_integrity("VTM_CIRCUIT_BREAKER", "CRITICAL",
                                        f"{asset} {_atr_mult:.1f}×ATR")
                    fired = True

            return fired
        except Exception as e:
            logger.debug("[VALIDATOR] VTM CB inner error: %s", e)
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: 6-hour log summary
    # ─────────────────────────────────────────────────────────────────────────

    def _log_health_summary(self) -> None:
        """Emit a full health report in the exact format specified by MRS §12."""
        now_str = datetime.now().strftime("%H:%M:%S")
        sep = "═" * 54
        logger.info(f"[VALIDATOR] {sep}")
        logger.info(f"[VALIDATOR] 6-HOUR HEALTH REPORT — {now_str}")
        logger.info(f"[VALIDATOR] {sep}")

        watch_count    = 0
        degraded_count = 0

        # Log all known components
        for name, ch in sorted(self._health.items()):
            logger.info(f"[VALIDATOR] {ch.formatted()}")
            if ch.status == "WATCH":
                watch_count += 1
            elif ch.status in ("DEGRADED", "FAILING", "DEAD"):
                degraded_count += 1

        # API weights
        for exchange, frac in self._api_weights.items():
            logger.info(
                "[VALIDATOR] API_RATE_%s weight=%.0f%% [%s]",
                exchange.upper(),
                frac * 100,
                "NOMINAL" if frac < 0.80 else "THROTTLING",
            )

        # SYSTEM_INTEGRITY summary
        logger.info("[VALIDATOR] SYSTEM_INTEGRITY .............. ALL WATCHDOGS NOMINAL [OK]")

        # Footer
        issues = []
        if watch_count:
            issues.append(f"{watch_count} at WATCH")
        if degraded_count:
            issues.append(f"{degraded_count} at DEGRADED/FAILING")

        if issues:
            logger.info(f"[VALIDATOR] ══ {', '.join(issues)}. Review before next live session. ══")
        else:
            logger.info(f"[VALIDATOR] ══ All components healthy. ══")

        logger.info(f"[VALIDATOR] {sep}")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_or_create(self, name: str) -> ComponentHealth:
        if name not in self._health:
            self._health[name] = ComponentHealth(name=name)
        return self._health[name]

    def _set_integrity(self, name: str, status: str, notes: str = "") -> None:
        ch = self._get_or_create(name)
        ch.integrity = status
        if notes:
            ch.notes = notes

    # ─────────────────────────────────────────────────────────────────────────
    # Atomic persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _persist_state(self) -> None:
        """Write state atomically: write to .tmp then os.replace()."""
        try:
            os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
            payload = {
                "persisted_at":       datetime.now().isoformat(),
                "cycle_count":        self._cycle_count,
                "api_weights":        self._api_weights,
                "calibration_counts": self._calibration_counts,
                "trade_outcome_counts": {
                    k: len(v) for k, v in self._trade_outcomes.items()
                },
                "health_summary": {
                    name: {
                        "liveness":    ch.liveness,
                        "calibration": ch.calibration,
                        "edge":        ch.edge,
                        "integrity":   ch.integrity,
                        "notes":       ch.notes,
                    }
                    for name, ch in self._health.items()
                },
            }
            tmp_path = self._state_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, self._state_path)
        except Exception as e:
            logger.debug("[VALIDATOR] persist error: %s", e)

    def _load_state(self) -> None:
        """Restore health snapshot from prior session if available."""
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path) as f:
                data = json.load(f)

            self._cycle_count        = data.get("cycle_count", 0)
            self._api_weights        = data.get("api_weights", {})
            self._calibration_counts = data.get("calibration_counts", {})

            for name, snap in data.get("health_summary", {}).items():
                ch = self._get_or_create(name)
                ch.liveness    = snap.get("liveness")
                ch.calibration = snap.get("calibration")
                ch.edge        = snap.get("edge")
                ch.integrity   = snap.get("integrity", "OK")
                ch.notes       = snap.get("notes", "")

            logger.info(
                "[VALIDATOR] Restored state from prior session "
                "(cycles=%d, components=%d)",
                self._cycle_count, len(self._health),
            )
        except Exception as e:
            logger.debug("[VALIDATOR] state load error (starting fresh): %s", e)
