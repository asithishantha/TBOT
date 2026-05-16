"""
Enhanced PerformanceWeightedAggregator with AI Safety Features
===============================================================
IMPROVEMENTS:
- AI circuit breaker to prevent over-filtering
- Regime context passed to AI validator
- Better cold-start handling for regime detection
- AI performance tracking
- Graceful degradation if AI fails
"""

import pandas as pd
import logging
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
from datetime import datetime, timedelta
from src.utils.trap_filter import validate_candle_structure
from src.indicators.divergence import RSIDivergenceDetector
from src.analysis.break_retest import BreakRetestValidator
from src.execution.transition_detector import TransitionDetector

logger = logging.getLogger(__name__)


class PerformanceWeightedAggregator:
    """
Enhanced Signal Aggregator with World-Class Filters
====================================================
Adds Governor + Volatility + Sniper checks to existing aggregator
    """

    def __init__(
        self,
        mean_reversion_strategy,
        trend_following_strategy,
        ema_strategy,
        volume_flow_strategy=None,
        asset_type: str = "BTC",
        config: Dict = None,
        ai_validator=None,
        mtf_integration=None,  # For Governor access
        enable_world_class_filters: bool = True,
        enable_ai_circuit_breaker: bool = False,
        enable_detailed_logging: bool = False,
        strong_signal_bypass_threshold: float = 0.70,
        use_macro_governor: bool = True,
        use_gatekeeper: bool = True
    ):
        self.s_mean_reversion = mean_reversion_strategy
        self.s_trend_following = trend_following_strategy
        self.s_ema = ema_strategy
        self.s_volume_flow = volume_flow_strategy
        self.asset_type = asset_type.upper()
        self.use_macro_governor = use_macro_governor
        self.use_gatekeeper = use_gatekeeper

        # Initialize regime tracking
        self.previous_regime = None
        self.regime_initialized = False

        # Logging and Thresholds
        self.detailed_logging = enable_detailed_logging
        self.strong_signal_bypass = strong_signal_bypass_threshold

        # ================================================================
        # AI VALIDATOR SETUP
        # ================================================================
        self.ai_validator = None
        self.ai_enabled = True
        
        # ✨ NEW: Store MTF integration for Governor
        self.mtf_integration = mtf_integration
        self.enable_filters = enable_world_class_filters

        # ✨ NEW: Initialize filter thresholds
        # Volatility gate: asset-class-specific defaults so FX pairs (EURJPY ATR ~0.07%)
        # aren't permanently blocked by a crypto-calibrated 0.35% threshold.
        #   FX  (EUR*, GBP*, USD*, JPY*, CHF*, AUD*, NZD*, CAD*) → 0.03% (0.0003)
        #   Metals / Indices (XAU, GOLD, USTEC, NAS*, SP5*, GER*) → 0.10% (0.0010)
        #   Crypto (BTC*, ETH*, BNB*)                             → 0.20% (0.0020)
        # Config override still wins if explicitly set.
        _asset_upper = asset_type.upper()
        _is_fx = any(
            fx in _asset_upper
            for fx in ("EUR", "GBP", "USD", "JPY", "CHF", "AUD", "NZD", "CAD")
        ) and "BTC" not in _asset_upper and "ETH" not in _asset_upper
        _is_crypto = any(c in _asset_upper for c in ("BTC", "ETH", "BNB", "SOL", "XRP"))
        _is_metals_indices = any(
            m in _asset_upper
            for m in ("XAU", "GOLD", "USTEC", "NAS", "SP5", "GER", "UK1", "NDX")
        )
        if _is_fx:
            _default_vol_threshold = 0.0003   # 0.03% — FX pairs
        elif _is_metals_indices:
            _default_vol_threshold = 0.0010   # 0.10% — metals / indices
        elif _is_crypto:
            _default_vol_threshold = 0.0020   # 0.20% — crypto (relaxed from original 0.35%)
        else:
            _default_vol_threshold = 0.0010   # 0.10% — safe generic fallback

        self.filter_thresholds = {
            'volatility_gate': config.get('world_class_filters', {}).get(
                'volatility_gate_threshold', _default_vol_threshold
            ),
            'sniper_confidence': config.get('world_class_filters', {}).get(
                'sniper_pattern_confidence', 0.60
            ),
            'min_profit': config.get('world_class_filters', {}).get(
                'min_profit_potential', 0.005
            ),
        }
        
        if self.enable_filters:
            logger.info(f"[FILTERS] World-Class Filters ENABLED for {asset_type}")
            logger.info(f"  Volatility Gate: {self.filter_thresholds['volatility_gate']:.3%}")
            logger.info(f"  Sniper Min:      {self.filter_thresholds['sniper_confidence']:.0%}")
            logger.info(f"  Min Profit:      {self.filter_thresholds['min_profit']:.2%}")
            
            
        if ai_validator is not None:
            try:
                # Validate AI is properly initialized
                assert hasattr(ai_validator, "sniper"), "Sniper not initialized"
                assert hasattr(ai_validator.sniper, "model"), "Model not loaded"
                assert hasattr(
                    ai_validator, "pattern_id_map"
                ), "Pattern mapping missing"
                assert len(ai_validator.pattern_id_map) > 0, "Pattern mapping empty"

                self.ai_validator = ai_validator
                self.ai_enabled = True

                logger.info(f"[AGGREGATOR] AI validation: ✓ ENABLED")
                logger.info(
                    f"[AGGREGATOR] Patterns loaded: {len(ai_validator.pattern_id_map)}"
                )

            except (AssertionError, AttributeError) as e:
                logger.error(f"[AGGREGATOR] AI validation setup failed: {e}")
                logger.warning("[AGGREGATOR] Continuing without AI validation")
                self.ai_validator = None
                self.ai_enabled = False

        # AI statistics tracking
        if self.ai_enabled:
            self.ai_stats = {
                "mr_signals_checked": 0,
                "mr_approved": 0,
                "mr_rejected": 0,
                "tf_signals_checked": 0,
                "tf_approved": 0,
                "tf_rejected": 0,
                "bypassed_strong_signal": 0,
            }

            # Circuit breaker configuration
            self.enable_circuit_breaker = enable_ai_circuit_breaker
            self.ai_rejection_window = deque(maxlen=50)
            self.ai_bypass_active = False
            self.ai_bypass_threshold = 0.85
            self.ai_bypass_cooldown = 0

            logger.info(
                f"[AGGREGATOR] AI circuit breaker: {'ENABLED' if enable_ai_circuit_breaker else 'DISABLED'}"
            )
            logger.info(
                f"[AGGREGATOR] Strong signal bypass: {self.strong_signal_bypass:.2%}"
            )
            logger.info(
                f"[AGGREGATOR] Detailed logging: {'ENABLED' if self.detailed_logging else 'DISABLED'}"
            )

        # ✨ NEW: Advanced Confluence Engines
        self.divergence_detector = RSIDivergenceDetector(pivot_window=5)
        self.break_retest_validator = BreakRetestValidator(lookback=50)

        # Strategy weights — read from config, used for priority when multiple strategies fire
        # NOT for consensus voting. Hardcoded 0.50/0.50 was ignoring mean_reversion_weight: 0.0
        # in all presets, causing MR opposition penalty to bleed into every BTC TF score.
        # Will be updated after config merge below so we use a temporary default here.
        self.weights = {"mean_reversion": 0.50, "trend_following": 0.50}

        # ================================================================
        # CONFIGURATION MERGE (Safety Fix)
        # ================================================================
        # 1. Define Defaults first (guarantees all keys exist)
        _is_fx_asset = any(
            fx in self.asset_type.upper()
            for fx in ("EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD")
        ) and "BTC" not in self.asset_type.upper()

        if self.asset_type == "BTC":
            self.config = {
                "buy_threshold": 0.30,
                "sell_threshold": 0.26,
                "two_strategy_bonus": 0.25,
                "three_strategy_bonus": 0.30,
                "four_strategy_bonus": 0.35,
                "bull_buy_boost": 0.25,
                "bull_sell_penalty": 0.20,
                "bear_sell_boost": 0.25,
                "bear_buy_penalty": 0.30,
                "min_confidence_to_use": 0.08,
                "min_signal_quality": 0.28,
                "hold_contribution_pct": 0.0,
                "opposition_penalty": 0.40,
            }
        elif _is_fx_asset:
            # FX pairs (EURUSD, EURJPY, GBPUSD, etc.) move in smaller, more
            # gradual increments than BTC or GOLD. Lowering thresholds prevents
            # valid setups from being blocked by score calculations tuned for
            # higher-volatility assets.
            # single_override_threshold 0.60 vs 0.72: FX strategies are configured
            # with min_confidence=0.45 — a 0.72 bar for independent firing almost
            # never gets reached, silently killing solo TF/EMA signals.
            self.config = {
                "buy_threshold": 0.26,
                "sell_threshold": 0.22,
                "two_strategy_bonus": 0.22,
                "three_strategy_bonus": 0.28,
                "four_strategy_bonus": 0.35,
                "bull_buy_boost": 0.20,
                "bull_sell_penalty": 0.12,
                "bear_sell_boost": 0.20,
                "bear_buy_penalty": 0.22,
                "min_confidence_to_use": 0.05,
                "min_signal_quality": 0.22,
                "hold_contribution_pct": 0.0,
                "opposition_penalty": 0.35,
                "single_override_threshold": 0.60,
                "allow_single_override": True,
            }
        else:  # GOLD, USTEC, indices (Default)
            self.config = {
                "buy_threshold": 0.30,
                "sell_threshold": 0.24,
                "two_strategy_bonus": 0.25,
                "three_strategy_bonus": 0.30,
                "four_strategy_bonus": 0.35,
                "bull_buy_boost": 0.22,
                "bull_sell_penalty": 0.15,
                "bear_sell_boost": 0.22,
                "bear_buy_penalty": 0.28,
                "min_confidence_to_use": 0.06,
                "min_signal_quality": 0.25,
                "hold_contribution_pct": 0.0,
                "opposition_penalty": 0.40,
            }
        
        # 2. Update with passed config (Merge instead of Overwrite)
        if config is not None:
            # This ensures keys missing from 'config' are filled by defaults above
            self.config.update(config)

        # 3. Wire strategy weights from merged config (T1.2 fix)
        # Previously hardcoded to 0.50/0.50, ignoring mean_reversion_weight: 0.0 in presets.
        # EMA weight now included so all three strategies contribute to consensus scoring.
        self.weights = {
            "mean_reversion": self.config.get("mean_reversion_weight", 0.50),
            "trend_following": self.config.get("trend_following_weight", 0.50),
            "ema": self.config.get("ema_weight", 0.40),
        }
        logger.info(
            f"[AGGREGATOR] Strategy weights: MR={self.weights['mean_reversion']:.2f}, "
            f"TF={self.weights['trend_following']:.2f}, "
            f"EMA={self.weights['ema']:.2f}"
        )

        # 4. Independent strategy thresholds (T1.1 fix)
        # allow_single_override and single_override_threshold exist in presets but were
        # never read by this class — orphaned config keys. Now wired.
        self.independent_thresholds = {
            "trend_following": self.config.get("single_override_threshold", 0.72),
            "mean_reversion": self.config.get("single_override_threshold", 0.75),
            "ema": self.config.get("single_override_threshold", 0.72),
        }
        self.allow_independent = self.config.get("allow_single_override", True)
        logger.info(
            f"[AGGREGATOR] Independent firing: {'ENABLED' if self.allow_independent else 'DISABLED'} "
            f"(TF≥{self.independent_thresholds['trend_following']:.2f}, "
            f"MR≥{self.independent_thresholds['mean_reversion']:.2f})"
        )

        self.stats = {
            "total_evaluations": 0,
            "signals_generated": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "bull_regime_count": 0,
            "bear_regime_count": 0,
            "regime_changes": 0,
            "consensus_signals": 0,
            "single_strategy_signals": 0,
            "regime_detection_failures": 0,
        }

        # T1.5: Stale price detection state
        # Tracks (last_price, last_change_time) per asset to catch frozen data feeds.
        # MT5 assets trade market hours only — 90 min avoids false stale alerts
        # across the overnight close gap. Crypto is 24/7 so 30 min is tight enough.
        self._last_prices = {}
        self._stale_threshold_minutes = 65   # default — exceeds 1H candle duration
        self._stale_thresholds = {           # per-asset overrides (MT5 = 90 min, crypto = default 65)
            "GOLD":   90,
            "USTEC":  90,
            "EURUSD": 90,
            "EURJPY": 90,
            "USOIL":  90,
            "GBPAUD": 90,
            "GBPUSD": 90,
            "USDJPY": 90,
        }

        # T3.4: Economic calendar — loaded at startup, hot-reloaded by CalendarUpdater
        self._econ_cal_path = "config/economic_calendar.json"
        self._econ_events = []
        self._load_calendar_file()

        # ── CONTEXT ENGINE: new infrastructure ──────────────────────────────
        # B.3: Dynamic thresholds
        from src.utils.dynamic_thresholds import DynamicThresholds
        self.dynamic_thresholds = DynamicThresholds(lookback=100, min_samples=5)

        # D.1: Trend Lifecycle tracking
        self._previous_regime = {}    # {asset: regime_name}
        self._regime_start_time = {}  # {asset: datetime}
        self._regime_durations = {}   # {asset: [list of durations in hours]}
        self._transition_counts = {}  # {asset: {(from, to): count}}

        # E.2: MTF Structure Memory
        self._structure_levels = {}   # {asset: [{price, tf, type, age_hours, tests}]}

        # G.1: Liquidity sweep tracking
        self._pdh = {}         # {asset: price}
        self._pdl = {}         # {asset: price}
        self._asian_high = {}  # {asset: price}
        self._asian_low = {}   # {asset: price}
        self._pdh_date = None

        # G.5: Last loss tracking (populated externally by trade result callback)
        self._last_loss_time = {}  # {asset: datetime}

        # E.5: Squeeze state tracking
        self._squeeze_was_active = {}  # {asset: bool}

        # B.2: State cache slots (populated in get_aggregated_signal)
        self._cached_composite = None
        self._last_state_candle_time = None

        # F.7: Spread history for MT5 assets (per asset, last 20 values)
        self._spread_history = {}

        # B.4: State persistence — survive restarts
        self._state_persistence_path = "data/aggregator_state.json"
        self._load_persisted_state()
        # ────────────────────────────────────────────────────────────────────

        # Regime transition evidence collector (SLIGHTLY regimes only)
        self._transition_detector = TransitionDetector()

        self._log_initialization()

    # ── B.4: State Persistence ───────────────────────────────────────────────

    def _load_persisted_state(self):
        """Load cached state from disk to survive restarts."""
        try:
            import json, os
            if not os.path.exists(self._state_persistence_path):
                logger.info("[STATE] No persisted state file found — starting fresh.")
                return

            with open(self._state_persistence_path) as f:
                saved = json.load(f)

            # Restore dynamic threshold distributions
            if hasattr(self, 'dynamic_thresholds'):
                for key_str, values in saved.get("threshold_cache", {}).items():
                    parts = key_str.split("|")
                    if len(parts) == 2:
                        self.dynamic_thresholds._cache[tuple(parts)] = values

            # Restore structure memory
            self._structure_levels = saved.get("structure_levels", {})

            # Restore regime tracking
            self._previous_regime = saved.get("previous_regime", {})
            self._regime_start_time = {}
            for k, v in saved.get("regime_start_times", {}).items():
                try:
                    from datetime import datetime as _dtp
                    self._regime_start_time[k] = _dtp.fromisoformat(v)
                except Exception:
                    pass
            self._regime_durations = saved.get("regime_durations", {})
            
            # Restore transition counts (convert string keys back to tuples)
            saved_tc = saved.get("transition_counts", {})
            self._transition_counts = {}
            for asset, counts in saved_tc.items():
                self._transition_counts[asset] = {}
                if isinstance(counts, dict):
                    for k_str, v in counts.items():
                        if "|" in k_str:
                            parts = k_str.split("|")
                            self._transition_counts[asset][tuple(parts)] = v
                        else:
                            self._transition_counts[asset][k_str] = v

            # Restore sweep levels
            self._pdh = saved.get("pdh", {})
            self._pdl = saved.get("pdl", {})
            self._asian_high = saved.get("asian_high", {})
            self._asian_low = saved.get("asian_low", {})

            # Restore squeeze tracking
            self._squeeze_was_active = saved.get("squeeze_was_active", {})

            # Restore spread history (F.7)
            self._spread_history = saved.get("spread_history", {})

            _n_levels = sum(len(v) for v in self._structure_levels.values()
                            if isinstance(v, (list, dict)))
            _n_thresh = len(saved.get("threshold_cache", {}))
            logger.info(
                f"[STATE] ✅ Loaded persisted state: "
                f"{_n_levels} structure levels, "
                f"{_n_thresh} threshold distributions, "
                f"{len(self._previous_regime)} regime histories"
            )
        except Exception as e:
            logger.warning(f"[STATE] Could not load persisted state: {e}. Starting fresh.")

    def _persist_state(self):
        """Save critical state to disk. Called once per candle close."""
        try:
            import json, os

            # Convert tuple keys to pipe-separated strings for JSON
            _tc = {}
            if hasattr(self, 'dynamic_thresholds'):
                for key_tuple, values in self.dynamic_thresholds._cache.items():
                    if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
                        _tc[f"{key_tuple[0]}|{key_tuple[1]}"] = list(values)[-100:]

            from datetime import datetime as _dtj
            state_data = {
                "threshold_cache": _tc,
                "structure_levels": getattr(self, '_structure_levels', {}),
                "previous_regime": getattr(self, '_previous_regime', {}),
                "regime_start_times": {
                    k: v.isoformat()
                    for k, v in getattr(self, '_regime_start_time', {}).items()
                },
                "regime_durations": getattr(self, '_regime_durations', {}),
                "transition_counts": {
                    asset: {f"{k[0]}|{k[1]}": v for k, v in counts.items()}
                    for asset, counts in getattr(self, '_transition_counts', {}).items()
                },
                "pdh": getattr(self, '_pdh', {}),
                "pdl": getattr(self, '_pdl', {}),
                "asian_high": getattr(self, '_asian_high', {}),
                "asian_low": getattr(self, '_asian_low', {}),
                "squeeze_was_active": getattr(self, '_squeeze_was_active', {}),
                "spread_history": getattr(self, '_spread_history', {}),
                "saved_at": _dtj.now().isoformat(),
            }

            os.makedirs(os.path.dirname(os.path.abspath(
                self._state_persistence_path)), exist_ok=True)
            tmp = self._state_persistence_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state_data, f, default=str)
            os.replace(tmp, self._state_persistence_path)
            logger.debug("[STATE] Persisted state to disk.")
        except Exception as e:
            logger.warning(f"[STATE] Persist failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────

    def _log_initialization(self):
        """Log configuration on startup"""
        logger.info("=" * 80)
        logger.info(f"🎯   PerformanceWeightedAggregator - {self.asset_type}")
        logger.info("=" * 80)
        logger.info("   ✓ STRICT MODE: Counter-trend trades blocked > 50% conf")
        logger.info("   ✓ RANGING SAFEGUARD: Max 1 trade when trend is weak")
        logger.info("   ✓ DYNAMIC threshold adjustment")
        if self.ai_enabled:
            logger.info("   ✓ AI VALIDATION: Active with circuit breaker")
        else:
            logger.info("   ⚠ AI VALIDATION: Disabled")
        logger.info("=" * 80)

    # ─────────────────────────────────────────────────────────────────────
    # CALENDAR HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _load_calendar_file(self):
        """Load economic events from the JSON file on disk."""
        try:
            import json as _json
            with open(self._econ_cal_path, encoding="utf-8") as _f:
                self._econ_events = _json.load(_f).get("events", [])
            logger.info(
                f"[CALENDAR] Loaded {len(self._econ_events)} events "
                f"from {self._econ_cal_path}"
            )
        except Exception as _e:
            logger.warning(f"[CALENDAR] Could not load {self._econ_cal_path}: {_e}")
            self._econ_events = []

    def reload_calendar(self):
        """
        Hot-reload the economic calendar from disk.
        Called by CalendarUpdater after each successful write so the
        aggregator picks up fresh data without a bot restart.
        """
        self._load_calendar_file()
        logger.info(
            f"[CALENDAR] 🔄 Hot-reloaded — "
            f"{len(self._econ_events)} active events in memory"
        )

    # ══════════════════════════════════════════════════════════════════════
    # CONTEXT ENGINE — Build composite state (called once per candle close)
    # ══════════════════════════════════════════════════════════════════════

    def _build_composite_state(self, df, df_4h, governor_data: dict):
        """Build a fresh CompositeState from closed-candle data."""
        from src.execution.composite_state import CompositeState
        import talib as ta
        from datetime import datetime
        state = CompositeState()

        if df is None or len(df) < 20:
            return state

        # ── D.3: Session Context ──────────────────────────────────────────
        try:
            _hour = datetime.utcnow().hour
            _dow = datetime.utcnow().weekday()
            if 0 <= _hour < 8:
                state.session_name = "ASIAN"
            elif 8 <= _hour < 12:
                state.session_name = "LONDON"
            elif 12 <= _hour < 17:
                state.session_name = "OVERLAP"
            elif 17 <= _hour < 21:
                state.session_name = "NY_CLOSE"
            else:
                state.session_name = "OFF_HOURS"
            state.is_friday_pm = (_dow == 4 and _hour >= 15)
        except Exception:
            pass

        # ── D.2: MTF Slope Agreement ──────────────────────────────────────
        try:
            _ema50_1h = df['close'].ewm(span=50, adjust=False).mean()
            _slope_1h = (_ema50_1h.iloc[-1] - _ema50_1h.iloc[-6]) / max(_ema50_1h.iloc[-6], 1)

            if df_4h is not None and len(df_4h) >= 10:
                _ema50_4h = df_4h['close'].ewm(span=50, adjust=False).mean()
                _slope_4h = (_ema50_4h.iloc[-1] - _ema50_4h.iloc[-6]) / max(_ema50_4h.iloc[-6], 1)

                state.slopes_aligned = (_slope_1h > 0 and _slope_4h > 0) or \
                                       (_slope_1h < 0 and _slope_4h < 0)
                state.slope_diverging = not state.slopes_aligned

            # Structural decay = old regime + slopes fighting
            if state.regime_age_ratio > 1.5 and state.slope_diverging:
                state.structural_decay = True
        except Exception:
            pass

        # ── Shared ATR (used by multiple sub-modules) ─────────────────────
        _atr = 0.0
        try:
            _atr_arr = ta.ATR(df['high'].values, df['low'].values,
                              df['close'].values, timeperiod=14)
            _atr = float(_atr_arr[-1]) if not np.isnan(_atr_arr[-1]) else 0.0
        except Exception:
            pass

        # ── E.1: ChoCh / BOS Detection ────────────────────────────────────
        self._update_structure(state, df)

        # ── E.2: MTF Structure Memory ─────────────────────────────────────
        self._update_structure_memory(state, df, df_4h)

        # ── E.3: MA Defense Validator ─────────────────────────────────────
        self._update_ma_defense(state, df)

        # ── E.4: Parabolic Space (Dynamic Z-Score) ────────────────────────
        try:
            _ema50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            _price = df['close'].iloc[-1]
            _distance = abs(_price - _ema50) / max(_atr, 0.0001)

            _extreme, _z, _thresh = self.dynamic_thresholds.check(
                self.asset_type, "ema50_distance", _distance,
                z_threshold=2.5, fallback=3.5
            )
            state.is_parabolic = _extreme
            state.distance_zscore = _z
        except Exception:
            pass

        # ── E.5: EMA Squeeze (ATR-Normalized) ────────────────────────────
        try:
            if _atr > 0:
                _ema20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
                _ema50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                _ema200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]
                _spread = (max(_ema20, _ema50, _ema200) - min(_ema20, _ema50, _ema200)) / max(_atr, 0.0001)

                state.squeeze_active = _spread < 0.5
                state.squeeze_strength = max(0.0, 1.0 - _spread)

                _prev_squeeze = self._squeeze_was_active.get(self.asset_type, False)
                if _prev_squeeze and not state.squeeze_active and _spread > 1.0:
                    state.coiled_spring = True
                self._squeeze_was_active[self.asset_type] = state.squeeze_active
        except Exception:
            pass

        # ── E.6: Inside/Outside Bar + Failed Breakout ─────────────────────
        try:
            if len(df) >= 4:
                _prev_h = df['high'].iloc[-2]
                _prev_l = df['low'].iloc[-2]
                _curr_h = df['high'].iloc[-1]
                _curr_l = df['low'].iloc[-1]
                _curr_c = df['close'].iloc[-1]

                state.inside_bar = _curr_h <= _prev_h and _curr_l >= _prev_l
                state.outside_bar = _curr_h > _prev_h and _curr_l < _prev_l

                if state.squeeze_active and state.inside_bar:
                    state.coiled_spring = True

                _recent_high = df['high'].iloc[-4:-1].max()
                if _curr_h > _recent_high and _curr_c < _recent_high:
                    state.failed_breakout = True
        except Exception:
            pass

        # ── F.1: Effort vs Result (All Assets) ────────────────────────────
        try:
            _tick_vol = df['volume'].iloc[-1]
            _body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
            _er = _tick_vol / max(_body, 0.0001)

            _extreme, _z, _ = self.dynamic_thresholds.check(
                self.asset_type, "effort_result", _er,
                z_threshold=2.0, fallback=None
            )
            state.effort_result_zscore = _z
            if _extreme and _z > 2.0 and abs(governor_data.get("regime_score", 0)) >= 0.5:
                state.absorption_detected = True
        except Exception:
            pass

        # ── F.2: Candle Body Ratio Trend ─────────────────────────────────
        try:
            _bodies = abs(df['close'] - df['open']).tail(10).values
            if len(_bodies) >= 8:
                _recent = _bodies[-3:].mean()
                _older = _bodies[:5].mean()
                state.body_trend_ratio = _recent / max(_older, 0.0001)
                state.conviction_dying = state.body_trend_ratio < 0.5
        except Exception:
            pass

        # ── F.5: BTC VPD (Volume-Price Divergence) ────────────────────────
        if self.asset_type == "BTC" and 'volume' in df.columns:
            try:
                _vol = df['volume'].iloc[-1]
                _vol_sma = df['volume'].tail(20).mean()
                _regime_score = governor_data.get("regime_score", 0)

                if abs(_regime_score) >= 1.0 and _vol < _vol_sma * 0.80:
                    state.vpd_diverging = True
            except Exception:
                pass

        # ── G.1: Unified Liquidity Sweeps ─────────────────────────────────
        self._update_sweeps(state, df)

        # ── G.2: Rejection Profiling ──────────────────────────────────────
        try:
            if _atr > 0:
                _o = df['open'].iloc[-1]
                _h = df['high'].iloc[-1]
                _l = df['low'].iloc[-1]
                _c = df['close'].iloc[-1]
                _total = _h - _l
                if _total > 0:
                    _upper_wick = _h - max(_o, _c)
                    _lower_wick = min(_o, _c) - _l
                    _wick_ratio = max(_upper_wick, _lower_wick) / _total

                    if _wick_ratio > 0.75 and state.nearby_4h_level is not None:
                        _dist_to_level = abs(_c - state.nearby_4h_level) / max(_atr, 0.001)
                        if _dist_to_level < 0.5:
                            state.rejection_at_level = True
                            state.rejection_strength = _wick_ratio
                            state.level_defended = True
        except Exception:
            pass

        # ── G.3: Session VWAP ─────────────────────────────────────────────
        try:
            if 'volume' in df.columns and _atr > 0:
                _midnight_mask = df.index.hour == 0
                if _midnight_mask.any():
                    _session_start = df[_midnight_mask].index[-1]
                else:
                    _session_start = df.index[0]
                _session = df[df.index >= _session_start]
                if len(_session) > 1:
                    _vwap = (_session['close'] * _session['volume']).cumsum() / \
                            _session['volume'].cumsum()
                    state.vwap_price = float(_vwap.iloc[-1])
                    state.distance_to_vwap_atr = abs(df['close'].iloc[-1] - state.vwap_price) / max(_atr, 0.001)
        except Exception:
            pass

        # ── G.5: Time since last loss ─────────────────────────────────────
        _last_loss = self._last_loss_time.get(self.asset_type)
        if _last_loss:
            from datetime import datetime as _dt2
            state.time_since_last_loss_hours = (_dt2.now() - _last_loss).total_seconds() / 3600

        # ── F.4: BTC CVD from WebSocket (injected via governor_data) ─────
        if self.asset_type in ("BTC", "BTCUSDT") and governor_data:
            state.cvd_trend = int(governor_data.get("cvd_trend", 0))
            state.cvd_stale = bool(governor_data.get("cvd_stale", True))
            # ── F.6: L2 Order Book Imbalance ─────────────────────────────
            state.order_book_imbalance = float(governor_data.get("order_book_imbalance", 0.0))
            state.order_book_wall_detected = bool(governor_data.get("order_book_wall_detected", False))

        # ── TRANSITION EVIDENCE (SLIGHTLY regimes only) ───────────────────
        # Must run AFTER CVD/order-book fields are populated above so the
        # order_flow sub-score has live data. df_4h is already available as
        # the second parameter of _build_composite_state.
        state._transition_evidence = None
        _regime_name = governor_data.get("consensus_regime", "UNKNOWN") if governor_data else "UNKNOWN"
        # ✅ M-4 FIX: Also fire transition detector for NEUTRAL+TRANSITION path.
        # EURJPY / EURUSD are often NEUTRAL regime but reach the TRANSITION
        # branch via the Governor. Without this flag they never get transition
        # evidence scoring, wasting the detector entirely for those assets.
        _is_transition_trade = (
            governor_data.get("trade_type", "") == "TRANSITION"
            if governor_data else False
        )
        if _regime_name in ("SLIGHTLY_BEARISH", "SLIGHTLY_BULLISH") or _is_transition_trade:
            try:
                _depth = governor_data.get("depth_data") if governor_data else None
                state._transition_evidence = self._transition_detector.collect_evidence(
                    asset=self.asset_type,
                    regime=_regime_name,
                    df_4h=df_4h if df_4h is not None else pd.DataFrame(),
                    df_1h=df,
                    composite_state=state,
                    cvd_trend=state.cvd_trend,
                    order_book_imbalance=state.order_book_imbalance,
                    depth_data=_depth,
                )
            except Exception as _te_err:
                logger.debug(f"[TRANSITION] Evidence collection error: {_te_err}")
        # ─────────────────────────────────────────────────────────────────

        # ── F.7: MT5 Spread Velocity (synthetic L2 proxy for non-BTC) ────
        if self.asset_type not in ("BTC", "BTCUSDT") and governor_data:
            try:
                _current_spread = governor_data.get("current_spread", 0)
                if _current_spread and _current_spread > 0:
                    if self.asset_type not in self._spread_history:
                        self._spread_history[self.asset_type] = []
                    self._spread_history[self.asset_type].append(_current_spread)
                    if len(self._spread_history[self.asset_type]) > 20:
                        self._spread_history[self.asset_type] =                             self._spread_history[self.asset_type][-20:]
                    _spreads = self._spread_history[self.asset_type]
                    if len(_spreads) >= 10:
                        import numpy as _np
                        _avg = _np.mean(_spreads)
                        state.spread_ratio = float(_current_spread) / max(_avg, 0.0001)
                        state.spread_velocity_spike = state.spread_ratio > 2.5
            except Exception:
                pass

        return state

    # ── D.1: Trend Lifecycle Modifier ────────────────────────────────────

    def _update_trend_lifecycle(self, state, regime_name: str):
        """Classify where in the trend lifecycle this asset sits."""
        from datetime import datetime
        asset = self.asset_type
        now = datetime.now()

        prev = self._previous_regime.get(asset)

        # Detect transition
        if prev and prev != regime_name:
            duration = (now - self._regime_start_time.get(asset, now)).total_seconds() / 3600
            if asset not in self._regime_durations:
                self._regime_durations[asset] = []
            self._regime_durations[asset].append(duration)
            if len(self._regime_durations[asset]) > 50:
                self._regime_durations[asset] = self._regime_durations[asset][-50:]

            trans_key = (prev, regime_name)
            if asset not in self._transition_counts:
                self._transition_counts[asset] = {}
            self._transition_counts[asset][trans_key] = \
                self._transition_counts[asset].get(trans_key, 0) + 1

            self._regime_start_time[asset] = now
            state.transition_type = f"{prev}→{regime_name}"

            if "NEUTRAL" in prev and "SLIGHTLY" in regime_name:
                state.lifecycle_phase = "PICKUP"
            elif "SLIGHTLY" in prev and regime_name in ("BULLISH", "BEARISH"):
                state.lifecycle_phase = "CONFIRMATION"
            elif regime_name in ("BULLISH", "BEARISH") and prev in ("BULLISH", "BEARISH"):
                state.lifecycle_phase = "ESTABLISHED"
            elif prev in ("BULLISH", "BEARISH") and "SLIGHTLY" in regime_name:
                state.lifecycle_phase = "FADING"
            elif prev in ("BULLISH", "BEARISH", "SLIGHTLY_BULLISH", "SLIGHTLY_BEARISH") \
                 and regime_name == "NEUTRAL":
                state.lifecycle_phase = "EXHAUSTION"
            else:
                state.lifecycle_phase = "ESTABLISHED"
        elif prev == regime_name:
            pass  # Same regime — keep current phase, just update age
        else:
            # First observation (after restart or new asset)
            self._regime_start_time[asset] = now
            # ✅ FIX: Default to ESTABLISHED so pattern layer is active immediately
            state.lifecycle_phase = "ESTABLISHED"
            logger.info(f"[LIFECYCLE] {asset} initialized to ESTABLISHED (Startup)")

        self._previous_regime[asset] = regime_name

        # Regime age
        start = self._regime_start_time.get(asset, now)
        state.regime_age_hours = (now - start).total_seconds() / 3600

        # Median regime duration (dynamic per asset)
        durations = self._regime_durations.get(asset, [])
        state.median_regime_duration = float(np.median(durations)) if len(durations) >= 5 else 12.0
        state.regime_age_ratio = state.regime_age_hours / max(state.median_regime_duration, 1.0)

        # Transition probability (Markov)
        if state.transition_type and asset in self._transition_counts:
            total_from_current = sum(
                c for (f, t), c in self._transition_counts[asset].items()
                if f == regime_name
            )
            continues = sum(
                c for (f, t), c in self._transition_counts[asset].items()
                if f == regime_name and t == regime_name
            )
            if total_from_current >= 3:
                state.transition_probability = continues / total_from_current
            else:
                state.transition_probability = 0.5

    # ── E.1: ChoCh / BOS Detection ───────────────────────────────────────

    def _update_structure(self, state, df):
        """Detect Break of Structure and Change of Character using 5-bar swing pivots."""
        try:
            if len(df) < 15:
                return
            highs = df['high'].values
            lows = df['low'].values

            swing_highs = []
            swing_lows = []
            for i in range(len(highs) - 3, 4, -1):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                   highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    swing_highs.append(highs[i])
                    if len(swing_highs) >= 2:
                        break

            for i in range(len(lows) - 3, 4, -1):
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                   lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    swing_lows.append(lows[i])
                    if len(swing_lows) >= 2:
                        break

            if len(swing_highs) >= 2:
                if swing_highs[0] > swing_highs[1]:
                    state.bos_detected = True    # Higher high = trend continuing
                elif swing_highs[0] < swing_highs[1]:
                    state.choch_detected = True  # Lower high = trend may be ending

            if len(swing_lows) >= 2:
                if swing_lows[0] < swing_lows[1] and not state.bos_detected:
                    state.bos_detected = True    # Lower low in downtrend = continuation
                elif swing_lows[0] > swing_lows[1] and not state.choch_detected:
                    state.choch_detected = True  # Higher low in downtrend = reversal
        except Exception:
            pass

    # ── E.2: MTF Structure Memory ─────────────────────────────────────────

    def _update_structure_memory(self, state, df, df_4h):
        """Track 4H swing levels. Delete broken ones. Link to state."""
        import talib as ta
        asset = self.asset_type
        if asset not in self._structure_levels:
            self._structure_levels[asset] = []

        try:
            current_price = df['close'].iloc[-1]
            _atr_arr = ta.ATR(df['high'].values, df['low'].values,
                              df['close'].values, timeperiod=14)
            _atr = float(_atr_arr[-1])
            if np.isnan(_atr) or _atr <= 0:
                return

            # Garbage collection: remove broken and stale levels
            self._structure_levels[asset] = [
                lvl for lvl in self._structure_levels[asset]
                if abs(current_price - lvl["price"]) / _atr <= 0.5
                and lvl.get("age_hours", 0) < 336
            ]

            # Age all levels
            for lvl in self._structure_levels[asset]:
                lvl["age_hours"] = lvl.get("age_hours", 0) + 1

            # Add new 4H swing points if available
            if df_4h is not None and len(df_4h) >= 10:
                _4h_highs = df_4h['high'].values
                _4h_lows = df_4h['low'].values
                for i in range(len(_4h_highs) - 3, 4, -1):
                    if _4h_highs[i] > _4h_highs[i-1] and _4h_highs[i] > _4h_highs[i+1]:
                        _exists = any(
                            abs(lvl["price"] - _4h_highs[i]) / _atr < 0.3
                            for lvl in self._structure_levels[asset]
                        )
                        if not _exists:
                            self._structure_levels[asset].append({
                                "price": _4h_highs[i], "tf": "4H",
                                "type": "swing_high", "tests": 0, "age_hours": 0
                            })
                        break

            # Find nearest level to current price
            nearest = None
            nearest_dist = float('inf')
            for lvl in self._structure_levels[asset]:
                dist = abs(current_price - lvl["price"]) / _atr
                if dist < nearest_dist and dist < 2.0:
                    nearest = lvl
                    nearest_dist = dist

            if nearest:
                state.nearby_4h_level = nearest["price"]
                state.level_test_count = nearest.get("tests", 0)
                if nearest_dist < 0.3:
                    nearest["tests"] = nearest.get("tests", 0) + 1
        except Exception:
            pass

    # ── E.3: MA Defense Validator ─────────────────────────────────────────

    def _update_ma_defense(self, state, df):
        """Check if key EMAs were tested and defended on this closed candle."""
        try:
            candle = df.iloc[-1]
            _o = candle['open']
            _h = candle['high']
            _l = candle['low']
            _c = candle['close']
            _ema50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]

            _pierced_from_above = _l < _ema50 < _c  # Wick below, closed above
            _broke_down = _c < _ema50 and _o > _ema50

            if _pierced_from_above:
                state.ema_50_status = "DEFENDED"
                _wick = _ema50 - _l
                _body = abs(_c - _o)
                state.defense_strength = min(1.0, _wick / max(_body, 0.0001) / 3.0)

                if state.defense_strength > 0.5 and state.effort_result_zscore > 1.5:
                    state.ema_50_reclassified = "SUPPORT"
                    state.absorption_detected = True
                else:
                    state.ema_50_reclassified = "LINE"
            elif _broke_down:
                state.ema_50_status = "BROKEN"
                state.ema_50_reclassified = "RESISTANCE"
            else:
                state.ema_50_status = "UNTESTED"

            # Fix #15: MA Defense diagnostics
            logger.debug(
                f"[MA_DEFENSE] {self.asset_type}: "
                f"status={state.ema_50_status} "
                f"reclassified={state.ema_50_reclassified} "
                f"dist_to_50={abs(_c - _ema50):.4f} "
                f"defense_strength={state.defense_strength:.2f}"
            )
        except Exception:
            pass

    # ── G.1: Unified Liquidity Sweeps ────────────────────────────────────

    def _update_sweeps(self, state, df):
        """Check for PDH/PDL or Asian range sweeps (wicked through, closed back)."""
        asset = self.asset_type
        try:
            from datetime import datetime
            _h = df['high'].iloc[-1]
            _l = df['low'].iloc[-1]
            _c = df['close'].iloc[-1]
            _hour = datetime.utcnow().hour

            # Update Asian range (00:00-08:00 UTC)
            if 0 <= _hour < 8:
                self._asian_high[asset] = max(self._asian_high.get(asset, 0), _h)
                self._asian_low[asset] = min(self._asian_low.get(asset, float('inf')), _l)

            # Update PDH/PDL daily
            _today = datetime.utcnow().date()
            if self._pdh_date != _today:
                if len(df) > 24:
                    _yesterday = df.iloc[-25:-1]
                    self._pdh[asset] = _yesterday['high'].max()
                    self._pdl[asset] = _yesterday['low'].min()
                self._pdh_date = _today

            _asian_h = self._asian_high.get(asset)
            _asian_l = self._asian_low.get(asset)
            _pdh_val = self._pdh.get(asset)
            _pdl_val = self._pdl.get(asset)

            # Swept high = wicked above, closed below
            if _pdh_val and _h > _pdh_val and _c < _pdh_val:
                state.sweep_detected = True
                state.sweep_direction = 1
                state.sweep_level = _pdh_val
            elif _pdl_val and _l < _pdl_val and _c > _pdl_val:
                state.sweep_detected = True
                state.sweep_direction = -1
                state.sweep_level = _pdl_val
            elif _asian_h and 8 <= _hour <= 10 and _h > _asian_h and _c < _asian_h:
                state.sweep_detected = True
                state.sweep_direction = 1
                state.sweep_level = _asian_h
            elif _asian_l and 8 <= _hour <= 10 and _l < _asian_l and _c > _asian_l:
                state.sweep_detected = True
                state.sweep_direction = -1
                state.sweep_level = _asian_l
        except Exception:
            pass

    # ── Section I: Confluence Engine ─────────────────────────────────────

    def _score_confluence(self, state, tf_conf: float, mr_conf: float):
        """
        The Brain. Reads the complete state and applies adjustments
        based on PATTERNS first, individual evidence second.
        """

        # ─── STEP 1: INSTITUTIONAL PATTERN RECOGNITION ───────────────────

        # ─── PATTERN DIAGNOSTICS (Issue 1, Step 1) ──────────────────────
        # Shows exactly which conditions pass/fail for each pattern.
        # MISSING fields = upstream module not writing to CompositeState (bug).
        # ❌ fields = market conditions don't match (working as intended).
        _diag_fields = {
            "lifecycle_phase": state.lifecycle_phase,
            "regime_age_ratio": f"{state.regime_age_ratio:.2f}",
            "choch_detected": state.choch_detected,
            "structural_decay": state.structural_decay,
            "absorption_detected": state.absorption_detected,
            "conviction_dying": state.conviction_dying,
            "distance_zscore": f"{state.distance_zscore:.2f}",
            "bos_detected": state.bos_detected,
            "slopes_aligned": state.slopes_aligned,
            "sweep_detected": state.sweep_detected,
            "rejection_at_level": state.rejection_at_level,
            "effort_result_zscore": f"{state.effort_result_zscore:.2f}",
            "outside_bar": state.outside_bar,
            "failed_breakout": state.failed_breakout,
            "coiled_spring": state.coiled_spring,
            "ema_50_status": state.ema_50_status,
            "ema_50_reclassified": state.ema_50_reclassified,
        }
        logger.info(
            f"[PATTERN DIAG] {self.asset_type} state: "
            + " | ".join(f"{k}={v}" for k, v in _diag_fields.items())
        )

        # Evaluate each pattern and log which condition blocks it
        _dist_checks = {
            "phase∈ESTABLISHED/FADING": state.lifecycle_phase in ("ESTABLISHED", "FADING"),
            f"age_ratio>{1.3}": state.regime_age_ratio > 1.3,
            "choch_or_decay": state.choch_detected or state.structural_decay,
            "absorption_or_dying": state.absorption_detected or state.conviction_dying,
            f"dist_z>{1.5}": state.distance_zscore > 1.5,
        }
        _accum_checks = {
            "phase∈PICKUP/CONFIRM": state.lifecycle_phase in ("PICKUP", "CONFIRMATION"),
            f"age_ratio<{0.8}": state.regime_age_ratio < 0.8,
            "bos_detected": state.bos_detected,
            "slopes_aligned": state.slopes_aligned,
            "no_absorption": not state.absorption_detected,
        }
        _liq_checks = {
            "sweep_detected": state.sweep_detected,
            "rejection_at_level": state.rejection_at_level,
            f"effort_z>{2.0}": state.effort_result_zscore > 2.0,
            "outside_or_failed": state.outside_bar or state.failed_breakout,
        }
        _spring_checks = {
            "coiled_spring": state.coiled_spring,
            "bos_detected": state.bos_detected,
            "slopes_aligned": state.slopes_aligned,
        }
        _ma_checks = {
            "ema50=DEFENDED": state.ema_50_status == "DEFENDED",
            "ema50=SUPPORT": state.ema_50_reclassified == "SUPPORT",
            "phase∈CONFIRM/ESTAB": state.lifecycle_phase in ("CONFIRMATION", "ESTABLISHED"),
            f"age_ratio<{1.5}": state.regime_age_ratio < 1.5,
        }

        for _pname, _pchecks in [
            ("DISTRIBUTION", _dist_checks),
            ("ACCUMULATION", _accum_checks),
            ("LIQUIDITY_HUNT", _liq_checks),
            ("SPRING_BREAKOUT", _spring_checks),
            ("MA_DEFENSE", _ma_checks),
        ]:
            _passed = sum(_pchecks.values())
            _total = len(_pchecks)
            _blocker = next((k for k, v in _pchecks.items() if not v), None)
            _status = "✅ MATCHED" if all(_pchecks.values()) else f"❌ {_passed}/{_total}"
            logger.info(
                f"[PATTERN CHECK] {self.asset_type} {_pname}: {_status}"
                + (f" — first blocker: {_blocker}" if _blocker else "")
            )

        # PATTERN A: Institutional Distribution
        if all(_dist_checks.values()):
            tf_conf *= 0.45
            mr_conf *= 1.25
            state.institutional_pattern = "DISTRIBUTION"

        # PATTERN B: Institutional Accumulation
        elif all(_accum_checks.values()):
            tf_conf *= 1.30
            mr_conf *= 0.65
            state.institutional_pattern = "ACCUMULATION"

        # PATTERN C: Liquidity Hunt → Reversal
        elif all(_liq_checks.values()):
            mr_conf *= 1.35
            tf_conf *= 0.60
            state.institutional_pattern = "LIQUIDITY_HUNT"

        # PATTERN D: Coiled Spring Breakout
        elif (state.coiled_spring and state.bos_detected and state.slopes_aligned):
            tf_conf *= 1.25
            mr_conf *= 0.70
            state.institutional_pattern = "SPRING_BREAKOUT"

        # PATTERN E: MA Defense → Continuation
        elif (state.ema_50_status == "DEFENDED" and
              state.ema_50_reclassified == "SUPPORT" and
              state.lifecycle_phase in ("CONFIRMATION", "ESTABLISHED") and
              state.regime_age_ratio < 1.5):
            tf_conf *= 1.20
            state.institutional_pattern = "MA_DEFENSE"

        # ─── STEP 2: ADDITIVE CONFLUENCE (fallback if no pattern matched) ─
        else:
            state.institutional_pattern = None

            _exhaust = 0.0
            if state.choch_detected:            _exhaust += 2.0
            if state.is_parabolic:              _exhaust += 1.5
            if state.divergence_detected:       _exhaust += state.divergence_strength * 2
            if state.regime_age_ratio > 1.5:    _exhaust += min(2.0, state.regime_age_ratio - 1.5)
            if state.conviction_dying:          _exhaust += 1.0
            if state.structural_decay:          _exhaust += 1.5
            if state.absorption_detected:       _exhaust += 1.0
            if state.vpd_diverging:             _exhaust += 1.5
            # F.6: Order book wall blocking the signal direction
            if state.order_book_wall_detected:
                _tf_signal = 1 if tf_conf > 0 else -1  # approximate direction
                if (_tf_signal == 1 and state.order_book_imbalance < -0.5):
                    _exhaust += 1.5   # Sell wall blocking longs
                elif (_tf_signal == -1 and state.order_book_imbalance > 0.5):
                    _exhaust += 1.5   # Buy wall blocking shorts
            # F.7: Widening spread = liquidity withdrawal = volatility warning
            if state.spread_velocity_spike:
                _exhaust += 1.0
            if state.ai_reversal_probability > 0.75: _exhaust += 2.0
            if state.outside_bar:               _exhaust += 0.5

            _confirm = 0.0
            if state.bos_detected:              _confirm += 2.0
            if state.slopes_aligned:            _confirm += 1.0
            if state.lifecycle_phase == "PICKUP":        _confirm += 1.5
            if state.lifecycle_phase == "CONFIRMATION":  _confirm += 1.0
            if state.squeeze_active:            _confirm += 0.5
            if state.ema_50_status == "DEFENDED":        _confirm += 1.0
            if state.cvd_trend != 0 and not state.cvd_stale: _confirm += 1.0
            if state.level_defended:            _confirm += 1.5

            state.exhaustion_score = _exhaust
            state.confirmation_score = _confirm
            _net = _confirm - _exhaust
            state.net_conviction = _net

            if _net > 0:
                _boost = min(1.35, 1.0 + (_net * 0.05))
                tf_conf *= _boost
            elif _net < 0:
                _discount = max(0.40, 1.0 + (_net * 0.07))
                tf_conf *= _discount
                if _net < -3:
                    mr_conf *= min(1.30, 1.0 + (abs(_net) - 3) * 0.08)

        # ─── STEP 3: TRANSITION PROBABILITY MODIFIER ─────────────────────
        if state.transition_probability < 0.35:
            tf_conf *= 0.85
            mr_conf *= 0.85
        elif state.transition_probability > 0.70:
            tf_conf *= 1.10

        # Friday PM flag for VTM
        state.friday_tighten = state.is_friday_pm

        logger.info(
            f"[CONFLUENCE] {self.asset_type}: Phase={state.lifecycle_phase} "
            f"Pattern={state.institutional_pattern} "
            f"Exhaust={state.exhaustion_score:.1f} Confirm={state.confirmation_score:.1f} "
            f"Net={state.net_conviction:.1f} "
            f"TF={tf_conf:.3f} MR={mr_conf:.3f}"
        )

        # ✅ M-1 FIX: Confluence multipliers (1.10–1.30×) can push confidence
        # above 1.0, making downstream percentage calculations nonsensical.
        tf_conf = max(0.0, min(1.0, tf_conf))
        mr_conf = max(0.0, min(1.0, mr_conf))

        return tf_conf, mr_conf, state

    def get_statistics(self) -> Dict:
        """Return comprehensive statistics"""
        total = max(self.stats["total_evaluations"], 1)
        base_stats = {
            **self.stats,
            "signal_rate": (self.stats["signals_generated"] / total) * 100,
            "buy_rate": (self.stats["buy_signals"] / total) * 100,
            "sell_rate": (self.stats["sell_signals"] / total) * 100,
            "bull_regime_pct": (self.stats["bull_regime_count"] / total) * 100,
            "bear_regime_pct": (self.stats["bear_regime_count"] / total) * 100,
        }

        # Add AI statistics
        if self.ai_enabled and hasattr(self, "ai_stats"):
            mr_total = self.ai_stats["mr_signals_checked"]
            tf_total = self.ai_stats["tf_signals_checked"]

            base_stats["ai_validation"] = {
                "enabled": True,
                "circuit_breaker_active": self.ai_bypass_active,
                "mr_checked": mr_total,
                "mr_approved": self.ai_stats["mr_approved"],
                "mr_rejected": self.ai_stats["mr_rejected"],
                "mr_rejection_rate": (
                    (self.ai_stats["mr_rejected"] / mr_total * 100)
                    if mr_total > 0
                    else 0
                ),
                "tf_checked": tf_total,
                "tf_approved": self.ai_stats["tf_approved"],
                "tf_rejected": self.ai_stats["tf_rejected"],
                "tf_rejection_rate": (
                    (self.ai_stats["tf_rejected"] / tf_total * 100)
                    if tf_total > 0
                    else 0
                ),
            }
            

        return base_stats

    def _check_ai_circuit_breaker(self) -> bool:
        """
        Check if AI is rejecting too many signals
        Returns True if AI should be bypassed
        """
        if not self.enable_circuit_breaker or len(self.ai_rejection_window) < 20:
            return False

        # Calculate rejection rate (True = rejected, False = approved)
        rejection_rate = sum(self.ai_rejection_window) / len(self.ai_rejection_window)

        if rejection_rate > self.ai_bypass_threshold:
            if not self.ai_bypass_active:
                logger.warning("")
                logger.warning("=" * 70)
                logger.warning("⚠️  AI CIRCUIT BREAKER TRIGGERED")
                logger.warning(
                    f"   Rejection rate: {rejection_rate:.0%} (threshold: {self.ai_bypass_threshold:.0%})"
                )
                logger.warning(
                    f"   AI validation temporarily DISABLED for next 10 signals"
                )
                logger.warning("=" * 70)
                logger.warning("")
                self.ai_bypass_active = True
                self.ai_bypass_cooldown = 10  # Bypass next 10 signals

            return True

        # Check if cooldown expired
        if self.ai_bypass_active and self.ai_bypass_cooldown <= 0:
            logger.info("🔄 AI circuit breaker reset - validation RE-ENABLED")
            self.ai_bypass_active = False
            self.ai_rejection_window.clear()  # Reset tracking

        return self.ai_bypass_active

    def _detect_regime(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Multi-factor regime detection with cold-start handling
        Returns: (is_bull, confidence)
        """
        try:
            MIN_DATA_POINTS = 50

            # ===============================
            # 1️⃣ Cold-start & data sufficiency
            # ===============================
            if len(df) < MIN_DATA_POINTS:
                logger.warning(
                    f"Insufficient data for regime detection: {len(df)} rows"
                )
                self.stats["regime_detection_failures"] += 1

                if self.previous_regime is not None:
                    return self.previous_regime, 0.3

                if len(df) >= 20:
                    recent_momentum = (
                        df["close"].iloc[-1] - df["close"].iloc[-20]
                    ) / df["close"].iloc[-20]
                    emergency_regime = recent_momentum > 0
                    logger.info(
                        f"[REGIME] Emergency mode: {'BULL' if emergency_regime else 'BEAR'} "
                        f"(20-day momentum: {recent_momentum:.2%})"
                    )
                    return emergency_regime, 0.3

                logger.warning(
                    "[REGIME] Insufficient data - defaulting to BEAR (conservative)"
                )
                return False, 0.3

            # ===============================
            # 2️⃣ Feature generation
            # ===============================
            features_df = self.s_ema.generate_features(df.tail(250))
            if features_df.empty or len(features_df) < MIN_DATA_POINTS:
                logger.warning(f"EMA features insufficient: {len(features_df)} rows")
                self.stats["regime_detection_failures"] += 1
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3

            latest = features_df.iloc[-1]

            ema_fast = latest.get("ema_fast", np.nan)
            ema_slow = latest.get("ema_slow", np.nan)
            ema_diff_pct = latest.get("ema_diff_pct", 0.0)

            if pd.isna(ema_fast) or pd.isna(ema_slow):
                logger.warning("Invalid EMA values")
                self.stats["regime_detection_failures"] += 1
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3

            # ====================================================================
            # 3️⃣ Thresholds (Rolling Quantile) - ✅ TASK 21 (Phase 3)
            # ====================================================================
            # Reason: Fixed thresholds fail in different volatility regimes.
            # We use the last 100 bars to find the 65th/35th percentiles.
            ema_diff_series = features_df["ema_diff_pct"].tail(100).dropna()
            
            if len(ema_diff_series) >= 50: # Minimum bars for meaningful quantile
                # Calculate percentiles
                BULLISH_THRESHOLD = ema_diff_series.quantile(0.65)
                BEARISH_THRESHOLD = ema_diff_series.quantile(0.35)
                
                # Clamp to institutional bounds [0.05, 0.40]
                BULLISH_THRESHOLD = max(0.05, min(0.40, BULLISH_THRESHOLD))
                BEARISH_THRESHOLD = min(-0.05, max(-0.40, BEARISH_THRESHOLD))
            else:
                # Fallback to defaults
                BULLISH_THRESHOLD = 0.15 if self.asset_type == "BTC" else 0.10
                BEARISH_THRESHOLD = -0.15 if self.asset_type == "BTC" else -0.10

            close_prices = features_df["close"].values

            ret_20 = (
                (close_prices[-1] - close_prices[-20]) / close_prices[-20]
                if len(close_prices) >= 20
                else 0.0
            )
            ret_50 = (
                (close_prices[-1] - close_prices[-50]) / close_prices[-50]
                if len(close_prices) >= 50
                else 0.0
            )

            if len(close_prices) >= 21:
                returns = np.diff(close_prices[-21:]) / close_prices[-21:-1]
                vol_20 = np.std(returns) * np.sqrt(252)
            else:
                vol_20 = 0.2

            adx = latest.get("adx", 20)
            macd_hist = latest.get("macd_hist", 0)
            rsi = latest.get("rsi", 50)
            
            # Asset-specific ADX threshold
            adx_threshold = getattr(self.s_trend_following, 'adx_threshold', 25)

            # ===============================
            # 4️⃣ Multi-factor scoring
            # ===============================
            bullish_score = 0
            bearish_score = 0

            # EMA positioning (dominant factor)
            if ema_diff_pct > BULLISH_THRESHOLD:
                bullish_score += 3
            elif ema_diff_pct < BEARISH_THRESHOLD:
                bearish_score += 3

            # Short-term momentum
            if ret_20 > 0.02:
                bullish_score += 2
            elif ret_20 < -0.02:
                bearish_score += 2

            # Medium-term momentum
            if ret_50 > 0.05:
                bullish_score += 2
            elif ret_50 < -0.05:
                bearish_score += 2

            # MACD
            if macd_hist > 0:
                bullish_score += 1
            elif macd_hist < 0:
                bearish_score += 1

            # ADX trend strength
            if adx > adx_threshold:
                if ema_diff_pct > 0:
                    bullish_score += 1
                else:
                    bearish_score += 1

            # RSI
            if rsi > 60:
                bullish_score += 1
            elif rsi < 40:
                bearish_score += 1

            # ===============================
            # 5️⃣ Hysteresis-based decision
            # ===============================
            if self.previous_regime is None:
                is_bull = bullish_score > bearish_score
            else:
                if self.previous_regime:
                    is_bull = not (bearish_score > bullish_score + 2)
                else:
                    is_bull = bullish_score > bearish_score + 2

            # ===============================
            # 6️⃣ Confidence scoring
            # ===============================
            confidence = 0.5

            if abs(ema_diff_pct) > 0.5:
                confidence += 0.15

            if (is_bull and ret_20 > 0.03) or (not is_bull and ret_20 < -0.03):
                confidence += 0.15

            if adx > 25:
                confidence += 0.1

            if abs(bullish_score - bearish_score) >= 4:
                confidence += 0.1

            confidence = min(1.0, max(0.3, confidence))

            # ===============================
            # 7️⃣ Logging & stats
            # ===============================
            if self.previous_regime is not None and self.previous_regime != is_bull:
                self.stats["regime_changes"] += 1
                logger.info(
                    f"⚡ REGIME FLIP → {'BULL' if is_bull else 'BEAR'} | "
                    f"Scores B:{bullish_score} / R:{bearish_score} | "
                    f"Confidence: {confidence:.2f}"
                )

            elif not self.regime_initialized:
                logger.info(
                    f"🎬 INITIAL REGIME → {'BULL' if is_bull else 'BEAR'} | "
                    f"Confidence: {confidence:.2f}"
                )
                self.regime_initialized = True

            self.previous_regime = is_bull
            if is_bull:
                self.stats["bull_regime_count"] += 1
            else:
                self.stats["bear_regime_count"] += 1

            return is_bull, confidence

        # ======================================================
        # 8️⃣ HARD FALLBACK: EMA-only regime detection
        # ======================================================
        except Exception as e:
            logger.error(f"Primary regime detection failed: {e}", exc_info=True)
            self.stats["regime_detection_failures"] += 1

            try:
                ema_signal, ema_conf = self.s_ema.generate_signal(df)
                is_bull = ema_signal >= 0

                self.previous_regime = is_bull
                if is_bull:
                    self.stats["bull_regime_count"] += 1
                else:
                    self.stats["bear_regime_count"] += 1

                return is_bull, ema_conf

            except Exception as e:
                logger.error(f"EMA fallback failed: {e}", exc_info=True)
                fallback_regime = (
                    self.previous_regime if self.previous_regime is not None else False
                )
                return fallback_regime, 0.3


    def calculate_regime_adjusted_thresholds(
        self, is_bull: bool, regime_confidence: float
    ) -> Tuple[float, float]:
        """
        Dynamically adjust thresholds based on regime strength
        """
        base_buy = self.config["buy_threshold"]
        base_sell = self.config["sell_threshold"]

        # Fix E: proportional adjustments (percentage of base) instead of fixed offsets.
        # Fixed offsets were regime-blind: a 0.10 offset on a 0.23 scalper threshold
        # is a 43% swing, while the same offset on a 0.33 conservative is only 30%.
        strength = (regime_confidence - 0.5) * 2  # Map 0.5-1.0 to 0.0-1.0
        strength = max(0.0, min(1.0, strength))

        if is_bull:
            # Bull: ease buy gate by up to 18%, tighten sell gate by up to 15%
            adjusted_buy = base_buy * (1.0 - 0.18 * strength)
            adjusted_sell = base_sell * (1.0 + 0.15 * strength)
        else:
            # Bear: tighten buy gate by up to 20%, ease sell gate by up to 18%
            adjusted_buy = base_buy * (1.0 + 0.20 * strength)
            adjusted_sell = base_sell * (1.0 - 0.18 * strength)

        # Safety bounds
        adjusted_buy = max(0.15, min(0.60, adjusted_buy))
        adjusted_sell = max(0.15, min(0.60, adjusted_sell))

        # Log significant changes
        if abs(adjusted_buy - base_buy) > 0.05:
            logger.debug(
                f"[THRESHOLD] Buy: {base_buy:.2f}→{adjusted_buy:.2f} ({'BULL' if is_bull else 'BEAR'}, conf:{regime_confidence:.2f})"
            )

        return adjusted_buy, adjusted_sell

    def _format_ai_validation_for_viz(
        self, final_signal: int, details: dict, df: pd.DataFrame
    ) -> dict:
        """
        CRITICAL FIX: Format AI validation results for visualization
        ✅ FIXED: Proper type conversions for pattern_detected and near_sr_level
        """
        try:
            # Initialize with safe defaults
            viz_data = {
                "pattern_detected": False,  # ← Must be bool
                "validation_passed": False,
                "pattern_name": "None",
                "pattern_id": None,
                "pattern_confidence": 0.0,
                "top3_patterns": [],
                "top3_confidences": [],
                "sr_analysis": {
                    "near_sr_level": False,  # ← Must be bool
                    "level_type": "none",
                    "nearest_level": None,
                    "distance_pct": None,
                    "levels": [],
                    "total_levels_found": 0,
                },
                "action": "none",
                "rejection_reasons": [],
                "error": None,
            }

            # Check if AI validator exists
            if not self.ai_validator or not self.ai_enabled:
                viz_data["action"] = "ai_disabled"
                return viz_data

            current_price = float(df["close"].iloc[-1])

            # ================================================================
            # STEP 1: Get S/R Analysis
            # ================================================================
            try:
                sr_result = self.ai_validator._check_support_resistance_fixed(
                    asset=self.asset_type,
                    df=df,
                    current_price=current_price,
                    signal=final_signal,
                    threshold=self.ai_validator.current_sr_threshold,
                )

                # ✅ FIX: Convert numpy.bool to Python bool
                near_level = sr_result.get("near_level", False)
                if isinstance(near_level, np.bool_):
                    near_level = bool(near_level)

                viz_data["sr_analysis"] = {
                    "near_sr_level": near_level,  # ← Now guaranteed Python bool
                    "level_type": sr_result.get("level_type", "none"),
                    "nearest_level": sr_result.get("nearest_level"),
                    "distance_pct": sr_result.get("distance_pct"),
                    "levels": sr_result.get("all_levels", [])[:5],
                    "total_levels_found": len(sr_result.get("all_levels", [])),
                }

            except Exception as e:
                logger.error(f"[VIZ] S/R analysis failed: {e}")
                viz_data["error"] = f"S/R error: {str(e)}"

            # ================================================================
            # STEP 2: Get Pattern Detection
            # ================================================================
            try:
                pattern_result = self.ai_validator._check_pattern(
                    df=df,
                    signal=final_signal,
                    min_confidence=self.ai_validator.current_pattern_threshold,
                )

                # ✅ FIX: pattern_detected should be BOOL, not string
                pattern_confirmed = pattern_result.get("pattern_confirmed", False)
                pattern_name = pattern_result.get("pattern_name", "None")
                
                # Convert to proper bool
                if isinstance(pattern_confirmed, str):
                    pattern_confirmed = pattern_confirmed not in ["None", "Noise", ""]
                
                viz_data["pattern_detected"] = bool(pattern_confirmed)  # ← Force bool
                viz_data["pattern_name"] = pattern_name  # ← Separate field for name
                viz_data["pattern_id"] = pattern_result.get("pattern_id")
                viz_data["pattern_confidence"] = pattern_result.get("confidence", 0.0)

                # Get top 3 patterns
                if hasattr(self.ai_validator, "sniper") and self.ai_validator.sniper:
                    try:
                        snippet = df[["open", "high", "low", "close"]].iloc[-15:].values
                        first_open = snippet[0, 0]

                        if first_open > 0:
                            snippet_norm = snippet / first_open - 1
                            snippet_input = snippet_norm.reshape(1, 15, 4)

                            predictions = self.ai_validator.sniper.model.predict(
                                snippet_input, verbose=0
                            )[0]

                            top3_indices = predictions.argsort()[-3:][::-1]
                            top3_confidences = predictions[top3_indices]

                            top3_patterns = []
                            for idx in top3_indices:
                                pattern_name = self.ai_validator.reverse_pattern_map.get(
                                    idx, f"Pattern_{idx}"
                                )
                                top3_patterns.append(pattern_name)

                            viz_data["top3_patterns"] = top3_patterns
                            viz_data["top3_confidences"] = top3_confidences.tolist()

                    except Exception as e:
                        logger.debug(f"[VIZ] Top3 patterns failed: {e}")

            except Exception as e:
                logger.error(f"[VIZ] Pattern detection failed: {e}")
                viz_data["error"] = f"Pattern error: {str(e)}"

            # ================================================================
            # STEP 3: Determine Validation Status
            # ================================================================
            original_signal = details.get("original_signal", final_signal)

            if final_signal == 0 and original_signal != 0:
                viz_data["validation_passed"] = False
                viz_data["action"] = "rejected"

                reasons = []
                if not viz_data["sr_analysis"]["near_sr_level"]:
                    reasons.append("No nearby S/R level")
                if not viz_data["pattern_detected"]:
                    reasons.append("No pattern detected")
                if viz_data["pattern_confidence"] < self.ai_validator.current_pattern_threshold:
                    reasons.append(f"Low confidence ({viz_data['pattern_confidence']:.1%})")

                viz_data["rejection_reasons"] = reasons

            elif final_signal != 0:
                viz_data["validation_passed"] = True

                if details.get("ai_bypassed", False):
                    viz_data["action"] = "bypassed"
                elif details.get("signal_quality", 0) >= self.strong_signal_bypass:
                    viz_data["action"] = "bypassed_strong_signal"
                else:
                    viz_data["action"] = "approved"
            else:
                viz_data["action"] = "hold"

            # ================================================================
            # ✅ FINAL TYPE VALIDATION
            # ================================================================
            # Ensure all bools are Python bool, not numpy.bool
            viz_data["pattern_detected"] = bool(viz_data["pattern_detected"])
            viz_data["validation_passed"] = bool(viz_data["validation_passed"])
            viz_data["sr_analysis"]["near_sr_level"] = bool(viz_data["sr_analysis"]["near_sr_level"])

            return viz_data

        except Exception as e:
            logger.error(f"[VIZ] AI formatting failed: {e}", exc_info=True)
            return {
                "pattern_detected": False,
                "validation_passed": False,
                "pattern_name": "ERROR",
                "pattern_confidence": 0.0,
                "top3_patterns": [],
                "top3_confidences": [],
                "sr_analysis": {
                    "near_sr_level": False,
                    "level_type": "none",
                    "nearest_level": None,
                    "distance_pct": None,
                    "levels": [],
                    "total_levels_found": 0,
                },
                "action": "error",
                "error": str(e),
            }

    def _calculate_score(
        self,
        df: pd.DataFrame,
        target_signal: int,
        mr_signal: int,
        mr_conf: float,
        tf_signal: int,
        tf_conf: float,
        ema_signal: int,
        ema_conf: float,
        is_bull: bool,
    ) -> Tuple[float, str, int]:
        """Calculate aggregated score for all three strategies (MR + TF + EMA)."""
        components = []
        total_score = 0.0
        agreement_count = 0
        min_conf = self.config["min_confidence_to_use"]
        hold_contrib = self.config["hold_contribution_pct"]
        opposition_penalty = self.config["opposition_penalty"]

        # Mean Reversion contribution
        if mr_signal == target_signal:
            effective_conf = max(mr_conf, min_conf)
            contribution = effective_conf * self.weights["mean_reversion"]
            total_score += contribution
            components.append(f"MR_agree:{contribution:.3f}")
            agreement_count += 1
        elif mr_signal == 0:
            effective_conf = max(mr_conf, min_conf)
            contribution = (effective_conf * hold_contrib) * self.weights[
                "mean_reversion"
            ]
            total_score += contribution
            components.append(f"MR_hold:{contribution:.3f}")
        else:
            effective_conf = max(mr_conf, min_conf)
            penalty = (effective_conf * opposition_penalty) * self.weights[
                "mean_reversion"
            ]
            total_score -= penalty
            components.append(f"MR_oppose:-{penalty:.3f}")

        # Trend Following contribution
        if tf_signal == target_signal:
            effective_conf = max(tf_conf, min_conf)
            contribution = effective_conf * self.weights["trend_following"]
            total_score += contribution
            components.append(f"TF_agree:{contribution:.3f}")
            agreement_count += 1
        elif tf_signal == 0:
            effective_conf = max(tf_conf, min_conf)
            contribution = (effective_conf * hold_contrib) * self.weights[
                "trend_following"
            ]
            total_score += contribution
            components.append(f"TF_hold:{contribution:.3f}")
        else:
            effective_conf = max(tf_conf, min_conf)
            penalty = (effective_conf * opposition_penalty) * self.weights[
                "trend_following"
            ]
            total_score -= penalty
            components.append(f"TF_oppose:-{penalty:.3f}")

        # EMA contribution (previously excluded — now a full voting member)
        if ema_signal == target_signal:
            effective_conf = max(ema_conf, min_conf)
            contribution = effective_conf * self.weights["ema"]
            total_score += contribution
            components.append(f"EMA_agree:{contribution:.3f}")
            agreement_count += 1
        elif ema_signal == 0:
            effective_conf = max(ema_conf, min_conf)
            contribution = (effective_conf * hold_contrib) * self.weights["ema"]
            total_score += contribution
            components.append(f"EMA_hold:{contribution:.3f}")
        else:
            effective_conf = max(ema_conf, min_conf)
            penalty = (effective_conf * opposition_penalty) * self.weights["ema"]
            total_score -= penalty
            components.append(f"EMA_oppose:-{penalty:.3f}")

        # --- VolumeFlow vote ---
        if self.s_volume_flow is not None:
            try:
                vf_signal, vf_conf = self.s_volume_flow.generate_signal(df)
                if vf_signal == target_signal and vf_conf >= min_conf:
                    effective_conf = max(vf_conf, min_conf)
                    contribution = effective_conf * (1 - self.config.get("opposition_penalty", 0.40))
                    total_score += contribution
                    components.append(f"VF_agree:{contribution:.3f}")
                    agreement_count += 1
                elif vf_signal != 0 and vf_signal != target_signal:
                    penalty = vf_conf * self.config.get("opposition_penalty", 0.40)
                    total_score -= penalty
                    components.append(f"VF_oppose:-{penalty:.3f}")
                elif vf_signal == 0:
                    effective_conf = max(vf_conf, min_conf)
                    hold_contribution = effective_conf * hold_contrib
                    total_score += hold_contribution
                    if hold_contribution > 0:
                        components.append(f"VF_hold:{hold_contribution:.3f}")
            except Exception as _vf_e:
                logger.debug(f"[AGG] VolumeFlow signal error: {_vf_e}")

        explanation = " + ".join(components) if components else "no_agreement"

        # Agreement bonus — tiered (two_strategy_bonus and three_strategy_bonus now both active)
        if agreement_count == 4:
            bonus = self.config.get("four_strategy_bonus", self.config.get("three_strategy_bonus", 0.35))
            total_score += bonus
            explanation += f" + bonus4({bonus:.2f})"
        elif agreement_count == 3:
            bonus = self.config.get("three_strategy_bonus", self.config["two_strategy_bonus"])
            total_score += bonus
            explanation += f" + bonus3({bonus:.2f})"
        elif agreement_count == 2:
            bonus = self.config["two_strategy_bonus"]
            total_score += bonus
            explanation += f" + bonus2({bonus:.2f})"

        # Regime context
        if target_signal == 1:  # BUY
            if is_bull:
                regime_adj = self.config["bull_buy_boost"]
                total_score += regime_adj
                explanation += f" + bull({regime_adj:.2f})"
            else:
                # ✨ NEW: Explosive Momentum Overrule
                if self._is_explosive_momentum(df, target_signal):
                    logger.info("[MOMENTUM] Skipping bear-regime penalty due to explosive BUY momentum")
                    regime_adj = 0
                    explanation += " + V-Shape Overrule"
                else:
                    regime_adj = -self.config["bear_buy_penalty"]
                    total_score = max(0.0, total_score + regime_adj)
                    explanation += f" - bear({abs(regime_adj):.2f})"
        else:  # SELL
            if is_bull:
                # ✨ NEW: Explosive Momentum Overrule
                if self._is_explosive_momentum(df, target_signal):
                    logger.info("[MOMENTUM] Skipping bull-regime penalty due to explosive SELL momentum")
                    regime_adj = 0
                    explanation += " + V-Shape Overrule"
                else:
                    regime_adj = -self.config["bull_sell_penalty"]
                    total_score = max(0.0, total_score + regime_adj)
                    explanation += f" - bull({abs(regime_adj):.2f})"
            else:
                regime_adj = self.config["bear_sell_boost"]
                total_score += regime_adj
                explanation += f" + bear({regime_adj:.2f})"

        total_score = max(0.0, total_score)
        return total_score, explanation, agreement_count
    
    def _check_governor_filter(self, df: pd.DataFrame, signal: int) -> Tuple[bool, Optional[str]]:
        """
        Filter 1: Governor (Daily 200 EMA) Check
        
        Returns:
            (passed, trade_type)
        """
        if not self.use_macro_governor:
            return True, "TREND"

        if not self.enable_filters or not self.mtf_integration:
            return True, "TREND"  # Skip if disabled
        
        try:
            # Get Governor analysis from MTF
            regime_data = self.mtf_integration._current_regime_data.get(self.asset_type)
            
            if not regime_data:
                logger.debug(f"[GOV] No data for {self.asset_type}, allowing trade")
                return True, "TREND"
            
            # ✨ IMPROVED: Robust key check
            governor = regime_data.get('governor') or regime_data.get('full_regime_status')
            
            if not governor:
                logger.debug(f"[GOV] No governor object for {self.asset_type}, allowing trade")
                return True, "TREND"
            
            # ✨ IMPROVED: Handle Enum vs String vs Attribute
            raw_trade_type = getattr(governor, 'trade_type', None)
            if raw_trade_type is None:
                # Fallback to consensus_regime if trade_type is missing
                regime_name = getattr(governor, 'consensus_regime', "NEUTRAL")
                trade_type = "NEUTRAL" if regime_name == "NEUTRAL" else "TREND"
            else:
                trade_type = getattr(raw_trade_type, 'value', str(raw_trade_type))

            # T2.1 fix: NEUTRAL used to block all trading.
            # Simulation: 129 blocked signals at 70.5% WR, +70.2% P&L.
            # NEUTRAL is MR's best environment (+159% P&L, 71% WR).
            # Now returns TRANSITION so trades fire at 50% position size
            # (sizing reduction applied in get_aggregated_signal below).
            if trade_type == "NEUTRAL":
                logger.info("[GOV] ⚠️ TRANSITION — market neutral, allowing at 50% size")
                return True, "TRANSITION"

            return True, trade_type
        
        except Exception as e:
            logger.error(f"[GOV] Error: {e}")
            return True, "TREND"  # Fail-open
    
    def _check_volatility_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Filter 2: Volatility Gate
        
        Returns:
            (passed, atr_pct)
        """
        if not self.enable_filters:
            return True, 0.005
        
        try:
            if len(df) < 20:
                return True, 0.005
            
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            current_price = df['close'].iloc[-1]
            atr_pct = atr / current_price
            
            threshold = self.filter_thresholds['volatility_gate']
            passed = atr_pct >= threshold
            
            if not passed:
                logger.info(f"[VOL] ❌ BLOCKED - ATR {atr_pct:.3%} < {threshold:.3%}")
            
            return passed, atr_pct
        
        except Exception as e:
            logger.error(f"[VOL] Error: {e}")
            return True, 0.005
    
    def _check_sniper_filter(self, df: pd.DataFrame, signal: int, governor_data: Dict = None) -> Tuple[bool, Dict]:
        """
        Filter 3: Sniper Lock - Institutional Edge Confirmation
        =======================================================
        A trade is confirmed if ANY of the following institutional edge conditions are met.
        This prevents rejecting high-quality trades due to cosmetic candle issues.
        
        Confirmation Logic (OR-based):
        1. AI Pattern: A high-confidence AI pattern is detected.
        2. Momentum Candle: The candle body is at least 60% of the total range.
        3. Turtle Breakout: Price closes above the 20-period Donchian High or below the Low.
        4. Volume Surge: Volume is >= 150% of its 20-period rolling average.
        5. Volatility Breach: Price closes outside the 2.0 standard deviation Bollinger Bands.
        6. Trend Momentum: Macro regime and 1H momentum are strong and aligned.
        
        Returns:
            (passed, details)
        """
        if not self.enable_filters:
            return True, {'trigger_type': 'DISABLED'}

        try:
            # Trap filter moved to pre-consensus veto phase

            latest = df.iloc[-1]
            reasons = []

            # ================================================================
            # 1. AI Pattern Confidence
            # ================================================================
            # Reason: The AI model has already encoded a multi-factor edge.
            if self.ai_validator and hasattr(self.ai_validator, 'sniper'):
                pattern_result = self.ai_validator._check_pattern(
                    df=df,
                    signal=signal,
                    min_confidence=self.filter_thresholds['sniper_confidence']
                )
                if pattern_result.get('pattern_confirmed'):
                    reasons.append({
                        'passed': True,
                        'trigger_type': 'AI_PATTERN',
                        'pattern_name': pattern_result.get('pattern_name'),
                        'confidence': pattern_result.get('confidence'),
                    })

            # ================================================================
            # 2. Momentum Candle
            # ================================================================
            # Reason: Confirms strong conviction from buyers or sellers in the current period.
            body = abs(latest['close'] - latest['open'])
            total_range = latest['high'] - latest['low']
            if total_range > 0:
                body_ratio = body / total_range
                if body_ratio >= 0.60:
                    is_bullish_candle = latest['close'] > latest['open']
                    if (signal == 1 and is_bullish_candle) or (signal == -1 and not is_bullish_candle):
                        reasons.append({
                            'passed': True,
                            'trigger_type': 'MOMENTUM_CANDLE',
                            'body_ratio': body_ratio,
                        })

            # ================================================================
            # 3. Trend Momentum (Institutional Continuity)
            # ================================================================
            # Reason: If the macro regime and 1H momentum are both strong and 
            # aligned, we allow entry even without a classic breakout or pattern.
            if governor_data:
                _regime = governor_data.get("regime", "NEUTRAL")
                _is_bull = "BULL" in _regime.upper()
                _is_bear = "BEAR" in _regime.upper()
                _h1_dir = governor_data.get("h1_momentum_dir", "FLAT")
                
                _regime_aligned = (signal == 1 and _is_bull) or (signal == -1 and _is_bear)
                _h1_aligned = (signal == 1 and _h1_dir == "UP") or (signal == -1 and _h1_dir == "DOWN")
                
                if _regime_aligned and _h1_aligned:
                    reasons.append({
                        'passed': True,
                        'trigger_type': 'TREND_MOMENTUM',
                        'regime': _regime,
                        'h1_dir': _h1_dir,
                    })

            # Check if we have enough data for rolling indicators
            if len(df) < 21: # Need 20 periods + current
                if reasons:
                    logger.info(f"[SNIPER] ✅ PASSED - Trigger(s): {[r['trigger_type'] for r in reasons]}")
                    return True, reasons[0]
                else:
                    logger.warning(f"[SNIPER] ❌ BLOCKED - Insufficient data for full institutional checks (need 21 bars, have {len(df)}).")
                    return False, {'trigger_type': None, 'reason': f'Insufficient data for breakouts (have {len(df)})'}

            # ================================================================
            # 4. Turtle Breakout (20-period Donchian Channel)
            # ================================================================
            # Reason: Captures classic institutional breakout entries.
            # We look at the previous 20 candles to define the channel *before* the current candle.
            high_20 = df['high'].iloc[-21:-1].max()
            low_20 = df['low'].iloc[-21:-1].min()

            if signal == 1 and latest['close'] > high_20:
                reasons.append({
                    'passed': True,
                    'trigger_type': 'TURTLE_BREAKOUT',
                    'breakout_level': high_20,
                    'price': latest['close'],
                })
            elif signal == -1 and latest['close'] < low_20:
                reasons.append({
                    'passed': True,
                    'trigger_type': 'TURTLE_BREAKOUT',
                    'breakout_level': low_20,
                    'price': latest['close'],
                })

            # ================================================================
            # 5. Volume Surge
            # ================================================================
            # Reason: Confirms institutional participation and conviction behind a move.
            volume_rolling_avg = df['volume'].iloc[-21:-1].mean()
            if volume_rolling_avg > 0 and latest['volume'] >= (volume_rolling_avg * 1.5):
                reasons.append({
                    'passed': True,
                    'trigger_type': 'VOLUME_SURGE',
                    'volume': latest['volume'],
                    'avg_volume': volume_rolling_avg,
                    'surge_factor': latest['volume'] / volume_rolling_avg if volume_rolling_avg > 0 else 0,
                })

            # ================================================================
            # 6. Volatility Breach (Bollinger Bands)
            # ================================================================
            # Reason: Detects that price has moved into a new volatility regime.
            close_rolling_mean = df['close'].iloc[-21:-1].mean()
            close_rolling_std = df['close'].iloc[-21:-1].std()
            
            if close_rolling_std > 0:
                upper_band = close_rolling_mean + (2.0 * close_rolling_std)
                lower_band = close_rolling_mean - (2.0 * close_rolling_std)

                if signal == 1 and latest['close'] > upper_band:
                    reasons.append({
                        'passed': True,
                        'trigger_type': 'VOLATILITY_BREACH',
                        'band': 'upper',
                        'price': latest['close'],
                    })
                elif signal == -1 and latest['close'] < lower_band:
                    reasons.append({
                        'passed': True,
                        'trigger_type': 'VOLATILITY_BREACH',
                        'band': 'lower',
                        'price': latest['close'],
                    })
            
            # ================================================================
            # 7. Established Trend + BOS Confirmation (Institutional Continuation)
            # ================================================================
            # Reason: When the lifecycle is ESTABLISHED/CONFIRMATION with a fresh
            # Break-of-Structure and aligned slopes, this is a classic institutional
            # trend-continuation setup. It is valid regardless of the macro regime
            # label (which reads NEUTRAL during transitional phases).
            # The TREND_MOMENTUM trigger only fires for explicit BULL/BEAR regimes,
            # so without this trigger these high-quality setups would be silently
            # blocked despite a strong TF signal.
            _cs = getattr(self, '_cached_composite', None)
            if _cs is not None:
                if (
                    _cs.lifecycle_phase in ("CONFIRMATION", "ESTABLISHED")
                    and _cs.bos_detected
                    and _cs.slopes_aligned
                    and not _cs.structural_decay
                    and not _cs.absorption_detected
                    and _cs.regime_age_ratio < 2.0
                ):
                    reasons.append({
                        'passed': True,
                        'trigger_type': 'ESTABLISHED_BOS',
                        'phase': _cs.lifecycle_phase,
                        'age_ratio': round(_cs.regime_age_ratio, 2),
                    })

            # ================================================================
            # Final Decision
            # ================================================================
            if reasons:
                # Log all triggers that passed
                trigger_types = [r['trigger_type'] for r in reasons]
                logger.info(f"[SNIPER] ✅ PASSED - Trigger(s): {trigger_types}")
                # Return the details of the first trigger found
                return True, reasons[0]

            logger.info(f"[SNIPER] ❌ BLOCKED - No institutional edge confirmed.")
            return False, {'trigger_type': None, 'reason': 'No confirmation criteria met'}

        except Exception as e:
            logger.error(f"[SNIPER] Error in institutional edge check: {e}", exc_info=True)
            # Fail-open: If the filter fails, we allow the trade to avoid blocking valid signals due to code errors.
            return True, {'trigger_type': 'ERROR_FALLBACK'}
    
    def _check_profit_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Filter 4: Minimum Profit Potential
        
        Returns:
            (passed, potential_pct)
        """
        if not self.enable_filters:
            return True, 0.01
        
        try:
            if len(df) < 20:
                return True, 0.01
            
            # Use ATR as proxy for potential move
            high_low = df['high'] - df['low']
            atr = high_low.rolling(14).mean().iloc[-1]
            
            current_price = df['close'].iloc[-1]
            potential_pct = atr / current_price
            
            threshold = self.filter_thresholds['min_profit']
            passed = potential_pct >= threshold
            
            if not passed:
                logger.info(f"[PROFIT] ❌ BLOCKED - Potential {potential_pct:.2%} < {threshold:.2%}")
            
            return passed, potential_pct
        
        except Exception as e:
            logger.error(f"[PROFIT] Error: {e}")
            return True, 0.01
    


    def _check_atr_expansion_filter(self, df: pd.DataFrame, trade_type: str) -> bool:
        """
        Fix C: Replaced ATR Expansion (candle_range >= 1.5*ATR) with ADX Trend Confirmation.

        Old logic required the latest candle's range to exceed 1.5× ATR. This blocked
        valid signals in slow-grinding trends (GOLD, EURUSD) where candles are small but
        direction is clear. The 1.5× bar was consistently failing even when ADX showed a
        strong trend (ADX > 25).

        New logic: confirm a trend is in force (ADX > 18). This threshold is intentionally
        low — 18 separates genuine trend from pure noise without demanding strong momentum.
        Counter-trend and REVERSION trades bypass the check (trade_type != "TREND").
        """
        if trade_type != "TREND":
            return True

        try:
            if len(df) < 20:
                return True

            # Calculate ADX (14)
            try:
                import talib
                adx_series = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                adx = adx_series[-1]
            except Exception:
                # Manual ADX fallback: use DM-based approximation via TR rolling
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr14 = tr.rolling(14).mean()
                dm_plus = (df['high'].diff()).clip(lower=0)
                dm_minus = (-df['low'].diff()).clip(lower=0)
                # Use only the dominant direction
                dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
                dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
                di_plus = 100 * dm_plus.rolling(14).mean() / atr14
                di_minus = 100 * dm_minus.rolling(14).mean() / atr14
                dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
                adx = dx.rolling(14).mean().iloc[-1]

            if pd.isna(adx):
                return True

            ADX_MIN = 18
            passed = adx >= ADX_MIN

            if not passed:
                logger.info(f"[ADX_TREND] ❌ BLOCKED - ADX {adx:.1f} < {ADX_MIN} (insufficient trend strength)")
            else:
                logger.debug(f"[ADX_TREND] ✅ PASSED - ADX {adx:.1f}")

            return passed

        except Exception as e:
            logger.error(f"[ADX_TREND] Error: {e}")
            return True  # Fail-open

    def _is_explosive_momentum(self, df: pd.DataFrame, signal: int) -> bool:
        """
        Detects 'V-Shape' or 'Parabolic' price action that overrules macro bias.
        Criteria:
        1. ADX > 30 (Strong immediate trend)
        2. Velocity: Last 6 bars move > 2.0 * ATR14
        3. Alignment: Price > EMA20 > EMA50 (for Longs)
        """
        try:
            if len(df) < 50: return False
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # 1. Trend Strength
            adx = ta.ADX(high, low, close, timeperiod=14)[-1]
            if adx < 30: return False
            
            # 2. ATR-Scaled Velocity
            atr = ta.ATR(high, low, close, timeperiod=14)[-1]
            move = close[-1] - close[-6]
            velocity_ratio = abs(move) / (atr if atr > 0 else 1)
            
            if velocity_ratio < 2.0: return False
            
            # 3. Local Alignment
            ema20 = ta.EMA(close, timeperiod=20)[-1]
            ema50 = ta.EMA(close, timeperiod=50)[-1]
            
            if signal == 1: # Buying into a bear regime
                if move > 0 and close[-1] > ema20 > ema50:
                    return True
            elif signal == -1: # Selling into a bull regime
                if move < 0 and close[-1] < ema20 < ema50:
                    return True
                    
            return False
        except Exception as e:
            logger.debug(f"[MOMENTUM] Overrule check error: {e}")
            return False

    def get_aggregated_signal(
        self,
        df: pd.DataFrame,
        current_regime: str = "NEUTRAL",
        is_bull_market: bool = True,
        governor_data: Dict = None,
        live_price: Optional[float] = None # ✨ NEW: For accurate staleness check
    ) -> Tuple[int, Dict]:
        """
        Main aggregation logic with AI validation and external regime context.
        """
        self.stats["total_evaluations"] += 1
        try:
            timestamp = str(df.index[-1]) if len(df) > 0 else "unknown"

            # AI-5: Clear per-cycle pattern cache so sniper filter and format_viz share results.
            if self.ai_validator and hasattr(self.ai_validator, 'clear_pattern_cache'):
                self.ai_validator.clear_pattern_cache()

            # ═══════════════════════════════════════════════════════════════
            # T1.5: STALE PRICE DETECTION
            # Gold was frozen at 5021.08 for 47.6 hours (March 13–15), firing
            # 12 SELL signals on dead data. Block evaluation if price has not
            # moved by even 1 pip in over 30 minutes.
            # ═══════════════════════════════════════════════════════════════
            from datetime import datetime as _dt
            # Use live_price if provided (from exchange), fallback to last closed bar
            _current_price = live_price if live_price is not None else (float(df["close"].iloc[-1]) if len(df) > 0 else 0.0)
            _now = _dt.now()
            _last = self._last_prices.get(self.asset_type)
            if _last:
                _last_price, _last_time = _last
                _minutes_since_move = (_now - _last_time).total_seconds() / 60
                _price_moved = abs(_current_price - _last_price) / max(_last_price, 1) > 0.00001
                _stale_limit = self._stale_thresholds.get(
                    self.asset_type, self._stale_threshold_minutes
                )
                if not _price_moved and _minutes_since_move > _stale_limit:
                    logger.warning(
                        f"[STALE] ❌ {self.asset_type} price frozen at {_current_price} "
                        f"for {_minutes_since_move:.0f}min — blocking signal evaluation"
                    )
                    return 0, {
                        "timestamp": timestamp,
                        "regime": "UNKNOWN",
                        "reasoning": f"stale_price_{_minutes_since_move:.0f}min",
                        "final_signal": 0,
                        "signal_quality": 0.0,
                        "mr_signal": 0, "mr_confidence": 0.0,
                        "tf_signal": 0, "tf_confidence": 0.0,
                        "ema_signal": 0, "ema_confidence": 0.0,
                    }
            # Update last-seen price only when it actually moves
            if not _last or abs(_current_price - _last[0]) / max(_last[0], 1) > 0.00001:
                self._last_prices[self.asset_type] = (_current_price, _now)

            # ═══════════════════════════════════════════════════════════════
            # B.2: STATE CACHE — Heavy calculations run ONCE per candle close.
            # The 5-second loop reads the cached state for micro-execution checks only.
            # ═══════════════════════════════════════════════════════════════
            _candle_time = df.index[-1] if not df.empty else None
            _state_is_fresh = (
                _candle_time is not None and
                getattr(self, '_last_state_candle_time', None) == _candle_time
            )

            if not _state_is_fresh and _candle_time is not None:
                # New candle closed — rebuild the full composite state
                self._cached_composite = self._build_composite_state(df, governor_data.get('df_4h') if governor_data else None, governor_data or {})
                self._last_state_candle_time = _candle_time
                self._persist_state()
                logger.debug(f"[STATE] Rebuilt composite state for {self.asset_type} at {_candle_time}")

            # Use cached state for all downstream logic
            state = getattr(self, '_cached_composite', None)

            # ═══════════════════════════════════════════════════════════════
            # FLASH VETO — abnormal candle body detection
            # Hard-block above 5× ATR14; soft-discount (−40% quality) at 3–5×.
            # ═══════════════════════════════════════════════════════════════
            _flash_discount = 1.0
            try:
                if len(df) >= 15:
                    import numpy as _fnp
                    _hi = df["high"].values; _lo = df["low"].values
                    _cl = df["close"].values; _op = df["open"].values
                    _tr = _fnp.maximum(
                        _hi[1:] - _lo[1:],
                        _fnp.abs(_hi[1:] - _cl[:-1]),
                        _fnp.abs(_lo[1:] - _cl[:-1]),
                    )
                    _atr14 = float(_fnp.nanmean(_tr[-14:])) if len(_tr) >= 14 else 0.0
                    _last_body = abs(float(_cl[-1]) - float(_op[-1]))
                    if _atr14 > 0:
                        _body_ratio = _last_body / _atr14
                        if _body_ratio > 5.0:
                            logger.warning(
                                f"[FLASH] ⛔ Hard-veto: candle body {_body_ratio:.1f}× ATR "
                                f"— news spike detected, blocking signal"
                            )
                            return 0, {
                                "timestamp": timestamp, "regime": "UNKNOWN",
                                "reasoning": f"flash_veto_{_body_ratio:.1f}x_atr",
                                "final_signal": 0, "signal_quality": 0.0,
                                "mr_signal": 0, "mr_confidence": 0.0,
                                "tf_signal": 0, "tf_confidence": 0.0,
                                "ema_signal": 0, "ema_confidence": 0.0,
                            }
                        elif _body_ratio > 3.0:
                            logger.warning(
                                f"[FLASH] ⚠️ Soft-veto: candle body {_body_ratio:.1f}× ATR "
                                f"— quality discounted 40%"
                            )
                            _flash_discount = 0.60
            except Exception:
                _flash_discount = 1.0

            # ═══════════════════════════════════════════════════════════════
            # T3.3: NY OPEN HOUR BLOCK (13:00–13:59 UTC)
            # TF signals at NY open: 53% WR, -21.2% P&L (stop-hunting territory).
            # Trades 1–2 hours later: 60% WR, +101.5% P&L.
            # BTC trades 24/7 — only block market-hours assets.
            # NOTE: FX pairs (EURUSD, EURJPY) are intentionally excluded.
            # 13:00 UTC = London/NY overlap — the highest-liquidity, most
            # directional hour of the FX session. Blocking it kills best entries.
            # The stop-hunt data that justified this block was from USTEC/GOLD.
            # ═══════════════════════════════════════════════════════════════
            _hour_utc = _dt.utcnow().hour
            if _hour_utc == 13 and self.asset_type in ("USTEC", "GOLD", "USOIL", "GBPAUD"):
                logger.info(
                    f"[SESSION] ⏸️ NY open hour block — no new entries for {self.asset_type}"
                )
                return 0, {
                    "timestamp": timestamp,
                    "regime": "UNKNOWN",
                    "reasoning": "ny_open_block",
                    "final_signal": 0, "signal_quality": 0.0,
                    "mr_signal": 0, "mr_confidence": 0.0,
                    "tf_signal": 0, "tf_confidence": 0.0,
                    "ema_signal": 0, "ema_confidence": 0.0,
                }

            # ═══════════════════════════════════════════════════════════════
            # T3.4: ECONOMIC CALENDAR BLOCK
            # Trading through NFP/FOMC/CPI on a 1H timeframe is gambling.
            # Block N hours before each high-impact event.
            # ═══════════════════════════════════════════════════════════════
            if self._econ_events:
                from datetime import timezone as _tz, timedelta as _td
                _utc_now = _dt.now(_tz.utc)
                _asset = self.asset_type
                for _evt in self._econ_events:
                    try:
                        _evt_time = _dt.fromisoformat(_evt["datetime"].replace("Z", "+00:00"))
                        _hours_before = _evt.get("block_hours_before", 2)
                        _block_start = _evt_time - _td(hours=_hours_before)
                        if _block_start <= _utc_now < _evt_time:
                            _affected = _evt.get("currencies", [])
                            _blocked = (
                                (_asset in ("BTC", "BTCUSDT") and "USD" in _affected) or
                                (_asset in ("GOLD", "XAUUSD") and "USD" in _affected) or
                                (_asset == "EURUSD" and ("EUR" in _affected or "USD" in _affected)) or
                                (_asset == "EURJPY" and ("EUR" in _affected or "JPY" in _affected)) or
                                (_asset in ("USTEC", "US100", "NAS100") and "USD" in _affected) or
                                (not _affected)  # fallback: block all if no currencies listed
                            )
                            if _blocked:
                                _mins_to_evt = (_evt_time - _utc_now).total_seconds() / 60
                                logger.warning(
                                    f"[CALENDAR] ⏸️ Blocking {_asset} — "
                                    f"{_evt['event']} in {_mins_to_evt:.0f}min"
                                )
                                return 0, {
                                    "timestamp": timestamp,
                                    "regime": "UNKNOWN",
                                    "reasoning": f"econ_calendar_{_evt['event'].replace(' ', '_')}",
                                    "final_signal": 0, "signal_quality": 0.0,
                                    "mr_signal": 0, "mr_confidence": 0.0,
                                    "tf_signal": 0, "tf_confidence": 0.0,
                                    "ema_signal": 0, "ema_confidence": 0.0,
                                }
                    except Exception:
                        continue

            # Step 1: Prepare context
            is_bull = is_bull_market
            regime_conf = governor_data.get('confidence', 0.5) if governor_data else 0.5
            regime_name = governor_data.get('regime', 'NEUTRAL') if governor_data else "NEUTRAL"

            # ✨ NEW: Advanced Confluence Overlays
            div_res = self.divergence_detector.analyze(df)
            br_res = self.break_retest_validator.validate(df, self.asset_type)

            # D.1: Update trend lifecycle in composite state
            if state is not None:
                self._update_trend_lifecycle(state, regime_name)
                
                # Ensure regime_age_ratio is always populated (Issue 1, Step 2 fix)
                _start = self._regime_start_time.get(self.asset_type)
                if _start:
                    _age_hours = (datetime.now() - _start).total_seconds() / 3600
                    state.regime_age_hours = _age_hours
                    _median = state.median_regime_duration or 12.0
                    state.regime_age_ratio = _age_hours / _median

            # Update stats based on provided regime
            if self.previous_regime is not None and self.previous_regime != is_bull:
                self.stats["regime_changes"] += 1
            self.previous_regime = is_bull
            if is_bull:
                self.stats["bull_regime_count"] += 1
            else:
                self.stats["bear_regime_count"] += 1


            # STEP 2: Get strategy signals
            # Pass 4H context to strategies if available
            df_4h = governor_data.get('df_4h') if governor_data else None
            logger.debug(f"[MR INPUT] {self.asset_type}: df_4h={'present, ' + str(len(df_4h)) + ' bars' if df_4h is not None else 'MISSING'}")
            
            mr_signal, mr_conf = self.s_mean_reversion.generate_signal(df, df_4h=df_4h)
            tf_signal, tf_conf = self.s_trend_following.generate_signal(df, df_4h=df_4h)
            ema_signal, ema_conf = self.s_ema.generate_signal(df, df_4h=df_4h)

            # Store originals for logging
            mr_original = mr_signal
            tf_original = tf_signal

            # ═══════════════════════════════════════════════════════════════
            # T3.5: BTC FUNDING RATE Z-SCORE CONFIDENCE MULTIPLIER
            # Extreme funding rates (Z ≥ 2.0) indicate crowded positioning.
            # Over-leveraged longs → MR short setups become highest probability.
            # Z-score adapts to sustained bull runs; static threshold doesn't.
            # ═══════════════════════════════════════════════════════════════
            _funding_z = governor_data.get("funding_rate_zscore", 0.0) if governor_data else 0.0
            if self.asset_type in ("BTC", "BTCUSDT") and abs(_funding_z) >= 2.0:
                if mr_signal != 0:
                    mr_conf = min(1.0, mr_conf * 1.15)
                    logger.info(
                        f"[FUNDING] Extreme positioning (Z={_funding_z:+.1f}): "
                        f"MR conf boosted to {mr_conf:.2f}"
                    )

            # ═══════════════════════════════════════════════════════════════
            # T3.6: DXY PROXY CONFIDENCE MULTIPLIER
            # Rising EUR/USD = falling dollar = bullish for GOLD/USTEC/EURJPY.
            # Computed from already-traded EUR/USD data — zero API cost.
            # ═══════════════════════════════════════════════════════════════
            _dxy_falling = governor_data.get("dxy_falling") if governor_data else None
            if _dxy_falling is not None and self.asset_type in ("GOLD", "USTEC", "EURJPY", "USOIL"):
                if self.asset_type == "GOLD":
                    # Dollar weakness → gold strength
                    if _dxy_falling and tf_signal == 1:
                        tf_conf = min(1.0, tf_conf * 1.10)
                        logger.debug(f"[DXY] Weak dollar: GOLD TF BUY conf boosted to {tf_conf:.2f}")
                    elif not _dxy_falling and tf_signal == -1:
                        tf_conf = min(1.0, tf_conf * 1.10)
                        logger.debug(f"[DXY] Strong dollar: GOLD TF SELL conf boosted to {tf_conf:.2f}")
                elif self.asset_type == "USTEC":
                    # Dollar weakness generally supportive of risk assets
                    if _dxy_falling and tf_signal == 1:
                        tf_conf = min(1.0, tf_conf * 1.05)
                        logger.debug(f"[DXY] Weak dollar: USTEC TF BUY conf boosted to {tf_conf:.2f}")
                elif self.asset_type == "USOIL":
                    # Dollar weakness = oil strength (inverse correlation)
                    if _dxy_falling and tf_signal == 1:   # Weak dollar + BUY oil
                        tf_conf = min(1.0, tf_conf * 1.10)
                        logger.debug(f"[DXY] Weak dollar: USOIL TF BUY conf boosted to {tf_conf:.2f}")
                    elif not _dxy_falling and tf_signal == -1:  # Strong dollar + SELL oil
                        tf_conf = min(1.0, tf_conf * 1.10)
                        logger.debug(f"[DXY] Strong dollar: USOIL TF SELL conf boosted to {tf_conf:.2f}")

            # ═══════════════════════════════════════════════════════════════
            # T2.6: CONSECUTIVE CANDLE CONFIDENCE MULTIPLIER
            # BTC after 3 consecutive same-direction bars + low ADX: 66% MR WR
            # GOLD after 5 consecutive bars: 85% TF continue rate
            # ADX guard prevents counter-trend fading during strong momentum
            # (MR fading streaks in high ADX: 33% WR on GOLD, 56% on BTC).
            # This is a confidence bonus, not a new gate — fails silently.
            # ═══════════════════════════════════════════════════════════════
            try:
                _closes = df['close'].values
                _consec = 0
                for _i in range(len(_closes) - 1, max(len(_closes) - 10, 0), -1):
                    if _i == 0:
                        break
                    if _closes[_i] > _closes[_i - 1]:
                        if _consec >= 0:
                            _consec += 1
                        else:
                            break
                    else:
                        if _consec <= 0:
                            _consec -= 1
                        else:
                            break

                # Compute ADX for the guard
                _adx_guard = 25.0  # default if calculation fails
                try:
                    import talib as _talib_c
                    _adx_raw = _talib_c.ADX(
                        df['high'].values, df['low'].values, _closes, timeperiod=14
                    )[-1]
                    if not np.isnan(_adx_raw):
                        _adx_guard = _adx_raw
                except Exception:
                    pass

                # BTC: boost MR when price has made 3+ consecutive candles in one
                # direction AND momentum is low — classic mean reversion setup
                if self.asset_type == "BTC" and abs(_consec) >= 3 and _adx_guard < 25:
                    if mr_signal != 0:
                        mr_conf = min(1.0, mr_conf * 1.20)
                        logger.debug(
                            f"[CANDLE] BTC {_consec}-bar streak + low ADX ({_adx_guard:.0f}): "
                            f"MR conf boosted to {mr_conf:.2f}"
                        )

                # GOLD: boost TF when riding a 5+ bar streak — trend continuation
                if self.asset_type == "GOLD" and abs(_consec) >= 5:
                    if tf_signal != 0:
                        tf_conf = min(1.0, tf_conf * 1.15)
                        logger.debug(
                            f"[CANDLE] GOLD {_consec}-bar streak: "
                            f"TF conf boosted to {tf_conf:.2f}"
                        )
            except Exception:
                pass  # Bonus only — never block execution on failure

            # Extract regime score for Gatekeeper (Phase 3)
            regime_score = governor_data.get("regime_score", 0.0) if governor_data else 0.0
            regime_is_bullish = governor_data.get("is_bullish", False) if governor_data else False
            regime_is_bearish = governor_data.get("is_bearish", False) if governor_data else False

            # ═══════════════════════════════════════════════════════════════
            # ENHANCED GATEKEEPER — Confidence Scaling + Transition Evidence
            # ═══════════════════════════════════════════════════════════════
            # FULL regimes (|regime_score| >= 1.0): hard block counter-trend.
            # SLIGHTLY regimes (|regime_score| < 1.0): penalise confidence,
            #   with penalty modulated by TransitionEvidence (2+ conditions
            #   required before any reduction is applied).
            # NEUTRAL: all strategies fire freely.
            # Explosive momentum overrule preserved for full-regime hard blocks.
            # ═══════════════════════════════════════════════════════════════
            if self.use_gatekeeper:
                is_neutral = (regime_score == 0.0) or (not regime_is_bullish and not regime_is_bearish)
                regime_strength = abs(regime_score)  # 0.5 for SLIGHTLY, 1.0 for full

                # Pull transition evidence if available
                _te = getattr(state, '_transition_evidence', None) if state else None
                _transition_score = _te.total_score if _te else 0.0
                _transition_conditions = _te.conditions_met if _te else 0

                if is_neutral:
                    # NEUTRAL: all strategies allowed in any direction
                    logger.debug(f"[GATEKEEPER] NEUTRAL — all strategies allowed ({self.asset_type})")

                elif regime_is_bullish:
                    if regime_strength >= 1.0:
                        # FULL BULLISH: hard block counter-trend shorts
                        if tf_signal < 0:
                            if self._is_explosive_momentum(df, -1):
                                logger.info(f"[GATEKEEPER] 🚀 EXPLOSIVE MOMENTUM - Overruling Bullish block for SHORT (TF)")
                            else:
                                logger.info(f"[GATEKEEPER] ❌ BLOCKED SHORT (TF): Strong bullish for {self.asset_type}")
                                tf_signal = 0; tf_conf = 0.0
                        if ema_signal < 0:
                            if self._is_explosive_momentum(df, -1):
                                logger.info(f"[GATEKEEPER] 🚀 EXPLOSIVE MOMENTUM - Overruling Bullish block for SHORT (EMA)")
                            else:
                                logger.info(f"[GATEKEEPER] ❌ BLOCKED SHORT (EMA): Strong bullish for {self.asset_type}")
                                ema_signal = 0; ema_conf = 0.0
                        if mr_signal < 0:
                            logger.info(f"[GATEKEEPER] ❌ BLOCKED SHORT (MR): Counter-trend in strong Bullish for {self.asset_type}")
                            mr_signal = 0; mr_conf = 0.0
                        elif mr_signal > 0:
                            logger.info(f"[GATEKEEPER] ✅ ALLOWED LONG (MR): Dip buy in strong Bullish for {self.asset_type}")

                    else:
                        # SLIGHTLY BULLISH: penalise shorts, don't kill them
                        # Bearish reversal evidence in a slightly bullish zone reduces penalty
                        _penalty = 0.50  # base: halve confidence
                        if _transition_conditions >= 2 and _transition_score < -0.15:
                            _penalty = max(0.30, _penalty + _transition_score)
                            logger.info(
                                f"[GATEKEEPER] TRANSITION evidence reduces SHORT penalty: "
                                f"{_penalty:.2f} (score={_transition_score:+.3f}, "
                                f"conditions={_transition_conditions}/4)"
                            )
                        if tf_signal < 0:
                            tf_conf *= _penalty
                            logger.info(
                                f"[GATEKEEPER] ⚠️ PENALIZED SHORT (TF): Slightly bullish — "
                                f"conf reduced to {tf_conf:.2f}"
                            )
                        if ema_signal < 0:
                            # EMA is a slow-trend follower — still zero in counter trend
                            ema_signal = 0; ema_conf = 0.0
                        if mr_signal < 0:
                            mr_conf *= min(_penalty + 0.10, 0.80)  # MR slightly less penalised
                            logger.info(
                                f"[GATEKEEPER] ⚠️ PENALIZED SHORT (MR): Slightly bullish — "
                                f"conf reduced to {mr_conf:.2f}"
                            )
                        elif mr_signal > 0:
                            logger.info(f"[GATEKEEPER] ✅ ALLOWED LONG (MR): Dip buy in slightly Bullish for {self.asset_type}")

                elif regime_is_bearish:
                    if regime_strength >= 1.0:
                        # FULL BEARISH: hard block counter-trend longs
                        if tf_signal > 0:
                            if self._is_explosive_momentum(df, 1):
                                logger.info(f"[GATEKEEPER] 🚀 EXPLOSIVE MOMENTUM - Overruling Bearish block for LONG (TF)")
                            else:
                                logger.info(f"[GATEKEEPER] ❌ BLOCKED LONG (TF): Strong bearish for {self.asset_type}")
                                tf_signal = 0; tf_conf = 0.0
                        if ema_signal > 0:
                            if self._is_explosive_momentum(df, 1):
                                logger.info(f"[GATEKEEPER] 🚀 EXPLOSIVE MOMENTUM - Overruling Bearish block for LONG (EMA)")
                            else:
                                logger.info(f"[GATEKEEPER] ❌ BLOCKED LONG (EMA): Strong bearish for {self.asset_type}")
                                ema_signal = 0; ema_conf = 0.0
                        if mr_signal > 0:
                            logger.info(f"[GATEKEEPER] ❌ BLOCKED LONG (MR): Counter-trend in strong Bearish for {self.asset_type}")
                            mr_signal = 0; mr_conf = 0.0
                        elif mr_signal < 0:
                            logger.info(f"[GATEKEEPER] ✅ ALLOWED SHORT (MR): Rally short in strong Bearish for {self.asset_type}")

                    else:
                        # SLIGHTLY BEARISH: penalise longs, don't kill them
                        # Bullish reversal evidence in a slightly bearish zone reduces penalty
                        _penalty = 0.50
                        if _transition_conditions >= 2 and _transition_score > 0.15:
                            _penalty = max(0.30, _penalty - _transition_score)
                            logger.info(
                                f"[GATEKEEPER] TRANSITION evidence reduces LONG penalty: "
                                f"{_penalty:.2f} (score={_transition_score:+.3f}, "
                                f"conditions={_transition_conditions}/4)"
                            )
                        if tf_signal > 0:
                            tf_conf *= _penalty
                            logger.info(
                                f"[GATEKEEPER] ⚠️ PENALIZED LONG (TF): Slightly bearish — "
                                f"conf reduced to {tf_conf:.2f}"
                            )
                        if ema_signal > 0:
                            ema_signal = 0; ema_conf = 0.0
                        if mr_signal > 0:
                            mr_conf *= min(_penalty + 0.10, 0.80)
                            logger.info(
                                f"[GATEKEEPER] ⚠️ PENALIZED LONG (MR): Slightly bearish — "
                                f"conf reduced to {mr_conf:.2f}"
                            )
                        elif mr_signal < 0:
                            logger.info(f"[GATEKEEPER] ✅ ALLOWED SHORT (MR): Rally short in slightly Bearish for {self.asset_type}")
            # --- End Enhanced Gatekeeper ---
            
            # Initialize core variables for details building (prevents UnboundLocalError if we skip)
            buy_score = 0.0
            sell_score = 0.0
            signal_quality = 0.0
            ai_validation_details = {}
            original_signal = 0
            final_signal = 0
            reasoning = "hold (no strategy agreement)"
            trade_type = "TREND"

            # COMPUTATIONAL OPTIMIZATION: If all signals are zero, skip heavy validation
            if mr_signal == 0 and tf_signal == 0 and ema_signal == 0:
                logger.debug(f"[AGGREGATOR] {self.asset_type}: No signals to validate, skipping to end.")
                # We can skip to building the details dictionary
            else:
                # Ranging Detection — keeps position limits, counter-trend blocking
                # now handled exclusively by the Smart Gatekeeper above (T1.3 fix).
                is_ranging = regime_conf <= 0.50
                max_trades_override = None
                filter_reason = ""
                if is_ranging:
                    max_trades_override = 1
                    filter_reason = "Ranging Mode (Max 1 Trade)"

                signal_quality = max(mr_conf, tf_conf)

                # --- Directional Trap Filter Veto (T2.3: regime-aware) ---
                if mr_signal != 0 or tf_signal != 0 or ema_signal != 0:
                    test_direction = "long" if (mr_signal > 0 or tf_signal > 0 or ema_signal > 0) else "short"
                    # regime_aligned: signal direction matches macro regime.
                    # Fix #16: NEUTRAL regime has no directional opinion — both
                    # directions are valid so treat as aligned for both sides.
                    # Without this, LONG signals in NEUTRAL are always "not aligned"
                    # which triggers the 1.5× BTC volume check that doesn't apply
                    # to SHORT, creating a permanent short-bias in NEUTRAL.
                    _is_neutral_regime = (regime_score == 0.0) or (not regime_is_bullish and not regime_is_bearish)
                    _trap_aligned = (
                        _is_neutral_regime or
                        (test_direction == "long" and is_bull) or
                        (test_direction == "short" and not is_bull)
                    )
                    if not validate_candle_structure(
                        df, self.asset_type,
                        direction=test_direction,
                        regime_confidence=regime_conf,
                        regime_aligned=_trap_aligned,
                    ):
                        logger.info(f"[TRAP] VETO - Candidate rejected by structure check.")
                        # Pass the REAL strategy signals through so the shadow trader
                        # can record and learn from trap-filter blocks (Bug 2 fix).
                        # Zeroing these out was hiding ~47 signals/cycle from the
                        # gate scorecard (76.6% WR, +13.3% P&L invisible to ML labels).
                        return 0, {
                            "timestamp": timestamp,
                            "regime": regime_name,
                            "reasoning": "blocked_by_trap_filter",
                            "final_signal": 0,
                            "original_signal": mr_signal or tf_signal or ema_signal, # Pass the intended direction
                            "signal_quality": 0.0,
                            "mr_signal": mr_signal,
                            "mr_confidence": mr_conf,
                            "tf_signal": tf_signal,
                            "tf_confidence": tf_conf,
                            "ema_signal": ema_signal,
                            "ema_confidence": ema_conf,
                            # Raw pre-gatekeeper values for shadow trader gate scoring
                            "mr_signal_raw": mr_original,
                            "tf_signal_raw": tf_original,
                        }

                # STEP 3: PRE-SCORE AI VALIDATION — DISABLED (T2.2)
                # Previously killed individual MR/TF votes before scoring, destroying
                # the consensus and independent evaluation pipeline.
                # Blocked 13 signals with 92.3% WR and +17.2% P&L.
                # AI validation now runs post-score via hybrid_validator.py.
                # The circuit breaker and stats objects are preserved for post-score use.
                ai_bypass = False
                ai_validation_details = {}

                # STEP 4: Calculate scores (MR + TF + EMA all contribute)
                buy_score, buy_explanation, buy_agreement = self._calculate_score(df, 1, mr_signal, mr_conf, tf_signal, tf_conf, ema_signal, ema_conf, is_bull)
                sell_score, sell_explanation, sell_agreement = self._calculate_score(df, -1, mr_signal, mr_conf, tf_signal, tf_conf, ema_signal, ema_conf, is_bull)

                # STEP 5: Dynamic thresholds
                adj_buy_thresh, adj_sell_thresh = self.calculate_regime_adjusted_thresholds(is_bull, regime_conf)

                # STEP 6: Make decision
                if buy_score >= adj_buy_thresh and buy_score > sell_score:
                    final_signal = 1
                elif sell_score >= adj_sell_thresh and sell_score > buy_score:
                    final_signal = -1

                reasoning = f"BUY (score:{buy_score:.2f}, thresh:{adj_buy_thresh:.2f})" if final_signal == 1 else f"SELL (score:{sell_score:.2f}, thresh:{adj_sell_thresh:.2f})" if final_signal == -1 else f"hold (buy:{buy_score:.2f} vs sell:{sell_score:.2f})"
                original_signal = final_signal

                # Fix F: removed hard cap at 0.7 — score can now reflect true 3-strategy consensus
                raw_quality = max(buy_score, sell_score)
                if buy_agreement < 2 and sell_agreement < 2: raw_quality *= 0.7
                if (final_signal == 1 and is_bull) or (final_signal == -1 and not is_bull): raw_quality *= 1.15
                signal_quality = min(raw_quality, 1.0)

                # Section 2.4B: Boost quality when transition evidence strongly agrees
                if state and hasattr(state, '_transition_evidence') and state._transition_evidence:
                    if state._transition_evidence.conditions_met >= 3:
                        _te_boost = abs(state._transition_evidence.total_score) * 0.15
                        _te_dir = state._transition_evidence.direction
                        if (final_signal == 1 and _te_dir == "BULLISH_REVERSAL") or \
                           (final_signal == -1 and _te_dir == "BEARISH_REVERSAL"):
                            signal_quality = min(1.0, signal_quality * (1.0 + _te_boost))
                            logger.debug(
                                f"[QUALITY] Transition evidence boost: "
                                f"×{1.0 + _te_boost:.3f} → {signal_quality:.2f}"
                            )

                if final_signal != 0 and signal_quality < self.config["min_signal_quality"]:
                    final_signal = 0
                    reasoning = f"hold_lowquality (original:{reasoning}, quality:{signal_quality:.2f})"

                # ═══════════════════════════════════════════════════════════
                # INDEPENDENT STRATEGY EVALUATION (T1.1 fix)
                # Consensus failed (final_signal still 0). Check if any single
                # strategy has enough individual confidence to fire alone.
                # Priority: TF > EMA > MR (based on solo P&L simulation data).
                # allow_single_override and single_override_threshold are config
                # keys that existed in presets but were never read — now wired.
                # ═══════════════════════════════════════════════════════════
                if final_signal == 0 and self.allow_independent:
                    candidates = []

                    # TF: use post-gatekeeper signal (consistent with MR/EMA treatment).
                    # tf_original pre-bypass was causing asymmetric gatekeeper application.
                    if tf_signal != 0 and tf_conf >= self.independent_thresholds["trend_following"]:
                        candidates.append(("TF", tf_signal, tf_conf))

                    # EMA: evaluated post-gatekeeper (gatekeeper treats EMA same as TF)
                    if ema_signal != 0 and ema_conf >= self.independent_thresholds["ema"]:
                        candidates.append(("EMA", ema_signal, ema_conf))

                    # MR: use post-gatekeeper signal (Smart Gatekeeper already filtered it)
                    if mr_signal != 0 and mr_conf >= self.independent_thresholds["mean_reversion"]:
                        candidates.append(("MR", mr_signal, mr_conf))

                    if candidates:
                        # Sort by confidence descending; TF wins ties (listed first)
                        candidates.sort(key=lambda x: x[2], reverse=True)
                        best_name, best_signal, best_conf = candidates[0]
                        final_signal = best_signal
                        signal_quality = best_conf * 0.85  # Solo signals get a small quality discount

                        # Multi-strategy confirmation bonus: any agreeing strategy lifts quality
                        agreeing = [c for c in candidates if c[1] == best_signal]
                        if len(agreeing) >= 2:
                            signal_quality = min(1.0, best_conf * 1.1)

                        reasoning = (
                            f"{'BUY' if final_signal == 1 else 'SELL'} "
                            f"(independent:{best_name}, conf:{best_conf:.2f}, "
                            f"confirmations:{len(agreeing)})"
                        )
                        logger.info(
                            f"[INDEPENDENT] {self.asset_type}: {best_name} fires alone "
                            f"(conf={best_conf:.2f}, aligned={len(agreeing)} strategies)"
                        )

                # Update original_signal to capture any consensus OR independent signal
                # before final filters (volatility, governor, etc) are applied.
                original_signal = final_signal

                # World-Class Filters
                # Fix D: profit filter removed — it duplicated the volatility filter (both
                # measured ATR/price%) while adding an independent failure point that blocked
                # valid signals in low-ATR trending regimes (e.g. GOLD steady grind moves).
                # Fix C: ATR expansion filter replaced with ADX trend confirmation (see method).
                if final_signal != 0 and self.enable_filters:
                    gov_passed, trade_type = self._check_governor_filter(df, final_signal)
                    if not gov_passed: final_signal = 0; reasoning = "blocked_by_governor"
                    else:
                        vol_passed, _ = self._check_volatility_filter(df)
                        if not vol_passed: final_signal = 0; reasoning = "low_volatility"
                        else:
                            sniper_passed, _ = self._check_sniper_filter(df, final_signal, governor_data=governor_data)
                            if not sniper_passed: final_signal = 0; reasoning = "no_sniper_confirmation"
                            else:
                                atr_exp_passed = self._check_atr_expansion_filter(df, trade_type)
                                if not atr_exp_passed: final_signal = 0; reasoning = "insufficient_trend_strength"
                                else:
                                    # Error 7: Profit Economics Monitor (non-blocking log)
                                    try:
                                        if final_signal != 0 and len(df) >= 14:
                                            import numpy as _pm_np
                                            _pm_tr = _pm_np.maximum(
                                                df["high"].values[1:] - df["low"].values[1:],
                                                _pm_np.abs(df["high"].values[1:] - df["close"].values[:-1]),
                                                _pm_np.abs(df["low"].values[1:]  - df["close"].values[:-1]),
                                            )
                                            _pm_atr = float(_pm_np.nanmean(_pm_tr[-14:]))
                                            if _pm_atr > 0:
                                                _pm_rr = (2.5 * _pm_atr) / (1.5 * _pm_atr)
                                                if _pm_rr < 1.5:
                                                    logger.warning(
                                                        f"[PROFIT] ⚠️ Low R:R {_pm_rr:.2f} — monitor only"
                                                    )
                                    except Exception:
                                        pass

                # Apply flash veto soft-discount to final quality score
                if _flash_discount < 1.0 and final_signal != 0:
                    signal_quality = round(signal_quality * _flash_discount, 4)
                    reasoning += f" [flash_discount:{_flash_discount:.0%}]"

                # ═══════════════════════════════════════════════════════════
                # C. SESSION LIQUIDITY PENALTY (Extended to all MT5 Assets)
                # ═══════════════════════════════════════════════════════════
                try:
                    if final_signal != 0:
                        from src.utils.market_hours import MarketHours
                        _hour_utc_s = _dt.utcnow().hour
                        
                        # 1. BTC (Binance) is 24/7 - only check for global liquidity lows
                        if "BTC" in self.asset_type:
                            session_quality = MarketHours.get_btc_session_quality()
                            if session_quality == "LOW":
                                signal_quality *= 0.85
                                reasoning += " [session:LOW_LIQ]"
                                logger.info(f"[SESSION] ⚠️ BTC low liquidity: quality discounted")

                        # 2. MT5/Exness Assets - Apply Session Penalties
                        else:
                            is_off_session = False
                            asset = self.asset_type.upper()

                            if any(x in asset for x in ("EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD")):
                                if _hour_utc_s < 7 or _hour_utc_s >= 20:
                                    is_off_session = True
                                    logger.info(f"[SESSION] ⚠️ FX off-session ({_hour_utc_s}:00 UTC)")

                            elif "GOLD" in asset or "XAU" in asset:
                                if _hour_utc_s < 7 or _hour_utc_s >= 20:
                                    is_off_session = True
                                    logger.info(f"[SESSION] ⚠️ GOLD off-session ({_hour_utc_s}:00 UTC)")

                            elif any(x in asset for x in ("USTEC", "US100", "NAS", "US30", "SPX")):
                                if _hour_utc_s < 13 or _hour_utc_s >= 21:
                                    is_off_session = True
                                    logger.info(f"[SESSION] ⚠️ INDEX off-session ({_hour_utc_s}:00 UTC)")

                            elif "OIL" in asset:
                                if _hour_utc_s < 13 or _hour_utc_s >= 19:
                                    is_off_session = True
                                    logger.info(f"[SESSION] ⚠️ OIL off-session ({_hour_utc_s}:00 UTC)")

                            if is_off_session:
                                # In Performance mode, we discount the final quality score
                                signal_quality *= 0.80
                                reasoning += " [session:OFF]"
                                logger.info(f"[SESSION] Off-session discount applied to {asset}")

                except Exception as e:
                    logger.warning(f"[SESSION] Gate calculation failed: {e}")

            # ── CONTEXT ENGINE WIRING ─────────────────────────────────────
            # F.3: MR Divergence Cross-Signal (reads from MR strategy if available)
            if state is not None:
                try:
                    _mr_details = {}
                    if hasattr(self.s_mean_reversion, '_last_divergence_info'):
                        _mr_details = self.s_mean_reversion._last_divergence_info or {}
                    if _mr_details.get("divergence_detected"):
                        state.divergence_detected = True
                        state.divergence_strength = float(_mr_details.get("divergence_strength", 0.5))
                    if state.is_parabolic and state.divergence_detected:
                        state.reversal_imminent = True
                except Exception:
                    pass

                # H.1: Feed AI Sniper output into composite state
                try:
                    _ai_data = ai_validation_details if isinstance(ai_validation_details, dict) else {}
                    if _ai_data:
                        state.ai_pattern_name = _ai_data.get("pattern_name")
                        state.ai_pattern_confidence = float(_ai_data.get("confidence", 0.0))
                        _reversal_patterns = [
                            "Evening Star", "Bearish Engulfing", "Shooting Star",
                            "Morning Star", "Bullish Engulfing", "Hammer"
                        ]
                        if state.ai_pattern_name in _reversal_patterns and \
                           state.ai_pattern_confidence > 0.75:
                            state.ai_reversal_probability = state.ai_pattern_confidence
                except Exception:
                    pass

                # Section I: Confluence Engine — adjust tf_conf and mr_conf
                try:
                    tf_conf, mr_conf, state = self._score_confluence(state, tf_conf, mr_conf)
                except Exception as _ce:
                    logger.debug(f"[CONFLUENCE] Scoring failed: {_ce}")

            # ─────────────────────────────────────────────────────────────

            # STEP 7: Build base response
            # ✨ NEW: Confluence Reasoning Enhancement
            bonus_tags = []
            if div_res and div_res.type != "NONE":
                # Only add if aligned with signal
                if (final_signal == 1 and "BULLISH" in div_res.type) or (final_signal == -1 and "BEARISH" in div_res.type):
                    tag = div_res.explanation.split(":")[-1].split("(")[0].strip()
                    bonus_tags.append(f"✨ {tag}")
            
            if br_res and br_res.is_valid:
                if (final_signal == 1 and br_res.type == "BULLISH_RETEST") or (final_signal == -1 and br_res.type == "BEARISH_RETEST"):
                    bonus_tags.append(f"🚀 {br_res.type.replace('_', ' ').title()}")

            if bonus_tags:
                reasoning += " | " + " | ".join(bonus_tags[:2])

            details = {
                "timestamp": timestamp,
                "regime": regime_name,
                "regime_confidence": regime_conf,
                "original_signal": original_signal,
                "final_signal": final_signal,
                "reasoning": reasoning,
                "signal_quality": signal_quality,
                "buy_score": buy_score,
                "sell_score": sell_score,
                "mr_signal": mr_signal,
                "mr_confidence": mr_conf,
                "tf_signal": tf_signal,
                "tf_confidence": tf_conf,
                "ema_signal": ema_signal,
                "ema_confidence": ema_conf,
                "governor_data": governor_data, # Pass governor data through
                "ai_validation": ai_validation_details,
                "trade_type": trade_type,
                "viz_overlay": {
                    "divergence": div_res,
                    "break_retest": br_res
                }
            }

            # STEP 8: Format AI validation for visualization
            if self.ai_validator:
                try:
                    # Pass copies to avoid accidental modification
                    ai_validation_details = self._format_ai_validation_for_viz(
                        final_signal=final_signal,
                        details={**details},
                        df=df
                    )
                except Exception as e:
                    logger.error(f"[AGGREGATOR] AI formatting failed: {e}")

            # STEP 9: Final Response update
            # Derive the ai_validated boolean from the action field so the DB
            # and dashboard always have a correct True/False value.
            # "approved"/"bypassed*" → AI allowed the signal through.
            # "rejected" → AI blocked it.
            # "skipped*"/"none"/"ai_disabled"/"hold" → AI was not in the loop.
            _ai_action = ai_validation_details.get("action", "") if isinstance(ai_validation_details, dict) else ""
            _ai_validated = _ai_action == "approved" or _ai_action.startswith("bypassed")

            details.update({
                "ai_validation": ai_validation_details,
                "ai_validated": _ai_validated,
                "mr_signal_raw": mr_original,  # Ensure originals are present
                "tf_signal_raw": tf_original,
                # Composite state — used by VTM pattern-aware exits and shadow trader
                "institutional_pattern": state.institutional_pattern if state else None,
                "friday_tighten": state.friday_tighten if state else False,
                "composite_state": state.to_dict() if state else {},
            })

            # T2.1: TRANSITION sizing — governor approved but market is neutral.
            # Apply 50% risk multiplier so these trades fire at half normal size.
            # T1.7 already wires mtf_risk_multiplier into both execution handlers.
            if final_signal != 0 and trade_type == "TRANSITION":
                current_multiplier = details.get("mtf_risk_multiplier", 1.0)
                details["mtf_risk_multiplier"] = current_multiplier * 0.5
                logger.info(
                    f"[TRANSITION] {self.asset_type}: signal approved at 50% size "
                    f"(mtf_risk_multiplier={details['mtf_risk_multiplier']:.2f})"
                )

            return final_signal, details
        

        except Exception as e:
            logger.error(f"Error in aggregation: {e}", exc_info=True)
            return 0, {
                "error": str(e),
                "timestamp": timestamp,
                "reasoning": f"error: {str(e)[:50]}",
                "signal_quality": 0.0,
                "final_signal": 0,
                "mr_signal": 0,
                "mr_confidence": 0.0,
                "tf_signal": 0,
                "tf_confidence": 0.0,
                "ema_signal": 0,
                "ema_confidence": 0.0,
            }
