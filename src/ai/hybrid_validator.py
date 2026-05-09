"""
 AI Signal Validator with Realistic S/R Thresholds
========================================================
Key fixes:
1. Base S/R threshold: 0.5% → 2.5% (5x more realistic)
2. Directional S/R logic (BUY needs support, SELL needs resistance)
3. Strategy-aware adjustments (TF gets wider thresholds)
4. Better adaptive scaling based on volatility and regime
5. Comprehensive logging preserved
6. Weekly/Monthly AVWAP (Phase 3)
"""

import pandas as pd
import numpy as np
import logging
import talib as ta
from typing import Tuple, Dict, Optional
from collections import deque, defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HybridSignalValidator:
    """
    AI-powered signal validation with  realistic thresholds
    """

    # Pattern classifications
    BULLISH_PATTERNS = {
        "Engulfing",
        "Morning Star",
        "Hammer",
        "Inverted Hammer",
        "Three White Soldiers",
        "Piercing",
        "Harami",
        "Three Inside",
        "Dragonfly Doji",
        "Bullish Engulfing",
        "Bullish Harami",
        "Marubozu",
    }

    BEARISH_PATTERNS = {
        "Evening Star",
        "Shooting Star",
        "Hanging Man",
        "Three Black Crows",
        "Dark Cloud",
        "Gravestone Doji",
        "Bearish Engulfing",
        "Three Outside",
        "Dark Cloud Cover",
        "Bearish Harami",
    }
    NEUTRAL_PATTERNS = {
        "Doji",  # Context-dependent
        "Spinning Top",
    }

    def __init__(
        self,
        analyst,
        sniper,
        pattern_id_map,
        sr_threshold_pct=0.0035,
        pattern_confidence_min=0.65,
        use_ai_validation=True,
        enable_adaptive_thresholds=True,
        strong_signal_bypass_threshold=0.85,
        circuit_breaker_threshold=0.70,
        enable_detailed_logging=False,
    ):
        self.analyst = analyst
        self.sniper = sniper
        self.pattern_id_map = pattern_id_map
        self.reverse_pattern_map = {v: k for k, v in pattern_id_map.items()}

        # Configuration
        self.base_sr_threshold = sr_threshold_pct
        self.base_pattern_confidence = pattern_confidence_min
        self.use_ai_validation = use_ai_validation
        self.enable_adaptive = enable_adaptive_thresholds
        self.strong_signal_bypass = strong_signal_bypass_threshold
        self.bypass_threshold = circuit_breaker_threshold
        self.detailed_logging = enable_detailed_logging

        # Current adaptive thresholds
        self.current_sr_threshold = sr_threshold_pct
        self.current_pattern_threshold = pattern_confidence_min

        # S/R cache
        self.sr_cache: Dict[str, Dict] = {}
        self.sr_update_interval = 3600  # 1 hour

        # Circuit breaker
        self.rejection_window = deque(maxlen=50)
        self.bypass_mode = False
        self.bypass_cooldown = 0

        # Statistics tracking
        self.stats = {
            "total_checks": 0,
            "approved": 0,
            "rejected": 0,
            "rejected_no_sr": 0,
            "rejected_no_pattern": 0,
            "rejected_low_confidence": 0,
            "rejected_direction_mismatch": 0,
            "bypassed_strong_signal": 0,
            "bypassed_circuit_breaker": 0,
            "adaptive_adjustments": 0,
        }

        # Rejection reason tracking
        self.rejection_reasons = defaultdict(int)

        # Performance metrics per strategy
        self.strategy_stats = defaultdict(
            lambda: {
                "checks": 0,
                "approved": 0,
                "rejected": 0,
            }
        )

        # Historical validation data
        self.validation_history = deque(maxlen=1000)

        # Threshold adjustment history
        self.threshold_history = deque(maxlen=100)

        # AI-5: Within-cycle pattern result cache.
        # _check_pattern() runs the TF neural net — expensive. In the council path it
        # gets called up to 3× per signal (sniper penalty, validate_signal, format_viz).
        # Cache keyed on (last_close_hash, signal) so identical evaluations hit memory.
        # Cache is cleared at the start of each validate_signal() call via clear_pattern_cache().
        self._pattern_cache: dict = {}

        self._log_initialization()

    def _log_initialization(self):
        """Log initialization details"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("🤖  AI SIGNAL VALIDATOR (Realistic Thresholds)")
        logger.info("=" * 70)
        logger.info(
            f"  Status:           {'ENABLED' if self.use_ai_validation else 'DISABLED'}"
        )
        logger.info(f"  Base S/R:         {self.base_sr_threshold:.2%}")
        logger.info(f"  Base Pattern:     {self.base_pattern_confidence:.0%}")
        logger.info(f"  Adaptive:         {'ON' if self.enable_adaptive else 'OFF'}")
        logger.info(f"  Strong Bypass:    {self.strong_signal_bypass:.0%}")
        logger.info(f"  Circuit Breaker:  {self.bypass_threshold:.0%}")
        logger.info(f"  Detailed Logging: {'ON' if self.detailed_logging else 'OFF'}")
        logger.info(f"  Patterns Loaded:  {len(self.pattern_id_map)}")
        logger.info("=" * 70)
        logger.info("")

    def validate_signal(
        self, signal: int, signal_details: dict, df: pd.DataFrame
    ) -> Tuple[int, dict]:
        validation_start = datetime.now()
        self.stats["total_checks"] += 1

        asset = signal_details.get("asset", "UNKNOWN")
        strategy = signal_details.get("strategy", "UNKNOWN")

        self.strategy_stats[strategy]["checks"] += 1

        if not self.use_ai_validation:
            return self._skip_validation(signal, signal_details, "ai_disabled")

        if signal == 0:
            return self._skip_validation(signal, signal_details, "hold_signal")

        # Layer 1: Circuit Breaker
        if self.bypass_mode:
            self.bypass_cooldown -= 1
            if self.bypass_cooldown <= 0:
                self._reset_circuit_breaker()
            else:
                result = self._bypass_validation(
                    signal,
                    signal_details,
                    reason="circuit_breaker",
                    cooldown=self.bypass_cooldown,
                )
                self.stats["bypassed_circuit_breaker"] += 1
                self.strategy_stats[strategy]["approved"] += 1
                return result

        # Layer 1.5: Strong Signal Bypass
        signal_quality = signal_details.get("signal_quality", 0.0)
        if signal_quality >= self.strong_signal_bypass:
            result = self._bypass_validation(
                signal,
                signal_details,
                reason="strong_signal",
                quality=signal_quality,
                threshold=self.strong_signal_bypass,
            )
            self.stats["bypassed_strong_signal"] += 1
            self.strategy_stats[strategy]["approved"] += 1
            return result

        # Layer 2: Adaptive Thresholds
        if self.enable_adaptive:
            self._update_adaptive_thresholds_fixed(df, signal_details, strategy)

        # Layer 2.5: Flash Crash Breaker (Extreme Dislocation)
        # Reason: Prevents mean reversion during vertical moves or liquidity voids.
        try:
            current_price = float(df["close"].iloc[-1])
            ema_20 = df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
            distance = abs(current_price - ema_20)
            atr_fast = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)[-1]
            
            if distance > (4.0 * atr_fast):
                if strategy.upper() == "REVERSION" or strategy == "mean_reversion":
                    logger.warning(f"[AI] Flash Crash VETO: Distance {distance:.2f} > 4xATR. Blocking REVERSION.")
                    return self._reject_signal(
                        signal_details, 
                        {"near_level": False, "reason": "flash_crash"}, 
                        None, 
                        reason="flash_crash_breaker", 
                        strategy=strategy
                    )
        except Exception as e:
            logger.error(f"[AI] Flash crash check failed: {e}")

        # Layer 3: Support/Resistance Check
        current_price = float(df["close"].iloc[-1])
        sr_result = self._check_support_resistance_fixed(
            asset, df, current_price, signal, threshold=self.current_sr_threshold
        )

        # Layer 4: Pattern Confirmation
        pattern_result = self._check_pattern(
            df, signal, min_confidence=self.current_pattern_threshold, strategy=strategy
        )

        sr_passed = sr_result["near_level"]
        pattern_passed = pattern_result["pattern_confirmed"]
        model_uncertain = pattern_result.get("model_uncertain", False)

        # ─────────────────────────────────────────────────────────────────────
        # Weighted-OR Gate (replaces hard AND gate)
        #
        # Old: BOTH S/R AND Pattern required → killed every mid-trend signal
        # where price had moved past historical levels (no S/R) or market was
        # in a momentum phase (no clean pattern).
        #
        # New logic:
        #   Both pass                       → full approval + boost
        #   One passes, other uncertain/miss → soft approval, −0.05 quality
        #   Both fail                        → reject (as before)
        #
        # "Regime-aligned" means the signal direction matches the macro bias
        # (bull+buy or bear+sell). Counter-trend signals keep the hard AND gate
        # because they need every confirmation available.
        # ─────────────────────────────────────────────────────────────────────
        regime = signal_details.get("regime", "NEUTRAL")
        is_bull_regime = "BULL" in regime.upper() or "BULLISH" in regime.upper()
        is_bear_regime = "BEAR" in regime.upper() or "BEARISH" in regime.upper()
        is_neutral_regime = not is_bull_regime and not is_bear_regime
        
        # In NEUTRAL regimes, we treat the signal as "potentially aligned" to allow 
        # momentum-based soft passes (Phase 3 recovery logic).
        regime_aligned = (signal == 1 and is_bull_regime) or (signal == -1 and is_bear_regime) or is_neutral_regime

        # ── 1H session momentum alignment ───────────────────────────────────
        # Extracted from the MTF regime detector's new intraday slope fields.
        # Gives the AI a "what is price actually doing RIGHT NOW on the 1H"
        # signal, separate from the 4H structural regime label.
        _gov = signal_details.get("governor_data") or {}
        h1_dir = _gov.get("h1_momentum_dir", "FLAT")
        h1_lower_highs = _gov.get("h1_lower_highs", False)
        h1_higher_lows = _gov.get("h1_higher_lows", False)
        h1_momentum_aligned = (
            (signal == 1 and (h1_dir == "UP" or h1_higher_lows)) or
            (signal == -1 and (h1_dir == "DOWN" or h1_lower_highs))
        )
        h1_momentum_confirmed = h1_momentum_aligned and h1_dir != "FLAT"

        if sr_passed and pattern_passed:
            # Full approval — both layers confirmed
            pass  # fall through to _approve_signal
        elif sr_passed and not pattern_passed and regime_aligned and not model_uncertain:
            # S/R confirmed, no clean pattern, but trend-aligned → soft pass
            logger.info(
                f"[AI] Soft-pass: S/R ✓ | Pattern ✗ ({pattern_result.get('reason','?')}) "
                f"| regime-aligned → approve with quality penalty"
            )
            signal_details = {**signal_details, "ai_quality_penalty": 0.05}
        elif pattern_passed and not sr_passed and regime_aligned:
            # Pattern confirmed, no nearby S/R level, but trend-aligned → soft pass
            logger.info(
                f"[AI] Soft-pass: Pattern ✓ ({pattern_result.get('pattern_name','?')}) "
                f"| S/R ✗ ({sr_result.get('reason','?')}) | regime-aligned → approve with quality penalty"
            )
            signal_details = {**signal_details, "ai_quality_penalty": 0.05}
        elif model_uncertain and sr_passed:
            # Model couldn't read the candle (uncertain output) but S/R is solid → soft pass
            logger.info(f"[AI] Soft-pass: S/R ✓ | Model uncertain → approve with quality penalty")
            signal_details = {**signal_details, "ai_quality_penalty": 0.05}
        elif regime_aligned and h1_momentum_confirmed and (sr_passed or pattern_passed):
            # ── 1H MOMENTUM SOFT-PASS (new) ──────────────────────────────────
            # Regime aligns with signal direction AND the 1H session is actively
            # moving in the same direction (slope + structure) AND at least one
            # of S/R or pattern confirms.  This path catches mid-trend entries
            # like declining BTC SELLs or falling GOLD where the candle pattern
            # is ambiguous but the price action context is unambiguous.
            # USTEC-style false signals are NOT helped by this path because they
            # show h1_dir="DOWN" / h1_lower_highs=True while regime=BULLISH — the
            # regime_aligned gate prevents the conflict from slipping through.
            _confirm_src = "S/R" if sr_passed else "Pattern"
            logger.info(
                f"[AI] 1H-momentum soft-pass: {_confirm_src} ✓ | "
                f"1H dir={h1_dir} | lower_highs={h1_lower_highs} | higher_lows={h1_higher_lows} "
                f"| regime-aligned → approve with quality penalty"
            )
            signal_details = {**signal_details, "ai_quality_penalty": 0.04, "h1_momentum_pass": True}
        elif regime_aligned and h1_momentum_confirmed and signal_quality >= 0.65:
            # ── HIGH CONFIDENCE TREND BYPASS (new) ──────────────────────────
            # If the 4H trend and 1H momentum are perfectly aligned and signal
            # quality is high (>65%), we allow entry even if BOTH S/R and 
            # pattern filters fail. This prevents being 'locked out' of strong
            # moves that are mid-air (no S/R) and momentum-based (no pattern).
            logger.info(
                f"[AI] Trend-bypass: High Quality ({signal_quality:.2f}) | "
                f"1H dir={h1_dir} | regime-aligned → approve with quality penalty"
            )
            signal_details = {**signal_details, "ai_quality_penalty": 0.08, "trend_bypass": True}
        else:
            # Hard reject — either both failed, or counter-trend with one missing,
            # or 1H momentum contradicts the signal direction.
            rejection_reason = "both_gates_failed" if (not sr_passed and not pattern_passed) else \
                ("no_sr_level" if not sr_passed else pattern_result.get("reason", "no_pattern"))
            # Annotate when 1H momentum was the deciding factor
            if regime_aligned and not h1_momentum_aligned and h1_dir != "FLAT":
                rejection_reason = f"h1_momentum_contradict ({h1_dir})"
            result = self._reject_signal(
                signal_details, sr_result, pattern_result,
                reason=rejection_reason, strategy=strategy,
            )
            if not sr_passed:
                self.stats["rejected_no_sr"] += 1
                self.rejection_reasons["no_sr_level"] += 1
            else:
                self.stats["rejected_no_pattern"] += 1
                self.rejection_reasons[pattern_result.get("reason", "no_pattern")] += 1
            self.strategy_stats[strategy]["rejected"] += 1
            return result

        # Approval
        result = self._approve_signal(
            signal,
            signal_details,
            sr_result,
            pattern_result,
            strategy=strategy,
            validation_time=(datetime.now() - validation_start).total_seconds(),
            df=df,
        )

        self.stats["approved"] += 1
        self.strategy_stats[strategy]["approved"] += 1
        self.rejection_window.append(False)

        return result

    def _update_adaptive_thresholds_fixed(
        self, df: pd.DataFrame, signal_details: dict, strategy: str
    ):
        regime = signal_details.get("regime", "BEAR")
        signal_quality = signal_details.get("signal_quality", 0.0)

        if len(df) >= 20:
            returns = df["close"].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(252)
        else:
            volatility = 0.20

        sr_threshold = self.base_sr_threshold
        if strategy == "mean_reversion": sr_threshold *= 1.0
        elif strategy == "trend_following": sr_threshold *= 1.5
        else: sr_threshold *= 1.2

        if volatility > 0.40: sr_threshold *= 1.3
        elif volatility > 0.30: sr_threshold *= 1.15
        elif volatility < 0.15: sr_threshold *= 0.9

        self.current_sr_threshold = np.clip(sr_threshold, 0.015, 0.060)

        pattern_threshold = self.base_pattern_confidence
        if "BULL" in regime.upper(): pattern_threshold *= 0.90
        else: pattern_threshold *= 0.95

        # T2.7: Session-conditioned confidence thresholds (data-driven, inverted
        # from conventional wisdom — Asian session is NOT low quality for this bot).
        # Simulation data:
        #   TF Asian (00-08 UTC):    62% WR, +114% P&L → LOWER barrier (0.85x)
        #   MR London (08-12 UTC):   48% WR,  -6.5% P&L → RAISE barrier (1.20x)
        #   MR NY Close (17-21 UTC): 67% WR, +19.7% P&L → LOWER barrier (0.85x)
        try:
            current_hour_utc = datetime.utcnow().hour
            strategy_lower = strategy.lower() if strategy else ""
            if 0 <= current_hour_utc < 8:
                # Asian: TF thrives — lower the barrier for all strategies
                pattern_threshold *= 0.85
            elif 8 <= current_hour_utc < 12:
                # London open: MR specifically struggles here
                if "mean_reversion" in strategy_lower or "reversion" in strategy_lower:
                    pattern_threshold *= 1.20
            elif 17 <= current_hour_utc < 21:
                # NY Close: MR's best session
                if "mean_reversion" in strategy_lower or "reversion" in strategy_lower:
                    pattern_threshold *= 0.85
        except Exception:
            pass  # Never let session logic block execution

        self.current_pattern_threshold = np.clip(pattern_threshold, 0.40, 0.75)

    def _check_support_resistance_fixed(
        self, asset: str, df: pd.DataFrame, current_price: float, signal: int, threshold: float
    ) -> dict:
        now = datetime.now()
        asset_cache = self.sr_cache.get(asset, {})
        last_update = asset_cache.get('updated_at')

        if not last_update or (now - last_update).total_seconds() > self.sr_update_interval:
            self._update_sr_levels(asset, df)
            asset_cache = self.sr_cache.get(asset, {})

        all_levels = asset_cache.get("levels", [])

        if not all_levels:
            return {"near_level": False, "reason": "no_sr_levels_found", "all_levels": []}

        if signal == 1:
            relevant_levels = [l for l in all_levels if l < current_price]
            level_type = "support"
        else:
            relevant_levels = [l for l in all_levels if l > current_price]
            level_type = "resistance"

        if not relevant_levels:
            # T2.4: ATR-based 20 EMA fallback distance.
            # Old code used a fixed % threshold that failed across asset price scales:
            # 0.3% on BTC = $200; on EUR/USD = 30 pips. ATR scales automatically.
            ema_20_series = df["close"].ewm(span=20, adjust=False).mean()
            current_ema = ema_20_series.iloc[-1]
            prev_ema = ema_20_series.iloc[-2]

            try:
                import talib as _ta
                _atr = _ta.ATR(
                    df['high'].values, df['low'].values, df['close'].values,
                    timeperiod=14
                )[-1]
            except Exception:
                _atr = current_price * 0.01  # 1% fallback if TA-Lib unavailable

            # AI-3 Fix: widened from 0.25×ATR to 0.5×ATR for EMA fallback.
            # Old 0.25×ATR only allowed price hugging the EMA (range-bound behaviour).
            # In trending markets price legitimately sits 0.5–1.0×ATR from the EMA;
            # the fallback would always fail, causing a chain of "dynamic_ema_too_far"
            # rejections even when the trend was clear.
            max_dist = 0.5 * _atr

            if signal == 1 and current_price > current_ema and current_ema > prev_ema:
                if abs(current_price - current_ema) <= max_dist:
                    return {
                        "near_level": True,
                        "level_type": "dynamic_ema_support",
                        "nearest_level": current_ema,
                        "distance_pct": ((current_price - current_ema) / current_price) * 100,
                        "reason": "riding_dynamic_20_ema"
                    }
                # AI-3 Fix: trending continuation check — 3 consecutive closes above rising EMA
                # confirms price is in a sustained uptrend even beyond 0.5×ATR from EMA.
                if len(df) >= 4:
                    last_3_closes = df["close"].iloc[-4:-1]
                    last_3_emas = ema_20_series.iloc[-4:-1]
                    if all(last_3_closes.values > last_3_emas.values):
                        return {
                            "near_level": True,
                            "level_type": "trending_continuation_support",
                            "nearest_level": current_ema,
                            "distance_pct": ((current_price - current_ema) / current_price) * 100,
                            "reason": "3bar_trending_continuation_above_ema"
                        }
                return {
                    "near_level": False,
                    "reason": "dynamic_ema_too_far",
                    "all_levels": all_levels
                }

            elif signal == -1 and current_price < current_ema and current_ema < prev_ema:
                if abs(current_price - current_ema) <= max_dist:
                    return {
                        "near_level": True,
                        "level_type": "dynamic_ema_resistance",
                        "nearest_level": current_ema,
                        "distance_pct": ((current_ema - current_price) / current_price) * 100,
                        "reason": "riding_dynamic_20_ema"
                    }
                # AI-3 Fix: trending continuation check — 3 consecutive closes below falling EMA
                if len(df) >= 4:
                    last_3_closes = df["close"].iloc[-4:-1]
                    last_3_emas = ema_20_series.iloc[-4:-1]
                    if all(last_3_closes.values < last_3_emas.values):
                        return {
                            "near_level": True,
                            "level_type": "trending_continuation_resistance",
                            "nearest_level": current_ema,
                            "distance_pct": ((current_ema - current_price) / current_price) * 100,
                            "reason": "3bar_trending_continuation_below_ema"
                        }
                return {
                    "near_level": False,
                    "reason": "dynamic_ema_too_far",
                    "all_levels": all_levels
                }

            # FALLBACK: Check if at level (boundary detection)
            any_level_distances = [abs(current_price - l) / current_price for l in all_levels]
            min_any_dist = min(any_level_distances) if any_level_distances else float("inf")
            if min_any_dist < threshold:
                closest = all_levels[np.argmin(any_level_distances)]
                return {"near_level": True, "level_type": "boundary", "nearest_level": closest, "distance_pct": min_any_dist * 100, "reason": f"at_level_${closest:.2f}"}
            return {"near_level": False, "reason": f"no_{level_type}_below/above"}

        distances = [(abs(current_price - level) / current_price, level) for level in relevant_levels]
        min_distance_pct, nearest_level = min(distances)
        near_level = min_distance_pct < threshold

        return {
            "near_level": near_level,
            "level_type": level_type,
            "nearest_level": nearest_level,
            "distance_pct": min_distance_pct * 100,
            "all_levels": all_levels,
            "reason": f"near_{level_type}_${nearest_level:.2f}" if near_level else f"{level_type}_too_far"
        }

    def _update_sr_levels(self, asset: str, df: pd.DataFrame):
        """More robust S/R level extraction with Anchored VWAP (Phase 3)."""
        pivots = self._extract_pivots(df, window=7)
        avwap_levels = self._calculate_anchored_vwaps(df)

        if len(pivots) < 3:
            closes = df["close"].values
            levels = np.percentile(closes, [10, 25, 50, 75, 90]).tolist()
            all_levels = sorted(list(set(levels + list(avwap_levels.values()))))
            self.sr_cache[asset] = {"levels": all_levels, "avwaps": avwap_levels, "updated_at": datetime.now(), "fallback_mode": True}
            return

        try:
            levels = self.analyst.get_support_resistance_levels(
                pivot_points=pivots,
                highs=df["high"].values,
                lows=df["low"].values,
                closes=df["close"].values,
                n_levels=7,
            )
            if not levels: levels = sorted(np.unique(pivots).tolist())
            all_levels = sorted(list(set(levels + list(avwap_levels.values()))))
            self.sr_cache[asset] = {"levels": all_levels, "avwaps": avwap_levels, "updated_at": datetime.now(), "fallback_mode": False}
        except Exception as e:
            logger.error(f"[SR UPDATE] Failed for {asset}: {e}")
            closes = df["close"].values
            levels = np.percentile(closes, [10, 30, 50, 70, 90]).tolist()
            all_levels = sorted(list(set(levels + list(avwap_levels.values()))))
            self.sr_cache[asset] = {"levels": all_levels, "avwaps": avwap_levels, "updated_at": datetime.now(), "fallback_mode": True}

    def _calculate_anchored_vwaps(self, df: pd.DataFrame) -> Dict[str, float]:
        try:
            temp_df = df.copy()
            if not isinstance(temp_df.index, pd.DatetimeIndex):
                if 'timestamp' in temp_df.columns:
                    temp_df.index = pd.to_datetime(temp_df['timestamp'])
                else: return {}
            
            last_date = temp_df.index[-1]
            week_start = last_date - timedelta(days=last_date.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            weekly_data = temp_df[temp_df.index >= week_start]
            w_vwap = (weekly_data['close'] * weekly_data['volume']).sum() / weekly_data['volume'].sum() if not weekly_data.empty and weekly_data['volume'].sum() > 0 else temp_df['close'].iloc[-1]

            month_start = last_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            monthly_data = temp_df[temp_df.index >= month_start]
            m_vwap = (monthly_data['close'] * monthly_data['volume']).sum() / monthly_data['volume'].sum() if not monthly_data.empty and monthly_data['volume'].sum() > 0 else temp_df['close'].iloc[-1]

            return {"weekly_avwap": w_vwap, "monthly_avwap": m_vwap}
        except Exception as e:
            logger.error(f"[AVWAP] Error: {e}")
            return {}

    def clear_pattern_cache(self):
        """Clear the within-cycle pattern cache. Call at the start of each evaluation cycle."""
        self._pattern_cache.clear()

    def _check_pattern(self, df: pd.DataFrame, signal: int, min_confidence: float = 0.60, strategy: str = "UNKNOWN") -> dict:
        try:
            if len(df) < 15: return {"pattern_confirmed": False, "reason": "insufficient_data"}

            # AI-5: Cache check — avoid triple neural net inference per council signal.
            # Key: last close price (proxy for df identity) + signal direction.
            try:
                cache_key = (round(float(df["close"].iloc[-1]), 6), signal)
                if cache_key in self._pattern_cache:
                    cached = self._pattern_cache[cache_key]
                    # Re-apply min_confidence gate since different callers may use different thresholds
                    if cached.get("pattern_confirmed") and cached.get("confidence", 0) < min_confidence:
                        return {**cached, "pattern_confirmed": False, "reason": "low_confidence_recalc"}
                    return cached
            except Exception:
                cache_key = None  # Cache miss — proceed normally

            snippet = df[["open", "high", "low", "close"]].iloc[-15:].values
            if snippet[0, 0] <= 0: return {"pattern_confirmed": False, "reason": "invalid_data"}
            snippet_input = (snippet / snippet[0, 0] - 1).reshape(1, 15, 4)
            predicted_id, confidence = self.sniper.predict_single(snippet_input)
            pattern_name = self.reverse_pattern_map.get(predicted_id, "Unknown")
            
            # --- Noise Filter ---
            if "noise" in pattern_name.lower():
                return {
                    "pattern_confirmed": False,
                    "confidence": 0,
                    "reason": "noise_detected"
                }

            if predicted_id == 0:
                # Fix: old code returned pattern_confirmed=True when confidence < 0.70 —
                # inverse logic (uncertain "no pattern" = confirmed pattern). Corrected:
                # - High confidence no-pattern (≥ 0.70): model is sure there's nothing → fail
                # - Low confidence no-pattern (< 0.70): model is uncertain → soft-pass with flag
                #   so validate_signal() can use S/R-only approval path instead of hard-failing.
                if confidence >= 0.70:
                    return {"pattern_confirmed": False, "reason": "no_pattern_detected", "pattern_name": "Noise", "confidence": confidence, "model_uncertain": False}
                else:
                    return {"pattern_confirmed": False, "reason": "model_uncertain_no_pattern", "pattern_name": "Uncertain", "confidence": confidence, "model_uncertain": True}

            # Alignment check
            is_bullish = pattern_name in self.BULLISH_PATTERNS
            is_bearish = pattern_name in self.BEARISH_PATTERNS
            
            if signal == 1 and not is_bullish: return {"pattern_confirmed": False, "reason": "direction_mismatch"}
            if signal == -1 and not is_bearish: return {"pattern_confirmed": False, "reason": "direction_mismatch"}

            # ================================================================
            # MR PATTERN CONFIRMATION: Institutional Reversal Pattern List
            # AI-4 Fix: expanded from 3 to 7 per direction. Original list blocked
            # valid reversal patterns (Harami, Piercing, Inverted Hammer, Dragonfly
            # Doji) that are established in institutional reversal playbooks.
            # ================================================================
            if strategy.upper() == "REVERSION" or strategy == "mean_reversion":
                allowed_long = [
                    "hammer", "morning_star", "bullish_engulfing",
                    "harami", "bullish_harami", "piercing",
                    "inverted_hammer", "dragonfly_doji", "three_inside",
                ]
                allowed_short = [
                    "shooting_star", "evening_star", "bearish_engulfing",
                    "bearish_harami", "dark_cloud", "dark_cloud_cover",
                    "gravestone_doji", "hanging_man", "three_outside",
                ]

                # Normalize for matching
                norm_pattern = pattern_name.lower().replace(" ", "_")

                if signal == 1 and norm_pattern not in allowed_long:
                    logger.info(f"[AI] MR Blocked: Pattern '{pattern_name}' is not in allowed institutional list for LONG.")
                    return {
                        "pattern_confirmed": False,
                        "confidence": 0.0,
                        "reason": f"unsupported_mr_pattern_{norm_pattern}"
                    }

                if signal == -1 and norm_pattern not in allowed_short:
                    logger.info(f"[AI] MR Blocked: Pattern '{pattern_name}' is not in allowed institutional list for SHORT.")
                    return {
                        "pattern_confirmed": False,
                        "confidence": 0.0,
                        "reason": f"unsupported_mr_pattern_{norm_pattern}"
                    }
            
            # --- Volume Weighting ---
            if 'volume' in df.columns and len(df) > 20:
                avg_vol = df['volume'].iloc[-21:-1].mean()
                volume = df['volume'].iloc[-1]
                if volume > (2.0 * avg_vol):
                    min_confidence = max(0.45, min_confidence - 0.20)

            if confidence < min_confidence:
                result = {"pattern_confirmed": False, "reason": "low_confidence", "confidence": confidence}
                if cache_key:
                    self._pattern_cache[cache_key] = result
                return result

            result = {"pattern_confirmed": True, "pattern_name": pattern_name, "confidence": confidence}
            if cache_key:
                self._pattern_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"[PATTERN] Error: {e}")
            return {"pattern_confirmed": False, "reason": "error"}

    def _approve_signal(self, signal: int, signal_details: dict, sr_result: dict, pattern_result: dict, strategy: str, validation_time: float, df: Optional[pd.DataFrame] = None) -> Tuple[int, dict]:
        self.rejection_window.append(False)
        pattern_conf = pattern_result.get("confidence", 0)
        boost = 0.10
        if pattern_conf > 0.80: boost += 0.05
        
        regime = signal_details.get("regime", "NEUTRAL")
        current_price = float(df['close'].iloc[-1]) if df is not None else 0

        # AI Confluence Bonus
        if df is not None and len(df) >= 14:
            import talib as ta
            atr = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)[-1]
            distance = sr_result.get("distance_pct", 999) / 100 * current_price
            if distance < (0.25 * atr): boost += 0.15

        # AVWAP Floor Bonus
        asset = signal_details.get("asset", "UNKNOWN")
        asset_cache = self.sr_cache.get(asset, {})
        weekly_avwap = asset_cache.get("avwaps", {}).get("weekly_avwap")
        
        if signal == 1 and regime == "SLIGHTLY_BULLISH" and weekly_avwap:
            if abs(current_price - weekly_avwap) / weekly_avwap < 0.003:
                boost += 0.25
                logger.info(f"  🏛️ INSTITUTIONAL FLOOR: +25% boost (Price at Weekly AVWAP)")

        return signal, {**signal_details, "ai_validation": "approved", "ai_sr_check": sr_result, "ai_pattern_check": pattern_result, "confidence_boost": boost}

    def _extract_pivots(self, df: pd.DataFrame, window=7) -> np.ndarray:
        highs, lows = df["high"].values, df["low"].values
        pivots = []
        for w in [window, 5, 3]:
            pivots = []
            for i in range(w, len(df) - w):
                if highs[i] == max(highs[i - w : i + w + 1]): pivots.append(highs[i])
                if lows[i] == min(lows[i - w : i + w + 1]): pivots.append(lows[i])
            if len(pivots) >= 3: break
        return np.array(pivots)

    def _reject_signal(self, details: dict, sr: dict, pattern: Optional[dict], reason: str, strategy: str) -> Tuple[int, dict]:
        self.rejection_window.append(True)
        self._check_circuit_breaker()
        return 0, {
            **details, 
            "ai_validation": "rejected", 
            "ai_rejection_reason": reason, 
            "ai_sr_check": sr,
            "ai_pattern_check": pattern,
            "final_signal": 0
        }

    def _skip_validation(self, signal: int, details: dict, reason: str) -> Tuple[int, dict]:
        return signal, {**details, "ai_validation": f"skipped_{reason}"}

    def _bypass_validation(self, signal: int, details: dict, reason: str, **kwargs) -> Tuple[int, dict]:
        return signal, {**details, "ai_validation": f"bypassed_{reason}"}

    def _check_circuit_breaker(self):
        if len(self.rejection_window) < 30: return
        if sum(self.rejection_window) / len(self.rejection_window) > self.bypass_threshold:
            self.bypass_mode = True
            # AI-4 Fix: raised from 15 to 30 cycles (was 75 min, now 150 min).
            # At 15 cycles the circuit breaker reset too quickly — the model would
            # re-enter rejection mode within the same session if the root cause
            # (bad market regime, stale model) hadn't been resolved.
            self.bypass_cooldown = 30

    def _reset_circuit_breaker(self):
        self.bypass_mode = False
        self.rejection_window.clear()

    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics for the monitor.
        """
        total = max(self.stats["total_checks"], 1)
        
        # Calculate rates
        approval_rate = (self.stats["approved"] / total) * 100
        rejection_rate = (self.stats["rejected"] / total) * 100
        
        # Sort and get top rejection reasons
        top_reasons = dict(sorted(self.rejection_reasons.items(), key=lambda item: item[1], reverse=True)[:5])
        
        return {
            "total_checks": self.stats["total_checks"],
            "approved": self.stats["approved"],
            "rejected": self.stats["rejected"],
            "approval_rate": f"{approval_rate:.1f}%",
            "rejection_rate": f"{rejection_rate:.1f}%",
            "rejection_breakdown": {
                "no_sr_level": self.stats["rejected_no_sr"],
                "no_pattern": self.stats["rejected_no_pattern"],
                "low_confidence": self.stats["rejected_low_confidence"],
                "direction_mismatch": self.stats["rejected_direction_mismatch"],
            },
            "bypasses": {
                "strong_signal": self.stats["bypassed_strong_signal"],
                "circuit_breaker": self.stats["bypassed_circuit_breaker"],
            },
            "current_thresholds": {
                "sr_threshold": f"{self.current_sr_threshold:.2%}",
                "pattern_confidence": f"{self.current_pattern_threshold:.0%}",
            },
            "adaptive_adjustments": self.stats["adaptive_adjustments"],
            "circuit_breaker": {
                "active": self.bypass_mode,
                "cooldown": self.bypass_cooldown,
            },
            "top_rejection_reasons": top_reasons,
            "per_strategy": dict(self.strategy_stats)
        }
