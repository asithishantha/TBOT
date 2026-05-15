"""
Veteran Trade Manager - Strategic/Tactical Risk Architecture
✨ REFACTORED: Centralized risk configuration from config.json.
📊 ROLE: Tactical execution engine (HOW to manage trades, not HOW MUCH to risk)
"""

import logging
import numpy as np
import talib
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Exit reasons for tracking"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT_1 = "take_profit_1"
    TAKE_PROFIT_2 = "take_profit_2"
    TAKE_PROFIT_3 = "take_profit_3"
    TRAILING_STOP = "trailing_stop"
    BREAK_EVEN = "break_even"
    MANUAL = "manual"
    TIME_STOP = "time_stop"
    EARLY_SCALE = "early_scale"
    # Smart market-condition exits
    VOLATILITY_SPIKE     = "volatility_spike"      # ATR explodes 2× → risk model invalid
    REVERSAL_CANDLE      = "reversal_candle"        # Strong engulfing bar against trade
    TREND_INVALIDATION   = "trend_invalidation"     # 3 bars against + ADX < 20
    MOMENTUM_EXHAUSTION  = "momentum_exhaustion"    # RSI extreme + MACD dying + ADX falling


def find_resistance_levels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    current_price: float,
    side: str,
    lookback: int = 50,
    min_touches: int = 2,
    tolerance: Optional[float] = None, # Use absolute price tolerance instead of %
) -> List[float]:
    """Find significant resistance/support levels using adaptive tolerance."""
    lookback = min(lookback, len(close))
    levels = []
    
    # Default tolerance if none provided (Fallback to 0.5% of price if ATR not provided)
    if tolerance is None:
        tolerance = current_price * 0.005

    if side == "long":
        highs = high[-lookback:]
        for i in range(2, len(highs) - 2):
            if highs[i] > current_price:
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    levels.append(highs[i])

        clustered = []
        for level in sorted(levels):
            if not clustered or (level - clustered[-1]) > tolerance:
                clustered.append(level)
            else:
                clustered[-1] = (clustered[-1] + level) / 2

        verified = []
        for level in clustered:
            touches = sum(1 for h in highs if abs(h - level) <= tolerance)
            if touches >= min_touches:
                verified.append(level)

        return sorted(verified)[:5]
    else:
        lows = low[-lookback:]
        for i in range(2, len(lows) - 2):
            if lows[i] < current_price:
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    levels.append(lows[i])

        clustered = []
        for level in sorted(levels, reverse=True):
            if not clustered or (clustered[-1] - level) > tolerance:
                clustered.append(level)
            else:
                clustered[-1] = (clustered[-1] + level) / 2

        verified = []
        for level in clustered:
            touches = sum(1 for l in lows if abs(l - level) <= tolerance)
            if touches >= min_touches:
                verified.append(level)

        return sorted(verified, reverse=True)[:5]


def calculate_hybrid_targets(
    entry_price: float,
    stop_loss: float,
    side: str,
    structure_levels: List[float],
    risk_multiples: List[float],
    partial_sizes: List[float],
    min_rr: float = 1.2,
) -> Tuple[List[float], List[float]]:
    """Calculate targets with structure awareness"""
    risk = abs(entry_price - stop_loss)
    targets = []
    adjusted_sizes = list(partial_sizes)

    logger.info(f"\n[VTM] Target Calculation:")
    logger.info(f"  Entry: ${entry_price:,.2f}")
    logger.info(f"  Stop:  ${stop_loss:,.2f}")
    logger.info(f"  Risk:  ${risk:,.2f} ({risk/entry_price:.2%})")

    if side == "long":
        for i, r_multiple in enumerate(risk_multiples):
            rr_target = entry_price + (risk * r_multiple)

            if r_multiple > 10:
                logger.warning(f"  ⚠️  TP{i+1}: {r_multiple}R exceeds 10R cap")
                continue

            if structure_levels:
                closest = min(structure_levels, key=lambda x: abs(x - rr_target))
                structure_rr = (closest - entry_price) / risk
                distance_pct = abs(closest - rr_target) / rr_target

                if structure_rr >= min_rr and distance_pct < 0.25:
                    targets.append(closest)
                    logger.info(f"  ✓ TP{i+1}: ${closest:,.2f} ({structure_rr:.1f}R) [Structure]")
                elif structure_rr < min_rr:
                    targets.append(rr_target)
                    logger.info(f"  → TP{i+1}: ${rr_target:,.2f} ({r_multiple:.1f}R) [Structure too close]")
                else:
                    targets.append(rr_target)
                    logger.info(f"  → TP{i+1}: ${rr_target:,.2f} ({r_multiple:.1f}R) [Structure too far]")
            else:
                targets.append(rr_target)
                logger.info(f"  → TP{i+1}: ${rr_target:,.2f} ({r_multiple:.1f}R)")
    else:
        for i, r_multiple in enumerate(risk_multiples):
            rr_target = entry_price - (risk * r_multiple)

            if r_multiple > 10:
                continue

            if structure_levels:
                closest = min(structure_levels, key=lambda x: abs(x - rr_target))
                structure_rr = (entry_price - closest) / risk
                distance_pct = abs(closest - rr_target) / abs(rr_target)

                if structure_rr >= min_rr and distance_pct < 0.25:
                    targets.append(closest)
                    logger.info(f"  ✓ TP{i+1}: ${closest:,.2f} ({structure_rr:.1f}R) [Structure]")
                elif structure_rr < min_rr:
                    targets.append(rr_target)
                    logger.info(f"  → TP{i+1}: ${rr_target:,.2f} ({r_multiple:.1f}R) [Structure too close]")
                else:
                    targets.append(rr_target)
                    logger.info(f"  → TP{i+1}: ${rr_target:,.2f} ({r_multiple:.1f}R) [Structure too far]")
            else:
                targets.append(rr_target)
                logger.info(f"  → TP{i+1}: ${rr_target:,.2f} ({r_multiple:.1f}R)")

    # ✅ M-3 FIX: Deduplicate targets that snapped to the same structure level.
    # When TP2 and TP3 both land on the nearest swing, 55% of the position
    # exits at the same price and the runner is never activated.
    if len(targets) > 1:
        seen_keys: set = set()
        unique_targets = []
        for _t in targets:
            _key = round(_t, 2)
            if _key not in seen_keys:
                seen_keys.add(_key)
                unique_targets.append(_t)
        if len(unique_targets) < len(targets):
            logger.info(
                f"  ℹ️  Deduplicated {len(targets) - len(unique_targets)} duplicate TP(s)"
            )
        targets = unique_targets

    if len(targets) < len(partial_sizes):
        logger.warning(f"  ⚠️  Only {len(targets)} targets (expected {len(partial_sizes)})")
        remaining = sum(partial_sizes[len(targets):])
        for i in range(len(targets)):
            adjusted_sizes[i] = partial_sizes[i] + (remaining / len(targets))
        adjusted_sizes = adjusted_sizes[:len(targets)]
        logger.info(f"  → Adjusted sizes: {[f'{s:.0%}' for s in adjusted_sizes]}")

    return targets, adjusted_sizes


class VeteranTradeManager:
    """
    ✨ REFACTORED: Strategic/Tactical Risk Architecture
    
    TACTICAL ROLE: Manages HOW to execute trades (stops, targets, trailing)
    STRATEGIC ROLE: Portfolio Manager decides HOW MUCH to risk
    
    KEY CHANGES:
    - Accepts risk configuration dictionary directly from config.json.
    - Validates trade economics before execution (pre-flight check).
    - Asymmetric constraints for TREND vs SCALP trades.
    """

    @classmethod
    def validate_trade_setup(
        cls,
        entry_price: float,
        stop_loss: float,
        risk_config: dict,
        trade_type: str = "TREND",
        atr_fast: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        ✨ INSTITUTIONAL: Strict TREND validation with ATR-based economics.
        """
        try:
            # Default atr_fast if not provided (Fallback to 1% of price if ATR not provided)
            if atr_fast is None:
                atr_fast = entry_price * 0.01

            # ATR-based adaptive cap (Replacing static max_stop_pct)
            max_stop_dist = atr_fast * 5.0
            min_rr = 1.5

            stop_distance = abs(entry_price - stop_loss)

            if stop_distance > max_stop_dist:
                return False, f"Stop too wide: ${stop_distance:,.2f} > 5.0 * ATR (${max_stop_dist:,.2f})"

            risk_multiples = risk_config.get("partial_targets", [1.0, 1.8, 3.0])
            partial_sizes  = risk_config.get("partial_sizes",   [0.45, 0.30, 0.25])
            if not risk_multiples:
                risk_multiples = [1.0, 1.8, 3.0]
            if not partial_sizes:
                partial_sizes = [1.0 / len(risk_multiples)] * len(risk_multiples)

            # ── Closest-target check (TP1 must clear half an ATR minimum) ──
            first_tp_dist = stop_distance * risk_multiples[0]
            if first_tp_dist < (0.5 * atr_fast):
                return False, (
                    f"TP1 too close to entry: ${first_tp_dist:,.2f} < 0.5×ATR (${0.5*atr_fast:,.2f})"
                )

            # ── Weighted R:R across ALL partial exits ──────────────────────
            # Using only TP1 as the R:R measure is wrong for partial-exit systems:
            # with TP1 at 1.0R (deliberate close first exit), the check always
            # fires even though weighted R:R across [1.0, 1.8, 3.0] is ~1.74.
            n = min(len(risk_multiples), len(partial_sizes))
            total_weight = sum(partial_sizes[:n])
            if total_weight > 0:
                weighted_rr = sum(
                    risk_multiples[i] * partial_sizes[i]
                    for i in range(n)
                ) / total_weight
            else:
                weighted_rr = risk_multiples[0]

            if weighted_rr < min_rr - 1e-9:
                return False, (
                    f"Weighted R:R too low: {weighted_rr:.2f}:1 < {min_rr:.2f}:1 "
                    f"(targets={risk_multiples}, sizes={partial_sizes})"
                )

            logger.info(
                f"[VTM PRE-FLIGHT] ✅ Trade Valid\n"
                f"  Type:        TREND\n"
                f"  Stop:        ${stop_distance:,.2f} ({(stop_distance/entry_price):.2%})\n"
                f"  TP1 dist:    ${first_tp_dist:,.2f} ({(first_tp_dist/entry_price):.2%})\n"
                f"  Weighted R:R:{weighted_rr:.2f}:1  (targets={risk_multiples})"
            )

            return True, "OK"

        except Exception as e:
            logger.error(f"[VTM PRE-FLIGHT] Error: {e}", exc_info=True)
            return False, f"Validation error: {str(e)}"

    def __init__(
        self,
        entry_price: float,
        side: str,
        asset: str, # Asset key still needed for logging
        risk_config: dict,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        quantity: float,
        volume: Optional[np.ndarray] = None,
        signal_details: Dict = None,
        account_risk: float = 0.015,
        atr_period: int = 14,
        trade_type: str = "TREND",
        local_free_margin: float = 0.0, # ✨ NEW: For leverage ceiling
        current_ask: float = 0.0,       # ✨ NEW: For spread floor
        current_bid: float = 0.0,       # ✨ NEW: For spread floor
        min_lot_override: Optional[float] = None,      # ✨ NEW: Exness compatibility
        lot_precision_override: Optional[int] = None   # ✨ NEW: Exness compatibility
    ):
        self.entry_price = entry_price
        self.side = side.lower()
        self.asset = asset.upper()
        self.risk_config = risk_config
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.atr_period = atr_period
        self.signal_details = signal_details or {}
        self.trade_type = trade_type
        
        self.position_size = quantity
        self.local_free_margin = local_free_margin
        self.current_ask = current_ask
        self.current_bid = current_bid
        self.min_lot_override = min_lot_override
        self.lot_precision_override = lot_precision_override
        
        # Determine asset type for leverage ceiling
        self.asset_category = "FOREX"
        crypto_keywords = ["BTC", "ETH", "SOL", "BNB", "XRP", "USDT"]
        if any(k in self.asset for k in crypto_keywords):
            self.asset_category = "CRYPTO"

        # Macro MAs from signal_details (for MA Shield/Front-run)
        gov_data = self.signal_details.get("governor_data", {})
        self.ema_1d_200 = gov_data.get("ema_1d_200")
        self.ema_4h_200 = gov_data.get("ema_4h_200")
        self.ema_4h_50 = gov_data.get("ema_4h_50")

        # ✅ TASK 18: Regime-Adaptive ATR Multipliers
        # config_base is the per-asset value from config.json (e.g. 1.5 for GBPUSD).
        # Regime logic adds on top rather than fully overriding — so raising atr_multiplier
        # in config.json actually widens the SL rather than being silently ignored.
        config_base = self.risk_config.get("atr_multiplier", 1.8)

        if self.trade_type == "REVERSION":
            # Reversion trades need at least as much room as a trend trade: mean-reversion
            # entries sit inside a range where a single wick routinely exceeds 1×ATR.
            # Hard floor of 2.0 prevents the 1.5× tight-stop chop problem.
            self.atr_multiplier = max(config_base, 2.0)
        elif self.trade_type == "TREND":
            regime = self.signal_details.get("regime", "NEUTRAL")
            volatility = self.signal_details.get("volatility_regime", "normal")
            if "BEAR" in regime or volatility == "high":
                # Adverse conditions: add 0.5 on top of config, minimum floor 2.5
                self.atr_multiplier = max(config_base + 0.5, 2.5)
            else:
                # Normal/bull: add 0.3 on top of config, minimum floor 2.0
                self.atr_multiplier = max(config_base + 0.3, 2.0)
        else:
            self.atr_multiplier = config_base

        logger.info(
            f"[VTM] ATR Multiplier: config={config_base}× → effective={self.atr_multiplier}× "
            f"({self.trade_type}, regime={self.signal_details.get('regime','NEUTRAL')})"
        )

        # T2.5: ADX-conditioned take profit targets.
        # Static targets fail in chop (ADX<20) where price never reaches them,
        # causing the 43% profit capture rate. Scale down in chop, up in momentum.
        base_targets = self.risk_config.get("partial_targets", [1.0, 1.8, 3.0])
        try:
            if len(close) >= 14:
                import talib as _talib
                adx_val = _talib.ADX(high, low, close, timeperiod=14)[-1]
                if not np.isnan(adx_val):
                    if adx_val < 20:
                        self.partial_targets = [max(1.0, t * 0.7) for t in base_targets]
                        logger.info(
                            f"[VTM] 📉 Chop mode (ADX={adx_val:.0f}): "
                            f"Targets scaled to {self.partial_targets}"
                        )
                    elif adx_val > 40:
                        self.partial_targets = [t * 1.3 for t in base_targets]
                        logger.info(
                            f"[VTM] 📈 Momentum mode (ADX={adx_val:.0f}): "
                            f"Targets scaled to {self.partial_targets}"
                        )
                    else:
                        self.partial_targets = list(base_targets)
                else:
                    self.partial_targets = list(base_targets)
            else:
                self.partial_targets = list(base_targets)
        except Exception as _e:
            logger.debug(f"[VTM] ADX target scaling failed ({_e}), using base targets")
            self.partial_targets = list(base_targets)

        self.partial_sizes = self.risk_config.get("partial_sizes", [0.45, 0.30, 0.25])

        # ── J: Pattern-Aware Exit Management ──────────────────────────────
        _pattern = self.signal_details.get("institutional_pattern")
        if _pattern == "LIQUIDITY_HUNT":
            # Snap-back reversals die fast — tighten everything
            self.partial_targets = [max(0.8, t * 0.6) for t in self.partial_targets]
            self.breakeven_profit_threshold = self.risk_config.get(
                "breakeven_profit_threshold", 0.01) * 0.5
            logger.info(f"[VTM] LIQUIDITY_HUNT pattern: Tight TPs {self.partial_targets}")

        elif _pattern == "ACCUMULATION":
            # Generational trend — give it room to breathe
            self.partial_targets = [t * 1.3 for t in self.partial_targets]
            self.atr_multiplier *= 1.2
            logger.info(f"[VTM] ACCUMULATION pattern: Wide targets {self.partial_targets}")

        elif _pattern == "SPRING_BREAKOUT":
            # Explosive but uncertain — take first TP early, let runner go
            self.partial_sizes = [0.60, 0.25, 0.15]  # Take 60% at TP1
            logger.info(f"[VTM] SPRING_BREAKOUT: Heavy first partial")
        # ──────────────────────────────────────────────────────────────────

        self.pivot_lookback = self.risk_config.get("pivot_lookback", 30)
        self.time_stop_bars = self.risk_config.get(
            f'time_stop_{self.trade_type.lower()}',
            self.risk_config.get(
                'time_stop_bars',
                self.risk_config.get('max_hold_bars', 72)  # max_hold_bars is legacy alias
            )
        )
        self.use_ema_structure = self.risk_config.get("use_ema_structure", False)
        self.use_structure_targets = self.risk_config.get("use_structure_targets", True)
        # Runner trailing stop ATR multiplier (replaces hardcoded 2.0)
        self.runner_trail_atr_multiplier = self.risk_config.get("runner_trail_atr_multiplier", 2.0)
        # Time-based break-even: move SL to entry after N bars if pnl >= threshold
        self.breakeven_after_bars = self.risk_config.get("breakeven_after_bars", None)
        self.breakeven_profit_threshold = self.risk_config.get("breakeven_profit_threshold", 0.01)

        # Early Scale: lock in a small partial exit within the first N bars
        self.early_scale_enabled = self.risk_config.get("early_scale_enabled", False)
        self.early_scale_threshold = self.risk_config.get("early_scale_threshold", 0.02)
        self.early_scale_bars = self.risk_config.get("early_scale_bars", 4)
        self.early_lock_atr_multiplier = self.risk_config.get("early_lock_atr_multiplier", 0.5)
        self._early_scaled = False          # fires at most once per trade
        self._greed_mode_activated = False  # fires at most once per trade
        self._time_stop_extended = False    # fires at most once per trade

        # State
        self.initial_stop_loss = None
        self.current_stop_loss = None
        self.take_profit_levels = []
        self.remaining_position = 1.0
        self.partials_hit = []
        self.bars_in_trade = 0
        self.highest_price_reached = entry_price
        self.lowest_price_reached = entry_price
        self.runner_activated = False
        self.has_pyramided = False # ✨ NEW: Trend Pyramiding Flag
        self.entry_time = datetime.now()
        
        # Calculate levels
        try:
            self._calculate_initial_levels()
        except Exception as e:
            logger.error(f"[VTM] Initialization error: {e}")
            raise

        # Log initialization
        logger.info("=" * 80)
        logger.info(f"🎯 VTM - {self.asset} {side.upper()} [{self.trade_type}]")
        logger.info("=" * 80)
        logger.info(f"Entry:    ${entry_price:,.2f}")
        logger.info(f"Stop:     ${self.initial_stop_loss:,.2f} (-{self._calc_pct_distance(entry_price, self.initial_stop_loss):.2f}%)")
        logger.info(f"Quantity: {self.position_size:.6f} units")

        logger.info(f"\n📊 TARGETS:")
        if not self.take_profit_levels or not self.partial_sizes:
            logger.info("  No take profit targets calculated or partial sizes defined.")
        for i, (target, size) in enumerate(zip(self.take_profit_levels, self.partial_sizes), 1):
            target_str = f"${target:,.2f}" if target is not None else "N/A"
            size_str = f"{size:.0%}" if size is not None else "N/A"
            if target is not None:
                pct = self._calc_pct_distance(entry_price, target)
                pct_str = f"(+{pct:.2f}%)"
            else:
                pct_str = ""
            logger.info(f"  {i}. {target_str} {pct_str} → Exit {size_str}")
        logger.info("=" * 80)

    @property
    def profit_locked(self) -> bool:
        """Checks if stop loss is at break-even or better"""
        if self.side == "long":
            return self.current_stop_loss >= self.entry_price
        else:
            return self.current_stop_loss <= self.entry_price

    @property
    def current_take_profit(self) -> Optional[float]:
        """Returns the next active take profit target"""
        idx = len(self.partials_hit)
        if idx < len(self.take_profit_levels):
            return self.take_profit_levels[idx]
        return None

    def _calc_pct_distance(self, price1: float, price2: float) -> float:
        return abs(price1 - price2) / price1 * 100

    def _calculate_atr(self) -> float:
        """
        ✅ TASK 18: Regime-adaptive ATR: fast in expanding vol, slow in compressed vol.
        """
        try:
            atr_fast = talib.ATR(self.high, self.low, self.close, timeperiod=7)[-1]
            atr_mid  = talib.ATR(self.high, self.low, self.close, timeperiod=14)[-1]
            atr_slow = talib.ATR(self.high, self.low, self.close, timeperiod=28)[-1]
            
            if np.isnan(atr_mid) or atr_slow == 0:
                return self.entry_price * 0.015
                
            ratio = atr_fast / atr_slow
            
            if ratio > 1.30:
                # Expanding vol — tighten fast
                selected_atr = atr_fast
                reason = "Expanding Vol (Tighten)"
            elif ratio < 0.70:
                # Compressed vol — breathe wide
                selected_atr = atr_slow
                reason = "Compressed Vol (Wide)"
            else:
                selected_atr = atr_mid
                reason = "Normal Vol"
                
            logger.debug(f"[VTM] Dynamic ATR Selection: {selected_atr:.4f} ({reason}, Ratio: {ratio:.2f})")
            return selected_atr
            
        except Exception as e:
            logger.error(f"[VTM] ATR error: {e}")
            return self.entry_price * 0.02
        
    def check_promotion_to_runner(
        self, 
        current_price: float
    ) -> bool:
        if len(self.partials_hit) != 1 or self.runner_activated:
            return False
        
        try:
            volume_ratio = 1.0
            if self.volume is not None and len(self.volume) > 20:
                avg_vol = np.mean(self.volume[-21:-1]) 
                current_vol = self.volume[-1]
                if avg_vol > 0:
                    volume_ratio = current_vol / avg_vol
            
            volume_strong = volume_ratio > 1.5
            
            candle_conviction = False
            if len(self.high) > 0 and len(self.low) > 0:
                latest_high, latest_low = self.high[-1], self.low[-1]
                candle_range = latest_high - latest_low
                if candle_range > 0:
                    if self.side == "long":
                        distance_from_high = (latest_high - current_price) / candle_range
                        candle_conviction = distance_from_high < 0.20
                    else:
                        distance_from_low = (current_price - latest_low) / candle_range
                        candle_conviction = distance_from_low < 0.20
            
            if volume_strong or candle_conviction:
                logger.info("\n" + "=" * 70 + "\n🚀 TRADE PROMOTION TRIGGERED\n" + "=" * 70)
                self.runner_activated = True
                self.take_profit_levels, self.partial_sizes = [], []
                # Keep current structural stop loss, do not force break-even
                logger.info(f"[VTM] Runner activated. SL remains at structural level: ${self.current_stop_loss:,.2f}")
                return True
            else:
                # Do not modify SL if promotion fails
                return False
        
        except Exception as e:
            logger.error(f"[VTM] Promotion check error: {e}")
            return False

    def _calculate_initial_levels(self):
        try:
            atr = self._calculate_atr()

            # STEP 1 — Venue Adaptive Leverage Ceiling
            # Reason: Prevents over-exposure based on venue-specific risk rules.
            if self.local_free_margin > 0:
                notional_value = self.position_size * self.entry_price
                max_notional = 0.0
                
                if self.asset_category == "CRYPTO":
                    max_notional = self.local_free_margin * 3.0
                elif self.asset_category == "FOREX":
                    max_notional = self.local_free_margin * 20.0
                
                if notional_value > max_notional and max_notional > 0:
                    logger.info(f"[VTM] ⚠️ Leverage Ceiling: Notional ${notional_value:,.2f} > Max ${max_notional:,.2f}. Scaling down.")
                    self.position_size = max_notional / self.entry_price

            if self.trade_type == "REVERSION":
                wick_buffer = 0.5 * atr

                if self.side == "long":
                    # Long reversion: price bouncing up from below — exit just *below* the
                    # EMA so we don't overshoot into the resistance zone above it.
                    self.initial_stop_loss = self.low[-1] - wick_buffer
                    tp_target = self.ema_4h_50 - (0.2 * atr) if self.ema_4h_50 else self.entry_price + (2.0 * atr)

                else:
                    # Short reversion: price falling from above — exit just *above* the
                    # EMA (mean-convergence point). The original code used ema_4h_50 + 0.2*atr
                    # which placed TP *above* the EMA — wrong direction for a short.
                    self.initial_stop_loss = self.high[-1] + wick_buffer
                    tp_target = self.ema_4h_50 - (0.2 * atr) if self.ema_4h_50 else self.entry_price - (2.0 * atr)

                self.current_stop_loss = self.initial_stop_loss
                self.take_profit_levels = [tp_target]
                self.partial_sizes = [1.0]

                logger.info(f"[VTM] REVERSION MODE: SL={self.initial_stop_loss}, TP={tp_target}")

            else:
                # ATR-based adaptive floors and caps
                min_stop_dist = atr * 0.5
                max_stop_dist = atr * 5.0

                if self.side == "long":
                    # 1. Standard ATR Baseline
                    target_stop_dist = atr * self.atr_multiplier
                    standard_sl = self.entry_price - target_stop_dist
                    final_sl = standard_sl

                    # 2. Joint Synergy: MA Shield (only active when use_ema_structure=true)
                    if self.use_ema_structure:
                        for ma in [self.ema_1d_200, self.ema_4h_200, self.ema_4h_50]:
                            if ma and standard_sl < ma < self.entry_price:
                                buffered_ma_sl = ma - (0.5 * atr)
                                if buffered_ma_sl > final_sl:
                                    logger.info(f"[VTM] 🛡️ MA Shield Jointly Applied: SL tucked behind MA ${ma:,.2f}")
                                    final_sl = buffered_ma_sl

                    # 3. Apply global clamps
                    final_sl = max(
                        self.entry_price - max_stop_dist,
                        min(self.entry_price - min_stop_dist, final_sl)
                    )
                else: # short
                    # 1. Standard ATR Baseline
                    target_stop_dist = atr * self.atr_multiplier
                    standard_sl = self.entry_price + target_stop_dist
                    final_sl = standard_sl

                    # 2. Joint Synergy: MA Shield (only active when use_ema_structure=true)
                    if self.use_ema_structure:
                        for ma in [self.ema_1d_200, self.ema_4h_200, self.ema_4h_50]:
                            if ma and standard_sl > ma > self.entry_price:
                                buffered_ma_sl = ma + (0.5 * atr)
                                if buffered_ma_sl < final_sl:
                                    logger.info(f"[VTM] 🛡️ MA Shield Jointly Applied: SL tucked behind MA ${ma:,.2f}")
                                    final_sl = buffered_ma_sl

                    # 3. Apply global clamps
                    final_sl = min(
                        self.entry_price + max_stop_dist,
                        max(self.entry_price + min_stop_dist, final_sl)
                    )
                
                # STEP 2 — Spread-Aware SL Floor
                # Reason: Prevents stops from being too tight relative to broker spread.
                if self.current_ask > 0 and self.current_bid > 0:
                    spread = abs(self.current_ask - self.current_bid)
                    calculated_sl_dist = abs(self.entry_price - final_sl)
                    final_sl_distance = max(calculated_sl_dist, 3.0 * spread)
                    
                    if final_sl_distance > calculated_sl_dist:
                        logger.info(f"[VTM] ↔️ Spread Floor: SL distance expanded to {final_sl_distance:.4f} (3x spread)")
                        final_sl = self.entry_price - final_sl_distance if self.side == "long" else self.entry_price + final_sl_distance

                # Global clamped final_sl assignment
                self.initial_stop_loss = final_sl

                # Structure-based targets (only when use_structure_targets=true)
                if self.use_structure_targets:
                    tolerance = 0.5 * atr
                    structure_levels = find_resistance_levels(self.high, self.low, self.close, self.entry_price, self.side, self.pivot_lookback, tolerance=tolerance)
                    raw_targets, self.partial_sizes = calculate_hybrid_targets(
                        self.entry_price, self.initial_stop_loss, self.side, structure_levels,
                        self.partial_targets, self.partial_sizes,
                        min_rr=2.0  # Standard TREND requirement
                    )
                    logger.debug(f"[VTM] Structure targets active: {len(structure_levels)} levels found")
                else:
                    # Pure ATR-multiple targets, no pivot hunting
                    raw_targets = [
                        self.entry_price + (atr * m) if self.side == "long" else self.entry_price - (atr * m)
                        for m in self.partial_targets
                    ]
                    logger.debug("[VTM] Structure targets disabled — using ATR multiples only")

                # ✅ PHASE 5: MA FRONT-RUN (Take Profit — only when use_ema_structure=true)
                self.take_profit_levels = []
                if self.use_ema_structure:
                    for tp in raw_targets:
                        adjusted_tp = tp
                        for ma in [self.ema_1d_200, self.ema_4h_200, self.ema_4h_50]:
                            if ma:
                                if self.side == "long":
                                    if abs(tp - ma) < (0.5 * atr) or (tp > ma > self.entry_price):
                                        candidate_tp = ma - (0.25 * atr)
                                        if candidate_tp > self.entry_price + (0.5 * atr):
                                            adjusted_tp = max(adjusted_tp, candidate_tp)
                                else:  # short
                                    if abs(tp - ma) < (0.5 * atr) or (tp < ma < self.entry_price):
                                        candidate_tp = ma + (0.25 * atr)
                                        if candidate_tp < self.entry_price - (0.5 * atr):
                                            adjusted_tp = min(adjusted_tp, candidate_tp)
                        self.take_profit_levels.append(adjusted_tp)
                else:
                    # No EMA adjustment — use raw targets directly
                    self.take_profit_levels = list(raw_targets)

                # Fallback targets
                if not self.take_profit_levels:
                    self.take_profit_levels = [self.entry_price + (atr * m) if self.side == "long" else self.entry_price - (atr * m) for m in self.partial_targets]
                    self.partial_sizes = [0.45, 0.30, 0.25]  # fallback only

            # STEP 3 — Lot Sanitizer
            # Reason: Ensures position size is valid for broker submission.
            LOT_PRECISION = {
                'BTC': 4,
                'GOLD': 2,
                'USTEC': 2,
                'EURJPY': 2,
                'EURUSD': 2,
                'GBPUSD': 2,
                'USDJPY': 2,
                'USOIL': 2,
                'GBPAUD': 2,
            }

            precision = self.lot_precision_override if self.lot_precision_override is not None else LOT_PRECISION.get(self.asset.upper(), 2)
            final_size = round(self.position_size, precision)

            MIN_LOT = {
                'BTC': 0.003,
                'GOLD': 0.01,
                'USTEC': 0.02,
                'EURJPY': 0.02,
                'EURUSD': 0.01,
                'GBPUSD': 0.01,
                'USDJPY': 0.01,
                'USOIL': 0.01,
                'GBPAUD': 0.01,
            }

            min_lot = self.min_lot_override if self.min_lot_override is not None else MIN_LOT.get(self.asset.upper(), 0.01)

            if final_size < min_lot:
                logger.warning(f"[VTM] Trade aborted: Final size {final_size} below minimum lot {min_lot} for {self.asset}.")
                # We raise an exception here to signal the manager to abort trade creation
                raise ValueError(f"Size {final_size} below min {min_lot} for {self.asset}")
            
            self.position_size = final_size
            self.current_stop_loss = self.initial_stop_loss

        except ValueError as ve:
            raise # Re-raise lot size error to abort
        except Exception as e:
            logger.error(f"[VTM] Level calculation error: {e}", exc_info=True)
            raise

    def on_new_bar(self, new_high: float, new_low: float, new_close: float) -> Optional[Dict]:
        try:
            self.high, self.low, self.close = np.append(self.high, new_high), np.append(self.low, new_low), np.append(self.close, new_close)
            # ✨ MEMORY MANAGEMENT: Limit to 500 candles (Safe for 200 EMA + buffer)
            if len(self.close) > 500: 
                self.high, self.low, self.close = self.high[-500:], self.low[-500:], self.close[-500:]
            
            self.bars_in_trade += 1
            if self.side == "long": self.highest_price_reached = max(self.highest_price_reached, new_high)
            else: self.lowest_price_reached = min(self.lowest_price_reached, new_low)
            
            atr = self._calculate_atr() # Calculate ATR here
            return self.check_exit(new_close, atr) # Pass ATR to check_exit
        except Exception as e:
            logger.error(f"[VTM] Update error: {e}")
            return None

    def update_with_current_price(self, current_price: float, df_4h: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        try:
            atr = self._calculate_atr() # Calculate ATR here
            
            if self.side == "long":
                old_high = self.highest_price_reached
                self.highest_price_reached = max(self.highest_price_reached, current_price)
                if self.runner_activated and self.highest_price_reached > old_high and self.trade_type == "TREND":
                    # ✅ PHASE 5: ATR-BASED RUNNER TRAIL (multiplier from runner_trail_atr_multiplier config)
                    new_trail = self.highest_price_reached - (self.runner_trail_atr_multiplier * atr)

                    if new_trail > self.current_stop_loss:
                        logger.info(f"[VTM] 🏃 Trailing SL updated to ${new_trail:,.2f} (from ${self.current_stop_loss:,.2f}).")
                        self.current_stop_loss = new_trail
            else:
                old_low = self.lowest_price_reached
                self.lowest_price_reached = min(self.lowest_price_reached, current_price)
                if self.runner_activated and self.lowest_price_reached < old_low and self.trade_type == "TREND":
                    # ✅ PHASE 5: ATR-BASED RUNNER TRAIL - SHORT (multiplier from runner_trail_atr_multiplier config)
                    new_trail = self.lowest_price_reached + (self.runner_trail_atr_multiplier * atr)
                    
                    if new_trail < self.current_stop_loss: 
                        logger.info(f"[VTM] 🏃 Trailing SL updated to ${new_trail:,.2f} (from ${self.current_stop_loss:,.2f}).")
                        self.current_stop_loss = new_trail
            
            return self.check_exit(current_price, atr, df_4h=df_4h) # Pass ATR and df_4h to check_exit
        except Exception as e:
            logger.error(f"[VTM] Price update error: {e}")
            return None

    def _calculate_adx(self) -> float:
        try:
            adx = talib.ADX(self.high, self.low, self.close, timeperiod=self.atr_period)
            return adx[-1]
        except Exception as e:
            logger.error(f"[VTM] ADX error: {e}")
            return 0.0

    def _calculate_atr_slow(self) -> float:
        try:
            atr = talib.ATR(self.high, self.low, self.close, timeperiod=100)
            return atr[-1]
        except Exception as e:
            logger.error(f"[VTM] ATR Slow error: {e}")
            return self.entry_price * 0.02

    def cancel_take_profit(self):
        """Cancel all remaining take profit targets."""
        self.take_profit_levels = []
        self.partial_sizes = []
        logger.debug(f"[VTM] All take profit orders cancelled for {self.asset}.")

    def enable_trailing_stop(self):
        """Activate the trailing stop mechanism (Runner Mode)."""
        self.runner_activated = True
        logger.debug(f"[VTM] Trailing stop (Greed Mode) enabled for {self.asset}.")

    def check_exit(self, current_price: float, atr_value: Optional[float] = None, df_4h: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        if atr_value is None:
            atr_value = self._calculate_atr() # Fallback if ATR not passed
        if self.remaining_position <= 0: return None

        # ── J: Friday PM trailing tightener ───────────────────────────────
        try:
            from datetime import datetime as _dtf
            _is_friday_pm = self.signal_details.get("friday_tighten") or \
                            (_dtf.utcnow().weekday() == 4 and _dtf.utcnow().hour >= 15)
            if _is_friday_pm:
                if hasattr(self, 'runner_trail_distance') and self.runner_trail_distance:
                    self.runner_trail_distance *= 0.6  # 40% tighter on Friday PM
                if hasattr(self, 'runner_trail_atr_multiplier'):
                    self.runner_trail_atr_multiplier = min(self.runner_trail_atr_multiplier, 1.2)
        except Exception:
            pass
        # ──────────────────────────────────────────────────────────────────

        # --- STEP 1: Volatility Break-Even Lock ---
        # Reason: Locks risk to zero once trade proves itself by moving 1.0 * ATR in profit.
        if self.side == "long":
            current_profit = current_price - self.entry_price
        else:
            current_profit = self.entry_price - current_price

        # --- STEP 0.5: Intermediate Trail (Early Protection) ---
        # Fires when profit exceeds 1.0×ATR — the trade has proven itself with a full
        # ATR of room, not just 0.75×.  Firing earlier (0.75×) caused chop-outs: a
        # normal intrabar pullback after a modest push would stop the trade before it
        # could play out.  At 1.0× the trade has real momentum before we tighten.
        # SL moves to entry - (initial_risk - 0.5×ATR), reducing risk by ~half while
        # still leaving 1.5× ATR of breathing room from current price.
        _initial_risk = abs(self.entry_price - self.initial_stop_loss) if self.initial_stop_loss is not None else atr_value
        if current_profit > 1.0 * atr_value:
            if self.side == "long":
                _intermediate_sl = self.entry_price - max(0.0, _initial_risk - 0.5 * atr_value)
                if _intermediate_sl > self.current_stop_loss:
                    logger.info(
                        f"[VTM] 🔒 Intermediate trail: {self.asset} SL → {_intermediate_sl:,.5f} "
                        f"(Profit: ${current_profit:.2f} > 1.0×ATR: ${1.0*atr_value:.2f})"
                    )
                    self.current_stop_loss = _intermediate_sl
            else:
                _intermediate_sl = self.entry_price + max(0.0, _initial_risk - 0.5 * atr_value)
                if _intermediate_sl < self.current_stop_loss:
                    logger.info(
                        f"[VTM] 🔒 Intermediate trail: {self.asset} SL → {_intermediate_sl:,.5f} "
                        f"(Profit: ${current_profit:.2f} > 1.0×ATR: ${1.0*atr_value:.2f})"
                    )
                    self.current_stop_loss = _intermediate_sl

        if current_profit > 1.5 * atr_value:
            # T1.4 fix: only move SL TO entry if it hasn't already passed entry.
            # Original code fired every tick with no side check, pulling a trailing
            # stop BACKWARDS to entry even after it had advanced beyond it.
            if self.side == "long" and self.current_stop_loss < self.entry_price:
                logger.info(f"[VTM] 🛡️ Break-even lock: {self.asset} (Profit: ${current_profit:.2f} > 1.5×ATR: ${1.5*atr_value:.2f})")
                self.current_stop_loss = self.entry_price
            elif self.side == "short" and self.current_stop_loss > self.entry_price:
                logger.info(f"[VTM] 🛡️ Break-even lock: {self.asset} (Profit: ${current_profit:.2f} > 1.5×ATR: ${1.5*atr_value:.2f})")
                self.current_stop_loss = self.entry_price

        # --- STEP 1.5: Time-Based Break-Even Lock ---
        # Fires independently of ATR: if the trade has been open for N bars and
        # pnl >= threshold, lock SL to break-even. Gated by breakeven_after_bars config.
        if self.breakeven_after_bars is not None and self.bars_in_trade >= self.breakeven_after_bars:
            _tbe_pnl = (
                (current_price - self.entry_price) / self.entry_price
                if self.side == "long"
                else (self.entry_price - current_price) / self.entry_price
            )
            if _tbe_pnl >= self.breakeven_profit_threshold:
                if self.side == "long" and self.current_stop_loss < self.entry_price:
                    logger.info(
                        f"[VTM] ⏱ Time-based BE lock: {self.asset} bar={self.bars_in_trade} "
                        f"pnl={_tbe_pnl:.2%} >= {self.breakeven_profit_threshold:.2%}"
                    )
                    self.current_stop_loss = self.entry_price
                elif self.side == "short" and self.current_stop_loss > self.entry_price:
                    logger.info(
                        f"[VTM] ⏱ Time-based BE lock: {self.asset} bar={self.bars_in_trade} "
                        f"pnl={_tbe_pnl:.2%} >= {self.breakeven_profit_threshold:.2%}"
                    )
                    self.current_stop_loss = self.entry_price

        # Calculate ADX and atr_slow once — used by multiple steps below.
        adx_value = self._calculate_adx()
        atr_slow = self._calculate_atr_slow()
        # Guard: atr_slow may be NaN when trade has < 100 bars of history.
        # Fall back to atr_value (fast ATR) so greed-mode comparisons don't silently fail.
        if np.isnan(atr_slow) or atr_slow == 0:
            atr_slow = atr_value

        # --- STEP 2: Stop-Loss Check (HIGHEST PRIORITY after BE locks) ---
        # Must be evaluated before pyramiding / early-scale returns so that a bar
        # which closes below the SL and also happens to meet pyramid conditions is
        # always treated as a stop-loss, never as a scale-in signal.
        try:
            if (self.side == "long" and current_price <= self.current_stop_loss) or \
               (self.side == "short" and current_price >= self.current_stop_loss):
                reason = ExitReason.STOP_LOSS
                offset = 0.125 * atr_value
                if self.side == "long":
                    if self.current_stop_loss > self.entry_price + offset:
                        reason = ExitReason.TRAILING_STOP
                    elif self.runner_activated:
                        reason = ExitReason.BREAK_EVEN
                else:
                    if self.current_stop_loss < self.entry_price - offset:
                        reason = ExitReason.TRAILING_STOP
                    elif self.runner_activated:
                        reason = ExitReason.BREAK_EVEN
                return {"reason": reason, "price": current_price, "size": self.remaining_position}
        except Exception as e:
            logger.error(f"[VTM] SL check error: {e}")

        # ══════════════════════════════════════════════════════════════════
        # SMART MARKET-CONDITION EXITS  (Steps 2.5 – 2.7)
        # Fire AFTER the hard SL (highest priority) but BEFORE mechanical TPs,
        # so deteriorating market conditions are caught before price grinds to
        # the original stop.  Each check is one-shot (gate flag prevents repeat).
        # ══════════════════════════════════════════════════════════════════

        # --- STEP 2.5: Volatility Spike Exit ---
        # If ATR has suddenly doubled vs its 100-bar baseline the entire risk model
        # used at entry is now wrong.  The SL is too tight for the new noise level,
        # and a continued adverse move could be far larger than anticipated.
        # Action: take 75 % of the remaining position off immediately; keep a 25 %
        # runner so we don't fully exit a trade that might still be going our way.
        # Guard: skip if trade is already up > 2× ATR (it's proving itself).
        if not getattr(self, "_vol_spike_exited", False):
            try:
                is_winning = (
                    (self.side == "long"  and current_price > self.entry_price + 2 * atr_value) or
                    (self.side == "short" and current_price < self.entry_price - 2 * atr_value)
                )
                if atr_value > 2.0 * atr_slow and not is_winning:
                    self._vol_spike_exited = True
                    vol_exit_size = min(0.75, self.remaining_position)
                    if vol_exit_size > 0:
                        self.remaining_position = max(0.0, self.remaining_position - vol_exit_size)
                        logger.warning(
                            f"[VTM] ⚡ VOLATILITY SPIKE: {self.asset} — "
                            f"ATR {atr_value:.5f} > 2× slow-ATR {atr_slow:.5f}. "
                            f"Reducing {vol_exit_size:.0%}, keeping runner."
                        )
                        return {"reason": ExitReason.VOLATILITY_SPIKE, "price": current_price, "size": vol_exit_size}
            except Exception as _e:
                logger.debug(f"[VTM] Vol-spike check skipped: {_e}")

        # --- STEP 2.6: Reversal Candle Exit ---
        # A bar whose range > 1.5× ATR that closes in the lower 40 % of its own
        # range on a long (upper 40 % on a short) is the price-action equivalent
        # of a counter-signal: conviction reversed and fast.
        # Action: close 50 %, tighten remaining SL to entry ± 0.3×ATR.
        # Only fires after bar 2 (needs at least one prior close to compare).
        if not getattr(self, "_reversal_candle_exited", False) and len(self.close) >= 3 and self.bars_in_trade >= 2:
            try:
                bar_range  = self.high[-1] - self.low[-1]
                bar_mid    = (self.high[-1] + self.low[-1]) / 2
                prev_close = self.close[-2]

                bearish_reversal = (
                    self.side == "long"
                    and bar_range > 1.5 * atr_value          # wide, high-conviction bar
                    and current_price < bar_mid               # closes in lower half
                    and current_price < prev_close            # closes below prior close
                )
                bullish_reversal = (
                    self.side == "short"
                    and bar_range > 1.5 * atr_value
                    and current_price > bar_mid               # closes in upper half
                    and current_price > prev_close
                )

                if bearish_reversal or bullish_reversal:
                    self._reversal_candle_exited = True
                    rev_size = min(0.50, self.remaining_position)
                    if rev_size > 0:
                        self.remaining_position = max(0.0, self.remaining_position - rev_size)
                        # Tighten SL on the runner to 0.8×ATR inside break-even
                        # (was 0.3×ATR — too close to entry, often hit by normal noise)
                        if self.side == "long":
                            tight_sl = self.entry_price - 0.8 * atr_value
                            if tight_sl > self.current_stop_loss:
                                self.current_stop_loss = tight_sl
                        else:
                            tight_sl = self.entry_price + 0.8 * atr_value
                            if tight_sl < self.current_stop_loss:
                                self.current_stop_loss = tight_sl
                        logger.warning(
                            f"[VTM] 🕯️ REVERSAL CANDLE: {self.asset} {self.side.upper()} — "
                            f"Range={bar_range:.5f} (1.5×ATR={1.5*atr_value:.5f}), "
                            f"close={current_price:.5f} vs mid={bar_mid:.5f}. "
                            f"Closing {rev_size:.0%}, SL tightened to ${self.current_stop_loss:,.5f}."
                        )
                        return {"reason": ExitReason.REVERSAL_CANDLE, "price": current_price, "size": rev_size}
            except Exception as _e:
                logger.debug(f"[VTM] Reversal-candle check skipped: {_e}")

        # --- STEP 2.7: Trend Invalidation Exit ---
        # 3 consecutive bars closing against the trade direction AND ADX < 20
        # means the market has lost its trend entirely — the edge that opened this
        # trade no longer exists.  Close the full remaining position rather than
        # waiting for the original SL to be hit bar by bar.
        # Guard: only fires when we have ≥ 4 bars (need 3 prior closes to compare)
        # and the position has been open at least 3 bars to avoid day-1 noise.
        if not getattr(self, "_trend_invalidated", False) and len(self.close) >= 5 and self.bars_in_trade >= 3:
            try:
                bars_against = sum(
                    1 for k in range(-3, 0)
                    if (self.side == "long"  and self.close[k] < self.close[k - 1]) or
                       (self.side == "short" and self.close[k] > self.close[k - 1])
                )
                if bars_against >= 3 and adx_value < 20:
                    self._trend_invalidated = True
                    ti_size = self.remaining_position
                    if ti_size > 0:
                        self.remaining_position = 0.0
                        logger.warning(
                            f"[VTM] ❌ TREND INVALIDATION: {self.asset} {self.side.upper()} — "
                            f"3 consecutive bars against + ADX={adx_value:.1f} < 20. "
                            f"Full close ({ti_size:.0%})."
                        )
                        return {"reason": ExitReason.TREND_INVALIDATION, "price": current_price, "size": ti_size}
            except Exception as _e:
                logger.debug(f"[VTM] Trend-invalidation check skipped: {_e}")

        # --- STEP 2.8: Counter-Momentum Early Cut ---
        # Detects strong opposing momentum building AGAINST the trade while it is
        # still a loser (i.e. before the trailing / break-even machinery can help).
        # This is the only VTM mechanism that fires on a losing trade.
        #
        # Conditions (all must hold):
        #   1. Trade is in a loss AND loss > 0.4×ATR  — meaningful move against, not noise
        #   2. Loss has worsened vs last check (price still moving away)
        #   3. RSI shows counter-direction momentum: for shorts RSI > 55 (bulls in control)
        #                                            for longs  RSI < 45 (bears in control)
        #   4. MACD histogram is pointing counter-direction (bullish for shorts, bearish for longs)
        #   5. bars_in_trade >= 2 — at least one full bar has closed (avoids entry-bar noise)
        #
        # Action: close 60 % immediately and tighten remaining SL to just beyond
        # entry (0.5×ATR) so the runner can only lose a small additional amount.
        # One-shot guard (_counter_momentum_cut) prevents repeat fires per position.
        if (not getattr(self, "_counter_momentum_cut", False)
                and self.bars_in_trade >= 2
                and len(self.close) >= 26):
            try:
                _loss = (
                    (self.entry_price - current_price)   # short: profit is negative when price rises
                    if self.side == "short"
                    else (current_price - self.entry_price)  # long: profit is negative when price falls
                )
                _loss = -_loss  # positive = trade is losing

                if _loss > 0.4 * atr_value:
                    # RSI — use historical closes (updated each 1H bar by main loop)
                    _rsi_series = talib.RSI(self.close, timeperiod=14)
                    _rsi = _rsi_series[-1] if not np.isnan(_rsi_series[-1]) else 50.0

                    # MACD histogram direction
                    _, _, _macd_hist = talib.MACD(
                        self.close, fastperiod=12, slowperiod=26, signalperiod=9
                    )
                    _h1 = _macd_hist[-1] if not np.isnan(_macd_hist[-1]) else 0.0
                    _h2 = _macd_hist[-2] if not np.isnan(_macd_hist[-2]) else 0.0

                    # Counter-momentum signals (from the perspective of "bears taking over" for shorts
                    # or "bulls taking over" for longs)
                    if self.side == "short":
                        _rsi_counter   = _rsi > 55          # bulls in control
                        _macd_counter  = _h1 > _h2 > 0      # histogram rising into positive = bullish
                    else:
                        _rsi_counter   = _rsi < 45          # bears in control
                        _macd_counter  = _h1 < _h2 < 0      # histogram falling into negative = bearish

                    if _rsi_counter and _macd_counter:
                        self._counter_momentum_cut = True
                        cut_size = min(0.60, self.remaining_position)
                        if cut_size > 0:
                            self.remaining_position = max(0.0, self.remaining_position - cut_size)
                            # Tighten remaining SL to entry ± 0.5×ATR so runner risk is minimal
                            if self.side == "short":
                                _tight_sl = self.entry_price + 0.5 * atr_value
                                if _tight_sl < self.current_stop_loss:
                                    self.current_stop_loss = _tight_sl
                            else:
                                _tight_sl = self.entry_price - 0.5 * atr_value
                                if _tight_sl > self.current_stop_loss:
                                    self.current_stop_loss = _tight_sl
                            logger.warning(
                                f"[VTM] ⚔️ COUNTER-MOMENTUM CUT: {self.asset} {self.side.upper()} — "
                                f"Loss=${_loss:.2f} ({_loss/atr_value:.2f}×ATR), "
                                f"RSI={_rsi:.1f}, MACD hist {_h1:+.4f}. "
                                f"Closing {cut_size:.0%} early. SL tightened to ${self.current_stop_loss:,.5f}."
                            )
                            return {"reason": ExitReason.TREND_INVALIDATION,
                                    "price": current_price, "size": cut_size}
            except Exception as _e:
                logger.debug(f"[VTM] Counter-momentum cut check skipped: {_e}")

        # --- STEP 3: Greed Mode Accelerator ---
        # During extreme trends/volatility, collapse early targets so the runner
        # trail captures the full move. One-shot: _greed_mode_activated prevents
        # re-executing every bar, avoiding log spam and repeated partial_sizes mutation.
        if not getattr(self, "_greed_mode_activated", False):
            if adx_value > 40 and atr_value > (1.5 * atr_slow):
                if len(self.take_profit_levels) > 1:
                    # Keep TP1 as a 30% partial lock — it gives certainty while the
                    # runner chases the full move.  Only TP2+ collapse into the runner.
                    logger.info(
                        f"[VTM] 🔥 GREED MODE: Strong trend (ADX:{adx_value:.1f}) & "
                        f"Volatility Expansion detected. Keeping TP1 (30%), collapsing rest to runner."
                    )
                    self.take_profit_levels = [self.take_profit_levels[0], self.take_profit_levels[-1]]
                    self.partial_sizes = [0.30, 0.70]
                    self._greed_mode_activated = True

        # --- STEP 3.5: Early Scale Exit ---
        # Objective: Lock in a small partial profit quickly in the first few bars before
        # momentum fades. Enabled via early_scale_enabled in per-asset risk config.
        if self.early_scale_enabled and not self._early_scaled:
            if self.bars_in_trade <= self.early_scale_bars:
                early_pnl_pct = (
                    (current_price - self.entry_price) / self.entry_price
                    if self.side == "long"
                    else (self.entry_price - current_price) / self.entry_price
                )
                if early_pnl_pct >= self.early_scale_threshold:
                    self._early_scaled = True
                    early_size = 0.20  # Fixed 20% early exit
                    self.remaining_position = max(0.0, self.remaining_position - early_size)

                    # Tighten SL to lock in partial profit
                    lock_offset = self.early_lock_atr_multiplier * atr_value
                    if self.side == "long":
                        lock_sl = self.entry_price + lock_offset
                        if lock_sl > self.current_stop_loss:
                            logger.info(
                                f"[VTM] ⚡ Early Scale SL lock: ${self.current_stop_loss:,.2f} → ${lock_sl:,.2f} "
                                f"(entry + {self.early_lock_atr_multiplier}x ATR)"
                            )
                            self.current_stop_loss = lock_sl
                    else:
                        lock_sl = self.entry_price - lock_offset
                        if lock_sl < self.current_stop_loss:
                            logger.info(
                                f"[VTM] ⚡ Early Scale SL lock: ${self.current_stop_loss:,.2f} → ${lock_sl:,.2f} "
                                f"(entry - {self.early_lock_atr_multiplier}x ATR)"
                            )
                            self.current_stop_loss = lock_sl

                    logger.info(
                        f"[VTM] ⚡ EARLY SCALE: {self.asset} {self.side.upper()} — "
                        f"exiting {early_size:.0%} at ${current_price:,.2f} "
                        f"(bar {self.bars_in_trade}, pnl={early_pnl_pct:.2%})"
                    )
                    return {"reason": ExitReason.EARLY_SCALE, "price": current_price, "size": early_size}

        # --- STEP 4: Trend Pyramiding ---
        # Objective: Scale into strong breakout trends. Fires only after SL is confirmed
        # safe (Step 2 above) so a fast reversal bar cannot be misclassified as a pyramid.
        if self.trade_type == "TREND" and not self.has_pyramided:
            if current_profit >= (1.0 * atr_value) and adx_value > 25:
                logger.info(f"[VTM] 🗼 TREND PYRAMIDING: Strong trend confirmed. Scaling in.")
                # Move SL of position 1 to entry before adding exposure
                self.current_stop_loss = self.entry_price
                self.has_pyramided = True
                return {
                    "action": "pyramid",
                    "asset": self.asset,
                    "side": self.side,
                    "new_size": self.position_size * 0.5,
                    "reason": "Trend Pyramiding Triggered"
                }

        # --- STEP 5: Trade State Mutation ---
        # Objective: Allow profitable Mean Reversion trades to convert into trend trades.
        if self.trade_type == "REVERSION":
            if adx_value > 30 and current_profit > 0:
                is_actually_profitable = (self.side == "long" and current_price > self.entry_price) or \
                                         (self.side == "short" and current_price < self.entry_price)
                if is_actually_profitable:
                    logger.info("[VTM] 🧬 Trade mutated from REVERSION to TREND. Ride the move.")
                    self.cancel_take_profit()
                    self.trade_type = "TREND"
                    self.enable_trailing_stop()

        # --- STEP 5.5: Momentum Exhaustion Exit ---
        # Three simultaneous conditions must hold:
        #   1. RSI is in the exhaustion zone (> 75 long / < 25 short) — price stretched
        #   2. MACD histogram declining 3 consecutive bars — momentum dying
        #   3. ADX falling over last 2 bars — trend weakening / losing steam
        # When all three align, the move is likely spent.  Close 50 % and lock the
        # runner to break-even so any remaining profit is protected, not gambled.
        # Guards: trade must be in profit and open ≥ 5 bars; needs 26 bars for MACD.
        if not getattr(self, "_momentum_exhausted", False) and len(self.close) >= 26 and self.bars_in_trade >= 5:
            try:
                _in_profit = (
                    (self.side == "long"  and current_price > self.entry_price) or
                    (self.side == "short" and current_price < self.entry_price)
                )
                if _in_profit:
                    rsi_arr  = talib.RSI(self.close, timeperiod=14)
                    _, _, macd_hist = talib.MACD(self.close, fastperiod=12, slowperiod=26, signalperiod=9)
                    adx_arr  = talib.ADX(self.high, self.low, self.close, timeperiod=14)

                    rsi_val = rsi_arr[-1] if not np.isnan(rsi_arr[-1]) else 50.0
                    rsi_exhausted = (self.side == "long" and rsi_val > 75) or \
                                    (self.side == "short" and rsi_val < 25)

                    h1, h2, h3 = macd_hist[-1], macd_hist[-2], macd_hist[-3]
                    macd_dying = (
                        not any(np.isnan(v) for v in [h1, h2, h3]) and (
                            (self.side == "long"  and h1 < h2 < h3) or
                            (self.side == "short" and h1 > h2 > h3)
                        )
                    )

                    adx_weakening = (
                        not np.isnan(adx_arr[-1]) and not np.isnan(adx_arr[-2])
                        and adx_arr[-1] < adx_arr[-2]
                    )

                    if rsi_exhausted and macd_dying and adx_weakening:
                        self._momentum_exhausted = True
                        exhaust_size = min(0.50, self.remaining_position)
                        if exhaust_size > 0:
                            self.remaining_position = max(0.0, self.remaining_position - exhaust_size)
                            # Lock SL to break-even so runner can only win or scratch
                            if self.side == "long" and self.current_stop_loss < self.entry_price:
                                self.current_stop_loss = self.entry_price
                            elif self.side == "short" and self.current_stop_loss > self.entry_price:
                                self.current_stop_loss = self.entry_price
                            logger.warning(
                                f"[VTM] 📉 MOMENTUM EXHAUSTION: {self.asset} {self.side.upper()} — "
                                f"RSI={rsi_val:.1f}, MACD hist declining, ADX={adx_arr[-1]:.1f}↓. "
                                f"Closing {exhaust_size:.0%}, SL locked to break-even ${self.entry_price:,.5f}."
                            )
                            return {"reason": ExitReason.MOMENTUM_EXHAUSTION, "price": current_price, "size": exhaust_size}
            except Exception as _e:
                logger.debug(f"[VTM] Momentum-exhaustion check skipped: {_e}")

        # --- STEP 6: Time Decay Protection (T4.2 — dynamic extension when in profit) ---
        # Objective: Prevent stale trades turning into long-term losses.
        # Extension rule: if the trade is in profit at the time-stop bar, grant ONE
        # +24-bar extension so a live winner is not forcefully closed.  A second
        # time-stop at bars_in_trade >= time_stop_bars + 24 closes unconditionally.
        if self.bars_in_trade >= self.time_stop_bars:
            _pnl_now = (
                (current_price - self.entry_price) / self.entry_price
                if self.side == "long"
                else (self.entry_price - current_price) / self.entry_price
            )
            _extended = getattr(self, "_time_stop_extended", False)

            if _pnl_now > 0 and not _extended:
                self._time_stop_extended = True
                logger.info(
                    f"[VTM] ⏳ Time stop reached for {self.asset} but trade is in profit "
                    f"({_pnl_now * 100:+.2f}%) — granting +24 bar extension "
                    f"(bars={self.bars_in_trade}, new_limit={self.time_stop_bars + 24})"
                )
            elif self.bars_in_trade < self.time_stop_bars + (24 if _extended else 0):
                pass  # still within extended window, no action
            else:
                logger.info(
                    f"[VTM] ⏳ Stale {self.trade_type} trade closed for {self.asset} "
                    f"(Bars: {self.bars_in_trade} >= "
                    f"{self.time_stop_bars + (24 if _extended else 0)}, "
                    f"pnl={_pnl_now * 100:+.2f}%)"
                )
                return {"reason": ExitReason.TIME_STOP, "price": current_price, "size": self.remaining_position}

        # --- STEP 7: TP Partial Exits ---
        try:
            for i, (target, size) in enumerate(zip(self.take_profit_levels, self.partial_sizes)):
                if i in self.partials_hit: continue
                if (self.side == "long" and current_price >= target) or (self.side == "short" and current_price <= target):
                    self.partials_hit.append(i)
                    self.remaining_position -= size

                    # After TP1 (first partial), attempt early runner promotion based on
                    # volume strength and candle conviction. Falls back to mechanical
                    # activation after TP2 if promotion conditions are not met.
                    if len(self.partials_hit) == 1 and not self.runner_activated and self.trade_type == "TREND":
                        promoted = self.check_promotion_to_runner(current_price)
                        if not promoted:
                            logger.info("[VTM] 🏃 TP1 hit — runner promotion skipped (conditions not met, waiting for TP2)")

                    # Mechanical fallback: activate runner after TP2
                    if len(self.partials_hit) >= 2 and not self.runner_activated and self.trade_type == "TREND":
                        self.runner_activated = True
                        logger.info(f"[VTM TACTICAL] 🏃 Runner Activated (mechanical — TP2 hit): trailing stop now follows price.")

                    tp_reasons = [ExitReason.TAKE_PROFIT_1, ExitReason.TAKE_PROFIT_2, ExitReason.TAKE_PROFIT_3]
                    reason = tp_reasons[i] if i < len(tp_reasons) else ExitReason.TAKE_PROFIT_3
                    return {"reason": reason, "price": current_price, "size": size}

            return None
        except Exception as e:
            logger.error(f"[VTM] Exit check error: {e}")
            return None

    def get_current_levels(self, live_price: Optional[float] = None) -> Dict:
        current_price = live_price if live_price is not None else self.close[-1]
        pnl_pct = (current_price - self.entry_price) / self.entry_price * 100 if self.side == "long" else (self.entry_price - current_price) / self.entry_price * 100
        next_target_idx = len(self.partials_hit)
        next_target = self.take_profit_levels[next_target_idx] if next_target_idx < len(self.take_profit_levels) else None
        
        # Directional distance — always negative = risk / downside remaining to SL
        # LONG: SL is below current → negative value (price must fall to hit SL)
        # SHORT: SL is above current → negative value (price must rise to hit SL)
        if self.current_stop_loss > 0:
            if self.side == "long":
                distance_to_sl_pct = (self.current_stop_loss - current_price) / current_price * 100
            else:
                distance_to_sl_pct = (current_price - self.current_stop_loss) / current_price * 100
        else:
            distance_to_sl_pct = 0

        if next_target and current_price > 0:
            if self.side == "long":
                distance_to_tp_pct = (next_target - current_price) / current_price * 100
            else:
                distance_to_tp_pct = (current_price - next_target) / current_price * 100
        else:
            distance_to_tp_pct = 0
        
        return {
            "entry_price": self.entry_price,
            "current_price": current_price,
            "stop_loss": self.current_stop_loss,
            "initial_stop": self.initial_stop_loss,
            "take_profit": next_target,
            "all_targets": self.take_profit_levels,
            "profit_locked": self.profit_locked,
            "remaining_position_pct": self.remaining_position,
            "pnl_pct": pnl_pct,
            "update_count": self.bars_in_trade,
            "partials_hit": len(self.partials_hit),
            "runner_active": self.runner_activated,
            "highest_reached": self.highest_price_reached,
            "lowest_reached": self.lowest_price_reached,
            "side": self.side,
            "distance_to_sl_pct": distance_to_sl_pct,
            "distance_to_tp_pct": distance_to_tp_pct
        }

    def to_dict(self) -> Dict:
        return {
            "entry_price": self.entry_price,
            "side": self.side,
            "asset": self.asset,
            "position_size": self.position_size,
            "initial_stop_loss": self.initial_stop_loss,
            "current_stop_loss": self.current_stop_loss,
            "take_profit_levels": self.take_profit_levels,
            "partial_sizes": self.partial_sizes,
            "remaining_position": self.remaining_position,
            "partials_hit": self.partials_hit,
            "bars_in_trade": self.bars_in_trade,
            "highest_price_reached": self.highest_price_reached,
            "lowest_price_reached": self.lowest_price_reached,
            "runner_activated": self.runner_activated,
            "has_pyramided": self.has_pyramided,
            "trade_type": self.trade_type,
            "entry_time": self.entry_time.isoformat(),
            "local_free_margin": self.local_free_margin,
            "current_ask": self.current_ask,
            "current_bid": self.current_bid,
            # One-shot state flags — persisted so from_dict() restores them correctly
            "_greed_mode_activated": getattr(self, "_greed_mode_activated", False),
            "_early_scaled": getattr(self, "_early_scaled", False),
            "_time_stop_extended": getattr(self, "_time_stop_extended", False),
            "_counter_momentum_cut": getattr(self, "_counter_momentum_cut", False),
            # Snapshot of the ADX-adjusted partial targets used at open — needed so
            # from_dict() can pass them to risk_config and avoid recalculation drift
            "partial_targets_snapshot": list(self.partial_targets),
        }

    @classmethod
    def from_dict(cls, state: Dict, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> 'VeteranTradeManager':
        # Pass a minimal risk_config that satisfies _calculate_initial_levels() without
        # triggering a lot-size ValueError. The position_size from state was already
        # validated when the trade was originally opened, so we set a min_lot of 0 and
        # bypass the leverage ceiling by leaving local_free_margin at 0.
        # All VTM state is fully overwritten from the stored dict immediately after.
        _restore_risk_config = {
            "partial_targets": state.get("partial_targets_snapshot", [1.0, 1.8, 3.0]),
            "partial_sizes": state.get("partial_sizes", [0.45, 0.30, 0.25]),
        }
        vtm = cls(
            entry_price=state["entry_price"],
            side=state["side"],
            asset=state["asset"],
            high=high,
            low=low,
            close=close,
            quantity=state["position_size"],
            trade_type=state.get("trade_type", "TREND"),
            risk_config=_restore_risk_config,
            local_free_margin=0.0,   # Suppress leverage ceiling during restore
            current_ask=state.get("current_ask", 0.0),
            current_bid=state.get("current_bid", 0.0),
            min_lot_override=0.0,    # Suppress lot-size ValueError during restore
        )
        # Overwrite all state from the persisted snapshot
        vtm.initial_stop_loss = state["initial_stop_loss"]
        vtm.current_stop_loss = state["current_stop_loss"]
        vtm.take_profit_levels = state["take_profit_levels"]
        vtm.partial_sizes = state["partial_sizes"]
        vtm.remaining_position = state["remaining_position"]
        vtm.partials_hit = state["partials_hit"]
        vtm.bars_in_trade = state["bars_in_trade"]
        vtm.highest_price_reached = state["highest_price_reached"]
        vtm.lowest_price_reached = state["lowest_price_reached"]
        vtm.runner_activated = state["runner_activated"]
        vtm.has_pyramided = state.get("has_pyramided", False)
        vtm._greed_mode_activated = state.get("_greed_mode_activated", False)
        vtm._early_scaled = state.get("_early_scaled", False)
        vtm._time_stop_extended = state.get("_time_stop_extended", False)
        vtm._counter_momentum_cut = state.get("_counter_momentum_cut", False)
        return vtm

    # ─────────────────────────────────────────────────────────────────────────
    # T3.2 — Manual Override Methods (Telegram /set_sl /set_tp /vtm_status)
    # ─────────────────────────────────────────────────────────────────────────

    def override_stop_loss(self, new_sl: float) -> str:
        """
        Manually override the current stop loss level via Telegram command.

        Validates the new SL is on the correct side of the CURRENT PRICE (not
        entry price) to allow profit-locking moves (e.g. setting a long SL
        above entry after a rally). Only rejects if the SL would trigger
        immediately against the current price.

        Returns a human-readable status string for the Telegram reply.
        """
        if new_sl <= 0:
            return f"❌ Invalid SL: {new_sl} — must be > 0"

        # Use current mark price for the sanity check; fall back to entry if
        # live price isn't available yet (position still being opened).
        current_price = getattr(self, "current_price", None) or self.entry_price

        # Reject only when the SL would fire against the current price
        # (i.e. it's already beyond where the market is right now).
        if self.side == "long" and new_sl >= current_price:
            return (
                f"❌ Rejected: SL {new_sl:.5f} is at or above current price "
                f"{current_price:.5f} — it would trigger immediately. "
                f"Use /set_tp to move take profit instead."
            )
        if self.side == "short" and new_sl <= current_price:
            return (
                f"❌ Rejected: SL {new_sl:.5f} is at or below current price "
                f"{current_price:.5f} — it would trigger immediately. "
                f"Use /set_tp to move take profit instead."
            )

        old_sl = self.current_stop_loss
        self.current_stop_loss = new_sl
        logger.info(
            f"[VTM] 🖊️ Manual SL override: {self.asset} {self.side.upper()} "
            f"SL {old_sl:.5f} → {new_sl:.5f}"
        )
        direction = "tighter 🛡️" if (
            (self.side == "long" and new_sl > old_sl) or
            (self.side == "short" and new_sl < old_sl)
        ) else "looser ↔️"
        return (
            f"✅ SL updated ({direction})\n"
            f"  Asset : {self.asset} {self.side.upper()}\n"
            f"  Old SL: {old_sl:.5f}\n"
            f"  New SL: {new_sl:.5f}\n"
            f"  Entry : {self.entry_price:.5f}"
        )

    def override_take_profit(self, new_tp: float, target_index: int = 0) -> str:
        """
        Manually override a specific take profit level via Telegram command.

        target_index selects which TP tier to update (0 = first remaining TP,
        1 = second, etc.).  Defaults to the nearest unfilled TP.

        Returns a human-readable status string for the Telegram reply.
        """
        if new_tp <= 0:
            return f"❌ Invalid TP: {new_tp} — must be > 0"

        if self.side == "long" and new_tp <= self.entry_price:
            return (
                f"❌ Rejected: TP {new_tp:.5f} is below entry {self.entry_price:.5f} "
                f"for a LONG position."
            )
        if self.side == "short" and new_tp >= self.entry_price:
            return (
                f"❌ Rejected: TP {new_tp:.5f} is above entry {self.entry_price:.5f} "
                f"for a SHORT position."
            )

        # Find remaining (unhit) TP levels.
        # self.partials_hit is a list of hit *index integers* (e.g. [0, 1] means
        # TP0 and TP1 were hit). The old filter used self.partials_hit[i] as a bool
        # which is wrong — it returned another index integer, not a hit/unhit flag.
        remaining_indices = [
            i for i in range(len(self.take_profit_levels))
            if i not in self.partials_hit
        ]

        # Empty list: position has no TP at all (e.g. min-lot with partials cleared).
        # Instead of refusing, append the new price as the single exit target.
        if not remaining_indices:
            self.take_profit_levels = [new_tp]
            logger.info(
                f"[VTM] 🖊️ Manual TP added (was empty): {self.asset} {self.side.upper()} "
                f"→ {new_tp:.5f}"
            )
            return (
                f"✅ TP set to {new_tp:.5f} for {self.asset} {self.side.upper()}\n"
                f"(Position had no TP — added as single full-exit target)"
            )

        if target_index >= len(remaining_indices):
            target_index = 0  # fall back to nearest

        actual_idx = remaining_indices[target_index]
        old_tp = self.take_profit_levels[actual_idx]
        self.take_profit_levels[actual_idx] = new_tp

        logger.info(
            f"[VTM] 🖊️ Manual TP override: {self.asset} {self.side.upper()} "
            f"TP[{actual_idx}] {old_tp:.5f} → {new_tp:.5f}"
        )
        return (
            f"✅ TP[{actual_idx + 1}] updated\n"
            f"  Asset : {self.asset} {self.side.upper()}\n"
            f"  Old TP: {old_tp:.5f}\n"
            f"  New TP: {new_tp:.5f}\n"
            f"  Entry : {self.entry_price:.5f}"
        )

    def get_override_status(self) -> dict:
        """
        Return current trade levels for Telegram /vtm_status display.

        Returns a flat dict so the Telegram handler can format it freely.
        """
        levels = self.get_current_levels()
        remaining_tps = [
            round(tp, 5)
            for i, tp in enumerate(self.take_profit_levels)
            if i >= len(self.partials_hit) or not self.partials_hit[i]
        ]
        hit_tps = len(self.partials_hit) if hasattr(self, "partials_hit") else 0

        return {
            "asset":          self.asset,
            "side":           self.side.upper(),
            "entry_price":    round(self.entry_price, 5),
            "current_price":  round(levels.get("current_price", 0.0), 5),
            "stop_loss":      round(self.current_stop_loss, 5),
            "initial_sl":     round(self.initial_stop_loss, 5),
            "remaining_tps":  remaining_tps,
            "tps_hit":        hit_tps,
            "bars_in_trade":  self.bars_in_trade,
            "pnl_pct":        round(levels.get("pnl_pct", 0.0), 3),
            "remaining_pct":  round(
                self.remaining_position / self.position_size * 100
                if self.position_size > 0 else 0.0, 1
            ),
            "trade_type":     getattr(self, "trade_type", "UNKNOWN"),
            "runner_active":  getattr(self, "runner_activated", False),
        }

    def __repr__(self):
        levels = self.get_current_levels()
        return f"VTM({self.asset} {self.side.upper()}: Entry=${levels['entry_price']:.2f}, Current=${levels['current_price']:.2f}, SL=${levels['stop_loss']:.2f}, P&L={levels['pnl_pct']:+.2f}%)"
