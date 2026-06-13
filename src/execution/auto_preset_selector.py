"""
Enhanced Dynamic Preset Selector - 4-Preset Support
====================================================
IMPROVEMENTS:
✅ 4-tier regime detection (mr → conservative → balanced → aggressive)
✅ Multi-factor scoring system for precise preset selection
✅ Asset-DNA Gating: MTF Regime becomes a HARD LOCK for Mean Reversion
✅ Better separation between presets using volatility + trend + momentum
✅ Scalper preset removed to avoid noise trading
✅ Telegram notifications with 4 presets
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class DynamicPresetSelector:
    """
    Real-time preset selector with 4-preset support:
    MR → CONSERVATIVE → BALANCED → AGGRESSIVE
    """
    
    def __init__(self, data_manager, config, telegram_bot=None):
        self.data_manager = data_manager
        self.config = config
        self.telegram_bot = telegram_bot
        
        # Track current presets
        self.current_presets = {}
        self.last_update = {}
        self.preset_history = {}
        
        # Cooldown to prevent excessive switching (minutes)
        self.min_switch_interval = 60
        
        # ================================================================
        # ENHANCED THRESHOLDS - 4 PRESET SYSTEM
        # ================================================================
        # Scoring system:
        # - High score (65+) = AGGRESSIVE (stable, trending)
        # - Medium (40-64) = BALANCED (normal conditions)
        # - Low (20-39) = CONSERVATIVE (high vol, weak trend, uncertainty)
        # - Very Low (0-19) = MR (Mean Reversion / Ranging)
        # ================================================================
        
        self.thresholds = {
            'BTC': {
                # Volatility thresholds (ATR ratio)
                'volatility_very_high': 3.0,      # Conservative trigger
                'volatility_high': 1.5,            # Balanced max
                'volatility_normal': 1.2,          # Aggressive max
                
                # Trend strength (ADX)
                'adx_very_weak': 20,               # Conservative zone
                'adx_weak': 20,                    # Balanced min
                'adx_moderate': 25,                # Aggressive min
                
                # Momentum thresholds (20-period %)
                'momentum_strong_bull': 1.5,       # Strong uptrend
                'momentum_bull': 2.0,              # Mild uptrend
                'momentum_bear': -2.0,             # Mild downtrend
                'momentum_strong_bear': -5.0,      # Strong downtrend
                
                # Price vs EMA (trend confirmation)
                'ema_distance_strong': 3.0,        # % away from EMA200
                'ema_distance_moderate': 1.5,
            },
            'GOLD': {
                # GOLD is less volatile than BTC - tighter ranges
                'volatility_very_high': 3.0,
                'volatility_high': 1.4,
                'volatility_normal': 1.15,

                'adx_very_weak': 20,
                'adx_weak': 18,
                'adx_moderate': 23,

                'momentum_strong_bull': 1.5,
                'momentum_bull': 1.5,
                'momentum_bear': -1.5,
                'momentum_strong_bear': -3.0,

                'ema_distance_strong': 2.0,
                'ema_distance_moderate': 1.0,
            },
            'FX': {
                # FX pairs: EURUSD, EURJPY, GBPUSD, GBPAUD, USDJPY
                # Tighter pip-range moves, lower ADX regimes, momentum in fractions of %.
                # Using GOLD thresholds here would misclassify most FX sessions as "stagnant".
                'volatility_very_high': 3.0,       # Ratio-based — comparable across assets
                'volatility_high': 1.4,
                'volatility_normal': 1.1,           # Slightly tighter than GOLD

                'adx_very_weak': 18,               # FX trends are valid at lower ADX
                'adx_weak': 15,
                'adx_moderate': 20,                # ADX 20 is a solid trend for FX

                'momentum_strong_bull': 0.5,       # 0.5% over 20 bars = strong for FX
                'momentum_bull': 0.8,
                'momentum_bear': -0.5,
                'momentum_strong_bear': -1.0,

                'ema_distance_strong': 1.0,        # FX rarely moves >1% from EMA200
                'ema_distance_moderate': 0.5,
            }
        }
    
    def update_market_regime(
        self, 
        asset_name: str, 
        market_data: pd.DataFrame,
        regime_data: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Analyze current market conditions and return optimal preset
        Uses multi-factor scoring for precise 4-preset classification
        ✅ ENHANCED: Asset-DNA Gating & MR Preset Support.
        """
        try:
            _FX_ASSETS = {"EURUSD", "EURJPY", "GBPUSD", "GBPAUD", "USDJPY"}
            if 'BTC' in asset_name.upper() or 'USTEC' in asset_name.upper():
                asset_type = 'BTC'
            elif asset_name.upper() in _FX_ASSETS:
                asset_type = 'FX'
            else:
                asset_type = 'GOLD'   # GOLD, USOIL, and anything else commodity-like
            thresholds = self.thresholds.get(asset_type)
            
            # Calculate metrics
            metrics = self._calculate_regime_metrics(market_data, asset_name)
            
            if metrics is None:
                logger.warning(f"[REGIME] {asset_name}: Failed to calculate metrics")
                return None

            # ================================================================
            # DIRTY RANGE OVERRIDE: Persistent Sideways Chop
            # Force Mean Reversion preset during low-quality sideways markets.
            # ================================================================
            adx_current = metrics['adx_series'].iloc[-1]
            adx_prev1 = metrics['adx_series'].iloc[-2]
            adx_prev2 = metrics['adx_series'].iloc[-3]
            adx_prev3 = metrics['adx_series'].iloc[-4]
            volatility_ratio = metrics['volatility_ratio']
            volume_trend = metrics.get('volume_trend', 0)
            
            # If ADX is rising meaningfully (> 2 pts over 4 bars), a new trend is
            # establishing — skip the chop override regardless of the raw ADX level.
            adx_slope_4bar = adx_current - adx_prev3
            adx_is_rising = adx_slope_4bar > 2.0

            if (
                adx_current < 20 and
                adx_prev1 < 20 and
                adx_prev2 < 20 and
                adx_prev3 < 20 and
                not adx_is_rising and
                0.45 <= volatility_ratio <= 1.0 and
                volume_trend <= 5.0
            ):
                logger.info(f"[PRESET] {asset_name} Persistent Sideways Chop: Mean Reversion forced.")
                
                # Update state to ensure the override is persistent
                old_preset = self.current_presets.get(asset_name)
                if old_preset != "mr":
                    self._record_preset_change(
                        asset=asset_name,
                        old_preset=old_preset,
                        new_preset="mr",
                        reason="Dirty Range Override: Persistent Sideways Chop (ADX < 20 for 4 bars)",
                        metrics=metrics
                    )
                    # Send notification for forced change
                    self._send_preset_change_notification(
                        asset=asset_name,
                        old_preset=old_preset,
                        new_preset="mr",
                        reason="Forced Mean Reversion: Low-quality sideways market",
                        metrics=metrics,
                        score=0
                    )
                
                return "mr"

            # Extract MTF Regime Context for Gating
            regime_name = "NEUTRAL"
            if regime_data:
                regime_name = regime_data.get('regime', 'NEUTRAL')
            
            # ================================================================
            # MULTI-FACTOR PRESET SCORING (0-100 scale)
            # ================================================================
            score = 50  # Start at neutral (balanced)
            decision_factors = []
            
            # FACTOR 1: Volatility (±30 points)
            atr_ratio = metrics['volatility_ratio']
            if atr_ratio > thresholds['volatility_very_high']:
                vol_score = -30
                decision_factors.append(f"Very High Vol ({atr_ratio:.2f}x): -30")
            elif atr_ratio > thresholds['volatility_high']:
                vol_score = -15
                decision_factors.append(f"High Vol ({atr_ratio:.2f}x): -15")
            elif atr_ratio < thresholds['volatility_normal']:
                vol_score = +15
                decision_factors.append(f"Low Vol ({atr_ratio:.2f}x): +15")
            else:
                vol_score = 0
                decision_factors.append(f"Moderate Vol ({atr_ratio:.2f}x): 0")
            
            score += vol_score
            
            # FACTOR 2: Trend Strength - ADX (±25 points)
            adx = metrics['adx']
            if adx > thresholds['adx_moderate']:
                adx_score = +25
                decision_factors.append(f"Strong Trend (ADX {adx:.1f}): +25")
            elif adx > thresholds['adx_weak']:
                adx_score = +10
                decision_factors.append(f"Moderate Trend (ADX {adx:.1f}): +10")
            elif adx < thresholds['adx_very_weak']:
                adx_score = -20
                decision_factors.append(f"Very Weak Trend (ADX {adx:.1f}): -20")
            else:
                adx_score = 0
                decision_factors.append(f"Weak Trend (ADX {adx:.1f}): 0")
            
            score += adx_score
            
            # FACTOR 3: Price Momentum (±20 points)
            momentum = metrics['price_momentum']
            if abs(momentum) > thresholds['momentum_strong_bull']:
                mom_score = +15
                decision_factors.append(f"Strong Momentum ({momentum:+.2f}%): +15")
            elif abs(momentum) > thresholds['momentum_bull']:
                mom_score = +8
                decision_factors.append(f"Moderate Momentum ({momentum:+.2f}%): +8")
            elif abs(momentum) < 0.5:
                mom_score = -15
                decision_factors.append(f"Stagnant ({momentum:+.2f}%): -15")
            else:
                mom_score = 0
                decision_factors.append(f"Normal Momentum ({momentum:+.2f}%): 0")
            
            score += mom_score
            
            # FACTOR 4: Trend Direction Alignment (±15 points)
            price = metrics['current_price']
            ema_200 = metrics['ema_200']
            ema_distance = abs((price - ema_200) / ema_200) * 100
            
            if metrics['trend_direction'] == 'BULL':
                if ema_distance > thresholds['ema_distance_strong']:
                    trend_score = +15
                    decision_factors.append(f"Strong Bull ({ema_distance:.2f}% above EMA): +15")
                elif ema_distance > thresholds['ema_distance_moderate']:
                    trend_score = +8
                    decision_factors.append(f"Bull Trend ({ema_distance:.2f}% above EMA): +8")
                else:
                    trend_score = 0
                    decision_factors.append(f"Weak Bull (near EMA): 0")
            else:  # BEAR
                if ema_distance > thresholds['ema_distance_strong']:
                    trend_score = -15
                    decision_factors.append(f"Strong Bear ({ema_distance:.2f}% below EMA): -15")
                elif ema_distance > thresholds['ema_distance_moderate']:
                    trend_score = -8
                    decision_factors.append(f"Bear Trend ({ema_distance:.2f}% below EMA): -8")
                else:
                    trend_score = 0
                    decision_factors.append(f"Weak Bear (near EMA): 0")
            
            score += trend_score
            
            # FACTOR 5: Volume Trend (±10 points) - if available
            vol_trend = metrics.get('volume_trend', 0)
            if abs(vol_trend) > 50:
                vol_trend_score = +10
                decision_factors.append(f"Strong Volume ({vol_trend:+.1f}%): +10")
            elif abs(vol_trend) > 20:
                vol_trend_score = +5
                decision_factors.append(f"Rising Volume ({vol_trend:+.1f}%): +5")
            elif abs(vol_trend) < -20:
                vol_trend_score = -8
                decision_factors.append(f"Declining Volume ({vol_trend:+.1f}%): -8")
            else:
                vol_trend_score = 0
            
            score += vol_trend_score
            
            # Clamp score to 0-100
            score = max(0, min(100, score))
            
            # ================================================================
            # MAP SCORE TO PRESET (4-TIER SYSTEM)
            # ================================================================
            # 65-100: AGGRESSIVE (good conditions - trending market)
            # 40-64:  BALANCED (normal conditions)
            # 20-39:  CONSERVATIVE (high vol, weak trend)
            # 0-19:   MR (Mean Reversion)
            # ================================================================
            
            if score >= 65:
                new_preset = 'aggressive'
                preset_reason = "AGGRESSIVE - Strong trend with manageable volatility"
            elif score >= 40:
                new_preset = 'balanced'
                preset_reason = "BALANCED - Normal market conditions"
            elif score >= 20:
                new_preset = 'conservative'
                preset_reason = "CONSERVATIVE - High risk or uncertain conditions"
            else:
                new_preset = 'mr'
                preset_reason = "MR - Neutral/Mean Reversion conditions (Low score)"
            
            # ================================================================
            # 🛡️ REGIME ALIGNMENT VETO
            # Reason: Ensure strategy matches the dominant market regime (Physics check).
            # Rising ADX exception: when ADX is sub-20 but climbing, a new trend is
            # establishing — treat as NEUTRAL so the score-based selection proceeds.
            # A rising ADX that breaks 20 while we're in MR is exactly the signal we
            # want to ride, not block.
            # ================================================================
            adx = metrics['adx']
            # Reuse slope already computed above (adx_slope_4bar / adx_is_rising).
            # adx_is_rising = True when ADX rose > 2 pts over the last 4 bars.
            market_regime_type = "NEUTRAL"
            if adx > 25:
                market_regime_type = "TREND"
            elif adx < 20 and not adx_is_rising:
                # Only classify as RANGE when ADX is flat or falling below 20.
                # A rising ADX (trend establishing) stays NEUTRAL — no veto fires.
                market_regime_type = "RANGE"

            # A. Trend Veto: No MR in a trending market — redirect to conservative
            if market_regime_type == "TREND" and new_preset == "mr":
                logger.warning(
                    f"[SELECTOR] 🛡️ ALIGNMENT VETO: MR blocked in TREND (ADX: {adx:.1f})"
                    f" — redirecting to conservative"
                )
                return "conservative"

            # B. Range Veto: No high-conviction trend presets in flat/falling range — redirect to MR
            if market_regime_type == "RANGE" and new_preset in ["balanced", "aggressive"]:
                logger.warning(
                    f"[SELECTOR] 🛡️ ALIGNMENT VETO: {new_preset} blocked in RANGE "
                    f"(ADX: {adx:.1f}, slope={adx_slope_4bar:+.1f}) — redirecting to mr"
                )
                return "mr"
            
            # ================================================================
            # 🛡️ ASSET-DNA HARD LOCKS (Gating)
            # ================================================================
            asset = asset_name.upper()
            
            _FX_ASSET_SET = {"EURUSD", "EURJPY", "GBPUSD", "GBPAUD", "USDJPY"}

            if new_preset == "mr":
                # BTC / USTEC: Block MR during BULLISH or SLIGHTLY_BULLISH
                # These assets trend strongly — MR in a bull regime is fighting the tape.
                if "BTC" in asset or "USTEC" in asset:
                    if regime_name in ["BULLISH", "SLIGHTLY_BULLISH"]:
                        logger.info(f"[SELECTOR] 🛡️ GATED - MR blocked for {asset} in {regime_name}. Falling back to conservative.")
                        new_preset = "conservative"

                # GOLD: Only allow MR in BEARISH or NEUTRAL (safety rule — gold has structural bull bias)
                elif "GOLD" in asset:
                    if regime_name not in ["BEARISH", "NEUTRAL"]:
                        logger.info(f"[SELECTOR] 🛡️ GATED - MR blocked for {asset} in {regime_name}. Falling back to conservative.")
                        new_preset = "conservative"

                # FX pairs: Allow MR in any regime — these are inherently mean-reverting intra-session.
                # The Regime Alignment Veto above already blocks MR in strong TREND (ADX > 25),
                # so no additional hard gate needed here.
                elif asset in _FX_ASSET_SET:
                    pass  # MR allowed — FX DNA is mean-reverting by design
            
            # Build detailed reason
            reason = self._format_reason_with_score(
                preset=new_preset,
                score=score,
                preset_reason=preset_reason,
                decision_factors=decision_factors,
                metrics=metrics
            )
            
            # Check if preset changed
            current_preset = self.current_presets.get(asset_name)
            
            if current_preset != new_preset:
                if self._can_switch_preset(asset_name):
                    logger.info(f"\n{'=' * 70}")
                    logger.info(f"[REGIME CHANGE] {asset_name}")
                    logger.info(f"{'=' * 70}")
                    logger.info(f"OLD: {current_preset or 'None'} → NEW: {new_preset.upper()}")
                    logger.info(f"SCORE: {score}/100")
                    logger.info(f"\n{reason}")
                    logger.info(f"{'=' * 70}\n")
                    
                    # Send Telegram notification
                    self._send_preset_change_notification(
                        asset=asset_name,
                        old_preset=current_preset,
                        new_preset=new_preset,
                        reason=reason,
                        metrics=metrics,
                        score=score
                    )
                    
                    # Record change
                    self._record_preset_change(
                        asset=asset_name,
                        old_preset=current_preset,
                        new_preset=new_preset,
                        reason=reason,
                        metrics=metrics
                    )
                    
                    return new_preset
                else:
                    logger.debug(
                        f"[REGIME] {asset_name}: Change detected "
                        f"({current_preset} → {new_preset}) but in cooldown"
                    )
                    return None
            
            # No change needed
            logger.debug(f"[REGIME] {asset_name}: Staying on {current_preset} (score: {score}/100)")
            return None
        
        except Exception as e:
            logger.error(f"[REGIME] Error updating {asset_name}: {e}", exc_info=True)
            return None
    
    def _format_reason_with_score(
        self,
        preset: str,
        score: int,
        preset_reason: str,
        decision_factors: list,
        metrics: Dict
    ) -> str:
        """Format detailed reason with scoring breakdown"""
        lines = [
            f"PRESET SELECTED: {preset.upper()} (Score: {score}/100)",
            f"",
            f"REASONING: {preset_reason}",
            f"",
            f"DECISION FACTORS:",
        ]
        
        for factor in decision_factors:
            lines.append(f"  • {factor}")
        
        lines.append(f"")
        lines.append(f"MARKET SNAPSHOT:")
        lines.append(f"  Price:        ${metrics['current_price']:,.2f}")
        lines.append(f"  Trend:        {metrics['trend_direction']}")
        lines.append(f"  ADX:          {metrics['adx']:.1f}")
        lines.append(f"  Volatility:   {metrics['volatility_ratio']:.2f}x normal")
        lines.append(f"  Momentum:     {metrics['price_momentum']:+.2f}% (20-period)")
        lines.append(f"  Volume Trend: {metrics.get('volume_trend', 0):+.1f}% vs avg")
        
        return '\n'.join(lines)
    
    def _send_preset_change_notification(
        self,
        asset: str,
        old_preset: str,
        new_preset: str,
        reason: str,
        metrics: Dict,
        score: int
    ):
        """Send Telegram notification with 4-preset support"""
        if not self.telegram_bot:
            return
        
        try:
            # Emoji mapping for 4 presets
            emoji_map = {
                'mr': '🔄',
                'conservative': '🛡️',
                'balanced': '⚖️',
                'aggressive': '⚡'
            }
            
            old_emoji = emoji_map.get(old_preset or 'balanced', '❓')
            new_emoji = emoji_map.get(new_preset, '❓')
            asset_emoji = '₿' if asset == 'BTC' else '🥇'
            
            # Build message
            msg = f"🔄 *PRESET CHANGE DETECTED*\n\n"
            msg += f"{asset_emoji} *Asset:* {asset}\n"
            msg += f"{old_emoji} Old Preset: `{(old_preset or 'None').upper()}`\n"
            msg += f"{new_emoji} New Preset: `{new_preset.upper()}`\n"
            msg += f"📊 Market Score: `{score}/100`\n\n"
            
            # Add preset description
            preset_desc = {
                'mr': '🔄 Mean Reversion - Optimal for Ranging/Neutral markets',
                'conservative': '🛡️ Most restrictive - High thresholds, safety-first',
                'balanced': '⚖️ Standard - Moderate thresholds, balanced approach',
                'aggressive': '⚡ Active - Lower thresholds, trend-following'
            }
            msg += f"*Profile:* {preset_desc.get(new_preset, 'Unknown')}\n\n"
            
            # Add key metrics
            msg += "*📊 Market Metrics:*\n"
            msg += f"• Price: ${metrics['current_price']:,.2f}\n"
            msg += f"• Trend: {metrics['trend_direction']}\n"
            msg += f"• ADX: {metrics['adx']:.1f} (Trend Strength)\n"
            msg += f"• Volatility: {metrics['volatility_ratio']:.2f}x normal\n"
            msg += f"• Momentum: {metrics['price_momentum']:+.2f}% (20d)\n"
            
            msg += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}"
            
            # Send via Telegram
            import asyncio
            
            if hasattr(self.telegram_bot, 'telegram_loop') and self.telegram_bot.telegram_loop:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self.telegram_bot.send_notification(msg, disable_preview=True),
                        self.telegram_bot.telegram_loop
                    )
                    future.result(timeout=5)
                    logger.info(f"[TELEGRAM] ✓ Preset change notification sent for {asset}")
                except Exception as e:
                    logger.debug(f"[TELEGRAM] Notification error: {e}")
            
        except Exception as e:
            logger.error(f"[TELEGRAM] Failed to send preset notification: {e}")
    
    def _calculate_regime_metrics(
        self, 
        df: pd.DataFrame, 
        asset_name: str
    ) -> Optional[Dict]:
        """Calculate all metrics needed for regime detection"""
        try:
            if len(df) < 200:
                return None
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # 1. ADX (Trend Strength)
            adx_series = self._calculate_adx_series(df, period=14)
            adx = adx_series.iloc[-1]
            
            # 2. ATR Ratio (Current volatility vs average)
            atr_current = self._calculate_atr(df, period=14)
            atr_avg = self._calculate_atr(df, period=50)
            volatility_ratio = atr_current / atr_avg if atr_avg > 0 else 1.0
            
            # 3. EMA 200 (Long-term trend)
            ema_200 = pd.Series(close).ewm(span=200, adjust=False).mean().iloc[-1]
            
            # 4. Current price & trend direction
            current_price = close[-1]
            trend_direction = 'BULL' if current_price > ema_200 else 'BEAR'
            
            # 5. Price momentum (20-period)
            price_momentum = ((close[-1] - close[-20]) / close[-20]) * 100
            
            # 6. Volume trend (if available)
            volume_trend = 0
            if 'volume' in df.columns:
                recent_vol = df['volume'].iloc[-20:].mean()
                older_vol = df['volume'].iloc[-50:-20].mean()
                volume_trend = ((recent_vol - older_vol) / older_vol) * 100 if older_vol > 0 else 0
            
            return {
                'adx': adx,
                'adx_series': adx_series,
                'volatility_ratio': volatility_ratio,
                'atr_current': atr_current,
                'atr_avg': atr_avg,
                'ema_200': ema_200,
                'current_price': current_price,
                'trend_direction': trend_direction,
                'price_momentum': price_momentum,
                'volume_trend': volume_trend,
                'timestamp': datetime.now(),
            }
        
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return None
    
    def _calculate_adx_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index Series"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # +DM and -DM
            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # True Range
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Smooth with EMA
            plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / \
                      pd.Series(tr).ewm(span=period, adjust=False).mean()
            minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / \
                       pd.Series(tr).ewm(span=period, adjust=False).mean()
            
            # DX and ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx_series = dx.ewm(span=period, adjust=False).mean()
            
            return adx_series.fillna(25.0)
        
        except Exception as e:
            logger.debug(f"ADX calculation error: {e}")
            return pd.Series([25.0] * len(df))
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index (legacy wrapper)"""
        return self._calculate_adx_series(df, period).iloc[-1]
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            atr = pd.Series(tr).rolling(period).mean().iloc[-1]
            return atr if not np.isnan(atr) else 0.0
        
        except Exception as e:
            logger.debug(f"ATR calculation error: {e}")
            return 0.0
    
    def _can_switch_preset(self, asset_name: str) -> bool:
        """Check if enough time has passed since last preset change"""
        last_update = self.last_update.get(asset_name)
        
        if last_update is None:
            return True
        
        elapsed_minutes = (datetime.now() - last_update).total_seconds() / 60
        return elapsed_minutes >= self.min_switch_interval
    
    def _record_preset_change(
        self, 
        asset: str, 
        old_preset: str, 
        new_preset: str, 
        reason: str,
        metrics: Dict
    ):
        """Record preset change for analysis"""
        change_record = {
            'timestamp': datetime.now(),
            'old_preset': old_preset,
            'new_preset': new_preset,
            'reason': reason,
            'metrics': metrics,
        }
        
        if asset not in self.preset_history:
            self.preset_history[asset] = []
        
        self.preset_history[asset].append(change_record)
        self.current_presets[asset] = new_preset
        self.last_update[asset] = datetime.now()
    
    def get_preset_for_asset(self, asset_name: str, regime_data: Optional[Dict] = None) -> Optional[str]:
        """
        Main method: Get current optimal preset for asset
        Call this before each trading cycle
        """
        try:
            asset_cfg = self.config['assets'][asset_name]
            exchange = asset_cfg.get('exchange', 'binance')
            
            # Fetch 4H data for regime analysis
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=90)
            
            if exchange == 'binance':
                df = self.data_manager.fetch_binance_data(
                    symbol=asset_cfg.get('symbol'),
                    interval='4h',
                    start_date=start_time.strftime('%Y-%m-%d'),
                    end_date=end_time.strftime('%Y-%m-%d %H:%M:%S'),
                )
            else:
                # Use mt5_symbol if available (e.g. BTCUSDm for BTC on Exness),
                # fallback to generic symbol field.
                _mt5_sym = asset_cfg.get('mt5_symbol') or asset_cfg.get('symbol')
                df = self.data_manager.fetch_mt5_data(
                    symbol=_mt5_sym,
                    timeframe='H4',
                    start_date=start_time.strftime('%Y-%m-%d'),
                    end_date=end_time.strftime('%Y-%m-%d %H:%M:%S'),
                )
            
            df = self.data_manager.clean_data(df)
            
            if len(df) < 200:
                logger.warning(
                    f"[REGIME] {asset_name}: Insufficient data ({len(df)}/200)"
                )
                return self.current_presets.get(asset_name, 'balanced')
            
            # Analyze and potentially switch preset
            new_preset = self.update_market_regime(asset_name, df, regime_data)
            
            # Return current preset (updated or unchanged)
            return self.current_presets.get(asset_name, new_preset or 'balanced')
        
        except Exception as e:
            logger.error(f"[REGIME] Error getting preset for {asset_name}: {e}")
            return self.current_presets.get(asset_name, 'balanced')
    
    def get_statistics(self) -> Dict:
        """Get statistics on preset changes"""
        stats = {
            'total_changes': sum(len(h) for h in self.preset_history.values()),
            'changes_by_asset': {
                asset: len(history) 
                for asset, history in self.preset_history.items()
            },
            'current_presets': self.current_presets.copy(),
            'last_update_times': self.last_update.copy(),
        }
        
        # Add preset distribution (4 presets)
        preset_counts = {'mr': 0, 'conservative': 0, 'balanced': 0, 'aggressive': 0}
        for history in self.preset_history.values():
            for record in history:
                preset = record.get('new_preset')
                if preset in preset_counts:
                    preset_counts[preset] += 1
        
        stats['preset_distribution'] = preset_counts
        
        return stats
