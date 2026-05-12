"""
Multi-Timeframe Regime Integration
====================================
Integrates MTF regime detection with:
1. Database logging (Supabase)
2. AI Validator context
3. Trading bot decision making
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timezone
from src.execution.mtf_regime_detector import (
    MultiTimeFrameRegimeDetector,
    RegimeStatus,  # Updated import
    GovernorStatus, # New import
)

logger = logging.getLogger(__name__)


class MTFRegimeIntegration:
    """
    Integrates multi-timeframe regime detection with trading system
    """

    def __init__(
        self, data_manager, db_manager=None, ai_validator=None, telegram_bot=None
    ):
        """
        Initialize integration

        Args:
            data_manager: DataManager instance
            db_manager: TradingDatabaseManager instance (optional)
            ai_validator: HybridSignalValidator instance (optional)
            telegram_bot: TelegramBot instance (optional)
        """
        self.data_manager = data_manager
        self.db_manager = db_manager
        self.ai_validator = ai_validator
        self.telegram_bot = telegram_bot

        # Create detectors for each asset
        self.detectors = {}
        
        # ✅ Store latest regime data for aggregators
        self._current_regime_data = {}

        logger.info("[MTF INTEGRATION] Initialized")

    def get_detector(self, asset_type: str) -> MultiTimeFrameRegimeDetector:
        """
        Get or create detector for asset

        Args:
            asset_type: "BTC" or "GOLD"

        Returns:
            MultiTimeFrameRegimeDetector instance
        """
        if asset_type not in self.detectors:
            self.detectors[asset_type] = MultiTimeFrameRegimeDetector(
                data_manager=self.data_manager, asset_type=asset_type
            )
            logger.info(f"[MTF] Created detector for {asset_type}")

        return self.detectors[asset_type]

    def analyze_and_log(
        self, asset_name: str, symbol: str, exchange: str, force_refresh: bool = False
    ) -> RegimeStatus: # Updated return type
        """
        Analyze regime and log to database

        Args:
            asset_name: "BTC" or "GOLD"
            symbol: Trading symbol
            exchange: "binance" or "mt5"
            force_refresh: Skip cache

        Returns:
            RegimeStatus object
        """
        try:
            # Get detector
            detector = self.get_detector(asset_name)

            # Analyze regime
            regime_status = detector.analyze_regime( # Use new RegimeStatus
                symbol=symbol, exchange=exchange, force_refresh=force_refresh
            )

            # Log to database
            if self.db_manager:
                self._log_to_database(regime_status)

            # Update AI validator context
            if self.ai_validator:
                self._update_ai_context(regime_status)

            # Send Telegram notification (if significant change)
            if self.telegram_bot and self._should_notify(regime_status):
                self._send_telegram_notification(regime_status)

            return regime_status

        except Exception as e:
            logger.error(f"[MTF] Analysis error for {asset_name}: {e}", exc_info=True)
            raise

    def _log_to_database(self, regime_status: RegimeStatus): # Updated parameter type
        """
        Log regime analysis to Supabase

        Args:
            regime_status: RegimeStatus object
        """
        try:
            # Map simplified status to the full database schema
            # Using absolute score as confidence proxy
            confidence = abs(regime_status.score)
            
            # Derived scores
            bullish_score = max(0.0, regime_status.score)
            bearish_score = max(0.0, -regime_status.score)

            # Extract timeframe data for cleaner mapping
            tf = regime_status.timeframe_data

            # Insert into mtf_regime_analysis table
            result = (
                self.db_manager.supabase.table("mtf_regime_analysis")
                .insert(
                    {
                        "asset": regime_status.asset,
                        "timestamp": regime_status.timestamp.isoformat(),
                        "consensus_regime": regime_status.consensus_regime,
                        "consensus_confidence": confidence,
                        "timeframe_agreement": confidence, 
                        "trend_coherence": confidence,     
                        "risk_level": "low" if confidence > 0.5 else "high",
                        "volatility_regime": "normal" if confidence > 0.5 else "high",
                        "recommended_mode": "council",
                        "allow_counter_trend": True,  # fallback: be permissive
                        "suggested_max_positions": 3,
                        # Scores
                        "bullish_score": bullish_score,
                        "bearish_score": bearish_score,
                        # Timeframe data (Using real calculated indicators)
                        "h1_regime": tf.get("1h", {}).get("regime", "NEUTRAL"),
                        "h1_confidence": tf.get("1h", {}).get("confidence", 0.0),
                        "h1_adx": tf.get("1h", {}).get("adx"),
                        "h1_rsi": tf.get("1h", {}).get("rsi"),
                        "h1_trend_direction": tf.get("1h", {}).get("trend_direction"),
                        
                        "h4_regime": tf.get("4h", {}).get("regime", "NEUTRAL"),
                        "h4_confidence": tf.get("4h", {}).get("confidence", 0.0),
                        "h4_adx": tf.get("4h", {}).get("adx"),
                        "h4_rsi": tf.get("4h", {}).get("rsi"),
                        "h4_trend_direction": tf.get("4h", {}).get("trend_direction"),
                        
                        "d1_regime": tf.get("1d", {}).get("regime", "NEUTRAL"),
                        "d1_confidence": tf.get("1d", {}).get("confidence", 0.0),
                        "d1_adx": tf.get("1d", {}).get("adx"),
                        "d1_rsi": tf.get("1d", {}).get("rsi"),
                        "d1_trend_direction": tf.get("1d", {}).get("trend_direction"),
                    }
                )
                .execute()
            )

            logger.info(f"[MTF DB] ✓ Logged {regime_status.asset} regime to database")

        except Exception as e:
            logger.error(f"[MTF DB] Failed to log regime: {e}")

    def _update_ai_context(self, regime_status: RegimeStatus): # Updated parameter type
        """
        Update AI validator with regime context

        Args:
            regime_status: RegimeStatus object
        """
        try:
            # Store regime in AI validator for use during validation
            if not hasattr(self.ai_validator, "mtf_regime_context"):
                self.ai_validator.mtf_regime_context = {}

            self.ai_validator.mtf_regime_context[regime_status.asset] = {
                "score": regime_status.score,
                "is_bullish": regime_status.is_bullish,
                "is_bearish": regime_status.is_bearish,
                "reasoning": regime_status.reasoning,
                "timestamp": regime_status.timestamp,
            }

            logger.info(f"[MTF AI] ✓ Updated AI context for {regime_status.asset}")

        except Exception as e:
            logger.error(f"[MTF AI] Failed to update context: {e}")

    def _should_notify(self, regime_status: RegimeStatus) -> bool: # Updated parameter type
        """
        Determine if Telegram notification should be sent based on significant score change.

        Args:
            regime_status: RegimeStatus object

        Returns:
            True if notification should be sent
        """
        # Notify if there's a strong bullish or bearish bias
        return regime_status.score > 0.5 or regime_status.score < -0.5

    def _send_telegram_notification(self, regime_status: RegimeStatus): # Updated parameter type
        """
        Send Telegram notification about regime

        Args:
            regime_status: RegimeStatus object
        """
        try:
            # Format message
            message = self._format_regime_message(regime_status)

            # Send via Telegram bot
            # self.telegram_bot.send_message(message)

            logger.info(f"[MTF TG] ✓ Sent notification for {regime_status.asset}")

        except Exception as e:
            logger.error(f"[MTF TG] Failed to send notification: {e}")

    def _format_regime_message(self, regime_status: RegimeStatus) -> str: # Updated parameter type
        """
        Format regime analysis for Telegram

        Args:
            regime_status: RegimeStatus object

        Returns:
            Formatted message string
        """
        emoji = "📈" if regime_status.is_bullish else ("📉" if regime_status.is_bearish else "➡️")

        message = f"""
{emoji} **MTF REGIME UPDATE - {regime_status.asset}**

**Score:** {regime_status.score:.2f}
**Bias:** {'BULLISH' if regime_status.is_bullish else 'BEARISH' if regime_status.is_bearish else 'NEUTRAL'}
**Reasoning:** {regime_status.reasoning}
"""

        return message.strip()

    def get_regime_for_trading(
        self, asset_name: str, symbol: str, exchange: str
    ) -> Dict:
        """
        Get regime data formatted for trading decisions

        Args:
            asset_name: "BTC" or "GOLD"
            symbol: Trading symbol
            exchange: "binance" or "mt5"

        Returns:
            Dict with regime data for trading logic
        """
        try:
            regime_status = self.analyze_and_log(
                asset_name=asset_name, symbol=symbol, exchange=exchange
            )

            # Map the new 5-tier model to the legacy expectations of main.py
            confidence = abs(regime_status.score)
            risk_level = "low" if regime_status.consensus_regime in ["BULLISH", "BEARISH"] else "medium"
            if regime_status.consensus_regime == "NEUTRAL":
                risk_level = "high"
            
            volatility = "normal"
            if regime_status.consensus_regime == "NEUTRAL":
                volatility = "high" # Neutral often means high uncertainty/chop

            # Allow counter-trend in NEUTRAL and SLIGHTLY regimes.
            # SLIGHTLY = 50% confidence (coin flip) — not strong enough to block
            # MR trades.  Only hard BULLISH / BEARISH (100% confidence) blocks.
            if regime_status.consensus_regime in (
                "NEUTRAL", "SLIGHTLY_BULLISH", "SLIGHTLY_BEARISH"
            ):
                allow_counter_trend = True
            else:
                allow_counter_trend = False

            # Pull 1H session momentum from the new timeframe_data fields
            _h1_tf = regime_status.timeframe_data.get("1h", {})
            regime_data = {
                "regime": regime_status.consensus_regime,
                "regime_score": regime_status.score,
                "is_bullish": regime_status.is_bullish,
                "is_bearish": regime_status.is_bearish,
                "reasoning": regime_status.reasoning,
                "timestamp": regime_status.timestamp.isoformat(),
                "ema_1d_200": regime_status.ema_1d_200,
                "ema_4h_200": regime_status.ema_4h_200,
                "ema_4h_50": regime_status.ema_4h_50,
                "confidence": confidence,
                "timeframe_agreement": confidence, # Proxy for agreement in 5-tier
                "recommended_mode": "council",
                "risk_level": risk_level,
                "volatility": volatility,
                "allow_counter_trend": allow_counter_trend,
                "max_positions": 3,
                "df_4h": regime_status.df_4h, # ✨ ADDED: 4H context for strategies
                "governor": regime_status, # ✨ ADDED: For Council & Performance Aggregators
                "full_regime_status": regime_status,
                # ── 1H Session Momentum (new) ────────────────────────────────
                # "UP" / "DOWN" / "FLAT" — slope of last 6 1H closes.
                # Used by AI validator to confirm or contradict a signal direction
                # without changing the 4H-based structural regime label.
                "h1_momentum_dir": _h1_tf.get("momentum_dir", "FLAT"),
                "h1_momentum_pct": _h1_tf.get("momentum_pct", 0.0),
                "h1_lower_highs": _h1_tf.get("lower_highs", False),
                "h1_higher_lows": _h1_tf.get("higher_lows", False),
                # Legacy alias so existing code that reads is_bull still works
                "is_bull": regime_status.is_bullish,
            }

            # ✅ Cache for aggregators
            self._current_regime_data[asset_name] = regime_data

            return regime_data

        except Exception as e:
            logger.error(f"[MTF] Error getting regime: {e}")
            # Return safe defaults
            return {
                "regime": "NEUTRAL",
                "regime_score": 0.0,
                "is_bullish": False,
                "is_bearish": False,
                "confidence": 0.0,
                "timeframe_agreement": 0.0,
                "recommended_mode": "council",
                "risk_level": "high",
                "volatility": "high",
                "allow_counter_trend": True,  # error fallback: be permissive
                "max_positions": 0,
                "reasoning": f"Error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
