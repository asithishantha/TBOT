"""
Binance Execution Handler with Hybrid Position Sizing + Order Tracking
ENHANCED: Asymmetric Hedging Support (Trend + Scalp simultaneously)
CLEANED: Removed duplicate methods
"""

import logging
import time
from binance.client import Client
from binance.enums import (
    SIDE_BUY,
    SIDE_SELL,
    ORDER_TYPE_MARKET,
    ORDER_TYPE_LIMIT,
    TIME_IN_FORCE_GTC,
)
from typing import Dict, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta, timezone
from src.execution.binance_futures import BinanceFuturesHandler
from src.global_error_handler import handle_errors, ErrorSeverity
from src.execution.position_rebalancer import PositionRebalancer
from src.execution.veteran_trade_manager import VeteranTradeManager
from src.utils.trade_logger import log_trade_event
from src.data.data_manager import CLOUDFRONT_HEADERS
from src.market.price_cache import price_cache

logger = logging.getLogger(__name__)


class SizingMode:
    """Position sizing modes"""

    AUTOMATED = "automated"
    MANUAL_OVERRIDE = "override"
    REDUCED_RISK = "reduced_risk"
    ELEVATED_RISK = "elevated"


class PositionSizingRequest:
    """Request object for position sizing with manual override support"""

    def __init__(
        self,
        asset: str,
        current_price: float,
        signal: int,
        mode: str = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        confidence_score: float = None,
        market_condition: str = None,
        override_reason: str = None,
        max_override_pct: float = 2.0,
    ):
        self.asset = asset
        self.current_price = current_price
        self.signal = signal
        self.mode = mode
        self.manual_size_usd = manual_size_usd
        self.confidence_score = confidence_score or 0.5
        self.market_condition = market_condition or "neutral"
        self.override_reason = override_reason
        self.max_override_pct = max_override_pct


class DynamicMarginCalculator:
    """
    Calculates available margin and maximum position sizes for Binance Futures
    Ensures positions never exceed available margin
    """

    def __init__(self, futures_handler, config: Dict):
        self.futures_handler = futures_handler
        self.config = config

    def get_available_margin_info(self, asset: str) -> Dict:
        """Get comprehensive margin information from Binance Futures"""
        try:
            # Get Futures account info
            account = self.futures_handler.client.futures_account()

            # Extract key metrics
            total_balance = float(account.get("totalWalletBalance", 0))
            available_balance = float(account.get("availableBalance", 0))
            total_unrealized_pnl = float(account.get("totalUnrealizedProfit", 0))
            total_margin_balance = float(account.get("totalMarginBalance", 0))

            # Get current positions to calculate used margin
            positions = self.futures_handler.client.futures_position_information()
            used_margin = 0.0

            for pos in positions:
                if pos["symbol"] == self.futures_handler.symbol:
                    pos_amt = abs(float(pos.get("positionAmt", 0)))
                    if pos_amt > 0:
                        entry_price = float(pos.get("entryPrice", 0))
                        leverage = float(pos.get("leverage", 1))
                        used_margin += (pos_amt * entry_price) / leverage

            # Get leverage setting
            leverage = self.config.get("assets", {}).get(asset, {}).get("leverage", 20)

            # Calculate max position based on available balance and leverage
            max_position_notional = available_balance * leverage

            logger.info(
                f"[MARGIN] Binance Futures Account Status:\n"
                f"  Total Balance:    ${total_balance:,.2f} USDT\n"
                f"  Available:        ${available_balance:,.2f} USDT\n"
                f"  Used Margin:      ${used_margin:,.2f} USDT\n"
                f"  Unrealized P&L:   ${total_unrealized_pnl:,.2f} USDT\n"
                f"  Leverage:         {leverage}x\n"
                f"  Max Position:     ${max_position_notional:,.2f} USDT"
            )

            return {
                "available_balance": available_balance,
                "total_balance": total_balance,
                "used_margin": used_margin,
                "unrealized_pnl": total_unrealized_pnl,
                "leverage": leverage,
                "max_position_notional": max_position_notional,
            }

        except Exception as e:
            logger.error(f"[MARGIN] Error getting margin info: {e}")
            return {
                "available_balance": 0.0,
                "total_balance": 0.0,
                "used_margin": 0.0,
                "unrealized_pnl": 0.0,
                "leverage": 1,
                "max_position_notional": 0.0,
            }

    def calculate_max_safe_position(
        self,
        available_margin: float,
        leverage: int,
        entry_price: float,
        stop_loss_price: float,
        buffer_pct: float = 0.10,
    ) -> Tuple[float, float]:
        """Calculate maximum safe position size that won't get liquidated"""
        try:
            max_position_from_margin = available_margin * leverage

            stop_distance = abs(entry_price - stop_loss_price)
            stop_distance_pct = stop_distance / entry_price

            denominator = (1 / leverage) + stop_distance_pct
            max_safe_position = available_margin / denominator

            # Apply safety buffer
            max_safe_position *= 1 - buffer_pct

            # Take minimum of margin limit and safe limit
            max_position_usd = min(max_position_from_margin, max_safe_position)

            # Calculate quantity
            max_quantity = max_position_usd / entry_price

            return max_position_usd, max_quantity

        except Exception as e:
            logger.error(f"[MARGIN] Error calculating max position: {e}")
            return 0.0, 0.0


class HybridPositionSizer:
    """
    Enhanced position sizer with dynamic margin awareness
    Automatically adjusts position sizes to fit available Binance Futures margin
    """

    def __init__(self, config: Dict, portfolio_manager, futures_handler=None):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.futures_handler = futures_handler
        self.portfolio_cfg = config["portfolio"]
        self.risk_cfg = config.get("risk_management", {})
        self.override_history = []

        # Initialize margin calculator if futures available
        self.margin_calculator = None
        if futures_handler:
            self.margin_calculator = DynamicMarginCalculator(futures_handler, config)

        # Risk parameters
        self.target_risk_pct = self.portfolio_cfg.get("target_risk_per_trade", 0.015)
        self.max_risk_pct = self.portfolio_cfg.get("max_risk_per_trade", 0.020)
        self.aggressive_threshold = self.portfolio_cfg.get(
            "aggressive_risk_threshold", 0.70
        )

        self.rebalancer = None

        logger.info(
            f"[RISK SIZER] Initialized\n"
            f"  Target Risk: {self.target_risk_pct:.2%}\n"
            f"  Max Risk:    {self.max_risk_pct:.2%}\n"
            f"  Futures:     {'✓ Dynamic Margin' if self.margin_calculator else '✗ Not available'}"
        )

    def set_rebalancer(self, rebalancer: PositionRebalancer):
        """Set the rebalancer (called by handler after initialization)"""
        self.rebalancer = rebalancer
        logger.info("[RISK SIZER] ✓ Rebalancer connected")

    def _get_available_balance(
        self,
        asset: str,
        is_futures: bool = False,
        entry_price: float = None,
        stop_loss_price: float = None,
    ) -> Tuple[float, Dict]:
        """Get available balance with margin info"""
        try:
            if is_futures and self.margin_calculator:
                # Get real-time margin info from Binance
                margin_info = self.margin_calculator.get_available_margin_info(asset)

                # If we have price info, calculate max safe position
                if entry_price and stop_loss_price:
                    max_pos_usd, max_qty = (
                        self.margin_calculator.calculate_max_safe_position(
                            available_margin=margin_info["available_balance"],
                            leverage=margin_info["leverage"],
                            entry_price=entry_price,
                            stop_loss_price=stop_loss_price,
                        )
                    )
                    margin_info["max_safe_position_usd"] = max_pos_usd
                    margin_info["max_safe_quantity"] = max_qty

                return margin_info["available_balance"], margin_info
            else:
                # Spot/Portfolio balance
                balance = self.portfolio_manager.get_asset_balance(asset)
                return balance, {"source": "portfolio", "balance": balance}

        except Exception as e:
            logger.error(f"[RISK] Error getting balance: {e}")
            return 0.0, {"error": str(e)}

    def calculate_size_risk_based(
        self,
        asset: str,
        entry_price: float,
        stop_loss_price: float,
        signal: int,
        confidence_score: float = None,
        market_condition: str = None,
        sizing_mode: str = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        override_reason: str = None,
        is_futures: bool = False,
        risk_pct: float = None,  # ✨ NEW: Accept external risk budget
    ) -> Tuple[float, Dict]:
        """
        ✨ REFACTORED: Accept risk budget from Portfolio Manager
        No longer calculates its own risk percentages
        """
        try:
            # Get balance
            asset_balance, margin_info = self._get_available_balance(
                asset=asset,
                is_futures=is_futures,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
            )

            if asset_balance <= 0:
                logger.error(f"[RISK] No available balance for {asset}")
                return 0.0, {"error": "insufficient_balance"}

            # ✨ CRITICAL: Use externally provided risk
            if risk_pct is None:
                logger.error("[RISK] No risk percentage provided!")
                return 0.0, {"error": "missing_risk_budget"}

            # Calculate position size from risk budget
            risk_amount = asset_balance * risk_pct

            stop_distance = abs(entry_price - stop_loss_price)
            stop_distance_pct = stop_distance / entry_price

            if stop_distance_pct < 0.005:
                stop_distance_pct = 0.005

            target_position_size = risk_amount / stop_distance_pct

            # Apply Futures margin limits if applicable
            final_position_size = target_position_size
            was_margin_limited = False

            if is_futures and "max_safe_position_usd" in margin_info:
                max_safe = margin_info["max_safe_position_usd"]

                if target_position_size > max_safe:
                    final_position_size = max_safe
                    was_margin_limited = True
                    logger.warning(
                        f"[MARGIN] Position reduced: ${final_position_size:,.2f}"
                    )

            # Apply asset max
            asset_cfg = self.config["assets"][asset]
            max_size = asset_cfg.get("max_position_usd", 100000)
            final_position_size = min(final_position_size, max_size)

            actual_risk = final_position_size * stop_distance_pct
            actual_risk_pct = actual_risk / asset_balance

            metadata = {
                "asset": asset,
                "is_futures": is_futures,
                "signal": signal,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "provided_risk_pct": risk_pct * 100,
                "target_position_size": target_position_size,
                "final_position_size": final_position_size,
                "actual_risk_usd": actual_risk,
                "actual_risk_pct": actual_risk_pct * 100,
                "margin_info": margin_info,
                "was_margin_limited": was_margin_limited,
            }

            return final_position_size, metadata

        except Exception as e:
            logger.error(f"[RISK] Error: {e}", exc_info=True)
            return 0.0, {"error": str(e)}


class BinanceExecutionHandler:
    """
    Binance Execution Handler with Hybrid Position Sizing + Order Tracking
    """

    def __init__(
        self, config: Dict, client: Client, portfolio_manager, data_manager=None
    ):
        self.config = config
        self.client = client
        self.client.session.headers.update(CLOUDFRONT_HEADERS)
        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager

        self.asset_config = config["assets"]["BTC"]
        self.risk_config = config["risk_management"]
        self.trading_config = config["trading"]
        self.error_handler = None
        self.trading_bot = None

        self.symbol = self.asset_config["symbol"]
        self.mode = self.trading_config.get("mode", "paper")
        self.max_positions_per_asset = config.get("trading", {}).get(
            "max_positions_per_asset", 3
        )
        self.is_paper_mode = self.mode.lower() == "paper"
        self.execution_lock = {}  # ✨ NEW: Prevent duplicate trades
        self.last_trade_time = {}  # ✨ NEW: Rapid-fire cooldown
        self.trade_timestamps_hourly = []  # ✨ NEW: Hourly trade limit

        # ✨ NEW: Standardized Hedging Config
        self.allow_hedging = self.trading_config.get(
            "allow_simultaneous_long_short", False
        )

        # ✅ STEP 1: Initialize Futures handler FIRST
        self.futures_handler = None
        futures_enabled = self.asset_config.get("enable_futures", False)

        if futures_enabled:
            try:
                logger.info("[HANDLER] Initializing Binance Futures...")
                self.futures_handler = BinanceFuturesHandler(
                    client=client, symbol=self.symbol, config=self.config
                )

                # Set leverage and margin type
                leverage = self.asset_config.get("leverage", 20)
                margin_type = self.asset_config.get("margin_type", "CROSSED")

                self.futures_handler.set_leverage(leverage)
                self.futures_handler.set_margin_type(margin_type)

                logger.info(
                    f"[HANDLER] ✓ Futures initialized\n"
                    f"  Leverage: {leverage}x\n"
                    f"  Margin:   {margin_type}"
                )

            except Exception as e:
                logger.error(f"[HANDLER] Futures initialization failed: {e}")
                self.futures_handler = None
        else:
            logger.info("[HANDLER] Futures trading disabled in config")

        # ✅ STEP 2: Initialize sizer WITH futures_handler reference
        self.sizer = HybridPositionSizer(
            config, portfolio_manager, futures_handler=self.futures_handler
        )

        # ✅ STEP 3: Initialize rebalancer and connect to sizer
        if self.futures_handler:
            rebalancer = PositionRebalancer(
                futures_handler=self.futures_handler,
                portfolio_manager=self.portfolio_manager,
            )
            self.sizer.set_rebalancer(rebalancer)
            logger.info("[HANDLER] ✓ Auto-rebalancing enabled")

        logger.info(f"BinanceExecutionHandler initialized - Mode: {self.mode.upper()}")

        # ✅ STEP 4: Auto-sync on startup (if enabled)
        if self.mode.lower() != "paper" and self.trading_config.get(
            "auto_sync_on_startup", True
        ):
            logger.info("[INIT] Auto-syncing positions with Binance...")
            self.sync_positions_with_binance("BTC")

    def can_open_position_side(self, asset_name: str, side: str) -> Tuple[bool, str]:
        """Check if we can open a position on a specific SIDE"""
        if side == "short":
            allow_shorts = self.config["assets"][asset_name].get("allow_shorts", False)
            if not allow_shorts:
                return False, f"Short trading disabled for {asset_name} in config"

        can_open_pm, pm_reason = self.portfolio_manager.can_open_position(
            asset_name, side
        )
        if not can_open_pm:
            return False, f"Portfolio limit: {pm_reason}"

        current_count = self.portfolio_manager.get_asset_position_count(
            asset_name, side
        )
        max_per_asset = self.max_positions_per_asset

        if current_count >= max_per_asset:
            return (
                False,
                f"Already have {current_count}/{max_per_asset} {side.upper()} positions",
            )

        return (
            True,
            f"OK - {current_count}/{max_per_asset} {side.upper()} positions open",
        )

    def _fetch_live_futures_price(self, symbol: str) -> Optional[float]:
        """
        Fetches the live price directly from the futures ticker endpoint.
        This method is guarded, has no retries, and returns None on any error.
        """
        try:
            # This is the only place that should call a live ticker endpoint.
            if self.futures_handler:
                # If in paper mode, we want a REAL price, not testnet.
                # Testnet prices can diverge significantly from the real market.
                if self.is_paper_mode:
                    try:
                        # Use a public client for live futures data (no keys needed)
                        from binance.client import Client as BClient
                        public_client = BClient("", "", requests_params={'timeout': 5})
                        public_client.API_URL = "https://fapi.binance.com"
                        ticker = public_client.futures_symbol_ticker(symbol=symbol)
                        return float(ticker["price"])
                    except Exception as e:
                        logger.debug(f"[PRICE] Could not fetch live futures ticker for {symbol}: {e}")
                
                # Fallback to the primary client (might be testnet or live keys)
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                return float(ticker["price"])
            else:
                logger.warning("[PRICE] Live futures price fetch skipped: Futures handler not available.")
                return None
        except Exception as e:
            # Catch all exceptions (APIError, JSONDecodeError, etc.) and return None
            # The error is already logged by the calling function, so we suppress it here.
            return None

    @handle_errors(
        component="binance_handler",
        severity=ErrorSeverity.ERROR,
        notify=True,
        reraise=False,
        default_return=None,
    )
    def get_current_price(self, symbol: str = None, force_live: bool = False) -> Optional[float]:
        """
        Unified price accessor for the entire system, with corrected force_live logic.
        """
        if symbol is None:
            symbol = self.symbol

        # 1. If a live price is forced, attempt to fetch it first.
        if force_live:
            # In paper mode, we attempt a live fetch but don't fail if it's unavailable
            live_price = self._fetch_live_futures_price(symbol)
            if live_price is not None:
                # Update cache and return the fresh price.
                price_cache.set(symbol, live_price)
                return live_price
            else:
                if self.is_paper_mode:
                    logger.debug(f"[PRICE] Live fetch failed in paper mode for {symbol}. Using cache/fallback.")
                else:
                    logger.warning(f"[PRICE] Live fetch failed. Falling back to cache for {symbol}.")

        # 2. If not forcing live, or if live fetch failed, try the cache.
        cached_price = price_cache.get(symbol)
        if cached_price is not None:
            return cached_price

        # 3. As a final fallback, check the last known price from the cache.
        last_known_price = price_cache.get_last_known(symbol)
        if last_known_price:
            logger.info(f"[PRICE] Using last known cached price for {symbol}: {last_known_price}")
            return last_known_price
        
        logger.error(f"Error fetching price for {symbol}: All methods (live and cache) failed.")
        return None


    @handle_errors(
        component="binance_handler",
        severity=ErrorSeverity.CRITICAL,
        notify=True,
        reraise=False,
        default_return=False,
    )
    def execute_signal(
        self,
        signal: int,
        current_price: float = None,
        asset_name: str = "BTC",
        confidence_score: float = None,
        market_condition: str = None,
        sizing_mode: str = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        override_reason: str = None,
        signal_details: Dict = None,
    ) -> bool:
        """
        ✅ ENHANCED: Execute trading signal with Asymmetric Hedging Support
        """

        if asset_name != "BTC":
            logger.error(f"[BINANCE HANDLER] Wrong Asset: {asset_name}")
            return False

        # ================================================================
        # RAPID-FIRE COOLDOWN (30s)
        # ================================================================
        now = time.time()
        
        # 1. Rapid-fire check (30s)
        last_time = self.last_trade_time.get(asset_name, 0)
        if now - last_time < 30:
            logger.warning(f"[COOLDOWN] Rapid-fire blocked for {asset_name} ({30 - (now - last_time):.1f}s remaining)")
            return False

        # 2. Hourly trade limit (Max 5 trades per hour)
        self.trade_timestamps_hourly = [
            t for t in self.trade_timestamps_hourly if now - t < 3600
        ]
        if len(self.trade_timestamps_hourly) >= 5:
            logger.warning(f"[THROTTLE] Hourly trade limit reached ({len(self.trade_timestamps_hourly)}/5). Blocking {asset_name}.")
            return False

        # ================================================================
        # DUPLICATE EXECUTION LOCK
        # ================================================================
        trade_type = "TREND"
        if signal_details:
            trade_type = signal_details.get("trade_type", "TREND")
            
        trade_key = f"{asset_name}_{trade_type}_{signal}"
        
        if self.execution_lock.get(trade_key, False):
            logger.warning(f"[LOCK] Duplicate execution blocked for {trade_key}")
            return False
            
        self.execution_lock[trade_key] = True
        
        try:
            if current_price is None:
                current_price = self.get_current_price(symbol=self.symbol, force_live=True)

            if current_price is None or current_price <= 0:
                logger.error(f"{asset_name}: Invalid price: {current_price}")
                return False

            existing_positions = self.portfolio_manager.get_asset_positions(asset_name)
            long_positions = [p for p in existing_positions if p.side == "long"]
            short_positions = [p for p in existing_positions if p.side == "short"]

            logger.info(
                f"\n{'='*80}\n"
                f"[SIGNAL] {asset_name} Signal: {signal:+2d} | Trade Type: {trade_type}\n"
                f"[STATE] Current Positions: {len(long_positions)} LONG, {len(short_positions)} SHORT\n"
                f"[CONFIG] Hedging Allowed: {self.allow_hedging}\n"
                f"{'='*80}"
            )

            # ================================================================
            # SCENARIO 1: SELL SIGNAL (-1)
            # ================================================================
            if signal == -1:
                # ✨ NEW HEDGING CHECK: Only close Longs if Hedging is DISABLED
                if not self.allow_hedging and long_positions:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"📉 SELL SIGNAL - Hedging Disabled: Closing ALL {len(long_positions)} LONG position(s)\n"
                        f"{'='*80}"
                    )

                    closed_count = 0
                    # Create a copy of the list to iterate over, as the underlying self.positions dict will be modified
                    for i, position in enumerate(list(long_positions), 1):
                        logger.info(f"  Closing LONG position: {position.position_id}")
                        # Correctly call the portfolio manager to close the position
                        result = self.portfolio_manager.close_position(
                            position_id=position.position_id,
                            exit_price=current_price,
                            reason="sell_signal",
                        )
                        if result:
                            closed_count += 1

                elif self.allow_hedging and long_positions:
                    logger.info(
                        f"\n[HEDGING] Keeping {len(long_positions)} LONG position(s) open "
                        f"(hedging enabled)"
                    )

                # Open SHORT
                can_open, reason = self.can_open_position_side(asset_name, "short")
                if not can_open:
                    logger.warning(f"[SKIP] Cannot open SHORT: {reason}")
                    # If hedging was enabled, we still have longs open, so return True
                    return True if (long_positions and self.allow_hedging) else False

                logger.info(
                    f"\n📉 SELL SIGNAL - Opening new SHORT position ({trade_type})"
                )

                return self._open_position(
                    signal=-1,
                    current_price=current_price,
                    asset_name=asset_name,
                    confidence_score=confidence_score,
                    market_condition=market_condition,
                    sizing_mode=sizing_mode,
                    manual_size_usd=manual_size_usd,
                    override_reason=override_reason,
                    signal_details=signal_details,
                )

            # ================================================================
            # SCENARIO 2: BUY SIGNAL (+1)
            # ================================================================
            elif signal == 1:
                # ✨ NEW HEDGING CHECK: Only close Shorts if Hedging is DISABLED
                if not self.allow_hedging and short_positions:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"📈 BUY SIGNAL - Hedging Disabled: Closing ALL {len(short_positions)} SHORT position(s)\n"
                        f"{'='*80}"
                    )

                    closed_count = 0
                    # Create a copy of the list to iterate over, as the underlying self.positions dict will be modified
                    for i, position in enumerate(list(short_positions), 1):
                        logger.info(f"  Closing SHORT position: {position.position_id}")
                        # Correctly call the portfolio manager to close the position
                        result = self.portfolio_manager.close_position(
                            position_id=position.position_id,
                            exit_price=current_price,
                            reason="buy_signal",
                        )
                        if result:
                            closed_count += 1

                elif self.allow_hedging and short_positions:
                    logger.info(
                        f"\n[HEDGING] Keeping {len(short_positions)} SHORT position(s) open "
                        f"(hedging enabled)"
                    )

                # Open LONG
                can_open, reason = self.can_open_position_side(asset_name, "long")
                if not can_open:
                    logger.warning(f"[SKIP] Cannot open LONG: {reason}")
                    return True if (short_positions and self.allow_hedging) else False

                logger.info(
                    f"\n📈 BUY SIGNAL - Opening new LONG position ({trade_type})"
                )

                return self._open_position(
                    signal=1,
                    current_price=current_price,
                    asset_name=asset_name,
                    confidence_score=confidence_score,
                    market_condition=market_condition,
                    sizing_mode=sizing_mode,
                    manual_size_usd=manual_size_usd,
                    override_reason=override_reason,
                    signal_details=signal_details,
                )

            # ================================================================
            # SCENARIO 3: HOLD SIGNAL (0)
            # ================================================================
            elif signal == 0:
                if not existing_positions:
                    return False

                positions_closed = False
                for position in existing_positions:
                    should_close, close_reason = self._check_stop_loss_take_profit(
                        position, current_price
                    )
                    if should_close:
                        logger.info(
                            f"[AUTO-CLOSE] {position.position_id}: {close_reason}"
                        )
                        if self._close_position(
                            position, current_price, asset_name, close_reason
                        ):
                            positions_closed = True

                return positions_closed

            return False

        except Exception as e:
            logger.error(f"Error executing {asset_name} signal: {e}", exc_info=True)
            return False
        finally:
            self.execution_lock[trade_key] = False

    def _calculate_asymmetric_risk(
        self, trade_type: str, base_risk: float = 0.015
    ) -> Tuple[float, Dict]:
        """Calculate risk based on trade type"""
        risk_profiles = {
            "TREND": {
                "multiplier": 1.33,  # 2% risk (1.5% * 1.33)
                "description": "Full trend trade",
                "break_even_trigger": 0.015,
                "trailing_stop": 0.025,
            },
            "SCALP": {
                "multiplier": 0.67,  # 1% risk (1.5% * 0.67)
                "description": "Conservative scalp",
                "break_even_trigger": 0.005,
                "trailing_stop": 0.015,
            },
            "V_SHAPE": {
                "multiplier": 1.0,  # 1.5% risk
                "description": "Recovery play",
                "break_even_trigger": 0.010,
                "trailing_stop": 0.020,
            },
        }

        profile = risk_profiles.get(trade_type, risk_profiles["TREND"])
        adjusted_risk = base_risk * profile["multiplier"]

        logger.info(f"\n[RISK CALC] Trade Type: {trade_type}")
        logger.info(f"  Adjusted Risk: {adjusted_risk:.2%}")

        return adjusted_risk, profile

    def _check_stop_loss_take_profit(
        self, position, current_price: float
    ) -> Tuple[bool, str]:
        """Check if stop-loss or take-profit is hit (fallback for non-VTM)"""
        try:
            if not position.stop_loss and not position.take_profit:
                return False, ""

            side = position.side
            entry_price = position.entry_price
            stop_loss = position.stop_loss
            take_profit = position.take_profit

            price_tolerance = 0.50

            if side == "long":
                if stop_loss and current_price <= (stop_loss + price_tolerance):
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    return True, f"stop_loss_hit ({pnl_pct:+.2f}%)"

                if take_profit and current_price >= (take_profit - price_tolerance):
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    return True, f"take_profit_hit ({pnl_pct:+.2f}%)"

            elif side == "short":
                if stop_loss and current_price >= (stop_loss - price_tolerance):
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    return True, f"stop_loss_hit ({pnl_pct:+.2f}%)"

                if take_profit and current_price <= (take_profit + price_tolerance):
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    return True, f"take_profit_hit ({pnl_pct:+.2f}%)"

            return False, ""

        except Exception as e:
            logger.error(f"Error checking SL/TP: {e}")
            return False, ""

    def _round_quantity(
        self, quantity: float, symbol: str = "BTCUSDT", is_futures: bool = False
    ) -> float:
        """Round quantity to correct lot size and precision for Binance"""
        try:
            if is_futures:
                exchange_info = self.client.futures_exchange_info()
            else:
                exchange_info = self.client.get_exchange_info()

            for s in exchange_info["symbols"]:
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step_size = float(f["stepSize"])
                            min_qty = float(f["minQty"])
                            precision = len(str(step_size).rstrip("0").split(".")[-1])

                            rounded_qty = round(quantity / step_size) * step_size
                            rounded_qty = round(rounded_qty, precision)
                            rounded_qty = max(min_qty, rounded_qty)

                            return rounded_qty

            return round(quantity, 5)

        except Exception as e:
            logger.error(f"Error rounding quantity: {e}")
            return round(quantity, 5)

    @handle_errors(
        component="binance_handler",
        severity=ErrorSeverity.CRITICAL,
        notify=True,
        reraise=False,
        default_return=False,
    )
    def _open_position(
        self,
        signal: int,
        current_price: float,
        asset_name: str,
        confidence_score: float = None,
        market_condition: str = None,
        sizing_mode: str = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        override_reason: str = None,
        signal_details: Dict = None,
    ) -> bool:
        """
        ✅ STRATEGIC/TACTICAL INTEGRATION
        Portfolio Manager controls HOW MUCH to risk (strategy)
        VTM validates HOW to execute (tactics)
        """
        try:
            side = "long" if signal == 1 else "short"

            # Extract trade type from signal details
            trade_type = "TREND"  # Default
            if signal_details:
                trade_type = signal_details.get("trade_type", "TREND")

            is_futures = (
                hasattr(self, "futures_handler")
                and self.futures_handler is not None
                and self.config.get("assets", {})
                .get(asset_name, {})
                .get("enable_futures", False)
            )

            # ================================================================
            # STEP 1: STRATEGIC - Get risk budget from Portfolio Manager
            # ================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"[STRATEGIC] Requesting risk budget from Portfolio Manager")
            logger.info(f"{'='*80}")

            risk_pct = self.portfolio_manager.get_risk_budget(
                asset=asset_name,
                strategy_type=trade_type,
                confidence_score=signal_details.get("mode_confidence"),
                market_condition=signal_details.get("regime")
            )

            if risk_pct <= 0:
                logger.error(
                    f"[STRATEGIC] ❌ Risk budget denied (0%)\n"
                    f"  → Trade rejected by Portfolio Manager"
                )
                return False

            # T1.7 fix: apply MTF regime multiplier computed in main.py but previously orphaned
            mtf_multiplier = signal_details.get("mtf_risk_multiplier", 1.0) if signal_details else 1.0
            if mtf_multiplier != 1.0:
                risk_pct *= mtf_multiplier
                logger.info(f"[RISK] MTF multiplier applied: {mtf_multiplier:.1f}x → risk_pct={risk_pct:.4f}")

            logger.info(f"[STRATEGIC] ✓ Risk budget approved: {risk_pct:.3%}")

            # ================================================================
            # STEP 2: Calculate initial stop loss for validation
            # ================================================================
            risk_config = self.asset_config.get("risk", {})
            
            # ATR-based adaptive stop loss distance
            atr_fast = signal_details.get("atr_fast") if signal_details else None
            atr_multiplier = risk_config.get("atr_multiplier", 1.8)
            
            if atr_fast:
                initial_sl_dist = atr_fast * atr_multiplier
                logger.info(f"[TACTICAL] Using ATR-based SL: {atr_multiplier}x ATR ({initial_sl_dist:.2f})")
            else:
                sl_pct = risk_config.get("stop_loss_pct", 0.02)
                initial_sl_dist = current_price * sl_pct
                logger.info(f"[TACTICAL] ⚠️ ATR not found, using static SL: {sl_pct:.2%}")

            if side == "long":
                initial_stop = current_price - initial_sl_dist
            else:
                initial_stop = current_price + initial_sl_dist

            # ================================================================
            # STEP 3: TACTICAL - VTM Pre-Flight Validation
            # ================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"[TACTICAL] VTM Pre-Flight Validation")
            logger.info(f"{'='*80}")

            is_valid, rejection_reason = VeteranTradeManager.validate_trade_setup(
                entry_price=current_price,
                stop_loss=initial_stop,
                risk_config=risk_config,
                trade_type=trade_type,
                atr_fast=signal_details.get("atr_fast") if signal_details else None
            )

            if not is_valid:
                logger.error(
                    f"[TACTICAL] ❌ Trade rejected by VTM\n"
                    f"  Reason: {rejection_reason}\n"
                    f"  → Aborting before paying fees"
                )
                return False

            logger.info(f"[TACTICAL] ✓ Trade validated by VTM")

            # ================================================================
            # STEP 4: Calculate position size using strategic risk budget
            # ================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"[SIZING] Calculating position size")
            logger.info(f"{'='*80}")

            # Get account balance
            account_balance = self.portfolio_manager.get_asset_balance(asset_name)

            # Calculate risk amount
            risk_amount_usd = account_balance * risk_pct

            # Calculate stop distance
            stop_distance = abs(current_price - initial_stop)
            stop_distance_pct = stop_distance / current_price

            # Position size = Risk Amount / Stop Distance %
            position_size_usd = risk_amount_usd / stop_distance_pct

            logger.info(
                f"[SIZING] Calculation:\n"
                f"  Account Balance: ${account_balance:,.2f}\n"
                f"  Risk Budget:     {risk_pct:.3%} = ${risk_amount_usd:.2f}\n"
                f"  Stop Distance:   {stop_distance_pct:.3%}\n"
                f"  Position Size:   ${position_size_usd:,.2f}"
            )

            if position_size_usd <= 0:
                logger.error(
                    f"[SIZING] ❌ Invalid position size: ${position_size_usd:.2f}"
                )
                return False

            # Apply margin limits if futures
            if is_futures and hasattr(self.sizer, "margin_calculator"):
                margin_info = self.sizer.margin_calculator.get_available_margin_info(
                    asset_name
                )
                max_safe = margin_info.get("max_safe_position_usd", position_size_usd)

                if position_size_usd > max_safe:
                    logger.warning(
                        f"[MARGIN] Position reduced to fit margin:\n"
                        f"  Calculated: ${position_size_usd:,.2f}\n"
                        f"  Max Safe:   ${max_safe:,.2f}"
                    )
                    position_size_usd = max_safe

            # Calculate quantity
            quantity = position_size_usd / current_price
            leverage = 1
            margin_type = "SPOT"

            if is_futures:
                asset_conf = self.config.get("assets", {}).get(asset_name, {})
                leverage = asset_conf.get("leverage", 20)
                margin_type = asset_conf.get("margin_type", "CROSSED")
                quantity = self.futures_handler._round_quantity(quantity)
            else:
                quantity = self._round_quantity(quantity, self.symbol, False)

            MIN_BTC = 0.00001
            if quantity < MIN_BTC:
                logger.warning(
                    f"[SIZING] ❌ Quantity {quantity:.8f} below minimum {MIN_BTC}"
                )
                return False

            # ================================================================
            # STEP 5: Execute order on exchange (with SAFE RETRY)
            # ================================================================
            order_id = None
            requested_price = current_price
            executed_price = current_price
            order = None
            MAX_RETRIES = 2

            for attempt in range(MAX_RETRIES):
                try:
                    if is_futures:
                        if side == "long":
                            order = self.futures_handler.open_long_position(
                                quantity=quantity,
                                stop_loss=initial_stop,
                                take_profit=None,
                            )
                        else:
                            order = self.futures_handler.open_short_position(
                                quantity=quantity,
                                stop_loss=initial_stop,
                                take_profit=None,
                            )
                    else:
                        if not self.is_paper_mode:
                            if side == "long":
                                order = self.client.order_market_buy(
                                    symbol=self.symbol, quantity=quantity
                                )
                            else:
                                logger.error("[SPOT] ❌ SHORT requires Futures API")
                                return False
                        else:
                            # Paper Mode Simulation
                            order = {
                                "status": "FILLED",
                                "orderId": f"PAPER_{side.upper()}_{int(time.time())}",
                                "avgPrice": current_price
                            }

                    # Validate response
                    # ✅ ENHANCED: If we have an orderId, the order was accepted by the exchange
                    if order and order.get("orderId"):
                        # For MARKET orders, NEW often means it was accepted and is being filled
                        if order.get("status") in ["FILLED", "PARTIALLY_FILLED", "NEW"]:
                            break
                    
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"[RETRY] Order failed or not filled (Attempt {attempt+1}/{MAX_RETRIES}). Retrying in 1s...")
                        time.sleep(1)
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"[RETRY] Exception during execution: {e}. Retrying in 1s...")
                        time.sleep(1)
                    else:
                        logger.error(f"[EXECUTION] ❌ Critical failure after {MAX_RETRIES} attempts: {e}")
                        return False

            # ✅ FINAL VALIDATION (Step 1)
            # If we have an orderId, we proceed. We'll fetch the price/status if NEW.
            if not order or not order.get("orderId"):
                logger.error(f"[EXECUTION] ❌ Order not filled properly after retries.")
                return False

            order_id = order.get("orderId")
            
            # ✅ EXTRACT EXECUTED PRICE (Step 2 & 3)
            # If status is NEW or price is 0, fetch the latest order status from the exchange
            raw_price = order.get("avgPrice", 0)
            executed_price = float(raw_price) if raw_price else 0
            
            if executed_price == 0 and is_futures and not self.is_paper_mode:
                try:
                    logger.info(f"[EXECUTION] Fetching actual fill price for order {order_id}...")
                    time.sleep(0.5) # Tiny wait for exchange to process fills
                    updated_order = self.client.futures_get_order(symbol=self.symbol, orderId=order_id)
                    raw_price = updated_order.get("avgPrice", 0)
                    executed_price = float(raw_price) if raw_price else 0
                    
                    if executed_price == 0:
                        # Try cumulative quote quantity
                        cum_quote = float(updated_order.get("cumQuote", 0))
                        exec_qty = float(updated_order.get("executedQty", 0))
                        if exec_qty > 0:
                            executed_price = cum_quote / exec_qty
                except Exception as e:
                    logger.warning(f"[EXECUTION] Failed to fetch updated order status: {e}")
                
            if executed_price <= 0:
                executed_price = current_price
                logger.warning(f"[EXECUTION] Using current_price fallback: ${executed_price:,.2f}")

            # ✅ TRACK SLIPPAGE
            slippage = abs(executed_price - requested_price)
            slippage_pct = (slippage / requested_price) * 100 if requested_price > 0 else 0
            logger.info(
                f"[SLIPPAGE] {asset_name} {side.upper()} | "
                f"Req: ${requested_price:,.2f}, Fill: ${executed_price:,.2f}, "
                f"Diff: ${slippage:,.2f} ({slippage_pct:.4f}%)"
            )

            logger.info(
                f"[EXECUTION] ✓ {side.upper()} opened & filled\n"
                f"  Order ID: {order_id}\n"
                f"  Fill Price: ${executed_price:,.2f}"
            )

            # ================================================================
            # STEP 6: Fetch OHLC for VTM
            # ================================================================
            ohlc_data = None
            if self.data_manager:
                try:
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - timedelta(days=10)

                    df = self.data_manager.fetch_binance_data(
                        symbol=self.symbol,
                        interval=self.asset_config.get("interval", "1h"),
                        start_date=start_time.strftime("%Y-%m-%d"),
                        end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    )

                    if len(df) > 0:
                        ohlc_data = {
                            "high": df["high"].values,
                            "low": df["low"].values,
                            "close": df["close"].values,
                            "volume": df["volume"].values,
                        }
                except Exception as e:
                    logger.warning(f"[VTM] ⚠️ OHLC fetch failed: {e}")

            # ================================================================
            # STEP 7: Add to Portfolio with VTM (TACTICAL)
            # ================================================================
            # Build complete signal details for VTM
            if signal_details is None:
                signal_details = {}

            signal_details.update(
                {
                    "trade_type": trade_type,
                    "strategic_risk_pct": risk_pct,
                    "tactical_validation": "passed",
                }
            )

            success = self.portfolio_manager.add_position(
                asset=asset_name,
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                position_size_usd=position_size_usd,
                stop_loss=None,  # VTM will calculate precise levels
                take_profit=None,
                trailing_stop_pct=None,
                binance_order_id=order_id,
                ohlc_data=ohlc_data,
                use_dynamic_management=True,
                signal_details=signal_details,
                leverage=leverage,
                margin_type=margin_type,
                is_futures=is_futures,
            )

            if success:
                # ✅ Standardized Log
                log_trade_event("ENTRY", {
                    "symbol": self.symbol,
                    "asset": asset_name,
                    "side": side,
                    "price": current_price,
                    "size": quantity,
                    "trade_type": trade_type,
                    "position_id": order_id
                })

                # ✅ Update last trade time for cooldown
                self.last_trade_time[asset_name] = time.time()
                self.trade_timestamps_hourly.append(time.time()) # Record for hourly limit
                
                logger.info(
                    f"\n{'='*80}\n"
                    f"✅ {asset_name} {side.upper()} POSITION OPENED\n"
                    f"{'='*80}\n"
                    f"Strategic Risk: {risk_pct:.3%}\n"
                    f"Trade Type:     {trade_type}\n"
                    f"Position Size:  ${position_size_usd:,.2f}\n"
                    f"VTM Active:     {'Yes' if ohlc_data else 'No'}\n"
                    f"⚠ SL Management:  VTM Only (No Exchange SL)\n"
                    f"{'='*80}"
                )
                return True
            else:
                logger.error(f"[PORTFOLIO] ❌ Position rejected")
                return False

        except Exception as e:
            logger.error(f"[OPEN] ❌ Error: {e}", exc_info=True)
            return False

    @handle_errors(
        component="binance_handler",
        severity=ErrorSeverity.CRITICAL,
        notify=True,
        reraise=False,
        default_return=False,
    )
    def _close_position(
        self, position, current_price: float, asset_name: str, reason: str
    ) -> bool:
        """Close LONG or SHORT position using the correct API (Futures or Spot)."""
        try:
            side = position.side
            quantity = position.quantity
            order_id = position.binance_order_id
            is_futures = getattr(position, "is_futures", False)

            logger.info(
                f"[CLOSE] Closing {asset_name} {side.upper()} position ({reason}) | Futures: {is_futures}"
            )

            # --- Case 1: Futures Position ---
            if is_futures:
                if not self.futures_handler:
                    logger.error(
                        "[FUTURES] Cannot close: Futures handler not available."
                    )
                    return False
                try:
                    if side == "long":
                        success = self.futures_handler.close_long_position(
                            quantity=quantity, order_id=order_id
                        )
                    else:  # short
                        success = self.futures_handler.close_short_position(
                            quantity=quantity, order_id=order_id
                        )

                    if success:
                        logger.info(
                            f"[FUTURES] ✓ Close order for {side.upper()} position succeeded."
                        )
                        return True
                    else:
                        logger.error(f"[FUTURES] ❌ Failed to close {side.upper()}")
                        return False
                except Exception as e:
                    logger.error(
                        f"[FUTURES] ❌ Exception during close: {e}", exc_info=True
                    )
                    return False

            # --- Case 2: Spot Position ---
            else:
                if self.is_paper_mode:
                    logger.info(
                        f"[PAPER] Simulated close for spot position: {order_id}"
                    )
                    # Explicitly call portfolio_manager to remove position in paper mode
                    self.portfolio_manager.close_position(
                        position_id=position.position_id,
                        exit_price=current_price,
                        reason=reason,
                    )
                    return True

                try:
                    if side == "long":
                        # To close a long spot position, you sell the asset
                        self.client.order_market_sell(
                            symbol=self.symbol, quantity=quantity
                        )
                        logger.info(
                            f"[SPOT] ✓ Market sell order to close long position was successful."
                        )
                        return True
                    else:  # short
                        logger.error(
                            "[SPOT] ❌ Cannot close short position on Spot market. Requires Futures."
                        )
                        return False
                except Exception as e:
                    logger.error(
                        f"[SPOT] ❌ Exception during close: {e}", exc_info=True
                    )
                    return False

        except Exception as e:
            logger.error(
                f"[CLOSE] Unhandled error in _close_position: {e}", exc_info=True
            )
            return False

    def check_and_update_positions_VTM(self, asset_name: str = "BTC", df_4h: Optional[pd.DataFrame] = None):
        """Check and update ALL positions with VTM"""
        try:
            positions = self.portfolio_manager.get_asset_positions(asset_name)
            if not positions:
                return False

            # Get the correct symbol for this asset from config
            symbol = self.config["assets"].get(asset_name, {}).get("symbol")
            if not symbol:
                logger.error(f"[VTM] Could not find symbol for asset {asset_name}")
                return False

            current_price = self.get_current_price(symbol=symbol, force_live=True)
            if not current_price:
                return False

            # ✅ RECONCILIATION: Fetch live positions from Binance
            is_futures = self.config.get("assets", {}).get(asset_name, {}).get("enable_futures", False)
            if is_futures and self.futures_handler:
                try:
                    active_positions = self.futures_handler.get_all_positions_info()
                    # Convert to simple format for PortfolioManager reconciliation
                    # Note: We match by 'side' for Binance as it aggregates positions
                    broker_data = [{'side': p['side'], 'quantity': abs(float(p['positionAmt']))} for p in active_positions]
                    self.portfolio_manager.reconcile_positions(asset_name, broker_data)
                except Exception as e:
                    logger.debug(f"[RECONCILE] Binance fetch failed: {e}")

            positions_closed = False
            pyramid_requests = []

            for position in positions:
                if position.trade_manager:
                    exit_signal = position.trade_manager.update_with_current_price(
                        current_price, df_4h=df_4h
                    )

                    if exit_signal:
                        # ✅ Check if it's an action (like pyramid) or an exit (reason)
                        if isinstance(exit_signal, dict) and "action" in exit_signal:
                            action = exit_signal["action"]
                            logger.info(f"[VTM] {position.position_id} triggered action: {action.upper()}")
                            
                            # Add to pyramid requests
                            pyramid_requests.append({
                                "asset": asset_name,
                                "side": position.side,
                                "action": action,
                                "original_position_id": position.position_id,
                                "signal_details": getattr(position, 'signal_details', {})
                            })
                            continue

                        # ✅ Handle standard exits
                        exit_reason = exit_signal.get("reason", "unknown") if isinstance(exit_signal, dict) else exit_signal
                        if hasattr(exit_reason, "value"):
                            exit_reason_str = exit_reason.value
                        else:
                            exit_reason_str = str(exit_reason)

                        logger.info(
                            f"[VTM] {position.position_id} triggered {exit_reason_str.upper()}"
                        )
                        self.portfolio_manager.close_position(
                            position_id=position.position_id, # Pass the string position_id
                            exit_price=current_price,
                            reason=f"VTM_{exit_reason_str}",
                        )
                        positions_closed = True
                        continue

                should_close, reason = self._check_stop_loss_take_profit(
                    position, current_price
                )
                if should_close:
                    self.portfolio_manager.close_position(
                        position_id=position.position_id, # Pass the string position_id
                        exit_price=current_price,
                        reason=reason,
                    )
                    positions_closed = True

            return {"closed": positions_closed, "pyramid_requests": pyramid_requests}

        except Exception as e:
            logger.error(f"[VTM] Error: {e}", exc_info=True)
            return False

    def check_and_update_positions(self, asset_name: str = "BTC"):
        """Actively check and update all positions"""
        try:
            return self.check_and_update_positions_VTM(asset_name)
        except Exception as e:
            logger.error(f"Error checking positions: {e}", exc_info=True)

    @handle_errors(
        component="binance_handler",
        severity=ErrorSeverity.WARNING,
        notify=True,
        reraise=False,
        default_return=False,
    )
    def sync_positions_with_binance(
        self, asset_name: str = "BTC", symbol: str = None
    ) -> bool:
        """
        ✅ REFACTORED: Syncs portfolio with Binance using full reconciliation.
        Handles multiple simultaneous positions (e.g., hedge mode).
        """
        if symbol is None:
            symbol = self.symbol

        if self.mode == "paper":
            logger.info("[SYNC] Sync disabled in paper mode.")
            return True

        if not self.futures_handler:
            logger.error("[SYNC] Futures handler not available, cannot sync.")
            return False

        logger.info(f"[SYNC] Starting full position reconciliation for {asset_name}...")

        try:
            # 1. Get current state from Portfolio and Binance
            portfolio_positions = self.portfolio_manager.get_asset_positions(asset_name)
            portfolio_map = {p.side: p for p in portfolio_positions}

            binance_positions = self.futures_handler.get_all_positions_info()
            binance_map = {p["side"]: p for p in binance_positions}

            logger.info(
                f"[SYNC] State Found: Portfolio({list(portfolio_map.keys())}) vs Binance({list(binance_map.keys())})"
            )

            current_price = self.get_current_price(symbol, force_live=True)
            if not current_price:
                logger.error("[SYNC] Could not fetch current price. Aborting sync.")
                return False

            # 2. Reconcile: Binance -> Portfolio (Import new positions)
            for side, binance_pos in binance_map.items():
                if side not in portfolio_map:
                    logger.warning(
                        f"[SYNC] ⚠️ Found position on Binance not in portfolio: {side.upper()}. Importing..."
                    )

                    pos_amt = abs(float(binance_pos.get("positionAmt", 0)))
                    entry_price = float(binance_pos.get("entryPrice", current_price))
                    position_size_usd = pos_amt * entry_price

                    # Fetch OHLC data for VTM
                    ohlc_data = None
                    try:
                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=10)
                        df = self.data_manager.fetch_binance_data(
                            symbol=symbol,
                            interval=self.config["assets"][asset_name].get(
                                "interval", "1h"
                            ),
                            start_date=start_time.strftime("%Y-%m-%d"),
                            end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        if len(df) > 50:
                            ohlc_data = {
                                "high": df["high"].values,
                                "low": df["low"].values,
                                "close": df["close"].values,
                                "volume": df["volume"].values,
                            }
                    except Exception as e:
                        logger.error(f"[VTM] Failed to fetch OHLC for import: {e}")

                    # Add to portfolio
                    success = self.portfolio_manager.add_position(
                        asset=asset_name,
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        position_size_usd=position_size_usd,
                        stop_loss=None,
                        take_profit=None,
                        trailing_stop_pct=None,
                        binance_order_id=None,  # No order ID for imported position
                        ohlc_data=ohlc_data,
                        use_dynamic_management=True,
                        signal_details={"source": "sync_import"},
                        is_futures=True,
                    )
                    if success:
                        logger.info(
                            f"[SYNC] ✅ Successfully imported {side.upper()} position."
                        )
                    else:
                        logger.error(
                            f"[SYNC] ❌ Failed to import {side.upper()} position."
                        )

            # 3. Reconcile: Portfolio -> Binance (Close desynced positions)
            for side, portfolio_pos in portfolio_map.items():
                if side not in binance_map:
                    logger.warning(
                        f"[SYNC] ⚠️ Found position in portfolio not on Binance: {side.upper()} "
                        f"(ID: {portfolio_pos.position_id}). Closing locally."
                    )
                    self.portfolio_manager.close_position(
                        position_id=portfolio_pos.position_id,
                        exit_price=current_price,
                        reason="sync_desync_from_exchange",
                    )

            logger.info("[SYNC] Reconciliation complete.")
            return True

        except Exception as e:
            logger.error(
                f"[SYNC] Critical error during reconciliation: {e}", exc_info=True
            )
            return False

    def _sync_futures_positions(self, asset_name: str, symbol: str) -> bool:
        """Sync Futures positions (supports LONG + SHORT)"""
        try:
            portfolio_positions = self.portfolio_manager.get_asset_positions(asset_name)
            futures_positions = (
                self.futures_handler.client.futures_position_information(symbol=symbol)
            )

            active_futures = []
            for pos in futures_positions:
                pos_amt = float(pos.get("positionAmt", 0))
                if pos_amt != 0:
                    side = "long" if pos_amt > 0 else "short"
                    active_futures.append(
                        {
                            "side": side,
                            "quantity": abs(pos_amt),
                            "entry_price": float(pos.get("entryPrice", 0)),
                        }
                    )

            logger.info(
                f"[SYNC] Portfolio: {len(portfolio_positions)} | Futures: {len(active_futures)}"
            )

            if active_futures and not portfolio_positions:
                import_enabled = self.config.get("portfolio", {}).get(
                    "import_existing_positions", False
                )
                if not import_enabled:
                    return True

                for fut_pos in active_futures:
                    success = self.portfolio_manager.add_position(
                        asset=asset_name,
                        symbol=symbol,
                        side=fut_pos["side"],
                        entry_price=fut_pos["entry_price"],
                        position_size_usd=fut_pos["quantity"] * fut_pos["entry_price"],
                        stop_loss=None,
                        take_profit=None,
                        trailing_stop_pct=None,
                        binance_order_id=None,
                        ohlc_data=None,
                        use_dynamic_management=True,
                        entry_time=datetime.now(),
                        signal_details={"imported": True},
                    )
                return True

            return True

        except Exception as e:
            logger.error(f"[SYNC] Futures sync error: {e}", exc_info=True)
            return False

    def _sync_spot_positions(self, asset_name: str, symbol: str) -> bool:
        """Sync Spot positions"""
        return True

    def _verify_vtm_status_after_sync(self, asset: str):
        """Verify VTM is working after position sync"""
        try:
            positions = self.portfolio_manager.get_asset_positions(asset)
            if not positions:
                return

            for pos in positions:
                if pos.trade_manager:
                    status = pos.get_vtm_status()
                    logger.info(f"  ✓ {pos.position_id}: VTM ACTIVE")
                else:
                    logger.warning(f"  ⚠️ {pos.position_id}: VTM NOT ACTIVE")

        except Exception as e:
            logger.error(f"[VTM VERIFICATION] Error: {e}")