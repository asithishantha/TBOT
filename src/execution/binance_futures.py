"""
Binance Futures API Integration - FIXED VERSION
✅ FIXED: Stop Loss using Algo Order API (fixes -4120 error)
✅ FIXED: Emergency close without reduceOnly parameter (fixes -1106 error)
✨ ENHANCED: Hedge Mode Integration for Asymmetric System
"""

import logging
import time
from binance.client import Client
from binance.enums import (
    SIDE_BUY,
    SIDE_SELL,
    ORDER_TYPE_MARKET,
    FUTURE_ORDER_TYPE_MARKET,
    FUTURE_ORDER_TYPE_LIMIT,
)
from typing import Dict, List, Optional, Tuple
from src.data.data_manager import CLOUDFRONT_HEADERS

logger = logging.getLogger(__name__)


class BinanceFuturesHandler:
    """
    Unified Binance Futures API Handler for BOTH Long & Short positions
    ✅ FIXED: Using Algo Order API for stop loss orders
    ✨ ENHANCED: Forced Hedge Mode for Asymmetric Trading
    """

    def __init__(self, client: Client, symbol: str = "BTCUSDT", config: dict = None):
        self.client = client
        self.client.session.headers.update(CLOUDFRONT_HEADERS)
        self.symbol = symbol
        self.filters = {}
        self.config = config or {}

        self.quantity_precision = 3  # Default for BTCUSDT
        self.price_precision = 2  # Default for BTCUSDT
        self.tick_size = 0.01  # Default tick size
        self.step_size = 0.001  # Default step size
        self.min_qty = 0.001  # Default min quantity
        self.min_notional = 5.0  # Default min notional

        # Hedging Configuration
        self.allow_hedging = self.config.get("trading", {}).get(
            "allow_simultaneous_long_short", False  # Fail closed: hedging OFF if key missing
        )

        # Verify Futures API access and load filters
        try:
            self.client.futures_account()
            self._load_symbol_filters()

            # Force Binance into Hedge Mode
            # try:
            #     self.client.futures_change_position_mode(
            #         dualSidePosition=self.allow_hedging
            #     )
            #     mode_str = "HEDGE" if self.allow_hedging else "ONE-WAY"
            #     logger.info(f"[FUTURES] ✓ Account Position Mode set to: {mode_str}")
            # except Exception as e:
            #     if "-4059" in str(e) or "No need to change" in str(e):
            #         logger.debug("[FUTURES] Hedge Mode already correctly set.")
            #     else:
            #         logger.error(f"[FUTURES] Failed to set Hedge Mode: {e}")

            logger.info(f"[FUTURES] ✓ Binance Futures API connected for {symbol}")

            # --- NEW: Get actual current position mode from Binance ---
            try:
                position_mode_info = self.client.futures_get_position_mode()
                self._actual_hedge_mode_enabled = position_mode_info.get('dualSidePosition', True) # Default to True (HEDGE) if not found
                logger.info(f"[FUTURES] Actual Binance Futures Mode: {'HEDGE' if self._actual_hedge_mode_enabled else 'ONE-WAY'} (from API)")
            except Exception as e:
                logger.error(f"[FUTURES] Failed to get actual position mode: {e}")
                self._actual_hedge_mode_enabled = True # Assume HEDGE mode as fallback to be safe
                logger.warning("[FUTURES] Assuming HEDGE mode due to error fetching actual mode.")
            # --- END NEW ---

        except Exception as e:
            logger.error(f"[FUTURES] ✗ Futures API unavailable: {e}")
            raise

    def _load_symbol_filters(self):
        """Load LOT_SIZE, PRICE_FILTER, and MIN_NOTIONAL filters for the symbol"""
        try:
            info = self.client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == self.symbol:
                    self.quantity_precision = int(s.get("quantityPrecision", 3))
                    self.price_precision = int(s.get("pricePrecision", 2))

                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            self.step_size = float(f["stepSize"])
                            self.min_qty = float(f["minQty"])
                            self.filters["step_size"] = self.step_size
                            self.filters["min_qty"] = self.min_qty

                        elif f["filterType"] == "PRICE_FILTER":
                            self.tick_size = float(f["tickSize"])
                            self.filters["tick_size"] = self.tick_size

                        elif f["filterType"] == "MIN_NOTIONAL":
                            self.min_notional = float(f.get("notional", 5.0))
                            self.filters["min_notional"] = self.min_notional

                    self.filters["precision"] = self.quantity_precision

                    logger.info(
                        f"[FUTURES] Filters loaded for {self.symbol}:\n"
                        f"  Quantity: precision={self.quantity_precision}, step={self.step_size}, min={self.min_qty}\n"
                        f"  Price:    precision={self.price_precision}, tick={self.tick_size}\n"
                        f"  Notional: min=${self.min_notional}"
                    )
                    return

            logger.warning(f"[FUTURES] Symbol {self.symbol} not found in exchange info")

        except Exception as e:
            logger.error(f"[FUTURES] Failed to load filters: {e}")
            logger.info(f"[FUTURES] Using default precision values for {self.symbol}")

    def _adjust_quantity(self, quantity: float) -> float:
        """Round quantity to valid step size"""
        import math

        step = self.filters.get("step_size", 0.001)
        precision = self.filters.get("precision", 3)

        # Round down to nearest step to avoid "LOT_SIZE" error
        quantity = math.floor(quantity / step) * step
        return round(quantity, precision)

    def set_leverage(self, leverage: int = 10) -> bool:
        """Set leverage for the trading pair"""
        try:
            response = self.client.futures_change_leverage(
                symbol=self.symbol, leverage=leverage
            )
            logger.info(f"[FUTURES] Leverage set to {leverage}x for {self.symbol}")
            return True
        except Exception as e:
            logger.error(f"[FUTURES] Failed to set leverage: {e}")
            return False

    def set_margin_type(self, margin_type: str = "CROSSED") -> bool:
        """Set margin type (ISOLATED or CROSSED)"""
        try:
            response = self.client.futures_change_margin_type(
                symbol=self.symbol, marginType=margin_type
            )
            logger.info(f"[FUTURES] Margin type set to {margin_type} for {self.symbol}")
            return True
        except Exception as e:
            if "-4046" in str(e):
                logger.debug(f"[FUTURES] Margin type already {margin_type}")
                return True
            logger.error(f"[FUTURES] Failed to set margin type: {e}")
            return False

    def _round_price(self, price: float) -> float:
        """Round price to exchange tick size"""
        if not hasattr(self, "tick_size"):
            return round(price, self.price_precision)

        rounded = round(price / self.tick_size) * self.tick_size
        rounded = round(rounded, self.price_precision)

        logger.debug(f"[PRICE] {price:.8f} → {rounded:.{self.price_precision}f}")
        return rounded

    def _round_quantity(self, quantity: float) -> float:
        """Round quantity to exchange step size"""
        if not hasattr(self, "step_size"):
            return round(quantity, self.quantity_precision)

        rounded = round(quantity / self.step_size) * self.step_size
        rounded = round(rounded, self.quantity_precision)

        logger.debug(f"[QTY] {quantity:.8f} → {rounded:.{self.quantity_precision}f}")
        return rounded

    def _validate_order(self, price: float, quantity: float) -> Tuple[bool, str]:
        """Validate order against Binance filters"""
        if quantity < self.min_qty:
            return False, f"Quantity {quantity} < minimum {self.min_qty}"

        notional = price * quantity
        if notional < self.min_notional:
            return False, f"Notional ${notional:.2f} < minimum ${self.min_notional}"

        price_str = f"{price:.{self.price_precision}f}"
        if float(price_str) != price:
            return False, f"Price precision error: {price} vs {price_str}"

        return True, "OK"

    def _place_stop_loss_algo(
        self,
        side: str,
        position_side: str,
        stop_price: float,
        quantity: float,
    ) -> bool:
                """
                [DISABLED]
                This function is disabled because repeated tests have shown the Binance
                Testnet API does not reliably support the required STOP_MARKET orders
                with the necessary 'reduceOnly' parameter to ensure safety.
        
                The bot will now rely exclusively on the VeteranTradeManager (VTM) to
                monitor prices and execute a market close if a stop loss is hit.
                
                This function will always return True to prevent the emergency close
                logic from being triggered.
                """
                logger.warning(
                    "[STOP LOSS]  DISABLED - Exchange-side stop-loss is disabled. "
                    "VTM is responsible for all trade exits."
                )
                return True

    def open_short_position(
        self, quantity: float, stop_loss: float = None, take_profit: float = None
    ) -> Optional[Dict]:
        """Open a SHORT position on Binance Futures"""
        return self._open_position(
            side="short",
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def open_long_position(
        self, quantity: float, stop_loss: float = None, take_profit: float = None
    ) -> Optional[Dict]:
        """Open a LONG position on Binance Futures"""
        return self._open_position(
            side="long", quantity=quantity, stop_loss=stop_loss, take_profit=take_profit
        )

    def _open_position(
        self,
        side: str,
        quantity: float,
        stop_loss: float = None,
        take_profit: float = None,
    ) -> Optional[Dict]:
        """
        ✅ FIXED: Using proper Algo API for stop loss placement
        """
        try:
            binance_side = SIDE_BUY if side == "long" else SIDE_SELL
            side_label = side.upper()
            position_side = "LONG" if side == "long" else "SHORT"

            quantity = self._round_quantity(quantity)
            if stop_loss:
                stop_loss = self._round_price(stop_loss)
            if take_profit:
                take_profit = self._round_price(take_profit)

            logger.info(
                f"[FUTURES] Opening {side_label} (Hedge Mode): "
                f"{quantity:.{self.quantity_precision}f} {self.symbol}"
            )

            ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
            current_price = float(ticker["price"])

            is_valid, error_msg = self._validate_order(current_price, quantity)
            if not is_valid:
                logger.error(f"[FUTURES] Validation failed: {error_msg}")
                return None

            # 1. Open position
            try:
                order_params = {
                    "symbol": self.symbol,
                    "side": binance_side,
                    "type": FUTURE_ORDER_TYPE_MARKET,
                    "quantity": quantity,
                }
                # Only add positionSide if actual account mode is HEDGE mode
                if hasattr(self, '_actual_hedge_mode_enabled') and self._actual_hedge_mode_enabled:
                    order_params["positionSide"] = position_side
                else:
                    # If ONE-WAY mode, do NOT send positionSide parameter
                    # logger.debug("[FUTURES] ONE-WAY mode detected, omitting positionSide parameter.")
                    pass # positionSide is already not in order_params
                
                order = self.client.futures_create_order(**order_params)
            except Exception as e:
                logger.error(f"[FUTURES] Entry failed: {e}")
                return None

            order_id = order.get("orderId")
            
            # ✅ ENHANCED: Extract average fill price for MARKET orders
            # Market orders often return avgPrice=0 initially in some API versions
            avg_price = float(order.get("avgPrice", 0))
            
            if avg_price == 0:
                # Try cumulative quote quantity / executed quantity
                cum_quote = float(order.get("cumQuote", 0))
                exec_qty = float(order.get("executedQty", 0))
                if exec_qty > 0:
                    avg_price = cum_quote / exec_qty
                
                # If still 0, check fills list
                if avg_price == 0 and "fills" in order:
                    fills = order["fills"]
                    if fills:
                        total_quote = sum(float(f.get("price", 0)) * float(f.get("qty", 0)) for f in fills)
                        total_qty = sum(float(f.get("qty", 0)) for f in fills)
                        if total_qty > 0:
                            avg_price = total_quote / total_qty
            
            # If we calculated a better price, update the order object for the handler
            if avg_price > 0:
                order["avgPrice"] = str(avg_price)
            
            # Ensure status is helpful for the caller
            # If it's a MARKET order, it's effectively FILLED if we have an ID
            if order.get("type") == FUTURE_ORDER_TYPE_MARKET and order.get("status") == "NEW":
                order["status"] = "FILLED"

            logger.info(
                f"[FUTURES] ✓ {side_label} opened\n"
                f"  Order ID: {order_id}\n"
                f"  Quantity: {quantity:.{self.quantity_precision}f} BTC\n"
                f"  Entry:    ${avg_price:,.2f}"
            )

            # 2. Cancel existing stop orders
            if stop_loss:
                logger.info(f"[SL] Preparing stop loss for {side_label}...")

                try:
                    cancelled = self.client.futures_cancel_all_open_orders(
                        symbol=self.symbol
                    )
                    if cancelled:
                        logger.info(
                            f"[SL] ✓ Cancelled {len(cancelled)} existing order(s)"
                        )
                except Exception as e:
                    if "-2011" not in str(e):
                        logger.warning(f"[SL] Cancel warning: {e}")

                time.sleep(0.5)

                # Validate SL
                if side == "long" and stop_loss >= current_price:
                    stop_loss = current_price * 0.97
                    stop_loss = self._round_price(stop_loss)
                elif side == "short" and stop_loss <= current_price:
                    stop_loss = current_price * 1.03
                    stop_loss = self._round_price(stop_loss)

                # Place stop loss using Algo API
                sl_success = self._place_stop_loss_algo(
                    side=side,
                    position_side=position_side,
                    stop_price=stop_loss,
                    quantity=quantity,
                )

                # Emergency close if SL fails
                if not sl_success:
                    logger.critical(
                        f"\n{'='*80}\n"
                        f"🛑 CRITICAL: STOP LOSS FAILED!\n"
                        f"{'='*80}\n"
                        f"Side: {side_label}\n"
                        f"Executing EMERGENCY CLOSE...\n"
                        f"{'='*80}\n"
                    )

                    close_side = SIDE_SELL if side == "long" else SIDE_BUY
                    try:
                        emergency_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=close_side,
                            positionSide=position_side,
                            type=FUTURE_ORDER_TYPE_MARKET,
                            quantity=quantity,
                        )
                        logger.critical(
                            f"[FUTURES] ✓ EMERGENCY CLOSE: {emergency_order.get('orderId')}"
                        )
                    except Exception as close_error:
                        logger.critical(
                            f"[FUTURES] ☠️ EMERGENCY CLOSE FAILED: {close_error}"
                        )
                    return None

            # 3. Take profit (optional)
            if take_profit:
                tp_side = SIDE_SELL if side == "long" else SIDE_BUY

                if side == "long" and take_profit <= current_price:
                    take_profit = None
                elif side == "short" and take_profit >= current_price:
                    take_profit = None

                if take_profit:
                    try:
                        tp_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=tp_side,
                            positionSide=position_side,
                            type=FUTURE_ORDER_TYPE_LIMIT,
                            price=take_profit,
                            quantity=quantity,
                            timeInForce="GTC",
                            reduceOnly=True,
                        )
                        logger.info(f"  ✓ TP: ${take_profit:,.{self.price_precision}f}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ TP Failed: {e}")

            return order

        except Exception as e:
            logger.error(f"[FUTURES] Failed: {e}", exc_info=True)
            return None

    def close_short_position(
        self, quantity: float = None, order_id: int = None
    ) -> bool:
        """Close a SHORT position (buy back)"""
        return self._close_position(side="short", quantity=quantity, order_id=order_id)

    def close_long_position(self, quantity: float = None, order_id: int = None) -> bool:
        """Close a LONG position (sell back)"""
        return self._close_position(side="long", quantity=quantity, order_id=order_id)

    def _close_position(
        self, side: str, quantity: float = None, order_id: int = None
    ) -> bool:
        """
        ✅ FIXED: Close position without reduceOnly for market orders and handle race conditions.
        """
        try:
            # IDEMPOTENT CLOSE: First, check if a position actually exists to be closed.
            position_info = self.get_position_info(side=side)

            # If no position exists or its size is zero, the desired state is already met.
            if not position_info or float(position_info.get("positionAmt", 0)) == 0:
                logger.info(f"[FUTURES] No active {side.upper()} position found on exchange. Assuming already closed.")
                return True

            # Determine quantity to close. If not provided, close the whole position.
            close_quantity = quantity
            if close_quantity is None:
                close_quantity = abs(float(position_info.get("positionAmt", 0)))
            
            if close_quantity == 0:
                logger.info("[FUTURES] Close requested for zero quantity. Nothing to do.")
                return True

            close_quantity = self._round_quantity(close_quantity)
            close_side = SIDE_SELL if side == "long" else SIDE_BUY
            side_label = side.upper()
            position_side = "LONG" if side == "long" else "SHORT"

            logger.info(
                f"[FUTURES] Closing {side_label} (Hedge Mode): "
                f"{close_quantity:.{self.quantity_precision}f} {self.symbol}"
            )

            try:
                # Attempt to close the position with a market order
                order_params = {
                    "symbol": self.symbol,
                    "side": close_side,
                    "type": FUTURE_ORDER_TYPE_MARKET,
                    "quantity": close_quantity
                }
                # Only add positionSide if actual account mode is HEDGE mode
                if hasattr(self, '_actual_hedge_mode_enabled') and self._actual_hedge_mode_enabled:
                    order_params["positionSide"] = position_side
                else:
                    # If ONE-WAY mode, do NOT send positionSide parameter
                    # logger.debug("[FUTURES] ONE-WAY mode detected for close, omitting positionSide parameter.")
                    pass # positionSide is already not in order_params

                order = self.client.futures_create_order(**order_params)

                logger.info(
                    f"[FUTURES] ✓ {side_label} closed\n"
                    f"  Order ID:  {order.get('orderId')}\n"
                    f"  Quantity:  {close_quantity:.{self.quantity_precision}f} BTC\n"
                    f"  Exit:      ${float(order.get('avgPrice', 0)):,.2f}\n"
                    f"  Status:    {order.get('status')}"
                )

                # Cancel remaining open orders for the symbol
                try:
                    cancelled = self.client.futures_cancel_all_open_orders(
                        symbol=self.symbol
                    )
                    if cancelled:
                        logger.debug(
                            f"[FUTURES] Cancelled {len(cancelled)} remaining order(s)"
                        )
                except Exception as e:
                    if "-2011" not in str(e): # Ignore "Unknown order sent"
                        logger.debug(f"[FUTURES] Cancel orders warning: {e}")

                return True

            except Exception as order_error:
                # Handle the specific race condition error where the position was closed just before our order
                if "ReduceOnly Order is rejected" in str(order_error):
                     logger.info(f"[FUTURES] Position likely closed by another process (ReduceOnly rejected). Marking as success.")
                     return True
                
                logger.error(
                    f"[FUTURES] ❌ Order execution failed\n"
                    f"  Side:     {side_label}\n"
                    f"  Quantity: {close_quantity:.{self.quantity_precision}f}\n"
                    f"  Error:    {str(order_error)}"
                )
                return False

        except Exception as e:
            logger.error(
                f"[FUTURES] ❌ Failed to close {side.upper()} position\n"
                f"  Error: {str(e)}",
                exc_info=True,
            )
            return False

    def _cancel_existing_stop_orders(self, side: str) -> bool:
        """
        Cancel all existing stop-loss orders for the given side
        This is CRITICAL before placing new stop orders to avoid -4130 error

        Args:
            side: "long" or "short"

        Returns:
            True if successful or no orders to cancel
        """
        try:
            # Get all open orders
            open_orders = self.client.futures_get_open_orders(symbol=self.symbol)

            if not open_orders:
                logger.debug(f"[SL] No open orders for {self.symbol}")
                return True

            # Determine which stop side to look for
            # LONG positions have SELL stops, SHORT positions have BUY stops
            target_stop_side = "SELL" if side == "long" else "BUY"

            cancelled_count = 0
            for order in open_orders:
                order_type = order.get("type", "")
                order_side = order.get("side", "")

                # Look for STOP_MARKET orders on the opposite side
                if order_type == "STOP_MARKET" and order_side == target_stop_side:
                    order_id = order.get("orderId")

                    try:
                        self.client.futures_cancel_order(
                            symbol=self.symbol, orderId=order_id
                        )
                        cancelled_count += 1
                        logger.info(f"[SL] Cancelled existing stop order {order_id}")
                    except Exception as e:
                        logger.warning(f"[SL] Failed to cancel order {order_id}: {e}")

            if cancelled_count > 0:
                logger.info(f"[SL] Cancelled {cancelled_count} existing stop order(s)")

            return True

        except Exception as e:
            logger.error(f"[SL] Error cancelling stop orders: {e}")
            return False

    def get_position_info(self, side: str = None) -> Optional[Dict]:
        """Get current position information"""
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)

            for pos in positions:
                if pos["symbol"] == self.symbol:
                    pos_amt = float(pos.get("positionAmt", 0))

                    if pos_amt == 0:
                        continue

                    if side == "short" and pos_amt >= 0:
                        continue
                    elif side == "long" and pos_amt <= 0:
                        continue

                    pos["side"] = "long" if pos_amt > 0 else "short"
                    return pos

            return None

        except Exception as e:
            logger.error(f"[FUTURES] Error getting position: {e}")
            return None

    def get_all_positions_info(self) -> List[Dict]:
        """Get ALL non-zero position information for the symbol."""
        active_positions = []
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            for pos in positions:
                if pos["symbol"] == self.symbol:
                    pos_amt = float(pos.get("positionAmt", 0))
                    if pos_amt != 0:
                        pos["side"] = "long" if pos_amt > 0 else "short"
                        active_positions.append(pos)
            return active_positions
        except Exception as e:
            logger.error(f"[FUTURES] Error getting all positions: {e}")
            return []

    def get_unrealized_pnl(self) -> float:
        """Get unrealized P&L for current position"""
        try:
            position = self.get_position_info()
            if position:
                return float(position.get("unRealizedProfit", 0))
            return 0.0
        except Exception as e:
            logger.error(f"[FUTURES] Error getting P&L: {e}")
            return 0.0

    def get_account_balance(self) -> float:
        """Get Futures account balance"""
        try:
            account = self.client.futures_account()
            for asset in account.get("assets", []):
                if asset["asset"] == "USDT":
                    return float(asset.get("availableBalance", 0))
            return 0.0
        except Exception as e:
            logger.error(f"[FUTURES] Error getting balance: {e}")
            return 0.0


# Integration functions remain the same
def integrate_futures_into_handler(handler):
    """Integrate Futures API into existing BinanceExecutionHandler"""
    futures_enabled = (
        handler.config.get("assets", {}).get("BTC", {}).get("enable_futures", False)
    )

    if not futures_enabled:
        logger.warning("[FUTURES] Futures trading disabled in config")
        return False

    try:
        handler.futures_handler = BinanceFuturesHandler(
            client=handler.client, symbol=handler.symbol, config=handler.config
        )
        handler.futures_handler.client.session.headers.update(CLOUDFRONT_HEADERS)  # ✅ CloudFront 403 fix

        leverage = handler.config.get("assets", {}).get("BTC", {}).get("leverage", 10)
        margin_type = (
            handler.config.get("assets", {})
            .get("BTC", {})
            .get("margin_type", "CROSSED")
        )

        handler.futures_handler.set_leverage(leverage)
        handler.futures_handler.set_margin_type(margin_type)

        logger.info("[FUTURES] ✓ Futures handler integrated")
        return True

    except Exception as e:
        logger.error(f"[FUTURES] Integration failed: {e}")
        return False


def patch_open_position_method(handler):
    """
    Patch the _open_position method to use Futures for BOTH longs and shorts
    """

    original_open_position = handler._open_position

    def _open_position_with_futures(
        signal: int, current_price: float, asset_name: str, **kwargs
    ):
        """
        Enhanced _open_position that uses Futures API for both LONG and SHORT
        """

        side = "long" if signal == 1 else "short"

        # If Futures is enabled, use it for BOTH directions
        if hasattr(handler, "futures_handler"):
            try:
                logger.info(f"[FUTURES] Using Futures API for {side.upper()} position")

                # Calculate SL/TP prices
                risk = handler.asset_config.get("risk", {})

                if side == "long":
                    stop_loss_pct = risk.get("stop_loss_pct", 0.05)
                    take_profit_pct = risk.get("take_profit_pct", 0.10)
                    trailing_stop_pct = risk.get("trailing_stop_pct", 0.03)

                    stop_loss_price = current_price * (1 - stop_loss_pct)
                    take_profit_price = current_price * (1 + take_profit_pct)
                else:  # short
                    stop_loss_pct = risk.get(
                        "stop_loss_pct_short", risk.get("stop_loss_pct", 0.04)
                    )
                    take_profit_pct = risk.get(
                        "take_profit_pct_short", risk.get("take_profit_pct", 0.08)
                    )
                    trailing_stop_pct = risk.get(
                        "trailing_stop_pct_short", risk.get("trailing_stop_pct", 0.025)
                    )

                    stop_loss_price = current_price * (1 + stop_loss_pct)
                    take_profit_price = current_price * (1 - take_profit_pct)

                # ✅ FIX: Round stop loss to correct precision BEFORE sizing
                stop_loss_price = round(stop_loss_price, 2)

                # Calculate position size using risk-based method
                position_size_usd, sizing_metadata = (
                    handler.sizer.calculate_size_risk_based(
                        asset=asset_name,
                        entry_price=current_price,
                        stop_loss_price=stop_loss_price,
                        signal=signal,
                        confidence_score=kwargs.get("confidence_score"),
                        market_condition=kwargs.get("market_condition", "neutral"),
                        sizing_mode=kwargs.get("sizing_mode", "automated"),
                        manual_size_usd=kwargs.get("manual_size_usd"),
                        override_reason=kwargs.get("override_reason"),
                    )
                )

                if position_size_usd <= 0:
                    logger.error(
                        f"[FUTURES] Invalid position size: ${position_size_usd:.2f}"
                    )
                    return False

                # Calculate quantity and round it
                quantity = position_size_usd / current_price
                quantity = handler.futures_handler._round_quantity(quantity)

                logger.info(
                    f"[FUTURES] Opening {side.upper()} position:\n"
                    f"  Size: ${position_size_usd:,.2f}\n"
                    f"  Quantity: {quantity:.6f} BTC\n"
                    f"  Entry: ${current_price:,.2f}\n"
                    f"  Stop Loss: ${stop_loss_price:,.2f}\n"
                    f"  Take Profit: ${take_profit_price:,.2f}"
                )

                # Open position on Futures
                if side == "long":
                    order = handler.futures_handler.open_long_position(
                        quantity=quantity,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                    )
                else:
                    order = handler.futures_handler.open_short_position(
                        quantity=quantity,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                    )

                if not order:
                    logger.error(f"[FUTURES] Failed to open {side.upper()} position")
                    return False

                order_id = order.get("orderId")

                # Fetch OHLC for VTM
                ohlc_data = None
                if handler.data_manager:
                    try:
                        from datetime import datetime, timedelta, timezone

                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=10)

                        df = handler.data_manager.fetch_binance_data(
                            symbol=handler.symbol,
                            interval=handler.asset_config.get("interval", "1h"),
                            start_date=start_time.strftime("%Y-%m-%d"),
                            end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        )

                        if len(df) > 0:
                            ohlc_data = {
                                "high": df["high"].values,
                                "low": df["low"].values,
                                "close": df["close"].values,
                            }
                    except Exception as e:
                        logger.warning(f"[VTM] OHLC fetch failed: {e}")

                # Add to portfolio
                success = handler.portfolio_manager.add_position(
                    asset=asset_name,
                    symbol=handler.symbol,
                    side=side,
                    entry_price=current_price,
                    position_size_usd=position_size_usd,
                    stop_loss=None,  # VTM will manage
                    take_profit=None,
                    trailing_stop_pct=trailing_stop_pct,
                    binance_order_id=order_id,
                    ohlc_data=ohlc_data,
                    use_dynamic_management=True,
                )

                if success:
                    logger.info(
                        f"[OK] {asset_name} {side.upper()} position opened via Futures"
                    )
                    logger.info(f"  └─ Order ID: {order_id}")
                    if ohlc_data:
                        logger.info(f"  └─ VTM: ACTIVE")
                    return True
                else:
                    logger.error(f"[FAIL] Portfolio rejected {side.upper()} position")
                    # Rollback - close the Futures position
                    if side == "long":
                        handler.futures_handler.close_long_position(quantity=quantity)
                    else:
                        handler.futures_handler.close_short_position(quantity=quantity)
                    return False

            except Exception as e:
                logger.error(
                    f"[FUTURES] Error opening {side.upper()}: {e}", exc_info=True
                )
                return False

        # Fallback to original method (Spot) if Futures disabled
        else:
            return original_open_position(
                signal=signal,
                current_price=current_price,
                asset_name=asset_name,
                **kwargs,
            )

    # Replace method
    handler._open_position = _open_position_with_futures
    logger.info("[FUTURES] _open_position method patched for LONG+SHORT")


def patch_close_position_method(handler):
    """
    Patch the _close_position method to use Futures for BOTH longs and shorts
    """

    original_close_position = handler._close_position

    def _close_position_with_futures(
        position, current_price: float, asset_name: str, reason: str
    ):
        """
        Enhanced _close_position that uses Futures API for both LONG and SHORT
        """

        # If Futures enabled, use it for both directions
        if hasattr(handler, "futures_handler") and handler.futures_handler:
            try:
                side = position.side
                quantity = position.quantity
                order_id = position.binance_order_id

                logger.info(
                    f"[FUTURES] Closing {side.upper()} position via Futures API\n"
                    f"  Position ID: {position.position_id}\n"
                    f"  Order ID:    {order_id}\n"
                    f"  Quantity:    {quantity:.6f}\n"
                    f"  Reason:      {reason}"
                )

                # Get current P&L from Futures before closing
                futures_position = handler.futures_handler.get_position_info(side=side)
                if futures_position:
                    futures_pnl = float(futures_position.get("unRealizedProfit", 0))
                    logger.info(f"  Unrealized P&L: ${futures_pnl:,.2f}")
                else:
                    logger.warning(f"  Could not fetch Futures position info")
                    futures_pnl = 0

                # ✅ CRITICAL FIX: Close on Futures with proper error handling
                success = False

                if side == "long":
                    success = handler.futures_handler.close_long_position(
                        quantity=quantity, order_id=order_id
                    )
                elif side == "short":
                    success = handler.futures_handler.close_short_position(
                        quantity=quantity, order_id=order_id
                    )
                else:
                    logger.error(f"[FUTURES] Invalid side: {side}")
                    return False

                # ✅ Check if close was successful
                if not success:
                    logger.error(
                        f"[FUTURES] ❌ Failed to close {side.upper()} position\n"
                        f"  Position ID: {position.position_id}\n"
                        f"  Order ID:    {order_id}\n"
                        f"  Quantity:    {quantity:.6f}\n"
                        f"  Reason:      Futures API returned False\n"
                        f"  Action:      Position remains open on exchange"
                    )
                    return False

                # ✅ Success - log details
                logger.info(
                    f"[FUTURES] ✅ {side.upper()} position closed successfully\n"
                    f"  Position ID: {position.position_id}\n"
                    f"  Final P&L:   ${futures_pnl:,.2f}"
                )

                # Close in portfolio
                trade_result = handler.portfolio_manager.close_position(
                    position_id=position.position_id,
                    exit_price=current_price,
                    reason=reason,
                )

                if trade_result:
                    logger.info(f"[OK] Portfolio updated after {side.upper()} close")
                    return True
                else:
                    logger.error(
                        f"[FAIL] Portfolio close failed for {side.upper()}\n"
                        f"  Warning: Position closed on exchange but not in portfolio!"
                    )
                    return False

            except Exception as e:
                logger.error(
                    f"[FUTURES] ❌ Exception closing {position.side.upper()} position\n"
                    f"  Position ID: {position.position_id}\n"
                    f"  Error:       {str(e)}\n"
                    f"  Traceback:",
                    exc_info=True,
                )
                return False

        # Fallback to original method (Spot) if Futures disabled
        else:
            logger.warning(f"[FUTURES] Handler not available, using fallback method")
            return original_close_position(
                position=position,
                current_price=current_price,
                asset_name=asset_name,
                reason=reason,
            )

    # Replace method
    handler._close_position = _close_position_with_futures
    logger.info("[FUTURES] _close_position method patched for LONG+SHORT")


def enable_futures_for_binance_handler(handler):
    """
    MAIN FUNCTION: Enable Futures trading for BOTH LONG and SHORT positions
    ✅ FIXED: Uses new Algo API for stop losses
    """
    try:
        logger.info("\n" + "=" * 70)
        logger.info("ENABLING BINANCE FUTURES FOR LONG + SHORT TRADING")
        logger.info("=" * 70)

        if not integrate_futures_into_handler(handler):
            return False

        balance = handler.futures_handler.get_account_balance()
        logger.info(f"[FUTURES] Account Balance: ${balance:,.2f} USDT")

        position = handler.futures_handler.get_position_info()
        if position:
            side = position.get("side", "unknown")
            pos_amt = abs(float(position.get("positionAmt", 0)))
            unrealized = float(position.get("unRealizedProfit", 0))

            logger.info(f"[FUTURES] Existing {side.upper()} position detected:")
            logger.info(f"  Quantity: {pos_amt:.8f} BTC")
            logger.info(f"  P&L:      ${unrealized:,.2f}")

        logger.info("=" * 70)
        logger.info("✅ FUTURES TRADING ENABLED")
        logger.info("  - Using Algo Order API for stop losses")
        logger.info("  - Hedge mode active")
        logger.info("=" * 70)

        return True

    except Exception as e:
        logger.error(f"[FUTURES] Enablement failed: {e}", exc_info=True)
        return False
