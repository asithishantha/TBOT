"""
Portfolio Manager - Enhanced with MT5 real-time profit tracking
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pickle
from pathlib import Path
from src.execution.veteran_trade_manager import VeteranTradeManager, ExitReason
from src.utils.trade_logger import log_trade_event
from src.utils.state_manager import save_system_state, load_system_state
from src.analytics.performance_tracker import PerformanceTracker
from src.audit_logger.audit_logger import log_trade
from src.utils.alert_manager import send_alert
from src.global_error_handler import handle_errors, ErrorSeverity
from datetime import datetime, timedelta, timezone


logger = logging.getLogger(__name__)

# Portfolio Exposure Control
USD_INVERSE_BUCKET = ["GOLD", "EURUSD", "BTC"]


class Position:
    """Represents a single trading position"""

    def __init__(
        self,
        asset: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        entry_time: datetime,
        risk_config: dict,
        signal_details: dict = None,
        position_id: str = None,
        stop_loss: float = None,
        take_profit: float = None,
        trailing_stop_pct: float = None,
        mt5_ticket: int = None,
        binance_order_id: int = None,
        ohlc_data: dict = None,
        account_balance: float = None,
        use_dynamic_management: bool = True,
        disable_partials: bool = False,
        vtm_overrides: Optional[Dict] = None,
        leverage: int = 1,
        margin_type: str = "SPOT",
        is_futures: bool = False,
        min_lot: Optional[float] = None,      # ✨ NEW: Exness compatibility
        lot_precision: Optional[int] = None   # ✨ NEW: Exness compatibility
    ):
        self.asset = asset
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.position_id = (
            position_id or f"{asset}_{side}_{int(entry_time.timestamp())}"
        )
        self.leverage = leverage
        self.margin_type = margin_type
        self.is_futures = is_futures
        self.closing = False
        self.min_lot = min_lot
        self.lot_precision = lot_precision
        self.disable_partials = disable_partials

        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop_pct = None
        self.highest_price = entry_price if side == "long" else entry_price
        self.lowest_price = entry_price if side == "short" else entry_price

        # Exchange-specific tracking
        self.mt5_ticket = mt5_ticket
        self.mt5_profit = 0.0
        self.mt5_last_update = None
        self.binance_order_id = binance_order_id
        self.binance_profit = 0.0
        self.binance_last_update = None

        self.session_start_time = None
        self.session_start_equity = None
        self.session_start_capital = None

        self.db_trade_id = None
        self.db_manager = None
        self.last_close_attempt = None

        # ✅ CRITICAL FIX: Initialize VTM with intelligent context
        self.trade_manager = None
        if use_dynamic_management and ohlc_data:
            try:
                # ✅ Extract hybrid context from signal_details (handle None case)
                if signal_details is None:
                    signal_details = {}

                hybrid_mode = signal_details.get("aggregator_mode")
                mode_confidence = signal_details.get("mode_confidence", 0.5)
                regime_analysis = signal_details.get("regime_analysis", {})

                # Get regime details
                trend_strength = regime_analysis.get("trend_strength", "weak")
                volatility_regime = regime_analysis.get("volatility_regime", "normal")
                price_clarity = regime_analysis.get("price_clarity", "mixed")
                momentum_aligned = regime_analysis.get("momentum_aligned", False)
                at_key_level = regime_analysis.get("at_key_level", False)

                logger.info(f"\n[VTM+HYBRID] Initializing with intelligent context:")
                logger.info(f"  Mode:        {hybrid_mode or 'N/A'}")
                logger.info(f"  Confidence:  {mode_confidence:.2%}")
                logger.info(f"  Trend:       {trend_strength}")
                logger.info(f"  Volatility:  {volatility_regime}")
                logger.info(f"  Clarity:     {price_clarity}")

                # --------------------------------------------------------
                # Adjust VTM parameters based on hybrid intelligence
                # --------------------------------------------------------
                account_risk = 0.010  # Base 1.0%

                # Council + strong trend = More aggressive
                if hybrid_mode == "council" and trend_strength == "strong":
                    account_risk *= 1.2  # 1.0% → 1.2%
                    early_lock_threshold_pct = 0.008  # Lock @ 0.8%
                    logger.info("  → Council strong trend: Risk↑ Lock↓")

                # Performance + choppy = More conservative
                elif hybrid_mode == "performance" and volatility_regime == "high":
                    account_risk *= 0.8  # 1.0% → 0.8%
                    early_lock_threshold_pct = 0.007  # Lock @ 0.7%
                    logger.info("  → Performance choppy: Risk↓ Lock↓")

                # High confidence boost
                if mode_confidence > 0.75 and momentum_aligned:
                    account_risk *= 1.1
                    logger.info(f"  → High confidence ({mode_confidence:.0%}): Risk↑")

                # Noisy price = Reduce risk
                if price_clarity == "noisy":
                    account_risk *= 0.85
                    logger.info("  → Noisy price: Risk↓")

                # Enforce bounds
                account_risk = max(0.008, min(account_risk, 0.025))

                # ✅ Initialize VTM with optimized parameters
                # Use Keyword arguments to ensure correct mapping even if VTM signature changes

                # Apply VTM overrides if they exist
                if vtm_overrides:
                    risk_config = risk_config.copy()
                    risk_config.update(vtm_overrides)
                    logger.info(f"[VTM] Overrides applied: {vtm_overrides}")

                self.trade_manager = VeteranTradeManager(
                    entry_price=entry_price,
                    side=side,
                    asset=asset,
                    risk_config=risk_config,
                    high=ohlc_data["high"],
                    low=ohlc_data["low"],
                    close=ohlc_data["close"],
                    volume=ohlc_data.get("volume"),
                    quantity=quantity,
                    account_risk=account_risk,
                    signal_details=signal_details,
                    trade_type=signal_details.get("trade_type", "TREND"),
                    min_lot_override=self.min_lot,
                    lot_precision_override=self.lot_precision
                )

                # ✅ Sync VTM's calculated levels back to the Position object
                if self.trade_manager:
                    # ✅ H-2 FIX: Min-lot positions (0.01 lot) cannot support
                    # partial exits — the fractional lot rounds to 0.  Suppress
                    # all TP tiers so VTM exits the full position via SL/trail
                    # only, rather than force-closing 100% at TP1 every time.
                    if self.disable_partials:
                        # Keep the last (most conservative/furthest) TP as a
                        # single full-exit target — partial sizes are wiped so
                        # the position exits 100% at that level instead of in
                        # fractions that would round to zero on a 0.01-lot trade.
                        _all_tps = self.trade_manager.take_profit_levels
                        if _all_tps:
                            # Use TP1 (nearest target) for min-lot positions.
                            # Previously used TP3 (furthest) which almost never hit —
                            # a min-lot position cannot partial-exit, so the single exit
                            # should be the highest-probability target, not the home-run.
                            self.trade_manager.take_profit_levels = [_all_tps[0]]
                        else:
                            self.trade_manager.take_profit_levels = []
                        self.trade_manager.partial_sizes = []
                        logger.info(
                            f"[VTM] ⚠️ Partials disabled for {asset} (min-lot position) "
                            f"— single full-exit TP set to "
                            f"{f'${_all_tps[0]:,.5f}' if _all_tps else 'none'}"
                        )

                    self.stop_loss = self.trade_manager.initial_stop_loss
                    if self.trade_manager.take_profit_levels:
                        self.take_profit = self.trade_manager.take_profit_levels[0]

                    logger.info(f"\n[VTM] ✓ Initialized with hybrid-optimized parameters")
                    logger.info(f"  Account Risk: {account_risk:.3f}")
                    logger.info(f"  Stop Loss:    ${self.stop_loss:,.2f}")
                    logger.info(f"  Take Profit:  {f'${self.take_profit:,.2f}' if self.take_profit is not None else 'N/A'}")

            except Exception as e:
                # Catch failures (including "Position size too large") so the object still initializes
                logger.error(f"[PORTFOLIO] VTM initialization failed for {asset}: {e}")
                self.trade_manager = None

                # ✅ Fallback to provided basic levels if VTM fails to init
                self.stop_loss = stop_loss if stop_loss else None
                self.take_profit = take_profit if take_profit else None
                self.trailing_stop_pct = trailing_stop_pct if trailing_stop_pct else None

        else:
            # ✅ No VTM requested or missing data - use passed levels
            logger.debug(f"[PORTFOLIO] VTM not initialized (missing data or disabled)")
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            self.trailing_stop_pct = trailing_stop_pct

    def update_with_new_bar(self, high: float, low: float, close: float):
        """
        ✅ CORRECTED: Update position with new OHLC bar
        Calls VTM's update method and handles exit signals properly
        """
        if self.trade_manager:
            try:
                old_stop = self.stop_loss
                # ✅ Call VTM's update method (returns Dict or None)
                exit_info = self.trade_manager.on_new_bar(
                    new_high=high, new_low=low, new_close=close
                )

                # ✨ NEW: Log VTM events
                if self.db_manager and self.db_trade_id:
                    # Log stop updates
                    if self.stop_loss != old_stop:
                        self.db_manager.update_trade_vtm_event(
                            trade_id=self.db_trade_id,
                            event_type="stop_updated",
                            old_value=old_stop,
                            new_value=self.stop_loss,
                            current_price=close,
                        )

                if exit_info:
                    # ✅ Check if it's an action (like pyramid) or an exit (reason)
                    if "action" in exit_info:
                        action = exit_info["action"]
                        logger.info(f"[VTM] {self.asset} action triggered: {action}")
                        return exit_info # Return the whole dict to the caller

                    # ✅ Handle standard exits — preserve size so caller can do partial close
                    reason = exit_info.get("reason")
                    if self.db_manager and self.db_trade_id:
                        self.db_manager.update_trade_vtm_event(
                            trade_id=self.db_trade_id,
                            event_type=reason.value if hasattr(reason, "value") else str(reason),
                            current_price=exit_info.get("price", close),
                            metadata={"size": exit_info.get("size", 1.0)},
                        )

                    # Convert enum to string for compatibility
                    from src.execution.veteran_trade_manager import ExitReason
                    if isinstance(reason, ExitReason):
                        reason_str = reason.value
                    else:
                        reason_str = str(reason)

                    exit_size = exit_info.get("size", 1.0)  # fraction of position (0–1)
                    exit_price_actual = exit_info.get("price", close)

                    logger.info(
                        f"[VTM] {self.asset} exit triggered: {reason_str} "
                        f"@ ${exit_price_actual:,.2f} (size={exit_size:.0%})"
                    )
                    # Return dict so PortfolioManager can route partial vs full close
                    return {"reason": reason_str, "size": exit_size, "price": exit_price_actual}

                # ✅ Update position's SL/TP with VTM's current levels
                # (VTM may trail stops or move to break-even)
                self.stop_loss = self.trade_manager.current_stop_loss

                return None

            except Exception as e:
                logger.error(f"[VTM] Error updating {self.asset}: {e}", exc_info=True)
                return None

        return None

    def update_with_current_price(self, current_price: float):
        """
        ✅ NEW: Real-time intra-bar update (for trailing stops)
        Call this more frequently than bar updates
        """
        if self.trade_manager:
            try:
                exit_info = self.trade_manager.update_with_current_price(current_price)

                if exit_info:
                    reason = exit_info["reason"]
                    reason_str = reason.value if isinstance(reason, ExitReason) else str(reason)
                    exit_size = exit_info.get("size", 1.0)

                    logger.info(
                        f"[VTM] {self.asset} real-time exit: {reason_str} "
                        f"@ ${current_price:,.2f} (size={exit_size:.0%})"
                    )
                    return {"reason": reason_str, "size": exit_size, "price": current_price}

                # Update position's stop loss (may have trailed)
                self.stop_loss = self.trade_manager.current_stop_loss

                return None

            except Exception as e:
                logger.error(f"[VTM] Real-time update error: {e}")
                return None

        return None

    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """
        ✅ CORRECTED: Check if position should close
        Prioritizes VTM exit signals over traditional SL/TP
        """
        # 1. Check VTM first (if active)
        if self.trade_manager:
            exit_info = self.trade_manager.check_exit(current_price)
            if exit_info:
                from src.execution.veteran_trade_manager import ExitReason
                reason = exit_info["reason"]
                exit_signal = (
                    reason.value if isinstance(reason, ExitReason) else str(reason)
                )
                return True, f"vtm_{exit_signal}"

        # 2. Fallback to traditional SL/TP (if no VTM)
        if self.stop_loss:
            if self.side == "long" and current_price <= self.stop_loss:
                return True, "stop_loss"
            elif self.side == "short" and current_price >= self.stop_loss:
                return True, "stop_loss"

        if self.take_profit:
            if self.side == "long" and current_price >= self.take_profit:
                return True, "take_profit"
            elif self.side == "short" and current_price <= self.take_profit:
                return True, "take_profit"

        # 3. Traditional trailing stop (only if no VTM)
        if not self.trade_manager:
            trail_stop = self.update_trailing_stop(current_price)
            if trail_stop:
                if self.side == "long" and current_price <= trail_stop:
                    return True, "trailing_stop"
                elif self.side == "short" and current_price >= trail_stop:
                    return True, "trailing_stop"

        return False, ""

    def get_vtm_status(self, live_price: Optional[float] = None) -> Optional[Dict]:
        """Get current VTM status for monitoring"""
        if not self.trade_manager:
            return None

        try:
            # VTM calculates the current price if not provided.
            levels = self.trade_manager.get_current_levels(live_price=live_price)
            if not levels:
                return None

            # ✅ FIX: Get the current_price that VTM calculated or used.
            current_price = levels["current_price"]
            
            next_target = levels.get("take_profit")

            # Calculate absolute P&L (Prioritize exchange-reported profit)
            exchange_pnl = self.get_exchange_pnl()
            position_notional = self.entry_price * self.quantity
            if exchange_pnl != 0.0:
                # Broker P&L includes swap/commission — keep abs and pct consistent
                pnl_abs = exchange_pnl
                pnl_pct = (exchange_pnl / position_notional * 100) if position_notional > 0 else levels["pnl_pct"]
            else:
                pnl_abs = (current_price - self.entry_price) * self.quantity if self.side == "long" else \
                          (self.entry_price - current_price) * self.quantity
                pnl_pct = levels["pnl_pct"]

            return {
                "side": self.side,
                "entry_price": levels["entry_price"],
                "current_price": levels["current_price"],
                "pnl_pct": pnl_pct,
                "pnl_abs": pnl_abs, # Absolute P&L (Exchange-synced)
                "stop_loss": levels["stop_loss"],
                "take_profit": (
                    next_target
                    if next_target
                    else levels.get("all_targets", [])[-1] if levels.get("all_targets") else None
                ),
                "distance_to_sl_pct": levels["distance_to_sl_pct"],
                "distance_to_tp_pct": levels["distance_to_tp_pct"],
                "profit_locked": levels["profit_locked"],
                "bars_in_trade": levels["update_count"],
                "partials_hit": levels["partials_hit"],
                "runner_active": levels["runner_active"],
                "update_count": levels["update_count"],
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                # Runner trail / early-lock dynamics
                "runner_trail_atr_multiplier": getattr(self.trade_manager, "runner_trail_atr_multiplier", None),
                "early_lock_atr_multiplier":   getattr(self.trade_manager, "early_lock_atr_multiplier", None),
                "current_early_lock_threshold_pct": (
                    # early_lock fires when profit > early_lock_atr_multiplier × ATR
                    # Express as % of entry so the display is meaningful
                    (getattr(self.trade_manager, "early_lock_atr_multiplier", 0.5)
                     * self.trade_manager._calculate_atr()
                     / self.entry_price)
                    if self.entry_price > 0 else 0.0
                ),
                "current_runner_trail_pct": (
                    # Actual trail width = gap between extreme price and current SL as % of current price
                    abs(
                        (self.trade_manager.highest_price_reached - self.trade_manager.current_stop_loss)
                        / current_price
                    ) if (
                        getattr(self.trade_manager, "runner_activated", False)
                        and self.side == "long"
                        and self.trade_manager.highest_price_reached > 0
                    ) else abs(
                        (self.trade_manager.current_stop_loss - self.trade_manager.lowest_price_reached)
                        / current_price
                    ) if (
                        getattr(self.trade_manager, "runner_activated", False)
                        and self.side == "short"
                        and self.trade_manager.lowest_price_reached > 0
                    ) else 0.0
                ),
            }
        except Exception as e:
            logger.error(f"Error getting VTM status: {e}")
            return None

    def get_position_value(self, current_price: float) -> float:
        """Get current position value in USD"""
        return self.quantity * current_price

    def get_pnl(self, current_price: float) -> float:
        """Get current profit/loss"""
        if self.side == "long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def get_mt5_pnl(self) -> float:
        """Get real-time P&L from MT5 position"""
        return self.mt5_profit

    def get_binance_pnl(self) -> float:
        """Get real-time P&L from Binance position"""
        return self.binance_profit

    def get_exchange_pnl(self) -> float:
        """Get real-time P&L from either exchange"""
        if self.mt5_ticket and self.mt5_profit != 0.0:
            return self.mt5_profit
        elif self.binance_order_id and self.binance_profit != 0.0:
            return self.binance_profit
        return 0.0

    def get_pnl_pct(self, current_price: float) -> float:
        """Get current profit/loss percentage"""
        position_value = self.entry_price * self.quantity
        return self.get_pnl(current_price) / position_value if position_value > 0 else 0

    def update_trailing_stop(self, current_price: float) -> Optional[float]:
        """Update trailing stop (only used if VTM is disabled)"""
        if self.trailing_stop_pct is None:
            return None

        if self.side == "long":
            if current_price > self.highest_price:
                self.highest_price = current_price
            return self.highest_price * (1 - self.trailing_stop_pct)
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            return self.lowest_price * (1 + self.trailing_stop_pct)

    def __getstate__(self):
        """
        Custom method for pickling. Excludes non-serializable attributes.
        """
        state = self.__dict__.copy()
        # Remove the unpickleable db_manager attribute
        if 'db_manager' in state:
            del state['db_manager']
        return state

    def __setstate__(self, state):
        """
        Custom method for unpickling. Restores state and re-initializes
        non-serializable attributes.
        """
        self.__dict__.update(state)
        # Re-initialize the db_manager attribute after unpickling.
        # It will need to be re-assigned by the PortfolioManager after loading.
        self.db_manager = None
        # Always reset closing flags on reload to prevent stuck positions
        self.closing = False
        self.last_close_attempt = None



class PortfolioManager:
    """
    Manages portfolio-level risk and position sizing
    Fetches actual capital from exchanges (MT5 and Binance)
    Tracks real-time MT5 profit for accurate P&L
    """

    def __init__(
        self,
        config: Dict,
        mt5_handler=None,
        binance_client=None,
        db_manager=None,
        execution_handlers: Dict = None,
        telegram_bot=None,
    ):
        self.config = config
        self.telegram_bot = telegram_bot
        self.portfolio_config = config["portfolio"]
        self.max_positions_per_asset = config.get("trading", {}).get(
            "max_positions_per_asset", 3
        )

        self.mt5_handler = mt5_handler
        self.binance_client = binance_client
        self.db_manager = db_manager
        self.execution_handlers = execution_handlers or {}
        self.risk_cfg = config.get("risk_management", {})
        
        self.correlation_threshold = self.portfolio_config.get("correlation_threshold", 0.65)
        logger.info(f"[RISK] Correlation threshold: {self.correlation_threshold:.0%}")

        self.mode = config["trading"].get("mode", "paper")
        self.is_paper_mode = self.mode.lower() == "paper"

        # ✅ FIX 1: Remove paper_capital usage in live mode
        self.paper_capital = self.portfolio_config["initial_capital"]

        # ✅ FIX 2: Initialize with live balances (will raise error if unavailable in live mode)
        self.initial_capital = self._fetch_total_capital(strict=True)
        self.current_capital = self.initial_capital
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital

        # ✅ FIX 3: Track last balance refresh time
        self.last_balance_refresh = datetime.now()
        self.balance_refresh_interval = timedelta(minutes=5)  # Refresh every 5 min

        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []
        self.price_history: Dict[str, List[float]] = {asset: [] for asset in config["assets"].keys()}
        self.realized_pnl_today = 0.0
        self.performance_tracker = PerformanceTracker()  # ✨ NEW: Strategy Performance Tracking
        self.loss_streak = 0  # ✨ NEW: Consecutive Loss Tracking
        self._loss_streak_alerted = False  # Guard: send alert only ONCE per streak
        self._circuit_breaker_override = False  # Manual override via /resume Telegram command

        self.session_start_time = None
        self.session_start_equity = None
        self.session_start_capital = None
        self.state_file = Path("data/portfolio_state.pkl")

        # Tracks whether the last close per asset was manual (Telegram command / force-close)
        # vs. natural (VTM, SL, TP).  Used by check_min_time_between_trades() to decide
        # whether the "no open positions" cooldown bypass should be suppressed.
        self.last_close_was_manual: Dict[str, bool] = {}

        logger.info(f"Portfolio Manager initialized in {self.mode.upper()} mode")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")

        if not self.is_paper_mode:
            logger.info("✓ Using LIVE account balances (will auto-refresh)")
            logger.info(
                f"  - MT5 Handler: {'Connected' if mt5_handler else 'Not Connected'}"
            )
            logger.info(
                f"  - Binance Client: {'Connected' if binance_client else 'Not Connected'}"
            )
        else:
            logger.info("✓ Using PAPER mode with simulated capital")

    def _resolve_symbol(self, asset_name: str) -> str:
        """Return the broker-correct symbol for the asset's configured exchange."""
        cfg = self.config.get("assets", {}).get(asset_name, {})
        exchange = cfg.get("exchange", "binance")
        if exchange == "mt5":
            return cfg.get("mt5_symbol") or cfg.get("symbol", asset_name)
        return cfg.get("binance_symbol") or cfg.get("symbol", asset_name)

    def save_portfolio_state(self):
        """Saves the current open positions to a file atomically."""
        if self.is_paper_mode:
            logger.info("[STATE] Paper mode, skipping state save.")
            return

        # If there are no positions, ensure no state file is left
        if not self.positions:
            if self.state_file.exists():
                try:
                    self.state_file.unlink()
                    logger.info("[STATE] No open positions. Removed stale state file.")
                except Exception as e:
                    logger.error(f"[STATE] Error removing stale state file: {e}")
            return

        temp_file_path = self.state_file.with_suffix('.pkl.tmp')
        try:
            self.state_file.parent.mkdir(exist_ok=True)
            with open(temp_file_path, "wb") as f:
                pickle.dump(self.positions, f)
            
            # Atomically rename the temp file to the final file
            temp_file_path.rename(self.state_file)
            logger.info(f"[STATE] Successfully saved {len(self.positions)} open positions to {self.state_file}")

            # ✨ NEW: Save non-picklable system metrics to JSON
            system_metrics = {
                "peak_equity": self.peak_equity,
                "loss_streak": self.loss_streak,
                "realized_pnl_today": self.realized_pnl_today,
                "mode": self.mode,  # Save the mode (live/paper)
                "timestamp": datetime.now().isoformat()
            }
            save_system_state(system_metrics)

        except Exception as e:
            logger.error(f"[STATE] Failed to save portfolio state: {e}", exc_info=True)
            if temp_file_path.exists():
                try:
                    temp_file_path.unlink()
                except Exception as e_del:
                    logger.error(f"[STATE] Failed to clean up temp file {temp_file_path}: {e_del}")

    def load_portfolio_state(self, data_manager):
        """Loads open positions from a file and re-initializes them."""
        if self.is_paper_mode:
            logger.info("[STATE] Paper mode, skipping state load.")
            return

        if not self.state_file.exists():
            logger.info("[STATE] No portfolio state file found. Starting fresh.")
            return
            
        # Check for empty file to prevent EOFError
        if self.state_file.stat().st_size == 0:
            logger.warning(f"[STATE] State file {self.state_file} is empty. Deleting and starting fresh.")
            self.state_file.unlink()
            return

        try:
            with open(self.state_file, "rb") as f:
                loaded_positions = pickle.load(f)

            if not loaded_positions:
                logger.info("[STATE] Portfolio state file is empty.")
                return

            # ✨ NEW: Restore non-picklable system metrics from JSON
            system_state = load_system_state()
            if system_state:
                saved_mode = system_state.get("mode")
                
                # Only restore metrics if the mode matches
                if saved_mode == self.mode:
                    self.peak_equity = system_state.get("peak_equity", self.peak_equity)
                    self.loss_streak = system_state.get("loss_streak", 0)
                    self.realized_pnl_today = system_state.get("realized_pnl_today", 0.0)
                    logger.info(
                        f"[STATE] Metrics restored for {self.mode.upper()} mode: "
                        f"Peak Equity=${self.peak_equity:,.2f}, Loss Streak={self.loss_streak}"
                    )
                else:
                    logger.warning(
                        f"[STATE] Skipping metrics restore: Mode mismatch "
                        f"(Current: {self.mode.upper()}, Saved: {str(saved_mode).upper()})"
                    )

            for position_id, position in loaded_positions.items():
                logger.info(f"[STATE] Reloading position: {position_id} ({position.asset} {position.side})")
                
                # Critical step: Re-initialize the VTM with fresh OHLC data
                try:
                    asset_config = self.config['assets'].get(position.asset)
                    if asset_config:
                        symbol = asset_config['symbol']
                        interval = asset_config.get('interval', '1h')
                        exchange = asset_config.get('exchange', 'binance')
                        
                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=10) # Fetch enough data

                        logger.info(f"[STATE] Fetching fresh OHLC data for {position.asset} ({symbol})...")
                        if exchange == 'binance':
                            df = data_manager.fetch_binance_data(
                                symbol=symbol, interval=interval,
                                start_date=start_time.strftime("%Y-%m-%d"),
                                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S")
                            )
                        else: # mt5
                            df = data_manager.fetch_mt5_data(
                                symbol=symbol, timeframe=interval,
                                start_date=start_time.strftime("%Y-%m-%d"),
                                end_date=end_time.strftime("%Y-%m-%d %H:%M:%S")
                            )
                        
                        if len(df) > 50:
                            if position.trade_manager:
                                # Re-sync existing trade manager
                                position.trade_manager.high = df['high'].values
                                position.trade_manager.low = df['low'].values
                                position.trade_manager.close = df['close'].values
                                position.trade_manager.volume = df['volume'].values if 'volume' in df else None
                                logger.info(f"[STATE] VTM for {position_id} successfully re-synced with {len(df)} candles.")
                            else:
                                # Create NEW trade manager if it was missing
                                logger.info(f"[STATE] VTM missing for {position_id}. Creating new manager...")
                                risk_config = getattr(position, 'risk_config', asset_config.get('risk', {}))
                                signal_details = getattr(position, 'signal_details', {})
                                
                                position.trade_manager = VeteranTradeManager(
                                    entry_price=position.entry_price,
                                    side=position.side,
                                    asset=position.asset,
                                    risk_config=risk_config,
                                    high=df['high'].values,
                                    low=df['low'].values,
                                    close=df['close'].values,
                                    volume=df['volume'].values if 'volume' in df else None,
                                    quantity=position.quantity,
                                    signal_details=signal_details,
                                    trade_type=signal_details.get("trade_type", "TREND"),
                                    # Restoring VTM for an already-open position: accept
                                    # whatever size the live trade actually has, even if it
                                    # has dipped below broker minimum after partial closes.
                                    min_lot_override=position.quantity,
                                )
                                logger.info(f"[STATE] VTM for {position_id} successfully created.")
                        else:
                            logger.warning(f"[STATE] Could not fetch enough OHLC data for {position_id}. VTM may be impaired.")
                    else:
                        logger.warning(f"[STATE] Asset config not found for {position.asset}. Cannot initialize VTM.")

                except Exception as e:
                    logger.error(f"[STATE] Failed to re-initialize VTM for {position_id}: {e}", exc_info=True)
                
                # Re-link the db_manager
                if self.db_manager:
                    position.db_manager = self.db_manager

                self.positions[position_id] = position

            logger.info(f"[STATE] Successfully loaded and re-initialized {len(self.positions)} positions.")
            
            # Clean up state file after successful load
            self.state_file.unlink()
            logger.info(f"[STATE] Removed state file {self.state_file} after successful load.")

        except Exception as e:
            logger.error(f"[STATE] Failed to load portfolio state: {e}", exc_info=True)
            # If loading fails, start with a clean slate to avoid corruption
            self.positions = {}
            
    def _get_quote_to_usd_rate(self, symbol: str) -> float:
        """
        Returns a multiplier to convert a value denominated in the symbol's
        quote currency into USD.

        • For USD-quoted instruments (EURUSD, XAUUSD, BTCUSDT, …) → 1.0
        • For JPY-quoted instruments (EURJPY, USDJPY, GBPJPY, …) → 1 / USD_JPY_rate

        Falls back to a hard-coded approximate rate if MT5 is unreachable.
        Result is cached for 5 minutes to avoid hammering MT5 on every call.
        """
        # Normalise: strip broker-specific suffix ('m', 'pro', etc.) before testing
        base = symbol.upper().replace("M", "").strip()

        # USD-quoted: no conversion needed
        if base.endswith("USD") or "USDT" in base or base in ("BTC", "GOLD", "USTEC", "USOIL"):
            return 1.0

        # JPY-quoted: need 1/USDJPY
        if base.endswith("JPY"):
            now = datetime.now()
            cached_rate, cached_ts = getattr(self, "_usdjpy_cache", (None, None))
            if cached_rate and cached_ts and (now - cached_ts).total_seconds() < 300:
                return 1.0 / cached_rate

            usdjpy_rate = None
            try:
                if self.mt5_handler:
                    for sym in ("USDJPYm", "USDJPY"):
                        price = self.mt5_handler.get_current_price(sym)
                        if price and price > 50:
                            usdjpy_rate = price
                            break
            except Exception:
                pass

            if not usdjpy_rate:
                usdjpy_rate = 155.0
                logger.debug("[CURRENCY] USDJPY rate unavailable — using fallback 155.0")

            self._usdjpy_cache = (usdjpy_rate, now)
            logger.debug(f"[CURRENCY] USDJPY rate: {usdjpy_rate:.3f} → conversion factor {1/usdjpy_rate:.6f}")
            return 1.0 / usdjpy_rate

        # AUD-quoted: need AUDUSD rate  (e.g. GBPAUD, EURAUD — price is in AUD)
        if base.endswith("AUD"):
            now = datetime.now()
            cached_rate, cached_ts = getattr(self, "_audusd_cache", (None, None))
            if cached_rate and cached_ts and (now - cached_ts).total_seconds() < 300:
                return cached_rate

            audusd_rate = None
            try:
                if self.mt5_handler:
                    for sym in ("AUDUSDm", "AUDUSD"):
                        price = self.mt5_handler.get_current_price(sym)
                        if price and 0.4 < price < 1.2:   # sanity: AUDUSD ~0.6–0.7
                            audusd_rate = price
                            break
            except Exception:
                pass

            if not audusd_rate:
                audusd_rate = 0.65   # fallback approximate rate
                logger.debug("[CURRENCY] AUDUSD rate unavailable — using fallback 0.65")

            self._audusd_cache = (audusd_rate, now)
            logger.debug(f"[CURRENCY] AUDUSD rate: {audusd_rate:.5f} → conversion factor {audusd_rate:.6f}")
            return audusd_rate

        # CAD-quoted: need 1/USDCAD
        if base.endswith("CAD"):
            now = datetime.now()
            cached_rate, cached_ts = getattr(self, "_usdcad_cache", (None, None))
            if cached_rate and cached_ts and (now - cached_ts).total_seconds() < 300:
                return 1.0 / cached_rate

            usdcad_rate = None
            try:
                if self.mt5_handler:
                    for sym in ("USDCADm", "USDCAD"):
                        price = self.mt5_handler.get_current_price(sym)
                        if price and 1.0 < price < 1.8:
                            usdcad_rate = price
                            break
            except Exception:
                pass

            if not usdcad_rate:
                usdcad_rate = 1.36
                logger.debug("[CURRENCY] USDCAD rate unavailable — using fallback 1.36")

            self._usdcad_cache = (usdcad_rate, now)
            return 1.0 / usdcad_rate

        # CHF-quoted: need 1/USDCHF
        if base.endswith("CHF"):
            now = datetime.now()
            cached_rate, cached_ts = getattr(self, "_usdchf_cache", (None, None))
            if cached_rate and cached_ts and (now - cached_ts).total_seconds() < 300:
                return 1.0 / cached_rate

            usdchf_rate = None
            try:
                if self.mt5_handler:
                    for sym in ("USDCHFm", "USDCHF"):
                        price = self.mt5_handler.get_current_price(sym)
                        if price and 0.7 < price < 1.3:
                            usdchf_rate = price
                            break
            except Exception:
                pass

            if not usdchf_rate:
                usdchf_rate = 0.90
                logger.debug("[CURRENCY] USDCHF rate unavailable — using fallback 0.90")

            self._usdchf_cache = (usdchf_rate, now)
            return 1.0 / usdchf_rate

        # Unknown quote currency — assume USD (safe default, logs a warning)
        logger.warning(f"[CURRENCY] Unknown quote currency for symbol '{symbol}' — assuming USD")
        return 1.0

    def get_risk_budget(
        self,
        asset: str,
        strategy_type: str = "TREND",
        confidence_score: Optional[float] = None,
        market_condition: Optional[str] = None
    ) -> float:
        """
        ✨ STRATEGIC RISK GOVERNOR
        
        Calculates risk budget for a new trade based on:
        1. Base risk (from config)
        2. Signal confidence adjustment (from hybrid_position.py)
        3. Market condition adjustment (from hybrid_position.py)
        4. Strategy type (SCALP vs TREND asymmetric adjustment)
        5. Correlation malus (reduce risk if holding correlated assets)
        6. Drawdown shield (reduce risk in drawdown)
        7. Total risk limit (cap aggregate open risk)
        
        Args:
            asset: Asset name (e.g., "BTC", "GOLD")
            strategy_type: "TREND" or "SCALP"
            confidence_score: Optional signal confidence (0.0 to 1.0)
            market_condition: Optional market regime description
            
        Returns:
            Risk percentage (e.g., 0.015 for 1.5%)
        """
        try:
            asset_cfg = self.config.get("assets", {}).get(asset, {})
            # ================================================================
            # STEP 1: Get base risk from config (Percentage or Fixed Dollar)
            # ================================================================
            fixed_risk_config = asset_cfg.get("fixed_risk_usd")
            base_risk = self.portfolio_config.get("target_risk_per_trade", 0.015)
            
            # ✅ T2.1: Apply logic ported from orphaned hybrid_position.py
            if confidence_score is not None:
                # Confidence-based scaling (0.3 to 1.5x)
                # Maps [0,1] to [0.5,1.5], clamped at 0.3 min
                confidence_scalar = 0.5 + (confidence_score * 1.0)
                confidence_scalar = max(0.3, min(1.5, confidence_scalar))
                base_risk *= confidence_scalar
                logger.info(f"  Confidence Adjustment: {confidence_scalar:.2f}x (Conf: {confidence_score:.2f})")

            if market_condition:
                condition = market_condition.lower()
                condition_scalars = {
                    "bullish": 1.1,
                    "neutral": 1.0,
                    "bearish": 0.8,
                    "uncertain": 0.6,
                    "extreme_volatility": 0.5,
                    "bear": 0.8, # Handle variants
                    "bull": 1.1
                }
                condition_scalar = condition_scalars.get(condition, 1.0)
                if condition_scalar != 1.0:
                    base_risk *= condition_scalar
                    logger.info(f"  Market Condition Adjustment: {condition_scalar:.2f}x (Regime: {market_condition})")

            strategy_multiplier = 1.0
            
            logger.info(f"\n[RISK BUDGET] Calculating for {asset} {strategy_type}")

            if fixed_risk_config and isinstance(fixed_risk_config, dict):
                # FIXED DOLLAR RISK LOGIC
                risk_usd = fixed_risk_config.get(strategy_type)
                if risk_usd:
                    if self.current_capital > 0:
                        risk_pct = risk_usd / self.current_capital
                        logger.info(f"  Fixed Dollar Risk: ${risk_usd} ({strategy_type})")
                        logger.info(f"  Account Capital: ${self.current_capital:,.2f}")
                        logger.info(f"  Calculated Risk: {risk_pct:.3%}")
                    else:
                        logger.error("[RISK] Cannot calculate fixed risk, current capital is zero.")
                        return 0.0
                else:
                    # Fallback to percentage if strategy type not in fixed config
                    risk_pct = base_risk
                    logger.info(f"  Base Risk: {risk_pct:.3%}")
            else:
                # ORIGINAL PERCENTAGE-BASED LOGIC
                logger.info(f"  Base Risk: {base_risk:.3%}")
            
                # ================================================================
                # STEP 2: Strategy Type Adjustment (Asymmetric)
                # ================================================================
                if strategy_type == "SCALP":
                    # Scalps: Lower risk (quick in/out)
                    strategy_multiplier = 1.25  # 1.5% → 1.875%
                    logger.info(f"  SCALP Multiplier: {strategy_multiplier:.2f}x")
                elif strategy_type == "TREND":
                    # Trends: Higher risk (riding momentum)
                    strategy_multiplier = 1.33  # 1.5% → 2.0%
                    logger.info(f"  TREND Multiplier: {strategy_multiplier:.2f}x")
                else:
                    strategy_multiplier = 1.0
                    logger.info(f"  Default Multiplier: {strategy_multiplier:.2f}x")
                
                risk_pct = base_risk * strategy_multiplier
            
            # ================================================================
            # STEP 3: Correlation Malus (Institutional Upgrade)
            # ✅ TASK 24: Rolling Pearson Correlation
            # ================================================================
            correlation_threshold = self.portfolio_config.get(
                "correlation_threshold", 0.65
            )
            correlation_malus = 1.0
            
            # Check if we hold correlated positions
            if len(self.positions) > 0:
                max_corr = 0.0
                correlated_asset = None
                
                for pos_asset, position in self.positions.items():
                    if pos_asset == asset: continue # Skip self
                    
                    # Numerical Correlation check (Pearson)
                    num_corr = abs(self.check_correlation(asset, pos_asset))
                    if num_corr > max_corr:
                        max_corr = num_corr
                        correlated_asset = pos_asset
                
                # Apply dynamic malus if threshold breached
                if max_corr > correlation_threshold:
                    # Dynamic reduction: e.g., 0.8 corr -> 0.2 multiplier (80% reduction)
                    # Clamped at 0.3 minimum multiplier (70% max reduction)
                    correlation_malus = max(0.3, 1.0 - max_corr)
                    
                    logger.warning(
                        f"  ⚠️ Correlation Malus: High linkage with {correlated_asset} ({max_corr:.2f})"
                    )
                    logger.info(f"  Risk reduced to {correlation_malus:.0%} of original budget")
            
            risk_pct *= correlation_malus
            
            # ================================================================
            # STEP 4: Drawdown Shield
            # ================================================================
            drawdown_threshold = self.portfolio_config.get("max_drawdown", 0.15)
            current_drawdown = 0.0
            
            if self.peak_equity > 0:
                current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
            
            drawdown_malus = 1.0
            
            if current_drawdown > 0.08:  # 8% drawdown trigger
                drawdown_malus = 0.65
                logger.warning(
                    f"  ⚠️ Drawdown Shield: {current_drawdown:.2%} drawdown detected"
                )
                logger.info(f"  Risk reduced by {1 - drawdown_malus:.0%}")
            
            risk_pct *= drawdown_malus
            
            # ================================================================
            # STEP 5: Total Risk Limit (Aggregate Cap)
            # ✅ IMPROVED: Use Dollar Risk at SL (not notional exposure)
            # ================================================================
            max_total_risk_pct = self.risk_cfg.get("max_total_open_risk", 0.10)
            
            # Calculate current total risk (Dollar amount at risk across all positions)
            current_total_risk_usd = 0.0
            for position in self.positions.values():
                # Conversion factor: 1.0 for USD-quoted, 1/USDJPY for JPY-quoted, etc.
                quote_to_usd = self._get_quote_to_usd_rate(position.symbol)
                if position.stop_loss:
                    # Risk = |Entry - SL| * Quantity  (in quote currency) → convert to USD
                    pos_risk = abs(position.entry_price - position.stop_loss) * position.quantity
                    current_total_risk_usd += pos_risk * quote_to_usd
                else:
                    # Fallback: estimate 5% of notional (in quote currency) → convert to USD
                    current_total_risk_usd += (position.entry_price * position.quantity * 0.05) * quote_to_usd
            
            current_total_risk_pct = (
                current_total_risk_usd / self.current_capital 
                if self.current_capital > 0 
                else 0
            )
            
            # Check if adding new trade would exceed limit
            remaining_risk_budget_pct = max_total_risk_pct - current_total_risk_pct
            
            if risk_pct > remaining_risk_budget_pct:
                logger.warning(
                    f"  ⚠️ Total Risk Limit: Current {current_total_risk_pct:.2%}, "
                    f"Max {max_total_risk_pct:.2%}"
                )
                logger.info(
                    f"  Risk capped from {risk_pct:.3%} to {max(0, remaining_risk_budget_pct):.3%}"
                )
                risk_pct = max(0, remaining_risk_budget_pct)
            
            # ================================================================
            # STEP 5.5: Risk Floor (Enforce Minimum Risk USD)
            # ================================================================
            min_risk_usd = asset_cfg.get("min_risk_usd")
            if min_risk_usd and self.current_capital > 0:
                min_risk_pct = min_risk_usd / self.current_capital
                if risk_pct < min_risk_pct:
                    logger.info(
                        f"  🛡️ Risk Floor Applied: {risk_pct:.3%} → {min_risk_pct:.3%} "
                        f"(${min_risk_usd:.2f} minimum)"
                    )
                    risk_pct = min_risk_pct
            
            # ================================================================
            # STEP 6: Final Validation
            # ================================================================
            # Ensure we don't go below minimum viable risk
            min_risk = 0.001  # 0.1% absolute minimum
            if risk_pct < min_risk:
                logger.error(
                    f"  ❌ Risk budget {risk_pct:.3%} below minimum {min_risk:.3%}"
                )
                logger.error(f"  → Trade should be rejected")
                return 0.0
            
            # Ensure we don't exceed maximum risk
            max_risk = self.portfolio_config.get("max_risk_per_trade", 0.025)
            if risk_pct > max_risk:
                logger.warning(f"  ⚠️ Risk capped at maximum {max_risk:.3%}")
                risk_pct = max_risk
            
            # ================================================================
            # STEP 7: Log Final Budget
            # ================================================================
            logger.info(f"\n[RISK BUDGET] FINAL: {risk_pct:.3%}")
            logger.info(f"  Breakdown:")
            logger.info(f"    Base:         {base_risk:.3%}")
            logger.info(f"    Strategy:     ×{strategy_multiplier:.2f}")
            logger.info(f"    Correlation:  ×{correlation_malus:.2f}")
            logger.info(f"    Drawdown:     ×{drawdown_malus:.2f}")
            logger.info(f"    Final:        {risk_pct:.3%}")
            logger.info(f"  → ${self.current_capital * risk_pct:,.2f} at risk\n")
            
            return risk_pct
            
        except Exception as e:
            logger.error(f"[RISK BUDGET] Error calculating risk: {e}", exc_info=True)
            # Return safe default
            return 0.01  # 1% fallback

    def _get_asset_group(self, asset: str) -> str:
        """
        Helper: Categorize assets into correlation groups
        
        Returns:
            Group name: "crypto", "precious_metals", "indices", "forex", "other"
        """
        asset = asset.upper()
        
        # Crypto group
        if asset in ["BTC", "BITCOIN", "BTCUSD", "BTCUSDT", "ETH", "ETHEREUM"]:
            return "crypto"
        
        # Precious metals group
        if asset in ["GOLD", "XAU", "XAUUSDm", "SILVER", "XAG", "XAGUSD"]:
            return "precious_metals"
        
        # Indices group
        if any(x in asset for x in ["SPX", "SPY", "QQQ", "NASDAQ", "DOW", "USTEC"]):
            return "indices"
        
        # Forex group
        if any(x in asset for x in ["EUR", "GBP", "JPY", "USD", "AUD", "CHF", "CAD"]):
            return "forex"
        
        return "other"

    def check_circuit_breaker(self) -> tuple:
        """Check if trading should be halted due to risk breaches"""
        # Manual override: /resume Telegram command bypasses all circuit-breaker conditions.
        # The override is cleared automatically at the start of the next trading day via
        # reset_daily_stats().  You can also re-engage the breaker with /stop_trading.
        if getattr(self, "_circuit_breaker_override", False):
            return False, ""

        if self.session_start_equity and self.session_start_equity > 0:
            daily_loss = (self.session_start_equity - self.equity) / self.session_start_equity
            limit = self.risk_cfg.get('max_daily_loss_pct', 0.03)

            if daily_loss > limit:
                return True, f'Daily loss {daily_loss:.1%} > limit {limit:.1%}'

        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.equity) / self.peak_equity
            
            # Layer 1: Hard Max Drawdown (15%) - Severe protection
            max_dd = self.portfolio_config.get('max_drawdown', 0.15)
            if drawdown > max_dd:
                return True, f'CRITICAL: Hard Drawdown {drawdown:.1%} > max {max_dd:.1%}'
            
            # Layer 2: Profit Lock (15%) - Protects recent gains from peak
            # Triggered when equity drops from its highest ever point based on config
            profit_lock_threshold = self.portfolio_config.get('profit_lock_threshold', 0.15)
            if drawdown > profit_lock_threshold:
                reason = f'PROFIT LOCK: Equity dropped {drawdown:.1%} from peak. Protecting gains.'
                send_alert(reason)
                return True, reason

        # Consecutive Loss Shield — alert fires ONCE when streak is first hit,
        # not on every subsequent 5-minute cycle.
        if self.loss_streak >= 3:
            reason = f'Consecutive loss streak of {self.loss_streak} trades'
            if not self._loss_streak_alerted:
                send_alert(reason)
                self._loss_streak_alerted = True
            return True, reason

        return False, ''

    def _fetch_total_capital(self, strict: bool = False) -> float:
        """
        ✅ FIXED: Fetch total available capital from ALL exchanges

        Args:
            strict: If True, raise error when live balances unavailable in live mode

        Returns:
            Total capital in USD (MT5 + Binance combined)
        """
        if self.is_paper_mode:
            logger.info(f"[PAPER] Using simulated capital: ${self.paper_capital:,.2f}")
            return self.paper_capital

        total_capital = 0.0
        errors = []
        balances_found = []

        # ================================================================
        # ✅ FIX 1: Check MT5 balance (GOLD) - Always try if handler exists
        # ================================================================
        if self.mt5_handler is not None:
            try:
                mt5_balance = self._fetch_mt5_balance()
                if mt5_balance is not None and mt5_balance > 0:
                    total_capital += mt5_balance
                    balances_found.append(f"MT5: ${mt5_balance:,.2f}")
                    logger.info(f"[MT5] ✓ Balance fetched: ${mt5_balance:,.2f}")
                else:
                    logger.warning(f"[MT5] Balance is 0 or None")
                    errors.append("MT5 balance unavailable or 0")
            except Exception as e:
                logger.error(f"[MT5] Error fetching balance: {e}", exc_info=True)
                errors.append(f"MT5 error: {str(e)}")
        else:
            logger.debug("[MT5] Handler not available, skipping MT5 balance")

        # ================================================================
        # ✅ FIX 2: Check Binance balance (BTC) - Always try if client exists
        # ================================================================
        if self.binance_client is not None:
            try:
                binance_balance = self._fetch_binance_balance()
                if binance_balance is not None and binance_balance > 0:
                    total_capital += binance_balance
                    balances_found.append(f"Binance: ${binance_balance:,.2f}")
                    logger.info(f"[BINANCE] ✓ Balance fetched: ${binance_balance:,.2f}")
                else:
                    logger.warning(f"[BINANCE] Balance is 0 or None")
                    errors.append("Binance balance unavailable or 0")
            except Exception as e:
                logger.error(f"[BINANCE] Error fetching balance: {e}", exc_info=True)
                errors.append(f"Binance error: {str(e)}")
        else:
            logger.debug("[BINANCE] Client not available, skipping Binance balance")

        # ================================================================
        # ✅ FIX 3: Log combined results clearly
        # ================================================================
        logger.info(
            f"\n{'='*80}\n"
            f"[BALANCE SUMMARY]\n"
            f"{'='*80}\n"
            f"Balances Found: {len(balances_found)}\n"
            f"  {chr(10).join(balances_found) if balances_found else 'None'}\n"
            f"Total Capital:  ${total_capital:,.2f}\n"
            f"Errors: {len(errors)}\n"
            f"  {chr(10).join(errors) if errors else 'None'}\n"
            f"{'='*80}"
        )

        # ================================================================
        # ✅ FIX 4: Strict mode enforcement (for live trading)
        # ================================================================
        if strict and not self.is_paper_mode and total_capital == 0:
            if self.mt5_handler is None and self.binance_client is None:
                # This is okay, handlers aren't ready yet, will refresh later
                logger.warning("[BALANCE] No handlers initialized yet, initial capital set to 0. Will refresh later.")
                return 0.0
            else:
                error_msg = (
                    f"CRITICAL: Unable to fetch live account balances!\n"
                    f"Errors: {', '.join(errors)}\n"
                    f"Cannot proceed with live trading without valid balances."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        # ================================================================
        # ✅ FIX 5: Fallback handling
        # ================================================================
        if total_capital == 0:
            if self.is_paper_mode:
                logger.warning("No balances fetched, using paper capital")
                return self.paper_capital
            else:
                logger.error(
                    "⚠️  NO BALANCES AVAILABLE FROM ANY EXCHANGE!\n"
                    "Check your MT5 connection and Binance API keys."
                )
                return 0.0

        return total_capital

    def _fetch_mt5_balance(self) -> Optional[float]:
        """
        ✅ FIXED: Fetch MT5 balance with better error handling
        """
        try:
            import MetaTrader5 as mt5

            if not mt5.initialize():
                logger.error("[MT5] Failed to initialize terminal")
                return None

            account_info = mt5.account_info()

            if account_info:
                balance = account_info.balance
                equity = account_info.equity
                margin = account_info.margin
                free_margin = account_info.margin_free

                logger.info(
                    f"[MT5] Account info:\n"
                    f"  Balance:      ${balance:,.2f}\n"
                    f"  Equity:       ${equity:,.2f}\n"
                    f"  Margin Used:  ${margin:,.2f}\n"
                    f"  Free Margin:  ${free_margin:,.2f}"
                )

                # Use equity (includes unrealized P&L)
                return equity
            else:
                logger.error("[MT5] No account info available")
                return None

        except Exception as e:
            logger.error(f"[MT5] Error fetching balance: {e}", exc_info=True)
            return None

    def _fetch_binance_balance(self) -> Optional[float]:
        """
        ✅ FIXED: Dynamically fetches Futures balance if enabled, otherwise Spot.
        """
        try:
            if not self.binance_client:
                logger.error("[BINANCE] Client not initialized")
                return None

            # Check if we are trading Futures or Spot
            is_futures = (
                self.config.get("assets", {})
                .get("BTC", {})
                .get("enable_futures", False)
            )

            if is_futures:
                logger.info("[BINANCE] Fetching FUTURES account info...")
                try:
                    try:
                        account = self.binance_client.futures_account()
                    except Exception as _e1021:
                        if "-1021" in str(_e1021):
                            logger.warning("[BINANCE] Clock drift detected (-1021), re-syncing time offset...")
                            from src.data.data_manager import _sync_time_offset
                            _sync_time_offset(self.binance_client)
                            account = self.binance_client.futures_account()  # one retry
                        else:
                            raise

                    # Log raw keys for debugging (safely)
                    available_keys = list(account.keys()) if isinstance(account, dict) else "None"
                    logger.debug(f"[BINANCE] Futures account keys found: {available_keys}")

                    total_balance = float(account.get("totalWalletBalance", 0))
                    available = float(account.get("availableBalance", 0))
                    unrealized_pnl = float(account.get("totalUnrealizedProfit", 0))
                    # Use margin balance (wallet + unrealized PnL) = true account equity.
                    # This prevents a false circuit-breaker trigger when an already-open
                    # losing position is imported at startup: the loss is already "priced
                    # in" to the margin balance, so closing the position produces zero
                    # delta rather than a sudden drop from a cash-only baseline.
                    margin_balance = total_balance + unrealized_pnl

                    logger.info(
                        f"[BINANCE FUTURES] Balance breakdown:\n"
                        f"  Wallet:     ${total_balance:,.2f}\n"
                        f"  USDT Free:  ${available:,.2f}\n"
                        f"  Unrealized: ${unrealized_pnl:+,.2f}\n"
                        f"  Equity:     ${margin_balance:,.2f}"
                    )
                    return margin_balance if margin_balance > 0 else 0.0
                except Exception as e:
                    logger.error(f"[BINANCE] Error calling futures_account: {e}")
                    return None

            else:
                # SPOT WALLET LOGIC (Original)
                logger.debug("[BINANCE] Fetching SPOT account info...")
                account = self.binance_client.get_account()

                total_balance = 0.0
                asset_details = []

                for balance in account["balances"]:
                    asset = balance["asset"]
                    free = float(balance["free"])
                    locked = float(balance["locked"])
                    total = free + locked

                    if total > 0.0001:
                        if asset == "USDT":
                            total_balance += total
                            asset_details.append(
                                f"  USDT: ${total:,.2f} (free: ${free:,.2f}, locked: ${locked:,.2f})"
                            )

                        elif asset == "BTC":
                            handler = self.execution_handlers.get("binance")
                            if not handler:
                                logger.error("[BINANCE] Cannot get BTC price, handler not available.")
                                continue
                            btc_price = handler.get_current_price("BTCUSDT")
                            if not btc_price:
                                logger.error("[BINANCE] Failed to get BTC price from handler.")
                                continue
                            
                            usd_value = total * btc_price
                            total_balance += usd_value
                            asset_details.append(
                                f"  BTC:  {total:.8f} @ ${btc_price:,.2f} = ${usd_value:,.2f}"
                            )

                if asset_details:
                    logger.info(
                        f"[BINANCE SPOT] Balance breakdown:\n"
                        + "\n".join(asset_details)
                        + f"\n  Total: ${total_balance:,.2f}"
                    )

                return total_balance if total_balance > 0 else None

        except BinanceAPIException as e:
            logger.error(f"[BINANCE] API error: {e.status_code} - {e.message}")
            return None
        except Exception as e:
            logger.error(f"[BINANCE] Error fetching balance: {e}", exc_info=True)
            return None

    def refresh_capital(self, force: bool = False) -> bool:
        """
        ✅ FIXED: Better logging for balance refresh
        """
        if self.is_paper_mode:
            return True

        # Check if refresh is needed
        now = datetime.now()
        time_since_refresh = now - self.last_balance_refresh

        if not force and time_since_refresh < self.balance_refresh_interval:
            logger.debug(
                f"[BALANCE] Skipping refresh (last: {time_since_refresh.seconds}s ago, "
                f"interval: {self.balance_refresh_interval.seconds}s)"
            )
            return True

        logger.info(
            f"\n{'='*80}\n"
            f"[BALANCE REFRESH]\n"
            f"{'='*80}\n"
            f"Last refresh: {time_since_refresh.seconds}s ago\n"
            f"Force:        {force}\n"
            f"{'='*80}"
        )

        # Fetch new balances
        new_capital = self._fetch_total_capital(strict=False)

        if new_capital > 0:
            old_capital = self.current_capital
            self.current_capital = new_capital
            self.equity = new_capital
            self.last_balance_refresh = now

            # ✅ Account-switch guard: if the fetched balance is dramatically
            # lower than peak_equity (>80% drop in one refresh), the previous
            # peak is almost certainly from a different broker account (e.g.
            # switching from Binance demo to a small live MT5 account).
            # Reset peak_equity and session_start_equity to the real balance so
            # the circuit-breaker doesn't fire a phantom 99% drawdown.
            if self.peak_equity > 0 and new_capital < self.peak_equity * 0.20:
                logger.warning(
                    f"[BALANCE] ⚠️  Peak equity reset: fetched balance ${new_capital:,.2f} is "
                    f"<20% of stored peak ${self.peak_equity:,.2f} — looks like an account "
                    f"switch.  Resetting peak and session baseline to current balance."
                )
                self.peak_equity = new_capital
                self.session_start_equity = new_capital
                self.session_start_capital = new_capital
                self.initial_capital = new_capital
            elif self.equity > self.peak_equity:
                self.peak_equity = self.equity

            change = new_capital - old_capital
            change_pct = (change / old_capital * 100) if old_capital > 0 else 0

            logger.info(
                f"[BALANCE] ✓ Refreshed successfully\n"
                f"  Old: ${old_capital:,.2f}\n"
                f"  New: ${new_capital:,.2f}\n"
                f"  Δ:   ${change:+,.2f} ({change_pct:+.2f}%)"
            )
            return True
        else:
            logger.error(
                f"[BALANCE] ✗ Failed to refresh balances!\n"
                f"  Check MT5 connection and Binance API"
            )
            return False

    def update_mt5_positions_profit(self):
        """
        Update all MT5 positions with real-time profit from MT5
        Call this periodically to sync P&L
        """
        try:
            import MetaTrader5 as mt5

            # Get all open MT5 positions
            mt5_positions = mt5.positions_get()

            if mt5_positions is None or len(mt5_positions) == 0:
                return

            # Update profit for each tracked position
            for asset, position in self.positions.items():
                if position.mt5_ticket is None:
                    continue  # Skip non-MT5 positions (e.g., Binance)

                # Find matching MT5 position
                for mt5_pos in mt5_positions:
                    if mt5_pos.ticket == position.mt5_ticket:
                        position.mt5_profit = mt5_pos.profit
                        position.mt5_last_update = datetime.now()

                        logger.debug(
                            f"[MT5] Updated {asset} profit: ${mt5_pos.profit:,.2f}"
                        )
                        break

        except Exception as e:
            logger.error(f"Error updating MT5 positions profit: {e}", exc_info=True)

    def update_binance_positions_profit(self):
        """
        Update all Binance positions with real-time profit
        Call this periodically to sync P&L
        """
        try:
            if self.binance_client is None:
                return

            # Get current prices for all Binance positions
            for asset, position in self.positions.items():
                if position.binance_order_id is None:
                    continue  # Skip non-Binance positions (e.g., MT5)

                # Get current price
                try:
                    handler = self.execution_handlers.get("binance")
                    if not handler:
                        logger.debug(f"Cannot update Binance profit for {asset}, handler not available.")
                        continue
                    
                    current_price = handler.get_current_price(position.symbol)
                    if not current_price:
                        logger.debug(f"Could not fetch price for {position.symbol} via handler.")
                        continue

                    # Calculate real-time P&L
                    if position.side == "long":
                        position.binance_profit = (
                            current_price - position.entry_price
                        ) * position.quantity
                    else:
                        position.binance_profit = (
                            position.entry_price - current_price
                        ) * position.quantity

                    position.binance_last_update = datetime.now()

                    logger.debug(
                        f"[BINANCE] Updated {asset} profit: ${position.binance_profit:,.2f}"
                    )

                except Exception as e:
                    logger.debug(f"Error fetching Binance price for {asset}: {e}")

        except Exception as e:
            logger.error(f"Error updating Binance positions profit: {e}", exc_info=True)

    def calculate_position_size(
        self, asset: str, entry_price: float, stop_loss: float, venue: str, confidence: float = 0.5
    ) -> float:
        """
        STEP 1 — Venue Isolation Dam
        Calculate position size based STRICTLY on venue-specific free margin.
        ✅ ENHANCED: Added confidence-based scaling
        """
        local_free_margin = 0.0

        if self.is_paper_mode:
            # Use asset balance estimate for paper mode
            local_free_margin = self.get_asset_balance(asset)
        else:
            try:
                if venue.upper() == "MT5":
                    import MetaTrader5 as mt5
                    if not mt5.initialize():
                        logger.error("[PORTFOLIO] Failed to initialize MT5 for margin check")
                        return 0.0
                    account_info = mt5.account_info()
                    if account_info:
                        local_free_margin = account_info.margin_free
                    else:
                        logger.error("[PORTFOLIO] Could not fetch MT5 account info")
                        return 0.0

                elif venue.upper() == "BINANCE":
                    if not self.binance_client:
                        logger.error("[PORTFOLIO] Binance client not initialized")
                        return 0.0
                    
                    # Check if futures enabled for this asset
                    is_futures = self.config.get("assets", {}).get(asset, {}).get("enable_futures", False)
                    
                    if is_futures:
                        try:
                            account = self.binance_client.futures_account()
                        except Exception as _e1021:
                            if "-1021" in str(_e1021):
                                logger.warning("[PORTFOLIO] Clock drift detected (-1021), re-syncing time offset...")
                                from src.data.data_manager import _sync_time_offset
                                _sync_time_offset(self.binance_client)
                                account = self.binance_client.futures_account()  # one retry
                            else:
                                raise
                        local_free_margin = float(account.get("availableBalance", 0))
                    else:
                        # For spot, available USDT
                        asset_balance = self.binance_client.get_asset_balance(asset="USDT")
                        if asset_balance:
                            local_free_margin = float(asset_balance.get("free", 0))
                
                else:
                    logger.error(f"[PORTFOLIO] Unknown venue: {venue}")
                    return 0.0

            except Exception as e:
                logger.error(f"[PORTFOLIO] Error fetching venue margin for {venue}: {e}")
                return 0.0

        if local_free_margin <= 0:
            logger.error(f"[PORTFOLIO] Cannot calculate size for {asset}: {venue} free margin is 0!")
            return 0.0

        # Risk calculation based on local free margin
        risk_percentage = self.portfolio_config.get("target_risk_per_trade", 0.015)
        risk_per_trade = risk_percentage * local_free_margin
        
        # Stop loss distance
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance == 0:
            logger.error(f"[PORTFOLIO] SL distance is 0 for {asset}")
            return 0.0

        # Position size = Risk Amount / Stop Distance %
        stop_distance_pct = sl_distance / entry_price
        position_size_usd = risk_per_trade / stop_distance_pct

        # ✨ STEP 1.5: Confidence-Based Scaling
        # Reason: Increase size for high-conviction signals, decrease for uncertain ones.
        scaling_factor = 1.0
        if confidence > 0.8:
            scaling_factor = 1.2
            logger.info(f"[PORTFOLIO] High confidence ({confidence:.2f}): Scaling size by 1.2x")
        elif confidence < 0.6:
            scaling_factor = 0.7
            logger.info(f"[PORTFOLIO] Low confidence ({confidence:.2f}): Scaling size by 0.7x")
        
        position_size_usd *= scaling_factor

        # Apply USD Correlation Shield
        asset_key = asset.upper()
        active_bucket_trades = sum(
            1 for pos in self.positions.values()
            if pos.asset.upper() in USD_INVERSE_BUCKET
        )

        if asset_key in USD_INVERSE_BUCKET and active_bucket_trades >= 1:
            logger.info(f"[PORTFOLIO] USD Correlation Shield Activated for {asset}. Reducing size.")
            position_size_usd *= 0.5

        # Apply asset weight and limits from config
        asset_weight = self.config["assets"].get(asset, {}).get("weight", 1.0)
        position_size_usd *= asset_weight

        # Hard Cap Check: Never allow a single trade to exceed 50% of the local free margin
        absolute_max = local_free_margin * 0.50
        position_size_usd = min(position_size_usd, absolute_max)

        logger.info(
            f"[PORTFOLIO] {asset} final size: ${position_size_usd:,.2f} "
            f"(Based on {venue} free margin: ${local_free_margin:,.2f}, Confidence Scaling: {scaling_factor}x)"
        )
        return position_size_usd

    def validate_balance_before_trade(self) -> Tuple[bool, str]:
        """
        ✅ NEW: Validate that we have valid balances before opening trades

        Returns:
            (is_valid, error_message)
        """
        if self.is_paper_mode:
            return True, "OK"

        # Force refresh to get latest balances
        if not self.refresh_capital(force=True):
            return False, "Failed to fetch account balances"

        if self.current_capital <= 0:
            return False, f"Invalid capital: ${self.current_capital:,.2f}"

        # Check minimum capital requirements
        min_capital = self.portfolio_config.get("min_capital_threshold", 1000)
        if self.current_capital < min_capital:
            return (
                False,
                f"Capital below minimum: ${self.current_capital:,.2f} < ${min_capital:,.2f}",
            )

        return True, "OK"

    def check_portfolio_limits(
        self, new_position_usd: float, new_side: str = None, asset: str = None
    ) -> bool:
        """
        ✅ FIXED: Check portfolio limits using margin exposure (not notional)
        """
        # Get limits
        max_exposure_pct = self.portfolio_config["max_portfolio_exposure"]
        
        # ✅ SMALL ACCOUNT PROTOCOL: Increase allowed exposure for very small accounts
        # to prevent single positions from locking out the entire bot.
        if self.current_capital < 200:
            # Allow up to 10x leverage equivalent for small accounts
            max_exposure_pct = max(max_exposure_pct, 10.0)
            
        max_exposure_usd = self.current_capital * max_exposure_pct
        
        # ✅ FIXED: Calculate current MARGIN exposure (not notional)
        long_margin = 0.0
        short_margin = 0.0
        
        for pos in self.positions.values():
            notional = pos.quantity * pos.entry_price
            leverage = getattr(pos, 'leverage', 1)
            margin = notional / leverage  # ← Use margin
            
            if pos.side == "long":
                long_margin += margin
            else:
                short_margin += margin
        
        current_gross_margin = long_margin + short_margin
        current_net_margin = abs(long_margin - short_margin)
        
        # ✅ Check hedging configuration
        allow_hedging = self.config.get("trading", {}).get(
            "allow_simultaneous_long_short", False
        )
        
        # Calculate new position margin
        # NOTE: new_position_usd should already be the NOTIONAL value
        # We need to get leverage for this asset to calculate margin
        
        # Get leverage from config (or default to 1)
        if asset:
            asset_cfg = self.config.get("assets", {}).get(asset, {})
            leverage = asset_cfg.get("leverage", 1)
        else:
            leverage = 1
        
        new_position_margin = new_position_usd / leverage
        
        # Check if it's a hedge
        is_hedge = False
        if asset and new_side:
            opposite_side = "short" if new_side == "long" else "long"
            opposite_positions = [
                p for p in self.positions.values()
                if p.asset == asset and p.side == opposite_side
            ]
            is_hedge = len(opposite_positions) > 0
        
        # Use NET margin for hedged strategies, GROSS for directional
        if allow_hedging and (is_hedge or new_side):
            if new_side == "long":
                new_net_margin = abs((long_margin + new_position_margin) - short_margin)
            elif new_side == "short":
                new_net_margin = abs(long_margin - (short_margin + new_position_margin))
            else:
                new_net_margin = current_net_margin + new_position_margin
            
            if new_net_margin > max_exposure_usd:
                logger.warning(
                    f"Portfolio NET margin limit exceeded:\n"
                    f"  Current Long Margin:  ${long_margin:,.2f}\n"
                    f"  Current Short Margin: ${short_margin:,.2f}\n"
                    f"  Current Net Margin:   ${current_net_margin:,.2f}\n"
                    f"  New {new_side or 'position'} (margin): ${new_position_margin:,.2f}\n"
                    f"  New Net Margin:       ${new_net_margin:,.2f}\n"
                    f"  Limit:                ${max_exposure_usd:,.2f}"
                )
                return False
            
            logger.info(
                f"[EXPOSURE] NET MARGIN: ${new_net_margin:,.2f} / ${max_exposure_usd:,.2f} "
                f"({new_net_margin/max_exposure_usd*100:.1f}%)"
                f"{' [HEDGE]' if is_hedge else ''}"
            )
        
        else:
            # Use GROSS margin for directional strategies
            new_gross_margin = current_gross_margin + new_position_margin
            
            if new_gross_margin > max_exposure_usd:
                logger.warning(
                    f"Portfolio GROSS margin limit: "
                    f"${new_gross_margin:,.2f} > ${max_exposure_usd:,.2f}"
                )
                return False
            
            logger.info(
                f"[EXPOSURE] GROSS MARGIN: ${new_gross_margin:,.2f} / ${max_exposure_usd:,.2f} "
                f"({new_gross_margin/max_exposure_usd*100:.1f}%)"
            )
        
        # Check drawdown limit (unchanged)
        drawdown = (
            (self.peak_equity - self.equity) / self.peak_equity
            if self.peak_equity > 0
            else 0
        )
        max_drawdown = self.portfolio_config["max_drawdown"]
        
        if drawdown >= max_drawdown:
            logger.warning(f"Max drawdown: {drawdown:.2%} >= {max_drawdown:.2%}")
            return False
        
        return True

    def get_asset_positions(self, asset: str, side: str = None) -> List[Position]:
        """
        Get all positions for a specific asset

        Args:
            asset: Asset name (e.g., "BTC", "GOLD")
            side: Optional side filter ("long" or "short")

        Returns:
            List of Position objects
        """
        positions = [pos for pos in self.positions.values() if pos.asset == asset]

        if side:
            positions = [pos for pos in positions if pos.side == side]

        return positions

    def get_asset_position_count(self, asset: str, side: str = None) -> int:
        """
        Count open positions for an asset

        Args:
            asset: Asset name
            side: Optional side filter

        Returns:
            Number of open positions
        """
        return len(self.get_asset_positions(asset, side))

    def check_correlation(self, asset1: str, asset2: str) -> float:
        """Calculate correlation between two assets"""
        if not self.portfolio_config["reduce_correlated_positions"]:
            return 0.0

        min_points = 30
        if (
            len(self.price_history.get(asset1, [])) < min_points
            or len(self.price_history.get(asset2, [])) < min_points
        ):
            return 0.0

        returns1 = np.diff(np.log(self.price_history[asset1][-min_points:]))
        returns2 = np.diff(np.log(self.price_history[asset2][-min_points:]))

        correlation = np.corrcoef(returns1, returns2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def should_reduce_position(self, new_asset: str) -> bool:
        """Check if position should be reduced due to correlation"""
        if not self.portfolio_config["reduce_correlated_positions"]:
            return False

        threshold = self.portfolio_config["correlation_threshold"]

        for existing_asset in self.positions.keys():
            if existing_asset != new_asset:
                corr = self.check_correlation(new_asset, existing_asset)
                if abs(corr) > threshold:
                    logger.warning(
                        f"High correlation detected between {new_asset} and {existing_asset}: "
                        f"{corr:.2f}"
                    )
                    return True

        return False

    def _get_asset_bucket(self, asset: str) -> Optional[str]:
        """Identify which exclusion bucket an asset belongs to"""
        asset_upper = asset.upper()
        
        # BTC is explicitly excluded from bucket logic
        if "BTC" in asset_upper:
            return None
            
        # Bucket A: Gold and NAS100
        if any(x in asset_upper for x in ["GOLD", "XAUUSD", "USTEC", "NAS100"]):
            return "A"
            
        # Bucket B: EURJPY and EURUSD
        if any(x in asset_upper for x in ["EURJPY", "EURUSD"]):
            return "B"
            
        return None

    def can_open_position(self, asset: str, side: str) -> Tuple[bool, str]:
        """Check both long and short separately"""
        # STEP 2 — Portfolio-Level Guard
        # Check current open positions for this asset against the cap.
        current_total = self.get_asset_position_count(asset)

        if current_total >= self.max_positions_per_asset:
            logger.info(f"[PORTFOLIO] Max positions reached for {asset}: {current_total}")
            return False, f"Max positions reached for {asset}"

        current_count = self.get_asset_position_count(asset, side)

        if current_count >= self.max_positions_per_asset:
            return False, f"Max {side} positions reached"

        # Check if opposite side exists (if simultaneous trading disabled)
        if not self.config.get("trading", {}).get(
            "allow_simultaneous_long_short", False
        ):
            opposite_side = "short" if side == "long" else "long"
            opposite_count = self.get_asset_position_count(asset, opposite_side)
            if opposite_count > 0:
                return False, f"Have opposite {opposite_side} position"

        return True, "OK"

    @handle_errors(
        component="portfolio_manager",
        severity=ErrorSeverity.ERROR,
        notify=True,
        reraise=False,
        default_return=False,
    )
    def add_position(
        self,
        asset: str,
        symbol: str,
        side: str,
        entry_price: float,
        position_size_usd: float,
        stop_loss: float = None,
        take_profit: float = None,
        trailing_stop_pct: float = None,
        mt5_ticket: int = None,
        binance_order_id: int = None,
        ohlc_data: dict = None,
        use_dynamic_management: bool = True,
        entry_time: datetime = None,
        signal_details: dict = None,
        vtm_overrides: Optional[Dict] = None,
        leverage: int = 1,
        margin_type: str = "CROSSED",
        is_futures: bool = True,
        disable_partials: bool = False,
        min_lot: Optional[float] = None,      # ✨ NEW: Exness compatibility
        lot_precision: Optional[int] = None   # ✨ NEW: Exness compatibility
    ) -> bool:
        """
        Add a new position with hybrid aware VTM support
        """

        # 1. Determine Max Positions allowed for this specific trade
        max_allowed = self.max_positions_per_asset

        # ✅ NEW: Check for Aggregator Override (Ranging Mode)
        if signal_details and signal_details.get("max_trades_override"):
            max_allowed = signal_details["max_trades_override"]

            # Check current open positions for this asset
            current_total = self.get_asset_position_count(asset)

            if current_total >= max_allowed:
                logger.warning(
                    f"[SAFEGUARD] 🛡️ Ranging Mode Active: Max positions capped at {max_allowed}. Cannot open new trade."
                )
                return False

        # ✅ NEW: Check if this is an import from sync
        is_sync_import = signal_details and signal_details.get("source") == "sync_import"
        # ✅ NEW: Check if this is a Small Account Protocol trade
        is_small_account_protocol_trade = signal_details and signal_details.get("small_account_protocol_active", False)

        # 2. Check portfolio exposure limits (SKIP IF IMPORTING OR SMALL ACCOUNT PROTOCOL TRADE)
        if not is_sync_import and not is_small_account_protocol_trade and not self.check_portfolio_limits(
            new_position_usd=position_size_usd, new_side=side, asset=asset
        ):
            logger.warning(f"Portfolio limits exceeded for {asset} {side.upper()}")
            return False

        # ✅ NEW: Log if the check was bypassed due to Small Account Protocol
        if is_small_account_protocol_trade:
            # Re-check the limits just to log the warning, but don't block the trade
            if not self.check_portfolio_limits(
                new_position_usd=position_size_usd, new_side=side, asset=asset
            ):
                logger.warning(
                    f"[SMALL ACCOUNT PROTOCOL] Bypassing portfolio exposure limits for {asset} (Sniper Mode active)."
                )

        # ✅ NEW: Log if the check was bypassed for sync import
        elif is_sync_import:
            # Re-check the limits just to log the warning, but don't block the trade
            if not self.check_portfolio_limits(
                new_position_usd=position_size_usd, new_side=side, asset=asset
            ):
                logger.warning(
                    f"[SYNC IMPORT] Bypassing portfolio exposure limits to import existing position for {asset}."
                )

        quantity = position_size_usd / entry_price
        logger.info(
            f"[PORTFOLIO] Adding {side.upper()} position:\n"
            f"  Size:     ${position_size_usd:,.2f}\n"
            f"  Quantity: {quantity:.8f}\n"
            f"  Entry:    ${entry_price:,.2f}"
        )

        # ============================================================================
        # 3. CREATE POSITION OBJECT
        # ✅ VTM is initialized INSIDE Position.__init__() - don't do it here!
        # ============================================================================
        risk_config = self.config.get("assets", {}).get(asset, {}).get("risk", {})
        
        position = Position(
            asset=asset,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=entry_time or datetime.now(),
            risk_config=risk_config,
            signal_details=signal_details,
            stop_loss=stop_loss,  # May be None if VTM will calculate
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            mt5_ticket=mt5_ticket,
            binance_order_id=binance_order_id,
            ohlc_data=ohlc_data,
            account_balance=self.current_capital,
            use_dynamic_management=use_dynamic_management,  # ← This triggers VTM init in Position.__init__()
            vtm_overrides=vtm_overrides,
            leverage=leverage,
            margin_type=margin_type,
            is_futures=is_futures,
            disable_partials=disable_partials,
            min_lot=min_lot,
            lot_precision=lot_precision
        )
        if use_dynamic_management and ohlc_data:
            if position.trade_manager:
                logger.info(
                    f"[VTM] Initialized for {asset}. SL: ${position.stop_loss:,.2f}"
                )
                # Pyramid scale-in positions must not pyramid themselves again — the
                # entry price is already inside a running trend, so profit >= 1 ATR
                # can be true almost immediately, causing a cascade of new positions.
                if signal_details and signal_details.get("is_pyramid_scale_in"):
                    position.trade_manager.has_pyramided = True
                    logger.info(f"[VTM] Pyramid scale-in — has_pyramided locked True for {asset}")
            else:
                logger.warning(f"[VTM] Failed to initialize for {asset}")

        # ============================================================================
        # 5. STORE POSITION
        # ============================================================================
        self.positions[position.position_id] = position

        # 6. Database Logging
        if self.db_manager:
            try:
                # ✅ TASK 25: Extract granular regime fields for trade record
                regime = signal_details.get("regime", "UNKNOWN") if signal_details else "UNKNOWN"
                quality = signal_details.get("signal_quality", 0.0) if signal_details else 0.0
                confidence = signal_details.get("mode_confidence", 0.0) if signal_details else 0.0

                trade_id, is_new = self.db_manager.insert_trade_entry(
                    asset=asset,
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    quantity=position.quantity,
                    position_size_usd=position_size_usd,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    position_id=position.position_id,
                    exchange=self.config["assets"][asset].get("exchange", "binance"),
                    strategy=signal_details.get("trade_type", "TREND") if signal_details else "TREND",
                    regime=regime,
                    signal_quality=quality,
                    confidence_score=confidence,
                    mt5_ticket=mt5_ticket,
                    binance_order_id=binance_order_id,
                    vtm_enabled=bool(position.trade_manager),
                    metadata={
                        "trailing_stop_pct": trailing_stop_pct,
                        "entry_time": position.entry_time.isoformat(),
                        "preset_used": (
                            signal_details.get("preset_config", {}).get(
                                "name", "default"
                            )
                            if signal_details
                            else "manual"
                        ),
                    },
                )

                position.db_trade_id = trade_id
                position.db_manager = self.db_manager

                if is_new:
                    logger.debug(f"[DB] New trade created: {trade_id}")

            except Exception as e:
                logger.error(f"[DB] Error logging trade entry: {e}")

        # 7. Final Logging
        current_count = self.get_asset_position_count(asset, side)
        
        # Standardized Log (ENTRY)
        log_trade_event("ENTRY", {
            "symbol": symbol,
            "asset": asset,
            "side": side,
            "price": entry_price,
            "quantity": quantity,
            "trade_type": signal_details.get("trade_type", "TREND") if signal_details else "TREND",
            "position_id": position.position_id
        })

        logger.info(
            f"✓ Position #{current_count} opened: {asset} {side.upper()} "
            f"@ ${entry_price:,.2f} | Size: ${position_size_usd:,.2f} "
            f"| ID: {position.position_id}"
        )

        return True

    def update_positions_with_ohlc(self, ohlc_data_dict: dict):
        """
        Update all positions with new OHLC data for dynamic management

        Args:
            ohlc_data_dict: Dict with asset keys and {'high': float, 'low': float, 'close': float} values

        Example:
            portfolio_manager.update_positions_with_ohlc({
                'BTC': {'high': 45000, 'low': 44500, 'close': 44800},
                'GOLD': {'high': 2050, 'low': 2045, 'close': 2048}
            })
        """
        positions_to_close = []   # (asset, exit_price, reason, size)

        for asset, position in list(self.positions.items()):
            if asset not in ohlc_data_dict:
                continue

            ohlc = ohlc_data_dict[asset]

            try:
                # Update position with new bar
                exit_signal = position.update_with_new_bar(
                    high=ohlc["high"], low=ohlc["low"], close=ohlc["close"]
                )

                # If VTM signals exit or action, handle it
                if exit_signal:
                    # ✅ T29: Handle Pyramid routing
                    if isinstance(exit_signal, dict) and exit_signal.get('action') == 'pyramid':
                        logger.info(f"[PYRAMID] 🗼 Triggered for {asset} {position.side}")

                        entry_price = ohlc['close']
                        position_size_usd = exit_signal['new_size'] * entry_price

                        # Inherit critical context from parent position
                        parent_ohlc = {
                            "high": position.trade_manager.high,
                            "low": position.trade_manager.low,
                            "close": position.trade_manager.close,
                            "volume": position.trade_manager.volume
                        }

                        self.add_position(
                            asset=asset,
                            symbol=position.symbol,
                            side=position.side,
                            entry_price=entry_price,
                            position_size_usd=position_size_usd,
                            ohlc_data=parent_ohlc,
                            signal_details={
                                **(getattr(position, 'signal_details', {}) or {}),
                                "source": "pyramid_trigger",
                                "parent_id": position.position_id,
                                "trade_type": "TREND" # Pyramiding is a trend behavior
                            },
                            leverage=getattr(position, 'leverage', 1),
                            margin_type=getattr(position, 'margin_type', "CROSSED"),
                            is_futures=getattr(position, 'is_futures', True)
                        )
                        continue # Do NOT close the current position

                    # ✅ Partial vs full exit routing
                    if isinstance(exit_signal, dict):
                        reason_str = exit_signal.get("reason", "unknown")
                        exit_price  = exit_signal.get("price", ohlc["close"])
                        exit_size   = exit_signal.get("size", 1.0)
                    else:
                        # Legacy string signal (fallback) — treat as full exit
                        reason_str  = str(exit_signal)
                        exit_price  = ohlc["close"]
                        exit_size   = 1.0

                    logger.info(
                        f"[VTM] {asset} triggered {reason_str.upper()} "
                        f"@ ${exit_price:,.2f} (size={exit_size:.0%})"
                    )
                    positions_to_close.append((asset, exit_price, reason_str, exit_size))

            except Exception as e:
                logger.error(f"[VTM] Error updating {asset}: {e}")

        # Close / partially-close positions that received exit signals
        for asset, exit_price, reason, exit_size in positions_to_close:
            if exit_size < 0.999:
                # Partial exit — close fraction, keep position alive
                self.partial_close_position(
                    asset=asset, partial_fraction=exit_size,
                    exit_price=exit_price, reason=f"VTM_{reason}"
                )
            else:
                self.close_position(
                    asset=asset, exit_price=exit_price, reason=f"VTM_{reason}"
                )

        return len(positions_to_close)

    def get_asset_balance(self, asset: str) -> float:
        """
        Get balance for a specific asset's exchange

        Args:
            asset: "BTC" or "GOLD"

        Returns:
            Balance in USD for that asset's exchange
        """
        if self.is_paper_mode:
            # In paper mode, split capital proportionally
            if asset == "BTC":
                return self.paper_capital * 0.9  # 90% for BTC
            elif asset == "GOLD":
                return self.paper_capital * 0.1  # 10% for Gold
            return self.paper_capital

        # Live mode - fetch from specific exchange
        asset_cfg = self.config.get("assets", {}).get(asset, {})
        exchange = asset_cfg.get("exchange", "binance").lower()
        
        if exchange == "mt5":
            balance = self._fetch_mt5_balance()
            if balance:
                logger.debug(f"[ASSET BALANCE] {asset}: ${balance:,.2f} (MT5)")
                return balance
        else: # binance
            balance = self._fetch_binance_balance()
            if balance:
                logger.debug(f"[ASSET BALANCE] {asset}: ${balance:,.2f} (Binance)")
                return balance

        # Fallback: use proportion of total capital
        logger.warning(
            f"[ASSET BALANCE] Could not fetch {asset} balance, using estimate"
        )

        # Estimate based on asset weight in config
        asset_weight = self.config["assets"].get(asset, {}).get("weight", 0.5)
        return self.current_capital * asset_weight

    def emergency_close_all(self):
        """
        🚨 EMERGENCY: Close ALL open positions immediately.
        Used when system health is compromised.
        """
        logger.critical("[EMERGENCY] Triggering global position exit due to system instability!")
        
        # We'll use the existing close_all_positions logic but with an emergency reason
        self.close_all_positions(reason="emergency_system_failure")
        
        if self.telegram_bot:
            asyncio.create_task(
                self.telegram_bot.notify_error("🚨 *EMERGENCY HALT*\nAll positions have been force-closed due to system instability!")
            )

    def close_all_positions(self, prices: Dict[str, float] = None, reason: str = "manual_close_all"):
        """
        ✅ FIXED: Close all open positions
        Fixes bug where exit_price was being interpreted as position_id
        """
        logger.info(f"Closing all positions (Reason: {reason})...")

        # Use list() to create a copy of keys since we modify dict during iteration
        position_ids = list(self.positions.keys())

        for pid in position_ids:
            if pid not in self.positions:
                continue

            position = self.positions[pid]
            asset_name = position.asset

            # Get correct exit price using ASSET name
            exit_price = (
                prices.get(asset_name, position.entry_price)
                if prices
                else position.entry_price
            )

            # ✅ FIX: Use keyword arguments to ensure data goes to correct parameters
            self.close_position(
                position_id=pid,  # ← String position ID
                exit_price=exit_price,  # ← Float price
                reason=reason,
            )

        logger.info("All positions closed")

    def close_all_positions_for_asset(
        self, asset: str, exit_price: float = None, reason: str = "manual_close_asset"
    ) -> List[Dict]:
        """
        Close ALL open positions for a specific asset (e.g., "GOLD").
        Used by Telegram /close commands.

        Strategy:
          1. Close any internally-tracked positions via the normal pipeline.
          2. If nothing is tracked, fall back to scanning the exchange directly
             for orphaned positions (e.g. opened/closed manually on MT5 terminal).
          3. Returns a sentinel {"already_closed": True} entry in the list when
             the position no longer exists on either system — so callers can
             distinguish "already closed" from "failed to close".

        Args:
            asset: Asset name (e.g., "BTC", "GOLD")
            exit_price: Exit price (optional, will fetch if not provided)
            reason: Close reason

        Returns:
            List of trade result dicts.  May include {"already_closed": True}
            to signal that the position was gone before this call.
        """
        logger.info(f"[CLOSE-ALL] Closing ALL positions for {asset}...")

        # ── Path 1: internally-tracked positions ────────────────────────────
        positions_to_close = [p for p in self.positions.values() if p.asset == asset]

        results = []
        for pos in positions_to_close:
            current_exit_price = exit_price

            if current_exit_price is None:
                if hasattr(self, "mt5_handler") and pos.mt5_ticket:
                    try:
                        import MetaTrader5 as mt5
                        tick = mt5.symbol_info_tick(pos.symbol)
                        if tick:
                            current_exit_price = (tick.ask + tick.bid) / 2
                    except Exception:
                        pass

                if current_exit_price is None:
                    current_exit_price = pos.entry_price

            result = self.close_position(
                position_id=pos.position_id,
                exit_price=current_exit_price,
                reason=reason,
            )
            if result:
                results.append(result)

        if results:
            logger.info(
                f"[CLOSE-ALL] Closed {len(results)}/{len(positions_to_close)} "
                f"tracked positions for {asset}"
            )
            return results

        # ── Path 2: nothing tracked — scan exchange for orphans ─────────────
        asset_cfg = self.config.get("assets", {}).get(asset, {})
        exchange = asset_cfg.get("exchange", "binance")
        symbol = self._resolve_symbol(asset)

        if exchange == "mt5" and symbol and self.mt5_handler and not self.is_paper_mode:
            try:
                import MetaTrader5 as _mt5
                live_positions = _mt5.positions_get(symbol=symbol)
                if live_positions:
                    logger.info(
                        f"[CLOSE-ALL] Found {len(live_positions)} orphaned MT5 position(s) "
                        f"for {asset} ({symbol}) — closing directly"
                    )
                    for mt5_pos in live_positions:
                        ticket = mt5_pos.ticket
                        side = "long" if mt5_pos.type == _mt5.POSITION_TYPE_BUY else "short"
                        close_result = self.mt5_handler._close_mt5_order(ticket, asset, side)
                        if close_result:
                            pnl = float(getattr(mt5_pos, "profit", 0.0) or 0.0)
                            fill_price = (
                                close_result.get("fill_price")
                                if isinstance(close_result, dict)
                                else mt5_pos.price_current
                            )
                            results.append({
                                "asset": asset,
                                "side": side,
                                "entry_price": mt5_pos.price_open,
                                "exit_price": fill_price or mt5_pos.price_current,
                                "quantity": mt5_pos.volume,
                                "pnl": (
                                    close_result.get("profit", pnl)
                                    if isinstance(close_result, dict)
                                    else pnl
                                ),
                                "pnl_pct": 0.0,
                                "mt5_ticket": ticket,
                                "reason": reason,
                                "orphan_close": True,
                            })
                            logger.info(
                                f"[CLOSE-ALL] ✅ Orphan MT5 ticket #{ticket} closed "
                                f"(side={side}, pnl=${pnl:,.2f})"
                            )
                        else:
                            logger.error(
                                f"[CLOSE-ALL] ❌ Failed to close orphan MT5 ticket #{ticket}"
                            )
                    if results:
                        return results
                else:
                    # Nothing on the exchange either — position is already gone
                    logger.info(
                        f"[CLOSE-ALL] No open positions on MT5 for {asset} ({symbol}). "
                        f"Position was already closed externally."
                    )
                    return [{"already_closed": True, "asset": asset}]

            except Exception as e:
                logger.error(f"[CLOSE-ALL] MT5 orphan-scan failed for {asset}: {e}")

        # Nothing tracked, exchange scan not applicable or failed
        if not positions_to_close:
            logger.warning(f"[CLOSE-ALL] No tracked positions for {asset}. "
                           f"Position may already be closed.")
            return [{"already_closed": True, "asset": asset}]

        logger.warning(f"[CLOSE-ALL] {len(positions_to_close)} tracked but 0 successfully closed for {asset}")
        return []

    def partial_close_position(
        self,
        asset: str,
        partial_fraction: float,
        exit_price: float,
        reason: str = "VTM_take_profit",
    ) -> Optional[Dict]:
        """
        Close a fractional portion of a position (VTM partial TP exits).

        Closes `partial_fraction` (e.g. 0.45) of the current quantity on the exchange,
        reduces position.quantity accordingly, and records the partial P&L — without
        removing the position from the portfolio so VTM can continue managing the remainder.
        """
        positions = self.get_asset_positions(asset)
        if not positions:
            logger.warning(f"[PARTIAL] No open position for {asset}")
            return None

        position = positions[0]

        if partial_fraction <= 0 or partial_fraction >= 1.0:
            logger.warning(f"[PARTIAL] Invalid fraction {partial_fraction:.2%} for {asset} — doing full close")
            return self.close_position(asset=asset, exit_price=exit_price, reason=reason)

        partial_qty  = position.quantity * partial_fraction
        partial_pnl  = (exit_price - position.entry_price) * partial_qty if position.side == "long" \
                       else (position.entry_price - exit_price) * partial_qty
        partial_pnl_pct = partial_pnl / (position.entry_price * position.quantity) if position.entry_price * position.quantity > 0 else 0

        # ── Send partial close to exchange ──────────────────────────────────
        exchange_ok = False
        if not self.is_paper_mode:
            asset_cfg  = self.config["assets"].get(position.asset, {})
            exchange   = asset_cfg.get("exchange", "binance")
            handler    = self.execution_handlers.get(exchange)

            if handler and hasattr(handler, "_partial_close_position"):
                try:
                    exchange_ok = handler._partial_close_position(
                        position=position,
                        partial_qty=partial_qty,
                        current_price=exit_price,
                        asset_name=asset,
                        reason=reason,
                    )
                except Exception as e:
                    logger.error(f"[PARTIAL] Handler error for {asset}: {e}", exc_info=True)
            else:
                logger.warning(f"[PARTIAL] Handler for {exchange} has no _partial_close_position — falling back to full close")
                return self.close_position(asset=asset, exit_price=exit_price, reason=reason)
        else:
            exchange_ok = True  # Paper mode always succeeds

        if not exchange_ok:
            logger.error(f"[PARTIAL] Exchange rejected partial close for {asset} — position unchanged")
            return None

        # ── Update portfolio position ────────────────────────��───────────────
        position.quantity -= partial_qty
        logger.info(
            f"[PARTIAL] {asset} {position.side.upper()} — closed {partial_fraction:.0%} "
            f"@ ${exit_price:,.2f} | P&L: ${partial_pnl:,.2f} | "
            f"Remaining qty: {position.quantity:.6f}"
        )

        # ── Record partial P&L ───────────────────────────────────────────────
        self.realized_pnl_today += partial_pnl
        if partial_pnl < 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0
            self._loss_streak_alerted = False

        trade_type = getattr(position, 'trade_type', 'TREND')
        self.performance_tracker.record_trade(trade_type, partial_pnl)

        # ── DB log ──────────────────────────────────────────���────────────────
        if self.db_manager and hasattr(position, "db_trade_id") and position.db_trade_id:
            try:
                self.db_manager.update_trade_vtm_event(
                    trade_id=position.db_trade_id,
                    event_type="partial_close",
                    current_price=exit_price,
                    metadata={
                        "partial_fraction": partial_fraction,
                        "partial_qty": partial_qty,
                        "partial_pnl": partial_pnl,
                        "reason": reason,
                    },
                )
            except Exception as e:
                logger.warning(f"[PARTIAL] DB log failed for {asset}: {e}")

        log_trade_event("TP_HIT", {
            "symbol": position.symbol,
            "asset": position.asset,
            "side": position.side,
            "price": exit_price,
            "quantity": partial_qty,
            "trade_type": trade_type,
            "reason": reason,
            "pnl": partial_pnl,
            "pnl_pct": partial_pnl_pct,
            "partial_fraction": partial_fraction,
            "position_id": position.position_id,
        })

        if self.telegram_bot and self.telegram_bot._current_loop:
            try:
                import asyncio
                asyncio.run_coroutine_threadsafe(
                    self.telegram_bot.notify_trade_closed(
                        asset=asset,
                        side=position.side,
                        pnl=partial_pnl,
                        pnl_pct=partial_pnl_pct * 100,
                        reason=reason,
                        partial=True,
                        partial_pct=partial_fraction * 100,
                    ),
                    self.telegram_bot._current_loop
                )
            except Exception:
                pass

        return {
            "asset": asset,
            "side": position.side,
            "exit_price": exit_price,
            "partial_fraction": partial_fraction,
            "partial_pnl": partial_pnl,
            "partial_pnl_pct": partial_pnl_pct,
            "reason": reason,
        }

    @handle_errors(
        component="portfolio_manager",
        severity=ErrorSeverity.ERROR,
        notify=True,
        reraise=False,
        default_return=None,
    )
    def close_position(
        self,
        asset: str = None,
        position_id: str = None,
        exit_price: float = None,
        reason: str = "manual",
    ) -> Optional[Dict]:
        """
        ✅ FIXED: Close position - Validates exchange close before removing from portfolio
        """
        # Find position to close
        if position_id:
            position = self.positions.get(position_id)
            if not position:
                logger.warning(f"Position {position_id} not found in portfolio")
                return None
        elif asset:
            positions = self.get_asset_positions(asset)
            if not positions:
                logger.warning(f"No positions to close for {asset}")
                return None
            position = positions[0]
            position_id = position.position_id
        else:
            logger.error("Must provide either asset or position_id")
            return None

        # Check if position is already being closed
        now = datetime.now()
        if position.closing:
            # If it's been "closing" for more than 60 seconds, assume it's stuck and allow retry
            if position.last_close_attempt and (now - position.last_close_attempt).total_seconds() > 60:
                logger.warning(f"Position {position_id} stuck in 'closing' for >60s. Overriding to allow retry.")
            else:
                logger.info(f"Position {position_id} is already in the process of being closed. Skipping.")
                return None
        
        if position.last_close_attempt:
            # Add a 30-second cooldown to prevent hammer on failing close attempts
            # (Unless we just overrode the "stuck" status above)
            if not position.closing and (now - position.last_close_attempt).total_seconds() < 30:
                logger.debug(f"Position {position_id} close attempt on cooldown. Skipping.")
                return None
            
        # Mark the position as closing and record attempt time
        position.closing = True
        position.last_close_attempt = now

        # ================================================================
        # ✅ STEP 1: CLOSE ON EXCHANGE FIRST (MT5 or Binance)
        # ================================================================
        exchange_closed = False
        close_error_msg = "Unknown handler error" # Default error message
        broker_close_data = None  # holds dict with authoritative fill+profit

        if not self.is_paper_mode:
            asset_cfg = self.config["assets"].get(position.asset, {})
            exchange = asset_cfg.get("exchange", "binance")
            handler = self.execution_handlers.get(exchange)

            if not handler:
                close_error_msg = f"{exchange.upper()} handler not available"
            else:
                try:
                    logger.info(f"[{exchange.upper()}] Attempting to close position {position.position_id}...")
                    handler_result = handler._close_position(
                        position=position,
                        current_price=exit_price,
                        asset_name=position.asset,
                        reason=reason,
                    )
                    # Handler may now return a dict with broker fill data, or a
                    # bare True/False for legacy/paper paths.
                    if isinstance(handler_result, dict):
                        broker_close_data = handler_result
                        exchange_closed = bool(handler_result.get("ok", True))
                    else:
                        exchange_closed = bool(handler_result)
                    if not exchange_closed:
                        close_error_msg = f"{exchange.upper()} order was rejected or failed. Check handler logs."

                except Exception as e:
                    close_error_msg = f"Handler exception: {str(e)}"
                    logger.error(f"[{exchange.upper()}] Error closing position: {e}", exc_info=True)
        else:
            # Paper mode always succeeds
            exchange_closed = True

        # ================================================================
        # ✅ STEP 2: ABORT OR PROCEED
        # ================================================================
        if not exchange_closed:
            logger.error(
                f"[CRITICAL] Position close failed on exchange!\n"
                f"  Position ID: {position_id}\n"
                f"  Asset:       {position.asset}\n"
                f"  Error:       {close_error_msg}\n"
                f"  ⚠️  Resetting 'closing' flag to allow future attempts."
            )
            # RESET THE FLAG so we can try again later
            position.closing = False
            return None

        # ================================================================
        # ✅ STEP 3: CALCULATE P&L (Only if exchange close succeeded)
        # ────────────────────────────────────────────────────────────────
        # Prefer the broker's authoritative numbers when we have them. The
        # local Python calc uses a cached/stale `exit_price` that can diverge
        # significantly from the actual fill (we've seen 50¢ on USOIL = $5+
        # under-reported), and it ignores swap + commission entirely.
        # ================================================================
        broker_fill_price = None
        broker_profit = None
        broker_swap = 0.0
        broker_commission = 0.0
        if broker_close_data:
            broker_fill_price = broker_close_data.get("fill_price")
            broker_profit = broker_close_data.get("profit")
            broker_swap = broker_close_data.get("swap", 0.0) or 0.0
            broker_commission = broker_close_data.get("commission", 0.0) or 0.0

        # If broker reported an actual fill price, use it as the canonical exit_price.
        if broker_fill_price is not None and broker_fill_price > 0:
            if exit_price and abs(broker_fill_price - exit_price) / max(abs(exit_price), 1e-9) > 0.0005:
                logger.warning(
                    f"[CLOSE] Stale-cache exit drift detected for {position.asset}: "
                    f"cached ${exit_price:,.5f} vs broker fill ${broker_fill_price:,.5f}. "
                    f"Using broker fill for P&L."
                )
            exit_price = broker_fill_price

        if broker_profit is not None:
            pnl = float(broker_profit) + float(broker_swap) + float(broker_commission)
            # Derive % from broker profit against the entry notional we tracked
            entry_notional = position.entry_price * position.quantity if position.entry_price else 0.0
            pnl_pct = (pnl / entry_notional) if entry_notional > 0 else 0.0
            logger.info(
                f"[CLOSE] Using BROKER P&L for {position.asset}: "
                f"${pnl:,.2f} (profit ${broker_profit:,.2f}, swap ${broker_swap:,.2f}, "
                f"commission ${broker_commission:,.2f})"
            )
        else:
            pnl = position.get_pnl(exit_price)
            pnl_pct = position.get_pnl_pct(exit_price)
            logger.debug(
                f"[CLOSE] Broker P&L unavailable for {position.asset}; "
                f"falling back to local calc using exit ${exit_price:,.5f}"
            )

        self.realized_pnl_today += pnl

        # ✨ NEW: Record strategy performance
        trade_type = getattr(position, 'trade_type', 'TREND')
        self.performance_tracker.record_trade(trade_type, pnl)

        # ✨ NEW: Track consecutive losses
        if pnl < 0:
            self.loss_streak += 1
            logger.warning(f"[STREAK] Loss streak incremented: {self.loss_streak}")
        else:
            if self.loss_streak > 0:
                logger.info(f"[STREAK] Loss streak of {self.loss_streak} reset to 0.")
            self.loss_streak = 0
            self._loss_streak_alerted = False  # Reset alert guard when streak clears

        # Update capital
        if self.is_paper_mode:
            self.current_capital += pnl
            self.equity = self.current_capital
        else:
            self.refresh_capital()

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # ================================================================
        # ✅ STEP 4: CREATE TRADE RESULT
        # ================================================================
        trade_result = {
            "asset": asset or position.asset,
            "position_id": position_id,
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "entry_time": position.entry_time,
            "exit_time": datetime.now(),
            "holding_time": (datetime.now() - position.entry_time).total_seconds() / 3600,
            "reason": reason,
            "mt5_ticket": position.mt5_ticket,
            "binance_order_id": position.binance_order_id,
            "exchange_closed": exchange_closed,
        }

        # ================================================================
        # ✅ STEP 5: LOG TO DATABASE
        # ================================================================
        if self.db_manager and hasattr(position, "db_trade_id") and position.db_trade_id:
            try:
                holding_time = (datetime.now() - position.entry_time).total_seconds() / 3600
                self.db_manager.update_trade_exit(
                    trade_id=position.db_trade_id,
                    exit_price=exit_price,
                    exit_reason=reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_time_hours=holding_time,
                    final_quantity=position.quantity,
                    metadata={"exit_time": datetime.now().isoformat(), "exchange_closed": exchange_closed},
                )
                logger.debug(f"[DB] Trade exit logged: {position.db_trade_id}")
            except Exception as e:
                logger.error(f"[DB] Error logging trade exit: {e}")

        # ================================================================
        # ✅ STEP 6: REMOVE FROM PORTFOLIO
        # ================================================================
        self.closed_positions.append(trade_result)
        # ✨ MEMORY MANAGEMENT: Limit to 100 entries
        if len(self.closed_positions) > 100:
            self.closed_positions.pop(0)

        # ✅ RACE-CONDITION GUARD: reconciliation loop may have already removed
        # this key from self.positions (broker reported 0 positions before we
        # reached this line).  Use pop() instead of del so a KeyError never
        # fires after a successful exchange close.
        if self.positions.pop(position_id, None) is None:
            logger.debug(
                f"[CLOSE] {position_id} already removed from portfolio by "
                f"reconciliation — exchange close still completed successfully."
            )

        # Track whether this close was manual (Telegram / force-close) so the
        # cooldown bypass in check_min_time_between_trades() works correctly.
        _reason_str = str(reason).lower()
        _is_manual = any(k in _reason_str for k in ("manual", "telegram", "force", "user"))
        self.last_close_was_manual[position.asset] = _is_manual

        remaining_count = self.get_asset_position_count(position.asset, position.side)

        # Standardize exit reason for logger
        exit_event_type = "EXIT"
        if "stop_loss" in str(reason).lower():
            exit_event_type = "SL_HIT"
        elif "take_profit" in str(reason).lower():
            exit_event_type = "TP_HIT"

        # ✅ Standardized Log
        log_trade_event(exit_event_type, {
            "symbol": position.symbol,
            "asset": position.asset,
            "side": position.side,
            "price": exit_price,
            "quantity": position.quantity,
            "trade_type": getattr(position, 'trade_type', 'UNKNOWN'),
            "reason": reason,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "position_id": position_id
        })

        logger.info(
            f"✓ Position closed successfully:\n"
            f"  Asset:     {position.asset} {position.side.upper()}\n"
            f"  Exit:      ${exit_price:,.2f}\n"
            f"  P&L:       ${pnl:,.2f} ({pnl_pct:.2%})\n"
            f"  Remaining: {remaining_count}/{self.max_positions_per_asset}"
        )

        # Send notification
        if self.telegram_bot and self.telegram_bot._current_loop:
            try:
                # Use a thread-safe method to call the async notification
                asyncio.run_coroutine_threadsafe(
                    self.telegram_bot.notify_trade_closed(
                        asset=trade_result["asset"],
                        side=trade_result["side"],
                        pnl=trade_result["pnl"],
                        pnl_pct=trade_result["pnl_pct"] * 100, # Convert to percentage points
                        reason=trade_result["reason"],
                    ),
                    self.telegram_bot._current_loop
                )
            except Exception as e:
                logger.error(f"[TELEGRAM] Failed to send close notification from PM: {e}")

        return trade_result

    def reconcile_positions(self, asset: str, broker_positions: List[Dict]):
        """
        ✅ RECONCILIATION: Ensure local positions match broker reality.
        Syncs local state if mismatches are detected.
        """
        local_positions = self.get_asset_positions(asset)
        
        # 1. Check for orphaned local positions (exist here but not on broker)
        # We check by specific exchange ID if available
        broker_ids = [str(p.get('id')) for p in broker_positions if p.get('id')]
        
        for pos in list(local_positions):
            # A. If we have a specific exchange ID, use it for exact matching
            local_exchange_id = str(pos.mt5_ticket) if pos.mt5_ticket else str(pos.binance_order_id)
            
            if local_exchange_id != "None" and broker_ids:
                if local_exchange_id not in broker_ids:
                    logger.error(f"[RECONCILE] Orphaned ID found: {pos.position_id} (ID: {local_exchange_id}). Removing.")
                    self.positions.pop(pos.position_id, None)
                    continue

            # B. If no IDs (rare) or for Binance where positions are aggregated by side
            # Check if at least one broker position exists for this side
            broker_sides = [p.get('side').lower() for p in broker_positions if p.get('side')]
            if pos.side.lower() not in broker_sides:
                logger.error(f"[RECONCILE] Orphaned SIDE found: {pos.position_id} ({pos.side.upper()}). No match on broker. Removing.")
                self.positions.pop(pos.position_id, None)

        # 2. Final check: If broker has 0 positions but we still have local ones, clear them
        if not broker_positions and local_positions:
            logger.warning(f"[RECONCILE] Broker reports 0 positions for {asset}. Clearing local state.")
            for pos in local_positions:
                self.positions.pop(pos.position_id, None)

    @handle_errors(
        component="portfolio_manager",
        severity=ErrorSeverity.WARNING,
        notify=False,  # Don't notify for update errors
        reraise=False,
        default_return=None,
    )
    def update_positions(self, prices: Dict[str, float] = None):
        """Update all positions with current prices and exchange profit"""
        # Update exchange positions with real-time profit
        if not self.is_paper_mode:
            self.update_mt5_positions_profit()
            self.update_binance_positions_profit()

        if prices:
            for asset, price in prices.items():
                if asset in self.price_history:
                    self.price_history[asset].append(price)
                    # ✨ MEMORY MANAGEMENT: Limit to 500 entries (Safe for 200 EMA + buffer)
                    if len(self.price_history[asset]) > 500:
                        self.price_history[asset].pop(0)

        # Calculate unrealized P&L
        total_unrealized_pnl = 0.0
        for pos in self.positions.values():
            # Prioritize exchange-reported profit
            if pos.mt5_ticket and pos.mt5_profit != 0.0:
                # Use MT5 profit for MT5 positions
                total_unrealized_pnl += pos.mt5_profit
            elif pos.binance_order_id and pos.binance_profit != 0.0:
                # Use Binance profit for Binance positions
                total_unrealized_pnl += pos.binance_profit
            elif prices and pos.asset in prices:
                # Calculate for positions without exchange tracking
                total_unrealized_pnl += pos.get_pnl(prices[pos.asset])

        if self.is_paper_mode:
            # In paper mode: equity = cash + unrealized P&L
            self.equity = self.current_capital + total_unrealized_pnl
        else:
            # In live mode: periodically refresh from exchanges
            pass

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def get_open_positions_count(self) -> int:
        """Get number of open positions"""
        return len(self.positions)

    def get_position(self, asset: str, position_id: str = None) -> Optional[Position]:
        """
        Get position(s) for an asset

        Args:
            asset: Asset name
            position_id: Optional specific position ID

        Returns:
            Position object if position_id provided, otherwise first position for asset
        """
        if position_id:
            return self.positions.get(position_id)

        # Return first position for asset (for backward compatibility)
        positions = self.get_asset_positions(asset)
        return positions[0] if positions else None

    def has_position(self, asset: str, side: str = None) -> bool:
        """
        Check if we have any open positions for an asset

        Args:
            asset: Asset symbol
            side: Optional side filter ('long' or 'short')
        """
        return self.get_asset_position_count(asset, side) > 0

    def reset_daily_pnl(self):
        """Reset realized P&L tracker (call this at start of each trading day)"""
        self.realized_pnl_today = 0.0
        logger.info("Daily P&L tracker reset")

    def start_trading_session(self):
        """Start trading session.

        session_start_equity is intentionally set to the CURRENT margin balance
        (wallet + unrealized PnL) so that positions imported from a prior session
        don't create a phantom loss when they are closed: their unrealized P&L is
        already baked into the baseline, and the circuit-breaker only fires on
        NEW losses incurred during this session.

        ✅ FIX: Force-refresh the real broker balance BEFORE locking in
        session_start_equity.  Without this, switching from a large demo/Binance
        account to a small live MT5 account causes the circuit-breaker to fire
        immediately (99% phantom drawdown) because session_start_equity is set
        from the previous session's stale in-memory equity value.
        """
        self.session_start_time = datetime.now()
        if not self.is_paper_mode:
            try:
                self.refresh_capital(force=True)
            except Exception as _e:
                logger.warning(f"[SESSION] Could not refresh capital before session start: {_e}")
        self.session_start_equity = self.equity
        self.session_start_capital = self.current_capital
        self.realized_pnl_today = 0.0
        # Clear any manual /resume override at the start of a new trading day
        if getattr(self, "_circuit_breaker_override", False):
            self._circuit_breaker_override = False
            logger.info("[SESSION] Circuit-breaker manual override cleared for new session.")
        open_count = len(self.positions)
        logger.info(
            f"Trading session started at {self.session_start_time}\n"
            f"  Session-start equity: ${self.session_start_equity:,.2f}"
            + (f"  ({open_count} imported position(s) already priced in)" if open_count else "")
        )

    def get_portfolio_status(self, current_prices: Dict[str, float] = None) -> Dict:
        """
        ✅ FIX: Auto-refresh balances when getting status
        """
        # ✅ Refresh if stale (respects time interval)
        self.refresh_capital(force=False)

        if current_prices is None:
            current_prices = {
                pos.asset: pos.entry_price for pos in self.positions.values()
            }

        total_exposure = 0.0
        total_unrealized_pnl = 0.0
        
        total_notional_value = 0.0
        total_margin_used = 0.0

        # ✅  Count positions per asset correctly
        asset_position_counts = {}
        asset_positions_detail = {}

        # Get all enabled assets from config
        enabled_assets = [a for a, cfg in self.config["assets"].items() if cfg.get("enabled", False)]

        for asset in enabled_assets:
            # Get all positions for this asset
            long_positions = [
                p
                for p in self.positions.values()
                if p.asset == asset and p.side == "long"
            ]
            short_positions = [
                p
                for p in self.positions.values()
                if p.asset == asset and p.side == "short"
            ]

            asset_position_counts[asset] = {
                "long": len(long_positions),
                "short": len(short_positions),
                "total": len(long_positions) + len(short_positions),
            }

            # Detailed info for debugging
            asset_positions_detail[asset] = {
                "long_ids": [p.position_id for p in long_positions],
                "short_ids": [p.position_id for p in short_positions],
                "long_tickets": [p.mt5_ticket for p in long_positions if p.mt5_ticket],
                "short_tickets": [
                    p.mt5_ticket for p in short_positions if p.mt5_ticket
                ],
            }

        # Calculate exposures and P&L
        for pos in self.positions.values():
            current_price = current_prices.get(pos.asset, pos.entry_price)
            notional_value = pos.quantity * current_price

            # Get leverage (defaults to 1 for spot trading)
            leverage = getattr(pos, 'leverage', 1)

            # ✅ Convert notional to USD before dividing by leverage.
            # For USD-quoted symbols this is a no-op (factor = 1.0).
            # For JPY-quoted symbols (EURJPY, USDJPY, …) this divides by
            # the current USD/JPY rate so we get a true USD margin figure.
            quote_to_usd = self._get_quote_to_usd_rate(pos.symbol)
            notional_usd = notional_value * quote_to_usd
            margin_used = notional_usd / leverage

            # Accumulate
            total_notional_value += notional_usd
            total_margin_used += margin_used
            total_exposure += margin_used  # ← Use USD margin, not raw notional

            # Calculate P&L (unchanged)
            if pos.mt5_ticket and pos.mt5_profit != 0.0:
                total_unrealized_pnl += pos.mt5_profit
            elif pos.binance_order_id and pos.binance_profit != 0.0:
                total_unrealized_pnl += pos.binance_profit
            else:
                total_unrealized_pnl += pos.get_pnl(current_price)
        
        if self.is_paper_mode:
            total_value = self.current_capital + total_unrealized_pnl
        else:
            total_value = self.current_capital

        # Calculate daily P&L
        if self.session_start_equity is not None:
            current_equity = self.current_capital + total_unrealized_pnl
            daily_pnl = current_equity - self.session_start_equity
        else:
            daily_pnl = self.realized_pnl_today + total_unrealized_pnl

        return {
        "mode": self.mode,
        "total_value": total_value,
        "capital": self.current_capital,
        "equity": self.equity,
        "cash": self.current_capital,
        
        # ✅ NEW: Separate notional vs actual exposure
        "total_notional_value": total_notional_value,      # For information
        "total_margin_used": total_margin_used,            # For risk limits
        "total_exposure": total_exposure,                  # ← This is margin_used
        
        "open_positions": len(self.positions),
        "daily_pnl": daily_pnl,
        "realized_pnl_today": self.realized_pnl_today,
        "total_unrealized_pnl": total_unrealized_pnl,
        "asset_position_counts": asset_position_counts,
        "asset_positions_detail": asset_positions_detail,
        "max_positions_per_asset": self.max_positions_per_asset,
        
        # Individual positions...
        "positions": {
            pos.position_id: {
                "asset": pos.asset,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "current_price": current_prices.get(pos.asset, pos.entry_price),
                "current_value": pos.quantity * current_prices.get(pos.asset, pos.entry_price),
                
                # ✅ NEW: Add leverage info to position details
                "leverage": getattr(pos, 'leverage', 1),
                "notional_value": pos.quantity * current_prices.get(pos.asset, pos.entry_price) * self._get_quote_to_usd_rate(pos.symbol),
                "margin_used": (pos.quantity * current_prices.get(pos.asset, pos.entry_price) * self._get_quote_to_usd_rate(pos.symbol)) / getattr(pos, 'leverage', 1),
                
                "pnl": (
                    pos.mt5_profit if (pos.mt5_ticket and pos.mt5_profit != 0.0)
                    else pos.binance_profit if (pos.binance_order_id and pos.binance_profit != 0.0)
                    else pos.get_pnl(current_prices.get(pos.asset, pos.entry_price))
                ),
                "pnl_pct": pos.get_pnl_pct(current_prices.get(pos.asset, pos.entry_price)),
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "mt5_ticket": pos.mt5_ticket,
                "mt5_profit": pos.mt5_profit if pos.mt5_ticket else None,
                "binance_order_id": pos.binance_order_id,
                "binance_profit": pos.binance_profit if pos.binance_order_id else None,
                "leverage": getattr(pos, "leverage", 1),
                "margin_type": getattr(pos, "margin_type", "SPOT"),
                "is_futures": getattr(pos, "is_futures", False),
            }
            for pos in self.positions.values()
        },
        }

    @handle_errors(
        component="portfolio_manager",
        severity=ErrorSeverity.ERROR,
        notify=True,
        reraise=False,
        default_return=None,
    )
    def close_position(
        self,
        asset: str = None,
        position_id: str = None,
        exit_price: float = None,
        reason: str = "manual",
    ) -> Optional[Dict]:
        """
        ✅ FIXED: Close position - Validates exchange close before removing from portfolio
        """
        # Find position to close
        if position_id:
            position = self.positions.get(position_id)
            if not position:
                logger.warning(f"Position {position_id} not found in portfolio")
                return None
        elif asset:
            positions = self.get_asset_positions(asset)
            if not positions:
                logger.warning(f"No positions to close for {asset}")
                return None
            position = positions[0]
            position_id = position.position_id
        else:
            logger.error("Must provide either asset or position_id")
            return None

        # Check if position is already being closed
        now = datetime.now()
        if position.closing:
            # If it's been "closing" for more than 60 seconds, assume it's stuck and allow retry
            if position.last_close_attempt and (now - position.last_close_attempt).total_seconds() > 60:
                logger.warning(f"Position {position_id} stuck in 'closing' for >60s. Overriding to allow retry.")
            else:
                logger.info(f"Position {position_id} is already in the process of being closed. Skipping.")
                return None
        
        if position.last_close_attempt:
            # Add a 30-second cooldown to prevent hammer on failing close attempts
            # (Unless we just overrode the "stuck" status above)
            if not position.closing and (now - position.last_close_attempt).total_seconds() < 30:
                logger.debug(f"Position {position_id} close attempt on cooldown. Skipping.")
                return None
            
        # Mark the position as closing and record attempt time
        position.closing = True
        position.last_close_attempt = now

        # ================================================================
        # ✅ STEP 1: CLOSE ON EXCHANGE FIRST (MT5 or Binance)
        # ================================================================
        exchange_closed = False
        close_error_msg = "Unknown handler error" # Default error message
        broker_close_data = None  # holds dict with authoritative fill+profit

        if not self.is_paper_mode:
            asset_cfg = self.config["assets"].get(position.asset, {})
            exchange = asset_cfg.get("exchange", "binance")
            handler = self.execution_handlers.get(exchange)

            if not handler:
                close_error_msg = f"{exchange.upper()} handler not available"
            else:
                try:
                    logger.info(f"[{exchange.upper()}] Attempting to close position {position.position_id}...")
                    handler_result = handler._close_position(
                        position=position,
                        current_price=exit_price,
                        asset_name=position.asset,
                        reason=reason,
                    )
                    # Handler may now return a dict with broker fill data, or a
                    # bare True/False for legacy/paper paths.
                    if isinstance(handler_result, dict):
                        broker_close_data = handler_result
                        exchange_closed = bool(handler_result.get("ok", True))
                    else:
                        exchange_closed = bool(handler_result)
                    if not exchange_closed:
                        close_error_msg = f"{exchange.upper()} order was rejected or failed. Check handler logs."

                except Exception as e:
                    close_error_msg = f"Handler exception: {str(e)}"
                    logger.error(f"[{exchange.upper()}] Error closing position: {e}", exc_info=True)
        else:
            # Paper mode always succeeds
            exchange_closed = True

        # ================================================================
        # ✅ STEP 2: ABORT OR PROCEED
        # ================================================================
        if not exchange_closed:
            logger.error(
                f"[CRITICAL] Position close failed on exchange!\n"
                f"  Position ID: {position_id}\n"
                f"  Asset:       {position.asset}\n"
                f"  Error:       {close_error_msg}\n"
                f"  ⚠️  Resetting 'closing' flag to allow future attempts."
            )
            # RESET THE FLAG so we can try again later
            position.closing = False
            return None

        # ================================================================
        # ✅ STEP 3: CALCULATE P&L (Only if exchange close succeeded)
        # ────────────────────────────────────────────────────────────────
        # Prefer the broker's authoritative numbers when we have them. The
        # local Python calc uses a cached/stale `exit_price` that can diverge
        # significantly from the actual fill (we've seen 50¢ on USOIL = $5+
        # under-reported), and it ignores swap + commission entirely.
        # ================================================================
        broker_fill_price = None
        broker_profit = None
        broker_swap = 0.0
        broker_commission = 0.0
        if broker_close_data:
            broker_fill_price = broker_close_data.get("fill_price")
            broker_profit = broker_close_data.get("profit")
            broker_swap = broker_close_data.get("swap", 0.0) or 0.0
            broker_commission = broker_close_data.get("commission", 0.0) or 0.0

        # If broker reported an actual fill price, use it as the canonical exit_price.
        if broker_fill_price is not None and broker_fill_price > 0:
            if exit_price and abs(broker_fill_price - exit_price) / max(abs(exit_price), 1e-9) > 0.0005:
                logger.warning(
                    f"[CLOSE] Stale-cache exit drift detected for {position.asset}: "
                    f"cached ${exit_price:,.5f} vs broker fill ${broker_fill_price:,.5f}. "
                    f"Using broker fill for P&L."
                )
            exit_price = broker_fill_price

        if broker_profit is not None:
            pnl = float(broker_profit) + float(broker_swap) + float(broker_commission)
            # Derive % from broker profit against the entry notional we tracked
            entry_notional = position.entry_price * position.quantity if position.entry_price else 0.0
            pnl_pct = (pnl / entry_notional) if entry_notional > 0 else 0.0
            logger.info(
                f"[CLOSE] Using BROKER P&L for {position.asset}: "
                f"${pnl:,.2f} (profit ${broker_profit:,.2f}, swap ${broker_swap:,.2f}, "
                f"commission ${broker_commission:,.2f})"
            )
        else:
            pnl = position.get_pnl(exit_price)
            pnl_pct = position.get_pnl_pct(exit_price)
            logger.debug(
                f"[CLOSE] Broker P&L unavailable for {position.asset}; "
                f"falling back to local calc using exit ${exit_price:,.5f}"
            )

        self.realized_pnl_today += pnl

        # ✨ NEW: Record strategy performance
        trade_type = getattr(position, 'trade_type', 'TREND')
        self.performance_tracker.record_trade(trade_type, pnl)

        # ✨ NEW: Track consecutive losses
        if pnl < 0:
            self.loss_streak += 1
            logger.warning(f"[STREAK] Loss streak incremented: {self.loss_streak}")
        else:
            if self.loss_streak > 0:
                logger.info(f"[STREAK] Loss streak of {self.loss_streak} reset to 0.")
            self.loss_streak = 0
            self._loss_streak_alerted = False  # Reset alert guard when streak clears

        # Update capital
        if self.is_paper_mode:
            self.current_capital += pnl
            self.equity = self.current_capital
        else:
            self.refresh_capital()

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # ================================================================
        # ✅ STEP 4: CREATE TRADE RESULT
        # ================================================================
        trade_result = {
            "asset": asset or position.asset,
            "position_id": position_id,
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "entry_time": position.entry_time,
            "exit_time": datetime.now(),
            "holding_time": (datetime.now() - position.entry_time).total_seconds() / 3600,
            "reason": reason,
            "mt5_ticket": position.mt5_ticket,
            "binance_order_id": position.binance_order_id,
            "exchange_closed": exchange_closed,
        }

        # ================================================================
        # ✅ STEP 5: LOG TO DATABASE
        # ================================================================
        if self.db_manager and hasattr(position, "db_trade_id") and position.db_trade_id:
            try:
                holding_time = (datetime.now() - position.entry_time).total_seconds() / 3600
                self.db_manager.update_trade_exit(
                    trade_id=position.db_trade_id,
                    exit_price=exit_price,
                    exit_reason=reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_time_hours=holding_time,
                    final_quantity=position.quantity,
                    metadata={"exit_time": datetime.now().isoformat(), "exchange_closed": exchange_closed},
                )
                logger.debug(f"[DB] Trade exit logged: {position.db_trade_id}")
            except Exception as e:
                logger.error(f"[DB] Error logging trade exit: {e}")

        # ================================================================
        # ✅ STEP 6: REMOVE FROM PORTFOLIO
        # ================================================================
        self.closed_positions.append(trade_result)
        # ✨ MEMORY MANAGEMENT: Limit to 100 entries
        if len(self.closed_positions) > 100:
            self.closed_positions.pop(0)

        # ✅ RACE-CONDITION GUARD: reconciliation loop may have already removed
        # this key from self.positions (broker reported 0 positions before we
        # reached this line).  Use pop() instead of del so a KeyError never
        # fires after a successful exchange close.
        if self.positions.pop(position_id, None) is None:
            logger.debug(
                f"[CLOSE] {position_id} already removed from portfolio by "
                f"reconciliation — exchange close still completed successfully."
            )

        # Track whether this close was manual (Telegram / force-close) so the
        # cooldown bypass in check_min_time_between_trades() works correctly.
        _reason_str = str(reason).lower()
        _is_manual = any(k in _reason_str for k in ("manual", "telegram", "force", "user"))
        self.last_close_was_manual[position.asset] = _is_manual

        remaining_count = self.get_asset_position_count(position.asset, position.side)

        # Standardize exit reason for logger
        exit_event_type = "EXIT"
        if "stop_loss" in str(reason).lower():
            exit_event_type = "SL_HIT"
        elif "take_profit" in str(reason).lower():
            exit_event_type = "TP_HIT"

        # ✅ Standardized Log
        log_trade_event(exit_event_type, {
            "symbol": position.symbol,
            "asset": position.asset,
            "side": position.side,
            "price": exit_price,
            "quantity": position.quantity,
            "trade_type": getattr(position, 'trade_type', 'UNKNOWN'),
            "reason": reason,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "position_id": position_id
        })

        logger.info(
            f"✓ Position closed successfully:\n"
            f"  Asset:     {position.asset} {position.side.upper()}\n"
            f"  Exit:      ${exit_price:,.2f}\n"
            f"  P&L:       ${pnl:,.2f} ({pnl_pct:.2%})\n"
            f"  Remaining: {remaining_count}/{self.max_positions_per_asset}"
        )

        # Send notification
        if self.telegram_bot and self.telegram_bot._current_loop:
            try:
                # Use a thread-safe method to call the async notification
                asyncio.run_coroutine_threadsafe(
                    self.telegram_bot.notify_trade_closed(
                        asset=trade_result["asset"],
                        side=trade_result["side"],
                        pnl=trade_result["pnl"],
                        pnl_pct=trade_result["pnl_pct"] * 100, # Convert to percentage points
                        reason=trade_result["reason"],
                    ),
                    self.telegram_bot._current_loop
                )
            except Exception as e:
                logger.error(f"[TELEGRAM] Failed to send close notification from PM: {e}")

        return trade_result

    def reconcile_positions(self, asset: str, broker_positions: List[Dict]):
        """
        ✅ RECONCILIATION: Ensure local positions match broker reality.
        Syncs local state if mismatches are detected.
        """
        local_positions = self.get_asset_positions(asset)
        
        # 1. Check for orphaned local positions (exist here but not on broker)
        # We check by specific exchange ID if available
        broker_ids = [str(p.get('id')) for p in broker_positions if p.get('id')]
        
        for pos in list(local_positions):
            # A. If we have a specific exchange ID, use it for exact matching
            local_exchange_id = str(pos.mt5_ticket) if pos.mt5_ticket else str(pos.binance_order_id)
            
            if local_exchange_id != "None" and broker_ids:
                if local_exchange_id not in broker_ids:
                    logger.error(f"[RECONCILE] Orphaned ID found: {pos.position_id} (ID: {local_exchange_id}). Removing.")
                    self.positions.pop(pos.position_id, None)
                    continue

            # B. If no IDs (rare) or for Binance where positions are aggregated by side
            # Check if at least one broker position exists for this side
            broker_sides = [p.get('side').lower() for p in broker_positions if p.get('side')]
            if pos.side.lower() not in broker_sides:
                logger.error(f"[RECONCILE] Orphaned SIDE found: {pos.position_id} ({pos.side.upper()}). No match on broker. Removing.")
                self.positions.pop(pos.position_id, None)

        # 2. Final check: If broker has 0 positions but we still have local ones, clear them
        if not broker_positions and local_positions:
            logger.warning(f"[RECONCILE] Broker reports 0 positions for {asset}. Clearing local state.")
            for pos in local_positions:
                self.positions.pop(pos.position_id, None)

    @handle_errors(
        component="portfolio_manager",
        severity=ErrorSeverity.WARNING,
        notify=False,  # Don't notify for update errors
        reraise=False,
        default_return=None,
    )
    def update_positions(self, prices: Dict[str, float] = None):
        """Update all positions with current prices and exchange profit"""
        # Update exchange positions with real-time profit
        if not self.is_paper_mode:
            self.update_mt5_positions_profit()
            self.update_binance_positions_profit()

        if prices:
            for asset, price in prices.items():
                if asset in self.price_history:
                    self.price_history[asset].append(price)
                    # ✨ MEMORY MANAGEMENT: Limit to 500 entries (Safe for 200 EMA + buffer)
                    if len(self.price_history[asset]) > 500:
                        self.price_history[asset].pop(0)

        # Calculate unrealized P&L
        total_unrealized_pnl = 0.0
        for pos in self.positions.values():
            # Prioritize exchange-reported profit
            if pos.mt5_ticket and pos.mt5_profit != 0.0:
                # Use MT5 profit for MT5 positions
                total_unrealized_pnl += pos.mt5_profit
            elif pos.binance_order_id and pos.binance_profit != 0.0:
                # Use Binance profit for Binance positions
                total_unrealized_pnl += pos.binance_profit
            elif prices and pos.asset in prices:
                # Calculate for positions without exchange tracking
                total_unrealized_pnl += pos.get_pnl(prices[pos.asset])

        if self.is_paper_mode:
            # In paper mode: equity = cash + unrealized P&L
            self.equity = self.current_capital + total_unrealized_pnl
        else:
            # In live mode: periodically refresh from exchanges
            pass

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def get_open_positions_count(self) -> int:
        """Get number of open positions"""
        return len(self.positions)

    def get_position(self, asset: str, position_id: str = None) -> Optional[Position]:
        """
        Get position(s) for an asset

        Args:
            asset: Asset name
            position_id: Optional specific position ID

        Returns:
            Position object if position_id provided, otherwise first position for asset
        """
        if position_id:
            return self.positions.get(position_id)

        # Return first position for asset (for backward compatibility)
        positions = self.get_asset_positions(asset)
        return positions[0] if positions else None

    def has_position(self, asset: str, side: str = None) -> bool:
        """
        Check if we have any open positions for an asset

        Args:
            asset: Asset symbol
            side: Optional side filter ('long' or 'short')
        """
        return self.get_asset_position_count(asset, side) > 0

    def reset_daily_pnl(self):
        """Reset realized P&L tracker (call this at start of each trading day)"""
        self.realized_pnl_today = 0.0
        logger.info("Daily P&L tracker reset")

    def start_trading_session(self):
        """Start trading session.

        session_start_equity is intentionally set to the CURRENT margin balance
        (wallet + unrealized PnL) so that positions imported from a prior session
        don't create a phantom loss when they are closed: their unrealized P&L is
        already baked into the baseline, and the circuit-breaker only fires on
        NEW losses incurred during this session.

        ✅ FIX: Force-refresh the real broker balance BEFORE locking in
        session_start_equity.  Without this, switching from a large demo/Binance
        account to a small live MT5 account causes the circuit-breaker to fire
        immediately (99% phantom drawdown) because session_start_equity is set
        from the previous session's stale in-memory equity value.
        """
        self.session_start_time = datetime.now()
        if not self.is_paper_mode:
            try:
                self.refresh_capital(force=True)
            except Exception as _e:
                logger.warning(f"[SESSION] Could not refresh capital before session start: {_e}")
        self.session_start_equity = self.equity
        self.session_start_capital = self.current_capital
        self.realized_pnl_today = 0.0
        # Clear any manual /resume override at the start of a new trading day
        if getattr(self, "_circuit_breaker_override", False):
            self._circuit_breaker_override = False
            logger.info("[SESSION] Circuit-breaker manual override cleared for new session.")
        open_count = len(self.positions)
        logger.info(
            f"Trading session started at {self.session_start_time}\n"
            f"  Session-start equity: ${self.session_start_equity:,.2f}"
            + (f"  ({open_count} imported position(s) already priced in)" if open_count else "")
        )

    def get_portfolio_status(self, current_prices: Dict[str, float] = None) -> Dict:
        """
        \u2705 FIX: Auto-refresh balances when getting status
        """
        # \u2705 Refresh if stale (respects time interval)
        self.refresh_capital(force=False)

        if current_prices is None:
            current_prices = {
                pos.asset: pos.entry_price for pos in self.positions.values()
            }

        total_exposure = 0.0
        total_unrealized_pnl = 0.0
        
        total_notional_value = 0.0
        total_margin_used = 0.0

        # \u2705  Count positions per asset correctly
        asset_position_counts = {}
        asset_positions_detail = {}

        # Get all enabled assets from config
        enabled_assets = [a for a, cfg in self.config["assets"].items() if cfg.get("enabled", False)]

        for asset in enabled_assets:
            long_positions = [
                p
                for p in self.positions.values()
                if p.asset == asset and p.side == "long"
            ]
            short_positions = [
                p
                for p in self.positions.values()
                if p.asset == asset and p.side == "short"
            ]

            asset_position_counts[asset] = {
                "long": len(long_positions),
                "short": len(short_positions),
                "total": len(long_positions) + len(short_positions),
            }

            asset_positions_detail[asset] = {
                "long_ids": [p.position_id for p in long_positions],
                "short_ids": [p.position_id for p in short_positions],
                "long_tickets": [p.mt5_ticket for p in long_positions if p.mt5_ticket],
                "short_tickets": [
                    p.mt5_ticket for p in short_positions if p.mt5_ticket
                ],
            }

        # Calculate exposures and P&L
        for pos in self.positions.values():
            current_price = current_prices.get(pos.asset, pos.entry_price)
            notional_value = pos.quantity * current_price

            # Get leverage (defaults to 1 for spot trading)
            leverage = getattr(pos, 'leverage', 1)

            # ✅ Convert notional to USD before dividing by leverage.
            # For USD-quoted symbols this is a no-op (factor = 1.0).
            # For JPY-quoted symbols (EURJPY, USDJPY, …) this divides by
            # the current USD/JPY rate so we get a true USD margin figure.
            quote_to_usd = self._get_quote_to_usd_rate(pos.symbol)
            notional_usd = notional_value * quote_to_usd
            margin_used = notional_usd / leverage

            # Accumulate
            total_notional_value += notional_usd
            total_margin_used += margin_used
            total_exposure += margin_used  # ← Use USD margin, not raw notional

            # Calculate P&L (unchanged)
            if pos.mt5_ticket and pos.mt5_profit != 0.0:
                total_unrealized_pnl += pos.mt5_profit
            elif pos.binance_order_id and pos.binance_profit != 0.0:
                total_unrealized_pnl += pos.binance_profit
            else:
                total_unrealized_pnl += pos.get_pnl(current_price)
        
        if self.is_paper_mode:
            total_value = self.current_capital + total_unrealized_pnl
        else:
            total_value = self.current_capital

        # Calculate daily P&L
        if self.session_start_equity is not None:
            current_equity = self.current_capital + total_unrealized_pnl
            daily_pnl = current_equity - self.session_start_equity
        else:
            daily_pnl = self.realized_pnl_today + total_unrealized_pnl

        return {
        "mode": self.mode,
        "total_value": total_value,
        "capital": self.current_capital,
        "equity": self.equity,
        "cash": self.current_capital,
        
        # ✅ NEW: Separate notional vs actual exposure
        "total_notional_value": total_notional_value,      # For information
        "total_margin_used": total_margin_used,            # For risk limits
        "total_exposure": total_exposure,                  # ← This is margin_used
        
        "open_positions": len(self.positions),
        "daily_pnl": daily_pnl,
        "realized_pnl_today": self.realized_pnl_today,
        "total_unrealized_pnl": total_unrealized_pnl,
        "asset_position_counts": asset_position_counts,
        "asset_positions_detail": asset_positions_detail,
        "max_positions_per_asset": self.max_positions_per_asset,
        
        # Individual positions...
        "positions": {
            pos.position_id: {
                "asset": pos.asset,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "current_price": current_prices.get(pos.asset, pos.entry_price),
                "current_value": pos.quantity * current_prices.get(pos.asset, pos.entry_price),
                
                # ✅ NEW: Add leverage info to position details
                "leverage": getattr(pos, 'leverage', 1),
                "notional_value": pos.quantity * current_prices.get(pos.asset, pos.entry_price) * self._get_quote_to_usd_rate(pos.symbol),
                "margin_used": (pos.quantity * current_prices.get(pos.asset, pos.entry_price) * self._get_quote_to_usd_rate(pos.symbol)) / getattr(pos, 'leverage', 1),
                
                "pnl": (
                    pos.mt5_profit if (pos.mt5_ticket and pos.mt5_profit != 0.0)
                    else pos.binance_profit if (pos.binance_order_id and pos.binance_profit != 0.0)
                    else pos.get_pnl(current_prices.get(pos.asset, pos.entry_price))
                ),
                "pnl_pct": pos.get_pnl_pct(current_prices.get(pos.asset, pos.entry_price)),
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "mt5_ticket": pos.mt5_ticket,
                "mt5_profit": pos.mt5_profit if pos.mt5_ticket else None,
                "binance_order_id": pos.binance_order_id,
                "binance_profit": pos.binance_profit if pos.binance_order_id else None,
                "leverage": getattr(pos, "leverage", 1),
                "margin_type": getattr(pos, "margin_type", "SPOT"),
                "is_futures": getattr(pos, "is_futures", False),
            }
            for pos in self.positions.values()
        },
        }
