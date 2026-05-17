"""
MT5 Execution Handler with Hybrid Position Sizing
INTEGRATED: Automated risk management + manual override support
✨ ENHANCED: Hedging enabled - Allows simultaneous Long/Short positions (Asymmetric System)
"""

import logging
import MetaTrader5 as mt5
import time  # Added for time.time()
from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
from datetime import datetime, timedelta, timezone
from src.global_error_handler import handle_errors, ErrorSeverity
from src.execution.veteran_trade_manager import VeteranTradeManager
from src.utils.trade_logger import log_trade_event
from src.market.price_cache import price_cache

logger = logging.getLogger(__name__)


def count_mt5_positions(symbol: str, side: str = None) -> int:
    """
    Count actual open positions on MT5
    """
    try:
        mt5_positions = mt5.positions_get(symbol=symbol)

        if mt5_positions is None:
            return 0

        if side is None:
            return len(mt5_positions)

        # Filter by side
        count = 0
        for pos in mt5_positions:
            pos_side = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
            if pos_side == side:
                count += 1

        return count

    except Exception as e:
        logger.error(f"Error counting MT5 positions: {e}")
        return 0


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
        self.error_handler = None
        self.trading_bot = None


class HybridPositionSizer:
    """
    ✅ FIXED: Position sizing that works for ALL account sizes

    Key Changes:
    1. Respects broker minimum lot sizes
    2. Scales appropriately for small accounts
    3. Prevents oversized positions on undercapitalized accounts
    """

    def __init__(self, config: Dict, portfolio_manager):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.portfolio_cfg = config["portfolio"]
        self.risk_cfg = config.get("risk_management", {})
        self.override_history = []

        self.target_risk_pct = self.portfolio_cfg.get("target_risk_per_trade", 0.015)
        self.max_risk_pct = self.portfolio_cfg.get("max_risk_per_trade", 0.020)
        self.aggressive_threshold = self.portfolio_cfg.get(
            "aggressive_risk_threshold", 0.70
        )

        logger.info(
            f"[RISK SIZER] Initialized\n"
            f"  Target Risk: {self.target_risk_pct:.2%}\n"
            f"  Max Risk:    {self.max_risk_pct:.2%}\n"
            f"  Aggressive:  >{self.aggressive_threshold:.0%} confidence"
        )

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
        risk_pct: float = None,  # ✨ NEW: Accept external risk budget
    ) -> Tuple[float, Dict]:
        """
        ✨ REFACTORED: Accept risk budget from Portfolio Manager
        """
        try:
            # Get account balance
            account_balance = self.portfolio_manager.get_asset_balance(asset)

            if account_balance <= 0:
                logger.error(f"[RISK] No available balance for {asset}")
                return 0.0, {"error": "insufficient_balance"}

            # ✨ CRITICAL: Use externally provided risk
            if risk_pct is None:
                logger.error("[RISK] No risk percentage provided!")
                return 0.0, {"error": "missing_risk_budget"}

            # Calculate position size from risk budget
            risk_amount = account_balance * risk_pct

            stop_distance = abs(entry_price - stop_loss_price)
            stop_distance_pct = stop_distance / entry_price

            if stop_distance_pct < 0.005:
                stop_distance_pct = 0.005

            position_size_usd = risk_amount / stop_distance_pct

            # Apply asset limits
            asset_cfg = self.config["assets"][asset]
            min_size = asset_cfg.get("min_position_usd", 100)
            max_size = asset_cfg.get("max_position_usd", 50000)

            if position_size_usd < min_size:
                logger.warning(
                    f"[RISK] Below minimum: ${position_size_usd:.2f} < ${min_size}"
                )
                return 0.0, {"error": "below_minimum"}

            position_size_usd = min(position_size_usd, max_size)

            actual_risk = position_size_usd * stop_distance_pct
            actual_risk_pct = actual_risk / account_balance

            metadata = {
                "asset": asset,
                "signal": signal,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "provided_risk_pct": risk_pct * 100,
                "position_size_usd": position_size_usd,
                "actual_risk_usd": actual_risk,
                "actual_risk_pct": actual_risk_pct * 100,
            }

            return position_size_usd, metadata

        except Exception as e:
            logger.error(f"[RISK] Error: {e}", exc_info=True)
            return 0.0, {"error": str(e)}


class MT5ExecutionHandler:
    """
    MT5 Execution Handler with Multi-Asset Support & Hybrid Position Sizing
    """

    def __init__(self, config: Dict, portfolio_manager, data_manager=None):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager
        self.sizer = HybridPositionSizer(config, portfolio_manager)

        self.risk_config = config.get("risk_management", {})
        self.trading_config = config.get("trading", {})
        self.mode = self.trading_config.get("mode", "paper")
        self.execution_lock = {}  # ✨ NEW: Prevent duplicate trades
        self.last_trade_time = {}  # ✨ NEW: Rapid-fire cooldown
        self.trade_timestamps_hourly = []  # ✨ NEW: Hourly trade limit

        self.max_positions_per_asset = config.get("trading", {}).get(
            "max_positions_per_asset", 3
        )

        # VTM-SL/TP: tracks last pushed values per ticket to suppress redundant SLTP orders
        self._last_pushed_sl: Dict[int, float] = {}
        self._last_pushed_tp: Dict[int, float] = {}

        logger.info("MT5ExecutionHandler with Multi-Asset support initialized")

        # Auto-sync enabled assets on startup
        auto_sync_enabled = bool(self.trading_config.get("auto_sync_on_startup", True))
        import_enabled = bool(
            self.config.get("portfolio", {}).get("import_existing_positions", True)
        )

        if auto_sync_enabled and import_enabled:
            logger.info("[INIT] Auto-syncing enabled MT5 assets...")
            mt5_assets = [
                name for name, cfg in self.config.get("assets", {}).items()
                if cfg.get("exchange", "mt5") == "mt5" and cfg.get("enabled", False)
            ]
            for asset in mt5_assets:
                if self.config["assets"].get(asset, {}).get("enabled", False):
                    symbol = self._resolve_symbol(asset)
                    self.sync_positions_with_mt5(asset, symbol)

    def _check_connection(self) -> bool:
        """Check if MT5 terminal is responsive and connected"""
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.error("[MT5] Terminal not responsive")
                return False
            
            if not terminal_info.connected:
                logger.warning("[MT5] Terminal disconnected")
                return False
                
            return True
        except Exception as e:
            logger.error(f"[MT5] Connection check failed: {e}")
            return False

    def _resolve_symbol(self, asset_name: str) -> str:
        """
        Return the correct MT5 broker symbol for an asset.
        When 'mt5_symbol' is present in config it takes precedence over 'symbol',
        allowing assets like BTC to use 'BTCUSDm' on Exness while keeping
        'BTCUSDT' as the Binance symbol in the same config block.
        """
        cfg = self.config["assets"].get(asset_name, {})
        return cfg.get("mt5_symbol") or cfg.get("symbol", asset_name)

    def get_current_price(
        self, symbol: str = None, force_live: bool = False
    ) -> Optional[float]:
        """
        Unified price accessor for MT5, using the central price cache.
        """
        if not symbol:
            # Fallback to a default if possible, or log error
            logger.error("[MT5] get_current_price called without symbol and no fallback available.")
            return None

        # 1. Try to get a fresh price from the cache
        cached_price = price_cache.get(symbol)
        if cached_price is not None and not force_live:
            return cached_price

        # 2. If cache is stale or a live price is forced, fetch from MT5
        if not self._check_connection():
            return price_cache.get_last_known(symbol)

        try:
            if self.mode.lower() == "paper" and (cached_price is None or force_live):
                # Simple mock prices for common assets
                mock_prices = {
                    "XAUUSD": 2000.00,
                    "XAUUSDm": 2000.00,
                    "USTEC": 15000.00,
                    "USTECm": 15000.00,
                    "EURJPY": 160.00,
                    "EURJPYm": 160.00,
                    "EURUSD": 1.1000,
                    "EURUSDm": 1.1000,
                    "GBPUSD": 1.2700,
                    "GBPUSDm": 1.2700,
                    "USDJPY": 150.00,
                    "USDJPYm": 150.00,
                    "GBPAUD": 1.9200,
                    "GBPAUDm": 1.9200,
                    "USOIL": 75.00,
                    "USOILm": 75.00,
                }
                # Handle suffixes like 'm'
                base_symbol = symbol.replace("m", "")
                mock_price = mock_prices.get(base_symbol, 100.0)
                
                price_cache.set(symbol, mock_price)
                logger.info(
                    f"[CACHE] Price cache updated with MOCK price (Paper Mode): {mock_price} for {symbol}"
                )
                return mock_price

            tick = mt5.symbol_info_tick(symbol)
            if tick:
                live_price = (tick.ask + tick.bid) / 2
                price_cache.set(symbol, live_price)  # Update cache
                # F.7: Capture spread for spread velocity (stored per symbol)
                _spread = tick.ask - tick.bid
                if _spread > 0:
                    if not hasattr(self, '_last_spread'):
                        self._last_spread = {}
                    self._last_spread[symbol] = _spread
                return live_price
            else:
                # Fallback to last known price if tick fails
                logger.warning(
                    f"Failed to get live tick for {symbol}, using last known price."
                )
                return price_cache.get_last_known(symbol)
        except Exception as e:
            logger.error(f"Error fetching MT5 price for {symbol}: {e}")
            # Fallback to last known price on error
            return price_cache.get_last_known(symbol)

    def can_open_position_side(self, asset_name: str, side: str) -> Tuple[bool, str]:
        """Check if we can open a position on a specific SIDE"""
        asset_cfg = self.config["assets"].get(asset_name)
        if not asset_cfg:
            return False, f"Asset {asset_name} not found in config"

        if side == "short":
            allow_shorts = asset_cfg.get("allow_shorts", False)
            if not allow_shorts:
                return False, f"Short trading disabled for {asset_name}"

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

        try:
            import MetaTrader5 as mt5

            account_info = mt5.account_info()
            if account_info:
                margin_free = account_info.margin_free
                
                # ✅ T38: Set sanity floor to $12.0
                # Real margin check happens below in _validate_margin()
                margin_required = 12.0

                if margin_free < margin_required:
                    return False, f"Insufficient margin: ${margin_free:.2f} free (need ${margin_required:.2f} sanity floor)"

        except Exception as e:
            logger.debug(f"[MT5] Margin check warning: {e}")

        return (
            True,
            f"OK - {current_count}/{max_per_asset} {side.upper()} positions open",
        )

    def _is_trading_allowed(self, symbol: str) -> bool:
        """
        Check if trading is currently allowed for the symbol
        """
        try:
            import MetaTrader5 as mt5

            # 1. Get symbol info
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.warning(f"[MT5] Symbol {symbol} not found")
                return False

            # 2. Check trade mode
            if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                logger.warning(
                    f"[MT5] Trading is DISABLED for {symbol} (Market closed or restricted)"
                )
                return False

            # 3. Check for tick data
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(
                    f"[MT5] Market likely CLOSED (no tick data for {symbol})"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"[MT5] Error checking trading status: {e}")
            return False

    @handle_errors(
        component="mt5_handler",
        severity=ErrorSeverity.CRITICAL,
        notify=True,
        reraise=False,
        default_return=False,
    )
    def _open_mt5_position(
        self,
        signal: int,
        current_price: float,
        symbol: str,
        asset: str,
        confidence_score: float = None,
        market_condition: str = None,
        sizing_mode: str = SizingMode.AUTOMATED,
        manual_size_usd: float = None,
        override_reason: str = None,
        signal_details: Dict = None,
    ) -> bool:
        """
        ✅ STRATEGIC/TACTICAL INTEGRATION
        Refactored for Multi-Asset support
        """
        try:
            # Get symbol info from MT5 for dynamic lot calculations
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"[MT5] Symbol {symbol} not found. Cannot open position.")
                return False

            # ================================================================
            # SMALL ACCOUNT PROTOCOL ("SNIPER MODE")
            # ================================================================
            is_small_account_mode = self.trading_config.get(
                "small_account_protocol", False
            )
            vtm_overrides = None
            force_lot_size = None

            asset_cfg = self.config["assets"].get(asset, {})
            trade_type = "TREND"  # Default
            if signal_details:
                trade_type = signal_details.get("trade_type", "TREND")

            if is_small_account_mode and asset_cfg.get("force_min_lot", False):
                # --- SNIPER MODE: Bypassing risk calculation ---
                volume_lots = symbol_info.volume_min
                logger.info(
                    f"[SNIPER MODE] Bypassing risk calculation, forcing MIN LOT ({volume_lots}) for {asset}."
                )
                position_size_usd = (
                    volume_lots * current_price * symbol_info.trade_contract_size
                )
                actual_usd = position_size_usd
                risk_pct = -1  # Indicate bypass
                
                if signal_details is None:
                    signal_details = {}
                signal_details["small_account_protocol_active"] = True
            else:
                # --- STANDARD RISK-BASED SIZING ---
                logger.info(f"\n{'='*80}")
                logger.info(
                    f"[STRATEGIC] Requesting risk budget from Portfolio Manager for {asset}"
                )
                logger.info(f"{'='*80}")
                # Pass signal direction so portfolio manager can apply USD clustering malus
                self.portfolio_manager._pending_signal_side = signal
                risk_pct = self.portfolio_manager.get_risk_budget(
                    asset=asset,
                    strategy_type=trade_type,
                    confidence_score=signal_details.get("mode_confidence"),
                    market_condition=signal_details.get("regime")
                )
                self.portfolio_manager._pending_signal_side = None

                if risk_pct <= 0:
                    logger.error(
                        f"[STRATEGIC] ❌ Risk budget denied (0%) for {asset}\n"
                        f"  → Trade rejected by Portfolio Manager"
                    )
                    return False

                # T1.7 fix: apply MTF regime multiplier computed in main.py but previously orphaned
                mtf_multiplier = signal_details.get("mtf_risk_multiplier", 1.0) if signal_details else 1.0
                if mtf_multiplier != 1.0:
                    risk_pct *= mtf_multiplier
                    logger.info(f"[RISK] MTF multiplier applied: {mtf_multiplier:.1f}x → risk_pct={risk_pct:.4f}")

                # Daily profit soft-lock: halve position size once realized PnL
                # crosses the soft-lock threshold (set in config.risk.daily_profit_lock)
                if signal_details and signal_details.get("profit_soft_lock_active"):
                    risk_pct *= 0.5
                    logger.info(f"[PROFIT LOCK] Soft lock: sizing halved → risk_pct={risk_pct:.4f}")

                logger.info(f"[STRATEGIC] ✓ Risk budget approved: {risk_pct:.3%}")

                # Calculate initial stop loss for validation
                risk_config = asset_cfg.get("risk", {})
                
                # ATR-based adaptive stop loss distance
                atr_fast = signal_details.get("atr_fast") if signal_details else None
                atr_multiplier = risk_config.get("atr_multiplier", 1.5)
                
                if atr_fast:
                    initial_sl_dist = atr_fast * atr_multiplier
                    logger.info(f"[TACTICAL] Using ATR-based SL: {atr_multiplier}x ATR ({initial_sl_dist:.2f})")
                else:
                    sl_pct = risk_config.get("stop_loss_pct", 0.01)
                    initial_sl_dist = current_price * sl_pct
                    logger.info(f"[TACTICAL] ⚠️ ATR not found, using static SL: {sl_pct:.2%}")

                side = "long" if signal == 1 else "short"
                if side == "long":
                    initial_stop = current_price - initial_sl_dist
                else:
                    initial_stop = current_price + initial_sl_dist

                # VTM Pre-Flight Validation
                logger.info(f"\n{'='*80}")
                logger.info(f"[TACTICAL] VTM Pre-Flight Validation for {asset}")
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

                # Sizing Calculation
                logger.info(f"\n{'='*80}")
                logger.info(f"[SIZING] Calculating position size for {asset}")
                logger.info(f"{'='*80}")

                # ── VENUE-SCOPED BALANCE ─────────────────────────────────────────
                # MT5 trades must be sized against the MT5/Exness account balance,
                # NOT the combined portfolio capital. Otherwise a $50 Exness account
                # can be told to open $11k notional because Binance has $14.9k —
                # which blows the free-margin budget on the first adverse tick.
                portfolio_balance = self.portfolio_manager.current_capital
                mt5_balance = portfolio_balance  # safe fallback
                try:
                    # mt5 is already imported at module scope — do NOT re-import
                    # locally, that creates an UnboundLocalError on the earlier
                    # mt5.symbol_info(symbol) call at the top of this function.
                    _ai = mt5.account_info()
                    if _ai is not None and _ai.balance is not None and float(_ai.balance) > 0:
                        mt5_balance = float(_ai.balance)
                except Exception as _e:
                    logger.debug(f"[SIZING] MT5 account_info fetch failed: {_e}")

                # Always size MT5 orders against MT5 equity. Cross-venue capital
                # cannot back an Exness margin call.
                account_balance = mt5_balance
                if abs(mt5_balance - portfolio_balance) > 1.0:
                    logger.info(
                        f"[SIZING] Using MT5 venue balance ${mt5_balance:,.2f} "
                        f"(portfolio combined: ${portfolio_balance:,.2f})"
                    )

                # ── SMALL-ACCOUNT GUARD (sub-$200 MT5 balance) ───────────────────
                # On a tiny Exness account, force min lot regardless of what the
                # risk-budget math says. Below the configured threshold we never
                # trust the risk calc — go straight to broker minimum lot.
                small_account_threshold = float(
                    self.trading_config.get("mt5_small_account_threshold_usd", 200.0)
                )
                force_small_account = mt5_balance < small_account_threshold

                if force_small_account:
                    logger.warning(
                        f"[SIZING] 🛡️ MT5 balance ${mt5_balance:,.2f} below "
                        f"${small_account_threshold:,.0f} threshold — "
                        f"forcing min lot for {asset}."
                    )
                    # Use min lot directly; downstream code will pick this up.
                    forced_lots = symbol_info.volume_min
                    # ✅ H-2 FIX: At min lot, partial exits round to 0 lots and
                    # exit the full position at TP1 — TP2/TP3 are dead.  Disable
                    # partials so VTM uses single-exit mode instead.
                    _min_lot_val = symbol_info.volume_min if symbol_info else 0.01
                    _disable_partials = forced_lots <= _min_lot_val * 1.5
                    _contract_size  = symbol_info.trade_contract_size
                    _notional_quote = forced_lots * current_price * _contract_size
                    # ✅ C-1 FIX: Store margin (notional / leverage) not raw
                    # notional so portfolio_manager risk checks are meaningful.
                    # Note: _notional_quote is in the symbol's quote currency
                    # (JPY for EURJPY etc.); portfolio_manager._get_quote_to_usd_rate()
                    # handles the final currency conversion on the display side.
                    _leverage = self.config.get("assets", {}).get(
                        asset, {}
                    ).get("leverage", 100)
                    position_size_usd = max(
                        _notional_quote / _leverage,
                        forced_lots * current_price * 0.01,
                    )
                    risk_amount_usd = position_size_usd  # for log clarity
                    stop_distance = abs(current_price - initial_stop)
                    stop_distance_pct = stop_distance / current_price if current_price else 0.0
                    if signal_details is None:
                        signal_details = {}
                    signal_details["small_account_protocol_active"] = True
                    signal_details["forced_min_lot"] = True
                else:
                    _disable_partials = False  # normal sizing — partials active
                    risk_amount_usd = account_balance * risk_pct
                    stop_distance = abs(current_price - initial_stop)
                    stop_distance_pct = stop_distance / current_price
                    position_size_usd = risk_amount_usd / stop_distance_pct

                # ── Asset-weight scaling (config["assets"][asset]["weight"]) ──────
                # portfolio_manager.calculate_position_size() applies this weight but
                # mt5_handler does its own sizing and was never calling that function,
                # so the weight was silently ignored for all MT5 assets.
                asset_weight = asset_cfg.get("weight", 1.0)
                if not force_small_account:
                    position_size_usd *= asset_weight

                # ── Config-level USD cap (max_position_usd) ──────────────────────
                # Same issue — the per-asset cap lived only in portfolio_manager.
                max_pos_usd = asset_cfg.get("max_position_usd")
                if not force_small_account and max_pos_usd and position_size_usd > max_pos_usd:
                    logger.info(
                        f"[SIZING] Capping ${position_size_usd:,.2f} at "
                        f"max_position_usd=${max_pos_usd:,.2f} for {asset}"
                    )
                    position_size_usd = max_pos_usd

                # ── Minimum SL distance floor (prevents tiny-SL → huge-notional) ─
                # When ATR is small (quiet session) a 2×ATR SL can be <0.3%, which
                # inflates position_size_usd explosively.  Per-asset floors:
                #   USTEC/indices : 0.5% (1H ATR ≈ 0.4-1.5% of price)
                #   EURUSD/EURJPY : 0.1% (FX ATR stops naturally 0.05-0.25%)
                #   GOLD/others   : 0.3% (XAU 1H ATR ≈ 0.3-0.9% of price)
                asset_type_upper = asset.upper()
                if asset_type_upper in ("USTEC", "US100", "NAS100"):
                    min_sl_pct = 0.005   # 0.5%
                elif asset_type_upper in ("EURUSD", "EURJPY"):
                    min_sl_pct = 0.001   # 0.1% — FX pairs have tight ATR-based stops
                else:
                    min_sl_pct = 0.003   # 0.3% (GOLD, others)
                min_sl_dist = current_price * min_sl_pct
                if not force_small_account and initial_sl_dist < min_sl_dist:
                    capped_stop_distance_pct = min_sl_pct
                    capped_size = risk_amount_usd / capped_stop_distance_pct
                    capped_size *= asset_weight
                    if max_pos_usd:
                        capped_size = min(capped_size, max_pos_usd)
                    logger.warning(
                        f"[SIZING] {asset}: SL distance {initial_sl_dist:.2f} "
                        f"({stop_distance_pct:.3%}) below min floor {min_sl_pct:.1%}. "
                        f"Capping position from ${position_size_usd:,.2f} "
                        f"→ ${capped_size:,.2f}"
                    )
                    position_size_usd = capped_size

                logger.info(
                    f"[SIZING] Calculation:\n"
                    f"  Account Balance: ${account_balance:,.2f}{' (MT5 venue)' if force_small_account or abs(mt5_balance - portfolio_balance) > 1.0 else ''}\n"
                    f"  Risk Budget:     {risk_pct:.3%} = ${risk_amount_usd:.2f}\n"
                    f"  Stop Distance:   {stop_distance_pct:.3%}\n"
                    f"  Asset Weight:    {asset_weight:.1f}x{' (bypassed: small-account)' if force_small_account else ''}\n"
                    f"  Position Size:   ${position_size_usd:,.2f}"
                    f"{'  [SMALL ACCOUNT — min lot forced]' if force_small_account else ''}"
                )
                if position_size_usd <= 0:
                    return False

                # Convert USD to MT5 lots
                contract_size = symbol_info.trade_contract_size
                if force_small_account:
                    # Already chose min lot; do not re-derive from notional.
                    volume_lots = symbol_info.volume_min
                else:
                    raw_volume_lots = position_size_usd / (current_price * contract_size)
                    volume_step = symbol_info.volume_step
                    volume_lots = round(raw_volume_lots / volume_step) * volume_step

                min_lot_notional_value = (
                    symbol_info.volume_min * current_price * contract_size
                )
                if not force_small_account and position_size_usd < min_lot_notional_value:
                    # ── Small-account fallback: use min lot instead of aborting ──
                    # Risk-based sizing produced a notional too small for the broker's
                    # minimum lot.  Rather than blocking the trade entirely, fall back
                    # to volume_min (the smallest tradeable unit) and let the margin
                    # check below decide whether the account can actually afford it.
                    logger.warning(
                        f"[SIZING] {asset}: Risk-based size (${position_size_usd:,.2f}) "
                        f"below min lot notional (${min_lot_notional_value:,.2f}). "
                        f"Falling back to min lot ({symbol_info.volume_min}) — "
                        f"Small Account Protocol activated."
                    )
                    volume_lots = symbol_info.volume_min
                    if signal_details is None:
                        signal_details = {}
                    signal_details["small_account_protocol_active"] = True
                else:
                    if volume_lots < symbol_info.volume_min:
                        volume_lots = symbol_info.volume_min

                    # Activate small-account protocol whenever we land at min lot
                    if volume_lots <= symbol_info.volume_min:
                        if signal_details is None:
                            signal_details = {}
                        signal_details["small_account_protocol_active"] = True
                        logger.info(
                            f"[MARGIN] 🛡️ Small Account Protocol: {asset} sized to min lot "
                            f"({symbol_info.volume_min}), enabling margin bypass."
                        )

                # Config-level per-asset max lot ceiling (harder than broker's volume_max)
                config_max_lots = asset_cfg.get("max_lots")
                if config_max_lots and volume_lots > config_max_lots:
                    logger.warning(
                        f"[SIZING] {asset}: lots {volume_lots:.2f} exceeds "
                        f"config max_lots={config_max_lots}. Capping."
                    )
                    volume_lots = config_max_lots

                volume_lots = min(symbol_info.volume_max, volume_lots)
                actual_usd = volume_lots * current_price * contract_size

            # Fetch OHLC for VTM initialization
            ohlc_data, df = self._fetch_ohlc_for_vtm(symbol, asset)

            # Margin validation
            small_account_active = signal_details.get("small_account_protocol_active", False) if signal_details else False
            if not self._validate_margin(volume_lots, current_price, symbol_info, small_account_active=small_account_active):
                return False

            # ── PRE-FLIGHT PORTFOLIO LIMITS CHECK ────────────────────────────
            # This must run BEFORE _execute_mt5_order so we never place a real
            # order on the exchange only to emergency-close it a millisecond
            # later because the NET margin limit is already breached.
            # Previously check_portfolio_limits() was only called inside
            # add_position() — AFTER the order was on the exchange, causing
            # ghost fills with no Telegram alert, no VTM, no portfolio tracking.
            side = "long" if signal == 1 else "short"
            if not small_account_active:
                if not self.portfolio_manager.check_portfolio_limits(
                    new_position_usd=actual_usd,
                    new_side=side,
                    asset=asset,
                ):
                    logger.warning(
                        f"[PRE-FLIGHT] ❌ {asset} {side.upper()} blocked — "
                        f"portfolio NET margin limit would be exceeded "
                        f"(notional ${actual_usd:,.2f}). Order NOT sent to exchange."
                    )
                    return False
                logger.info(
                    f"[PRE-FLIGHT] ✓ {asset} {side.upper()} passed portfolio "
                    f"margin check (notional ${actual_usd:,.2f})"
                )
            # ─────────────────────────────────────────────────────────────────

            # Execute order
            requested_price = current_price
            
            mt5_ticket, execution_price = self._execute_mt5_order(
                symbol, side, volume_lots, asset, trade_type, symbol_info
            )
            if not mt5_ticket:
                return False

            # ✅ TRACK SLIPPAGE
            slippage = abs(execution_price - requested_price)
            slippage_pct = (slippage / requested_price) * 100 if requested_price > 0 else 0
            logger.info(
                f"[SLIPPAGE] {asset} {side.upper()} | "
                f"Req: ${requested_price:,.2f}, Fill: ${execution_price:,.2f}, "
                f"Diff: ${slippage:,.2f} ({slippage_pct:.4f}%)"
            )

            # Add to Portfolio
            if signal_details is None:
                signal_details = {}
            signal_details.update(
                {"trade_type": trade_type, "strategic_risk_pct": risk_pct}
            )

            # ✅ Get lot precision from volume step
            import math
            lot_precision = 0
            if symbol_info.volume_step > 0:
                lot_precision = max(0, int(round(-math.log10(symbol_info.volume_step))))

            success = self.portfolio_manager.add_position(
                asset=asset,
                symbol=symbol,
                side=side,
                entry_price=execution_price,
                position_size_usd=actual_usd,
                mt5_ticket=mt5_ticket,
                ohlc_data=ohlc_data,
                use_dynamic_management=True,
                signal_details=signal_details,
                vtm_overrides=vtm_overrides,
                min_lot=symbol_info.volume_min,
                lot_precision=lot_precision,
                disable_partials=_disable_partials,
            )

            if success:
                # ✅ Standardized Log
                log_trade_event("ENTRY", {
                    "symbol": symbol,
                    "asset": asset,
                    "side": side,
                    "price": execution_price,
                    "size": volume_lots,
                    "trade_type": trade_type,
                    "position_id": str(mt5_ticket)
                })

                # ✅ Update last trade time for cooldown
                self.last_trade_time[asset] = time.time()
                self.trade_timestamps_hourly.append(time.time()) # Record for hourly limit
                
                # ✅ Update last trade time for cooldown
                self.last_trade_time[asset] = time.time()
                self.trade_timestamps_hourly.append(time.time()) # Record for hourly limit
                
                logger.info(
                    f"\n{'='*80}\n"
                    f"✅ {asset} {side.upper()} POSITION OPENED\n"
                    f"{'='*80}\n"
                    f"Trade Type:     {trade_type}\n"
                    f"Position Size:  ${actual_usd:,.2f}\n"
                    f"Lots:           {volume_lots:.2f}\n"
                    f"VTM Active:     {'Yes' if ohlc_data else 'No'}\n"
                    f"{'='*80}"
                )
                # ── VTM-SL/TP: push initial stop loss / take profit to exchange ───
                _asset_cfg_exchange = self.config.get("assets", {}).get(
                    asset, {}
                ).get("exchange", "mt5")
                
                _push_sl = self.trading_config.get("place_vtm_sl_on_exchange", False)
                _push_tp = self.trading_config.get("place_vtm_tp_on_exchange", False)

                if _asset_cfg_exchange == "mt5" and (_push_sl or _push_tp):
                    try:
                        _new_pos = next(
                            (p for p in self.portfolio_manager.positions.values()
                             if p.mt5_ticket == mt5_ticket
                             and getattr(p, "trade_manager", None) is not None),
                            None
                        )
                        if _new_pos:
                            if _push_sl:
                                _initial_sl = _new_pos.trade_manager.current_stop_loss
                                if _initial_sl:
                                    self._push_sl_to_exchange(mt5_ticket, symbol, _initial_sl)
                            
                            if _push_tp:
                                _initial_tp = _new_pos.trade_manager.current_take_profit
                                if _initial_tp:
                                    self._push_tp_to_exchange(mt5_ticket, symbol, _initial_tp)
                    except Exception as _e:
                        logger.warning(f"[VTM-EXCHANGE] Initial SL/TP push failed: {_e}")
                # ─────────────────────────────────────────────────────────────────
                return True
            else:
                if self.mode.lower() != "paper" and mt5_ticket:
                    logger.warning(f"[EMERGENCY] Closing orphaned MT5 #{mt5_ticket}")
                    self._emergency_close_mt5_position(
                        symbol,
                        volume_lots,
                        mt5.ORDER_TYPE_SELL if side == "long" else mt5.ORDER_TYPE_BUY,
                    )
                return False
        except Exception as e:
            logger.error(
                f"[MT5] ❌ Critical Error in _open_mt5_position for {asset}: {e}", exc_info=True
            )
            return False


    # ─────────────────────────────────────────────────────────────────────────
    # VTM-SL: push VTM-calculated stop loss onto the exchange
    # Only active when  trading.place_vtm_sl_on_exchange = true  in config.json
    # ─────────────────────────────────────────────────────────────────────────
    def _push_tp_to_exchange(self, ticket: int, symbol: str, new_tp: float) -> bool:
        """
        Modify the take-profit of an open MT5 position using TRADE_ACTION_SLTP.
        - Preserves the existing exchange SL.
        - Skips if TP has not meaningfully changed since the last push.
        - Always a no-op in paper mode.
        """
        if self.mode.lower() == "paper":
            return False
        try:
            # Guard: ignore micro-movements
            last = self._last_pushed_tp.get(ticket)
            if last is not None and abs(last - new_tp) < 0.00001:
                return False

            # Fetch live position to preserve its current SL
            live_positions = mt5.positions_get(ticket=ticket)
            if not live_positions:
                logger.warning(f"[VTM-TP] Ticket #{ticket} not found on exchange")
                return False
            existing_sl = live_positions[0].sl  # 0.0 if no SL set on exchange

            # Round to symbol precision
            sym_info = mt5.symbol_info(symbol)
            digits = sym_info.digits if sym_info else 5
            rounded_tp = round(new_tp, digits)

            request = {
                "action":   mt5.TRADE_ACTION_SLTP,
                "symbol":   symbol,
                "sl":       existing_sl,
                "tp":       rounded_tp,
                "position": ticket,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self._last_pushed_tp[ticket] = new_tp
                prev_str = f" (was {last:,.{digits}f})" if last is not None else " (initial)"
                logger.info(
                    f"[VTM-TP] ✅ #{ticket} {symbol}  TP → {rounded_tp:,.{digits}f}{prev_str}"
                )
                return True
            else:
                retcode = result.retcode if result else "N/A"
                comment = result.comment if result else "no result"
                logger.warning(
                    f"[VTM-TP] ⚠️ #{ticket} {symbol}  TP modify FAILED  "
                    f"retcode={retcode} — {comment}"
                )
                return False
        except Exception as e:
            logger.error(f"[VTM-TP] CRITICAL ERROR during TP push: {e}")
            return False

    def _push_sl_to_exchange(self, ticket: int, symbol: str, new_sl: float) -> bool:
        """
        Modify the stop-loss of an open MT5 position using TRADE_ACTION_SLTP.
        - Preserves the existing exchange TP.
        - Skips if SL has not meaningfully changed since the last push.
        - Always a no-op in paper mode.
        """
        if self.mode.lower() == "paper":
            return False
        try:
            # Guard: ignore micro-movements (< 1 pip equivalent)
            last = self._last_pushed_sl.get(ticket)
            if last is not None and abs(last - new_sl) < 0.00001:
                return False

            # Fetch live position to preserve its current TP
            live_positions = mt5.positions_get(ticket=ticket)
            if not live_positions:
                logger.warning(f"[VTM-SL] Ticket #{ticket} not found on exchange")
                return False
            existing_tp = live_positions[0].tp  # 0.0 if no TP set on exchange

            # Round to symbol precision
            sym_info = mt5.symbol_info(symbol)
            digits = sym_info.digits if sym_info else 5
            rounded_sl = round(new_sl, digits)

            request = {
                "action":   mt5.TRADE_ACTION_SLTP,
                "symbol":   symbol,
                "sl":       rounded_sl,
                "tp":       existing_tp,
                "position": ticket,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self._last_pushed_sl[ticket] = new_sl
                prev_str = f" (was {last:,.{digits}f})" if last is not None else " (initial)"
                logger.info(
                    f"[VTM-SL] ✅ #{ticket} {symbol}  SL → {rounded_sl:,.{digits}f}{prev_str}"
                )
                return True
            else:
                retcode = result.retcode if result else "N/A"
                comment = result.comment if result else "no result"
                logger.warning(
                    f"[VTM-SL] ⚠️ #{ticket} {symbol}  SL modify FAILED  "
                    f"retcode={retcode} — {comment}"
                )
                return False
        except Exception as e:
            logger.error(f"[VTM-SL] Error pushing SL to exchange: {e}")
            return False

    def _fetch_ohlc_for_vtm(self, symbol, asset):
        """Helper to fetch OHLC data for VTM."""
        ohlc_data, df = None, None
        if self.data_manager:
            try:
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=10)
                df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=self.config["assets"][asset].get("timeframe", "H1"),
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
                    logger.info(
                        f"[VTM] ✓ Fetched {len(df)} bars for dynamic management of {asset}"
                    )
                else:
                    logger.warning(f"[VTM] ⚠️ Insufficient data for {asset} ({len(df)} bars)")
            except Exception as e:
                logger.error(f"[VTM] ❌ OHLC fetch failed for {asset}: {e}")
        return ohlc_data, df

    def _validate_margin(self, volume_lots, current_price, symbol_info, small_account_active: bool = False):
        """Helper to perform pre-flight margin check."""
        if self.mode.lower() == "paper":
            return True
            
        # ✅ NEW: Bypass margin check for Small Account Protocol
        if small_account_active:
            logger.info("[MARGIN] 🛡️ Small Account Protocol active: Bypassing margin validation for MIN LOT.")
            return True
            
        try:
            account_info = mt5.account_info()
            if not account_info:
                return True

            # ✅ Use broker's native margin calculation (handles leverage/currency automatically)
            order_type = mt5.ORDER_TYPE_BUY 
            broker_margin = mt5.order_calc_margin(order_type, symbol_info.name, volume_lots, current_price)
            
            if broker_margin is None:
                logger.warning(f"[MARGIN] Broker calculation failed for {symbol_info.name}, using fallback.")
                leverage = account_info.leverage if account_info.leverage > 0 else 100
                # Fallback: Approximate notional (requires conversion usually, so this is very rough)
                estimated_margin = (volume_lots * symbol_info.trade_contract_size) / leverage
            else:
                estimated_margin = broker_margin

            estimated_margin *= 1.10  # 10% safety buffer
            
            logger.info(
                f"[MARGIN CHECK]\n"
                f"  Free Margin:      ${account_info.margin_free:,.2f}\n"
                f"  Required Margin:  ${estimated_margin:,.2f}\n"
                f"  Margin Level:     {account_info.margin_level:.2f}%"
            )
            
            if estimated_margin > account_info.margin_free:
                logger.error(
                    f"[MARGIN] ❌ Insufficient margin. Available: ${account_info.margin_free:,.2f}, Required: ${estimated_margin:,.2f}"
                )
                return False
                
        except Exception as e:
            logger.error(f"[MARGIN] Pre-flight error: {e}")
        return True

    def _execute_mt5_order(self, symbol, side, volume_lots, asset, trade_type, symbol_info):
        """Helper to execute order on MT5 or simulate in paper mode."""
        order_type = mt5.ORDER_TYPE_BUY if side == "long" else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        execution_price = (
            (tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid)
            if tick
            else None
        )
        if not execution_price:
            logger.error(f"[MT5] ❌ Could not get execution price for {symbol}")
            return None, None

        if self.mode.lower() == "paper":
            mt5_ticket = int(time.time())
            logger.info(
                f"[MT5] [PAPER MODE] ✓ {side.upper()} order simulated for {asset}: #{mt5_ticket}"
            )
            return mt5_ticket, execution_price

        if not self._is_trading_allowed(symbol):
            logger.error(f"[MT5] ❌ Trading not allowed for {symbol}")
            return None, None

        filling_mode = mt5.ORDER_FILLING_FOK
        if symbol_info.filling_mode == 2:
            filling_mode = mt5.ORDER_FILLING_IOC

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume_lots,
            "type": order_type,
            "price": execution_price,
            "sl": 0.0,
            "tp": 0.0,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Sig_{1 if side == 'long' else -1}_{asset}_{trade_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }
        MAX_RETRIES = 2
        result = None
        
        for attempt in range(MAX_RETRIES):
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                break
                
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"[RETRY] MT5 Order failed (Attempt {attempt+1}/{MAX_RETRIES}): "
                    f"{result.comment if result else 'No result'}. Retrying in 1s..."
                )
                time.sleep(1)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            actual_fill_price = result.price if result.price > 0 else execution_price
            logger.info(f"[MT5] ✓ {side.upper()} order placed for {asset}: #{result.order} @ ${actual_fill_price:,.2f}")
            return result.order, actual_fill_price
        else:
            last_error = mt5.last_error() or "Unknown error"
            logger.error(
                f"[MT5] ❌ Order Failed for {asset} after {MAX_RETRIES} attempts: "
                f"{result.comment if result else last_error}"
            )
            return None, None

    def _is_market_open_for_closing(self, symbol: str) -> Tuple[bool, str]:
        """
        ✅ NEW: Check if market is open for closing positions
        """
        try:
            import MetaTrader5 as mt5

            # 1. Get symbol info
            info = mt5.symbol_info(symbol)
            if info is None:
                return False, f"Symbol {symbol} not found"

            if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
                return False, "Market is CLOSED - Trading disabled"

            # Even in CLOSEONLY mode, we can close positions
            if info.trade_mode >= mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
                return True, "OK"

            # 3. Check for tick data
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False, "Market is CLOSED - No price quotes available"

            # 4. Check if last tick is recent (within 5 minutes)
            from datetime import datetime, timezone

            current_time = datetime.now(timezone.utc)
            tick_time = datetime.fromtimestamp(tick.time, timezone.utc)
            time_diff = (current_time - tick_time).total_seconds()

            if time_diff > 300:  # 5 minutes
                return (
                    False,
                    f"Market is CLOSED - Last quote was {int(time_diff/60)} minutes ago",
                )

            return True, "OK"

        except Exception as e:
            logger.error(f"[MT5] Error checking market status: {e}")
            return False, f"Error checking market status: {str(e)}"

    @handle_errors(
        component="mt5_handler",
        severity=ErrorSeverity.CRITICAL,
        notify=True,
        reraise=False,
        default_return=False,
    )
    def execute_signal(
        self,
        signal: int,
        symbol: str = None,
        asset_name: str = None,
        confidence_score: float = None,
        market_condition: str = None,
        sizing_mode: str = "automated",
        manual_size_usd: float = None,
        override_reason: str = None,
        signal_details: Dict = None,
    ) -> bool:
        """
        ✅ MT5 MULTI-ASSET TRADING: Refactored to support all enabled symbols
        """

        if not asset_name:
            logger.error("[MT5 HANDLER] ❌ No asset_name provided for execution")
            return False

        # ============================================================
        # RAPID-FIRE COOLDOWN (30s)
        # ============================================================
        now = time.time()
        last_time = self.last_trade_time.get(asset_name, 0)
        if now - last_time < 30:
            logger.warning(f"[COOLDOWN] Rapid-fire blocked for {asset_name} ({30 - (now - last_time):.1f}s remaining)")
            return False

        # ============================================================
        # DUPLICATE EXECUTION LOCK
        # ============================================================
        trade_type = "TREND"
        if signal_details:
            trade_type = signal_details.get("trade_type", "TREND")
            
        trade_key = f"{asset_name}_{trade_type}_{signal}"
        
        if self.execution_lock.get(trade_key, False):
            logger.warning(f"[LOCK] Duplicate execution blocked for {trade_key}")
            return False
            
        self.execution_lock[trade_key] = True

        try:
            # ============================================================
            # STEP 1: Get correct symbol and price
            # ============================================================
            if symbol is None:
                symbol = self._resolve_symbol(asset_name)

            if not symbol:
                logger.error(f"[MT5 HANDLER] ❌ Could not find symbol for asset: {asset_name}")
                return False

            current_price = self.get_current_price(symbol)
            if current_price == 0 or current_price is None:
                logger.error(f"{asset_name} ({symbol}): Failed to get current price")
                return False

            # ============================================================
            # STEP 2: Get existing positions & HEDGING CONFIG
            # ============================================================
            existing_positions = self.portfolio_manager.get_asset_positions(asset_name)

            long_positions = [p for p in existing_positions if p.side == "long"]
            short_positions = [p for p in existing_positions if p.side == "short"]

            allow_hedging = self.config.get("trading", {}).get(
                "allow_simultaneous_long_short", False
            )

            logger.info(
                f"\n{'='*80}\n"
                f"[SIGNAL] {asset_name} ({symbol}) Signal: {signal:+2d}\n"
                f"[STATE] Current Positions: {len(long_positions)} LONG, {len(short_positions)} SHORT\n"
                f"[CONFIG] Hedging Allowed: {allow_hedging}\n"
                f"{'='*80}"
            )

            # ============================================================
            # SCENARIO 1: SELL SIGNAL (-1) → Handle longs, Open short
            # ============================================================
            if signal == -1:
                if long_positions and not allow_hedging:
                    logger.info(f"📉 SELL SIGNAL - Closing {len(long_positions)} LONG position(s)")
                    for pos in long_positions:
                        self.portfolio_manager.close_position(
                            position_id=pos.position_id,
                            exit_price=current_price,
                            reason="sell_signal",
                        )
                    self._verify_position_sync(asset_name, symbol)

                can_open, reason = self.can_open_position_side(asset_name, "short")
                if not can_open:
                    logger.warning(f"⚠️  CANNOT OPEN SHORT for {asset_name}: {reason}")
                    return True if (long_positions and allow_hedging) else False

                return self._open_mt5_position(
                    signal=-1,
                    current_price=current_price,
                    symbol=symbol,
                    asset=asset_name,
                    confidence_score=confidence_score,
                    market_condition=market_condition,
                    sizing_mode=sizing_mode,
                    manual_size_usd=manual_size_usd,
                    override_reason=override_reason,
                    signal_details=signal_details,
                )

            # ============================================================
            # SCENARIO 2: BUY SIGNAL (+1) → Handle shorts, Open long
            # ============================================================
            elif signal == 1:
                if short_positions and not allow_hedging:
                    logger.info(f"📈 BUY SIGNAL - Closing {len(short_positions)} SHORT position(s)")
                    for pos in short_positions:
                        self.portfolio_manager.close_position(
                            position_id=pos.position_id,
                            exit_price=current_price,
                            reason="buy_signal",
                        )
                    self._verify_position_sync(asset_name, symbol)

                can_open, reason = self.can_open_position_side(asset_name, "long")
                if not can_open:
                    logger.warning(f"⚠️  CANNOT OPEN LONG for {asset_name}: {reason}")
                    return True if (short_positions and allow_hedging) else False

                return self._open_mt5_position(
                    signal=1,
                    current_price=current_price,
                    symbol=symbol,
                    asset=asset_name,
                    confidence_score=confidence_score,
                    market_condition=market_condition,
                    sizing_mode=sizing_mode,
                    manual_size_usd=manual_size_usd,
                    override_reason=override_reason,
                    signal_details=signal_details,
                )

            # ============================================================
            # SCENARIO 3: HOLD SIGNAL (0) → Check SL/TP
            # ============================================================
            elif signal == 0:
                if not existing_positions:
                    return False

                positions_closed = False
                for position in existing_positions:
                    should_close, close_reason = self._check_stop_loss_take_profit(
                        position, current_price
                    )
                    if should_close:
                        success = self.portfolio_manager.close_position(
                            position_id=position.position_id,
                            exit_price=current_price,
                            reason=close_reason,
                        )
                        if success:
                            positions_closed = True

                if positions_closed:
                    self._verify_position_sync(asset_name, symbol)
                return positions_closed

            return False

        except Exception as e:
            logger.error(f"Error executing {asset_name} signal: {e}", exc_info=True)
            return False
        finally:
            self.execution_lock[trade_key] = False

    # ============================================================================
    # HELPER METHOD - Position sync verification (already in your code)
    # ============================================================================

    def _verify_position_sync(self, asset_name: str, symbol: str):
        """
        ✅ Verify portfolio and MT5 are in sync after trade execution

        This is called after:
        - Opening new positions
        - Closing positions
        - Signal execution

        Logs detailed comparison and triggers re-sync if needed
        """
        if self.mode.lower() == "paper":
            logger.info(
                "[SYNC CHECK] Paper mode detected. Skipping MT5 position sync verification."
            )
            return True  # Always in sync in paper mode

        try:
            import MetaTrader5 as mt5

            # Get portfolio positions
            portfolio_positions = self.portfolio_manager.get_asset_positions(asset_name)
            portfolio_long = len([p for p in portfolio_positions if p.side == "long"])
            portfolio_short = len([p for p in portfolio_positions if p.side == "short"])

            # Get MT5 positions
            mt5_positions = mt5.positions_get(symbol=symbol)
            mt5_long = 0
            mt5_short = 0

            if mt5_positions:
                for pos in mt5_positions:
                    if pos.type == mt5.POSITION_TYPE_BUY:
                        mt5_long += 1
                    else:
                        mt5_short += 1

            # Compare
            sync_ok = (portfolio_long == mt5_long) and (portfolio_short == mt5_short)

            logger.info(
                f"\n{'='*80}\n"
                f"[SYNC CHECK] {asset_name}\n"
                f"{'='*80}\n"
                f"Portfolio:  {portfolio_long} LONG, {portfolio_short} SHORT (Total: {len(portfolio_positions)})\n"
                f"MT5:        {mt5_long} LONG, {mt5_short} SHORT (Total: {len(mt5_positions) if mt5_positions else 0})\n"
                f"Status:     {'✅ IN SYNC' if sync_ok else '⚠️  OUT OF SYNC'}\n"
                f"{'='*80}"
            )

            # If out of sync, trigger re-sync
            if not sync_ok:
                logger.warning(
                    f"[SYNC] Mismatch detected! Triggering automatic re-sync..."
                )
                self.sync_positions_with_mt5(asset_name, symbol)

            return sync_ok

        except Exception as e:
            logger.error(f"[SYNC CHECK] Error: {e}")
            return False

    @handle_errors(
        component="mt5_handler",
        severity=ErrorSeverity.CRITICAL,
        notify=True,
        reraise=False,
        default_return=False,
    )
    def _close_position(
        self, position, current_price: float, asset_name: str, reason: str
    ):
        """
        Close a single position.

        Returns
        -------
        dict | bool
            On success returns a dict with broker-authoritative fill data so the
            portfolio manager can use the *actual* fill price + broker P&L
            (including swap and commission) instead of computing P&L from a
            stale cached price.

            For backward compatibility, return value is still truthy on success
            and falsy on failure.
        """
        try:
            entry_price = position.entry_price
            quantity = position.quantity
            side = position.side
            position_id = position.position_id

            # Local pre-close estimate (kept for the log line only — the real
            # numbers come from the broker after the fill)
            est_size_usd = quantity * entry_price
            if side == "long":
                est_pnl = (current_price - entry_price) * quantity
            else:
                est_pnl = (entry_price - current_price) * quantity
            est_pnl_pct = (est_pnl / est_size_usd) * 100 if est_size_usd > 0 else 0

            logger.info(
                f"[CLOSE] {asset_name} {side.upper()} ({position_id}) — submitting close…\n"
                f"  Entry: ${entry_price:,.5f} → Pre-close cache: ${current_price:,.5f}\n"
                f"  Est P&L (pre-fill): ${est_pnl:,.2f} ({est_pnl_pct:+.2f}%)\n"
                f"  Reason: {reason}"
            )

            # Close on MT5 first (if live mode)
            if position.mt5_ticket:
                close_result = self._close_mt5_order(
                    position.mt5_ticket, asset_name, position.side
                )
            else:
                close_result = True  # No MT5 ticket, so no need to close on exchange

            if not close_result:
                return False

            # If we got a dict back, surface the broker fill data to the caller
            if isinstance(close_result, dict):
                fill_price = close_result.get("fill_price")
                broker_profit = close_result.get("profit")
                if fill_price is not None and broker_profit is not None:
                    logger.info(
                        f"[CLOSE] {asset_name} broker fill: ${fill_price:,.5f} | "
                        f"broker P&L: ${broker_profit:,.2f} "
                        f"(swap ${close_result.get('swap', 0):,.2f}, "
                        f"comm ${close_result.get('commission', 0):,.2f})"
                    )
                return close_result

            return True  # legacy True path (paper mode, no-ticket case)

        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
            return False

    def _partial_close_position(
        self, position, partial_qty: float, current_price: float, asset_name: str, reason: str
    ) -> bool:
        """
        Partially close an MT5 position by sending a reduce-only order for `partial_qty`
        (in base-asset units, e.g. oz for GOLD). Converts to lots using contract_size.
        The position ticket stays open with the remaining volume on the broker side.
        """
        try:
            if not position.mt5_ticket:
                logger.warning(f"[PARTIAL-MT5] No ticket for {asset_name} — cannot partial close")
                return False

            symbol = self._resolve_symbol(asset_name)
            if not symbol:
                logger.error(f"[PARTIAL-MT5] Symbol not found for {asset_name}")
                return False

            # Paper mode simulation
            if self.mode.lower() == "paper":
                logger.info(f"[PARTIAL-MT5] [PAPER] Simulated partial close {partial_qty:.6f} units for {asset_name}")
                return True

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"[PARTIAL-MT5] symbol_info not available for {symbol}")
                return False

            contract_size = symbol_info.trade_contract_size
            raw_lots = partial_qty / contract_size
            volume_step = symbol_info.volume_step
            partial_lots = round(raw_lots / volume_step) * volume_step
            partial_lots = max(symbol_info.volume_min, round(partial_lots, 8))

            is_open, market_msg = self._is_market_open_for_closing(symbol)
            if not is_open:
                logger.error(f"[PARTIAL-MT5] Market closed for {asset_name}: {market_msg}")
                return False

            mt5_positions = mt5.positions_get(ticket=position.mt5_ticket)
            if not mt5_positions:
                logger.warning(f"[PARTIAL-MT5] Ticket {position.mt5_ticket} not found on broker")
                return True  # Already closed externally

            mt5_pos = mt5_positions[0]
            order_type = mt5.ORDER_TYPE_SELL if mt5_pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"[PARTIAL-MT5] No tick data for {symbol}")
                return False

            close_price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

            # Cap to remaining broker volume to avoid INVALID_VOLUME errors
            partial_lots = min(partial_lots, mt5_pos.volume)

            request = {
                "action":      mt5.TRADE_ACTION_DEAL,
                "symbol":      symbol,
                "volume":      partial_lots,
                "type":        order_type,
                "position":    position.mt5_ticket,
                "price":       close_price,
                "deviation":   20,
                "magic":       234000,
                "comment":     f"PartialTP_{asset_name}",
                "type_time":   mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"[PARTIAL-MT5] ✓ Partial close {partial_lots:.4f} lots for {asset_name} "
                    f"@ ${close_price:,.2f} (ticket {position.mt5_ticket})"
                )
                return True
            else:
                error_msg  = result.comment if result else "No result"
                error_code = result.retcode if result else "N/A"
                logger.error(
                    f"[PARTIAL-MT5] ✗ Failed for {asset_name}: {error_msg} (code: {error_code})"
                )
                return False

        except Exception as e:
            logger.error(f"[PARTIAL-MT5] Exception for {asset_name}: {e}", exc_info=True)
            return False

    def _close_mt5_order(self, ticket: int, asset: str, side: str):
        """
        Close an MT5 order and return the broker's authoritative fill data.

        Returns
        -------
        dict | bool
            On success: {
                "ok": True,
                "fill_price": float,         # actual broker fill price
                "profit": float | None,      # broker P&L (incl. swap+commission) if available
                "swap": float,
                "commission": float,
                "deal_ticket": int | None,
                "volume_closed": float,      # in lots
            }
            On failure: False
            For backward compatibility, callers that only check truthiness still work.
        """
        # In paper mode, simulate successful closure
        if self.mode.lower() == "paper":
            logger.info(f"[MT5] [PAPER MODE] ✓ Simulated close of ticket {ticket}")
            return {"ok": True, "fill_price": None, "profit": None,
                    "swap": 0.0, "commission": 0.0,
                    "deal_ticket": None, "volume_closed": 0.0}

        try:
            # Dynamic symbol lookup
            symbol = self._resolve_symbol(asset)
            if not symbol:
                logger.error(f"[MT5] Cannot close ticket {ticket}: Symbol not found for asset {asset}")
                return False

            # ✅ Check if market is open for closing
            is_open, market_msg = self._is_market_open_for_closing(symbol)

            if not is_open:
                logger.error(
                    f"[MT5] ❌ CANNOT CLOSE POSITION\n"
                    f"  Ticket: {ticket}\n"
                    f"  Asset:  {asset}\n"
                    f"  Reason: {market_msg}\n"
                    f"  → Position will remain open until market reopens"
                )
                return False

            # Find the position by ticket
            mt5_positions = mt5.positions_get(ticket=ticket)

            if mt5_positions is None or len(mt5_positions) == 0:
                logger.warning(f"[MT5] Position ticket {ticket} not found on exchange. Clearing local record.")
                # Position already gone — try to look up the closing deal in history
                # so the caller still gets authoritative profit numbers.
                hist_data = self._fetch_broker_close_data(ticket)
                if hist_data:
                    return {"ok": True, **hist_data}
                return True  # legacy: missing position is treated as "already closed"

            mt5_position = mt5_positions[0]

            # Capture pre-close broker numbers as a sanity baseline
            pre_close_profit = float(getattr(mt5_position, "profit", 0.0) or 0.0)
            pre_close_swap = float(getattr(mt5_position, "swap", 0.0) or 0.0)

            # Determine close order type
            order_type = (
                mt5.ORDER_TYPE_SELL
                if mt5_position.type == mt5.POSITION_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )

            # Get current price
            tick = mt5.symbol_info_tick(mt5_position.symbol)
            if tick is None:
                logger.error(
                    f"[MT5] Cannot get current price for {mt5_position.symbol}"
                )
                return False

            close_price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

            # Build close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_position.symbol,
                "volume": mt5_position.volume,
                "type": order_type,
                "position": ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Close_{asset}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send order
            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # ── AUTHORITATIVE FILL DATA ─────────────────────────────────
                # result.price is the *actual* fill, not the snapshot we sent.
                actual_fill = float(getattr(result, "price", close_price) or close_price)

                # Pull the closing deal from history for true profit (incl. swap
                # & commission). MT5 server may take a moment; retry briefly.
                hist_data = self._fetch_broker_close_data(ticket, deal_id=getattr(result, "deal", None))

                logger.info(
                    f"[MT5] ✓ Closed ticket {ticket} @ ${actual_fill:,.5f} "
                    f"(req: ${close_price:,.5f})"
                )

                return {
                    "ok": True,
                    "fill_price": actual_fill,
                    "profit": (hist_data or {}).get("profit", pre_close_profit),
                    "swap": (hist_data or {}).get("swap", pre_close_swap),
                    "commission": (hist_data or {}).get("commission", 0.0),
                    "deal_ticket": getattr(result, "deal", None),
                    "volume_closed": float(mt5_position.volume),
                }
            else:
                error_msg = result.comment if result else "No result"
                error_code = result.retcode if result else "N/A"

                if result and result.retcode == 10018:  # Market closed
                    logger.error(
                        f"[MT5] ✗ MARKET CLOSED - Cannot close ticket {ticket}\n"
                        f"  Error: {error_msg}\n"
                        f"  → Position remains open, try again when market opens"
                    )
                else:
                    logger.error(
                        f"[MT5] ✗ Failed to close ticket {ticket}: {error_msg} (code: {error_code})"
                    )

                return False

        except Exception as e:
            logger.error(f"[MT5] Error closing ticket {ticket}: {e}", exc_info=True)
            return False

    def _fetch_broker_close_data(self, ticket: int, deal_id=None, max_attempts: int = 6):
        """Look up the closing deal in MT5 history to read authoritative P&L.

        MT5's history is sometimes briefly behind the order_send response, so we
        poll a few times. Returns dict with profit/swap/commission/fill_price, or
        None if we couldn't read it (caller falls back to local calc).
        """
        try:
            import time as _time
            from datetime import datetime as _dt, timedelta as _td

            for attempt in range(max_attempts):
                deals = None
                # Prefer fetching the specific deal if we have its id
                if deal_id:
                    try:
                        deals = mt5.history_deals_get(ticket=deal_id)
                    except Exception:
                        deals = None
                if not deals:
                    # Fall back to all deals for this position
                    try:
                        deals = mt5.history_deals_get(position=ticket)
                    except Exception:
                        deals = None
                if not deals:
                    # Last resort: scan a small recent window
                    try:
                        end = _dt.now()
                        start = end - _td(minutes=10)
                        deals = mt5.history_deals_get(start, end)
                        if deals:
                            deals = [d for d in deals if getattr(d, "position_id", None) == ticket]
                    except Exception:
                        deals = None

                if deals:
                    # Pick the deal with the largest |volume| that closed this
                    # position (entry deals have profit==0; exit deals carry it).
                    closing = [d for d in deals if float(getattr(d, "profit", 0.0) or 0.0) != 0.0]
                    if not closing:
                        # No profit-carrying deal yet (still settling); take last by time
                        closing = sorted(deals, key=lambda d: getattr(d, "time", 0))
                    if closing:
                        d = closing[-1]
                        return {
                            "profit": float(getattr(d, "profit", 0.0) or 0.0),
                            "swap": float(getattr(d, "swap", 0.0) or 0.0),
                            "commission": float(getattr(d, "commission", 0.0) or 0.0),
                            "fill_price": float(getattr(d, "price", 0.0) or 0.0) or None,
                        }

                _time.sleep(0.25)  # brief backoff before retry

            logger.debug(f"[MT5] Could not fetch closing deal for ticket {ticket} after {max_attempts} attempts")
            return None
        except Exception as e:
            logger.debug(f"[MT5] _fetch_broker_close_data error for ticket {ticket}: {e}")
            return None

    def _check_stop_loss_take_profit(
        self, position, current_price: float
    ) -> Tuple[bool, str]:
        """Check if stop-loss or take-profit is hit"""
        try:
            if hasattr(position, "entry_price"):
                entry_price = position.entry_price
                stop_loss = position.stop_loss
                take_profit = position.take_profit
                side = position.side
            else:
                entry_price = position.get("entry_price")
                stop_loss = position.get("stop_loss")
                take_profit = position.get("take_profit")
                side = position.get("side")

            price_tolerance = 0.01

            if side == "long":
                if stop_loss and current_price <= (stop_loss + price_tolerance):
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    return (
                        True,
                        f"stop_loss_hit (${current_price:.2f} <= ${stop_loss:.2f}, {pnl_pct:+.2f}%)",
                    )

                if take_profit and current_price >= (take_profit - price_tolerance):
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    return (
                        True,
                        f"take_profit_hit (${current_price:.2f} >= ${take_profit:.2f}, {pnl_pct:+.2f}%)",
                    )

            else:  # short
                if stop_loss and current_price >= (stop_loss - price_tolerance):
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    return (
                        True,
                        f"stop_loss_hit (${current_price:.2f} >= ${stop_loss:.2f}, {pnl_pct:+.2f}%)",
                    )

                if take_profit and current_price <= (take_profit + price_tolerance):
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    return (
                        True,
                        f"take_profit_hit (${current_price:.2f} <= ${take_profit:.2f}, {pnl_pct:+.2f}%)",
                    )

            return False, ""

        except Exception as e:
            logger.error(f"Error checking SL/TP: {e}", exc_info=True)
            return False, ""

    def check_and_update_positions(self, asset_name: str, df_4h: Optional[pd.DataFrame] = None):
        """Actively check and update all positions using VTM"""
        return self.check_and_update_positions_VTM(asset_name, df_4h=df_4h)

    def check_and_update_positions_VTM(self, asset_name: str, df_4h: Optional[pd.DataFrame] = None):
        """High-frequency VTM update loop for MT5 positions."""
        try:
            positions = self.portfolio_manager.get_asset_positions(asset_name)
            if not positions:
                return False

            symbol = self._resolve_symbol(asset_name)
            if not symbol:
                logger.error(f"[VTM LOOP] Could not find symbol for asset {asset_name}")
                return False

            current_price = self.get_current_price(symbol=symbol, force_live=True)
            if not current_price:
                return False

            # ✅ RECONCILIATION: Fetch live positions from broker
            try:
                broker_positions = mt5.positions_get(symbol=symbol)
                if broker_positions is not None:
                    # Convert to simple dicts for PortfolioManager
                    # Match by ticket ID for MT5
                    broker_data = [{'id': str(p.ticket), 'side': 'long' if p.type == mt5.POSITION_TYPE_BUY else 'short', 'quantity': p.volume} for p in broker_positions]
                    self.portfolio_manager.reconcile_positions(asset_name, broker_data)
            except Exception as e:
                logger.debug(f"[RECONCILE] MT5 fetch failed: {e}")

            positions_closed = False
            pyramid_requests = []

            for position in positions:
                # Update P&L from exchange if possible
                if position.mt5_ticket:
                    try:
                        import MetaTrader5 as mt5_internal
                        mt5_positions = mt5_internal.positions_get(ticket=position.mt5_ticket)
                        if mt5_positions:
                            position.mt5_profit = mt5_positions[0].profit
                            position.mt5_last_update = datetime.now()
                    except Exception as e:
                        logger.debug(f"Failed to update MT5 profit for {position.position_id}: {e}")

                if position.trade_manager:
                    # VTM-SL/TP: snapshot values before update to detect movement
                    _sl_before = position.trade_manager.current_stop_loss
                    _tp_before = position.trade_manager.current_take_profit
                    
                    exit_signal = position.trade_manager.update_with_current_price(
                        current_price, df_4h=df_4h
                    )
                    
                    _sl_after = position.trade_manager.current_stop_loss
                    _tp_after = position.trade_manager.current_take_profit

                    # Push SL/TP to exchange whenever VTM moves it (trailing, scaling, etc.)
                    _is_closing = (
                        exit_signal is not None
                        and not (isinstance(exit_signal, dict) and "action" in exit_signal)
                    )
                    _asset_exchange = self.config.get("assets", {}).get(
                        asset_name, {}
                    ).get("exchange", "mt5")
                    
                    if not _is_closing and position.mt5_ticket and _asset_exchange == "mt5":
                        # Always resolve via _resolve_symbol so that dual-exchange
                        # assets like BTC use their MT5 symbol ("BTCUSDm") instead of
                        # the Binance symbol ("BTCUSDT") that sits under the bare
                        # config["assets"][asset]["symbol"] key.  Using the wrong
                        # symbol causes every SL/TP push to fail silently, leaving the
                        # original exchange stop in place indefinitely.
                        _sym = self._resolve_symbol(asset_name)

                        if _sym:
                            # Push SL if moved
                            if (self.trading_config.get("place_vtm_sl_on_exchange", False)
                                and _sl_after is not None and _sl_before != _sl_after):
                                self._push_sl_to_exchange(position.mt5_ticket, _sym, _sl_after)
                                
                            # Push TP if moved
                            if (self.trading_config.get("place_vtm_tp_on_exchange", False)
                                and _tp_after is not None and _tp_before != _tp_after):
                                self._push_tp_to_exchange(position.mt5_ticket, _sym, _tp_after)

                    if exit_signal:
                        # ✅ Check if it's an action (like pyramid) or an exit (reason)
                        if isinstance(exit_signal, dict) and "action" in exit_signal:
                            action = exit_signal["action"]
                            logger.info(f"[VTM LOOP] {position.position_id} triggered action: {action.upper()}")
                            
                            # Add to pyramid requests to be returned to main loop
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
                        exit_reason_str = exit_reason.value if hasattr(exit_reason, "value") else str(exit_reason)

                        logger.info(f"[VTM LOOP] {position.position_id} triggered {exit_reason_str.upper()}")

                        self.portfolio_manager.close_position(
                            position_id=position.position_id,
                            exit_price=exit_signal.get("price", current_price) if isinstance(exit_signal, dict) else current_price,
                            reason=f"VTM_{exit_reason_str}",
                        )
                        positions_closed = True
            
            return {"closed": positions_closed, "pyramid_requests": pyramid_requests}

        except Exception as e:
            logger.error(f"[VTM LOOP] Error in MT5 VTM update: {e}", exc_info=True)
            return False

    def _emergency_close_mt5_position(
        self, symbol: str, volume: float, original_order_type: int
    ):
        """Emergency close of an unwanted MT5 position"""
        try:
            close_order_type = (
                mt5.ORDER_TYPE_SELL
                if original_order_type == mt5.ORDER_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )

            tick = mt5.symbol_info_tick(symbol)
            close_price = (
                tick.bid if close_order_type == mt5.ORDER_TYPE_SELL else tick.ask
            )

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_order_type,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "Emergency_close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[MT5] ✓ Emergency close successful")
            else:
                logger.error(f"[MT5] ✗ Emergency close failed: {result.comment if result else 'No result'}")

        except Exception as e:
            logger.error(f"[MT5] Emergency close error: {e}", exc_info=True)

    @handle_errors(
        component="mt5_handler",
        severity=ErrorSeverity.WARNING,
        notify=True,
        reraise=False,
        default_return=False,
    )
    def sync_positions_with_mt5(self, asset: str, symbol: str = None) -> bool:
        """
        ✅ FIXED: Import positions WITH multi-asset support
        """
        if symbol is None:
            symbol = self._resolve_symbol(asset)

        if not symbol:
            logger.error(f"[SYNC] No symbol found for {asset}")
            return False

        try:
            import MetaTrader5 as mt5
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"[SYNC] Symbol info for {symbol} unavailable; skipping.")
                return True

            logger.info(f"\n{'='*80}")
            logger.info(f"[SYNC] Starting position sync for {asset} ({symbol})")
            logger.info(f"{'='*80}")

            # Get MT5 positions
            mt5_positions = mt5.positions_get(symbol=symbol)
            mt5_count = len(mt5_positions) if mt5_positions else 0

            # Get portfolio positions
            portfolio_positions = self.portfolio_manager.get_asset_positions(asset)
            portfolio_count = len(portfolio_positions)

            # Count by side
            mt5_long = sum(1 for p in (mt5_positions or []) if p.type == mt5.POSITION_TYPE_BUY)
            mt5_short = sum(1 for p in (mt5_positions or []) if p.type == mt5.POSITION_TYPE_SELL)
            portfolio_long = sum(1 for p in portfolio_positions if p.side == "long")
            portfolio_short = sum(1 for p in portfolio_positions if p.side == "short")

            logger.info(f"[SYNC] MT5: {mt5_long}L / {mt5_short}S | Portfolio: {portfolio_long}L / {portfolio_short}S")

            # ================================================================
            # SCENARIO 1: IMPORT — handles both full (portfolio=0) and
            # partial (portfolio < mt5_count) mismatches.
            # Any MT5 position whose ticket is not already tracked is adopted.
            # ================================================================
            import_enabled = bool(self.config.get("portfolio", {}).get("import_existing_positions", True))

            if mt5_count > 0 and import_enabled:
                # Build set of tickets already tracked in portfolio
                tracked_tickets = {
                    pos.mt5_ticket
                    for pos in portfolio_positions
                    if pos.mt5_ticket is not None
                }

                orphaned = [p for p in mt5_positions if p.ticket not in tracked_tickets]

                if orphaned:
                    logger.info(
                        f"[SYNC] Found {len(orphaned)} untracked MT5 position(s) for {asset} "
                        f"(tracked tickets: {tracked_tickets}) — importing..."
                    )

                    # Fetch OHLC once for VTM initialisation
                    import math
                    ohlc_data = None
                    try:
                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=10)
                        df = self.data_manager.fetch_mt5_data(
                            symbol=symbol,
                            timeframe=self.config["assets"][asset].get("timeframe", "H1"),
                            start_date=start_time.strftime("%Y-%m-%d"),
                            end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        if len(df) > 50:
                            ohlc_data = {
                                "high": df["high"].values, "low": df["low"].values,
                                "close": df["close"].values, "volume": df["volume"].values,
                            }
                    except Exception as e:
                        logger.error(f"[SYNC] OHLC fetch failed: {e}")

                    lot_precision = 0
                    if symbol_info.volume_step > 0:
                        lot_precision = max(0, int(round(-math.log10(symbol_info.volume_step))))

                    imported_count = 0
                    for pos in orphaned:
                        pos_type = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"

                        success = self.portfolio_manager.add_position(
                            asset=asset,
                            symbol=symbol,
                            side=pos_type,
                            entry_price=pos.price_open,
                            position_size_usd=(pos.volume * pos.price_open * symbol_info.trade_contract_size),
                            stop_loss=pos.sl if pos.sl > 0 else None,
                            take_profit=pos.tp if pos.tp > 0 else None,
                            mt5_ticket=pos.ticket,
                            ohlc_data=ohlc_data,
                            use_dynamic_management=True,
                            entry_time=datetime.fromtimestamp(pos.time),
                            signal_details={"imported": True, "ticket": pos.ticket},
                            min_lot=symbol_info.volume_min,
                            lot_precision=lot_precision,
                        )
                        if success:
                            imported_count += 1
                            logger.info(
                                f"[SYNC] ✅ Adopted {pos_type} ticket #{pos.ticket} "
                                f"@ {pos.price_open} for {asset}"
                            )
                        else:
                            logger.warning(
                                f"[SYNC] ⚠️ Failed to adopt ticket #{pos.ticket} for {asset}"
                            )

                    logger.info(
                        f"[SYNC] Imported {imported_count}/{len(orphaned)} orphaned position(s) for {asset}"
                    )
                else:
                    logger.info(f"[SYNC] All MT5 positions for {asset} are already tracked — no import needed")

            # ================================================================
            # SCENARIO 2: CLEANUP — portfolio tracks positions MT5 no longer has
            # Ticket-level check: catches partial closes (e.g. one of two positions
            # closed directly on the exchange while the bot was running).
            # ================================================================
            if portfolio_count > 0:
                live_tickets = {p.ticket for p in (mt5_positions or [])}
                current_price = None  # Lazy fetch — only if needed

                for pos in list(portfolio_positions):
                    if pos.mt5_ticket is None:
                        continue  # Position has no ticket (paper / imported without ticket)
                    if pos.mt5_ticket not in live_tickets:
                        logger.warning(
                            f"[SYNC] ⚠️ Ticket #{pos.mt5_ticket} ({asset} {pos.side.upper()}) "
                            f"no longer on MT5 — externally closed, fetching broker close data."
                        )

                        # ── Query MT5 deal history for the authoritative fill price
                        # and P&L.  _fetch_broker_close_data polls with a short
                        # backoff so it handles the brief MT5 history lag after a close.
                        broker_data = self._fetch_broker_close_data(pos.mt5_ticket)

                        exit_price = None
                        if broker_data:
                            exit_price = broker_data.get("fill_price")
                            logger.info(
                                f"[SYNC] Broker close data for #{pos.mt5_ticket}: "
                                f"fill=${exit_price}, profit=${broker_data.get('profit', 'n/a')}, "
                                f"swap=${broker_data.get('swap', 0)}, "
                                f"commission=${broker_data.get('commission', 0)}"
                            )
                        else:
                            logger.debug(
                                f"[SYNC] No broker deal found for #{pos.mt5_ticket} — "
                                f"will fall back to current market price."
                            )

                        if exit_price is None or exit_price <= 0:
                            # Deal history unavailable; use current market price as fallback
                            if current_price is None:
                                current_price = self.get_current_price(symbol)
                            exit_price = current_price
                            logger.debug(
                                f"[SYNC] Falling back to market price ${exit_price} "
                                f"for #{pos.mt5_ticket} P&L calc."
                            )

                        self.portfolio_manager.close_position(
                            position_id=pos.position_id,
                            exit_price=exit_price,
                            reason="closed_on_exchange",
                            already_closed_on_exchange=True,
                            preloaded_broker_data=broker_data,
                        )

            return True

        except Exception as e:
            logger.error(f"[SYNC] Error for {asset}: {e}")
            return False

            # ================================================================
            # Dead code below — kept as reference only
            # ================================================================
            if mt5_count > 0 and portfolio_count == 0:
                import_enabled = bool(
                    self.config.get("portfolio", {}).get(
                        "import_existing_positions", False
                    )
                )

                logger.info(f"[SYNC] Config check:")
                logger.info(f"  portfolio.import_existing_positions = {import_enabled}")

                if import_enabled:
                    logger.info(
                        f"[SYNC] ✅ Import ENABLED - Importing {mt5_count} MT5 position(s) WITH VTM..."
                    )

                    # ============================================================
                    # STEP 1: Fetch OHLC data for VTM
                    # ============================================================
                    ohlc_data = None
                    df = None

                    try:
                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(days=10)

                        df = self.data_manager.fetch_mt5_data(
                            symbol=symbol,
                            timeframe=self.config["assets"][asset].get(
                                "timeframe", "H1"
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
                            logger.info(
                                f"[VTM] ✅ Fetched {len(df)} bars for dynamic management"
                            )
                        else:
                            logger.warning(
                                f"[VTM] ⚠️ Insufficient data ({len(df)} bars), VTM will be limited"
                            )

                    except Exception as e:
                        logger.error(f"[VTM] ❌ Failed to fetch OHLC: {e}")
                        ohlc_data = None
                        df = None

                    # ============================================================
                    # STEP 2: Get REAL market analysis from HybridAggregatorSelector
                    # ============================================================
                    signal_details_base = None

                    if df is not None and len(df) > 200:
                        try:
                            logger.info(
                                f"[HYBRID] Analyzing market for imported positions..."
                            )

                            # Try to get hybrid selector from parent bot
                            hybrid_selector = None
                            if hasattr(self, "trading_bot") and hasattr(
                                self.trading_bot, "hybrid_selector"
                            ):
                                hybrid_selector = self.trading_bot.hybrid_selector
                                logger.info(
                                    f"[HYBRID] Using existing hybrid_selector from bot"
                                )
                            else:
                                # Create temporary instance
                                from src.execution.hybrid_aggregator_selector import (
                                    HybridAggregatorSelector,
                                )

                                hybrid_selector = HybridAggregatorSelector(
                                    self.data_manager,
                                    self.config,
                                )
                                logger.info(
                                    f"[HYBRID] Created temporary hybrid_selector"
                                )

                            # Get current market analysis
                            mode_info = hybrid_selector.get_optimal_mode(asset, df)
                            analysis = mode_info["analysis"]

                            # Build REAL signal_details from market analysis
                            signal_details_base = {
                                "imported": True,
                                "import_time": datetime.now().isoformat(),
                                # Real aggregator mode
                                "aggregator_mode": mode_info["mode"],
                                "mode_confidence": mode_info["confidence"],
                                # Real regime analysis
                                "regime_analysis": {
                                    "regime_type": analysis["regime_type"],
                                    "trend_strength": analysis["trend"]["strength"],
                                    "trend_direction": analysis["trend"]["direction"],
                                    "adx": analysis["trend"]["adx"],
                                    "volatility_regime": analysis["volatility"][
                                        "regime"
                                    ],
                                    "volatility_ratio": analysis["volatility"]["ratio"],
                                    "price_clarity": analysis["price_action"][
                                        "clarity"
                                    ],
                                    "indecision_pct": analysis["price_action"][
                                        "indecision_pct"
                                    ],
                                    "momentum_aligned": analysis["momentum_aligned"],
                                    "at_key_level": analysis["at_key_level"],
                                },
                                "signal_quality": mode_info["confidence"],
                                "reasoning": f"Position imported from MT5 - {analysis['reasoning']}",
                            }

                            logger.info(f"[HYBRID] ✅ Market analysis complete:")
                            logger.info(f"  Mode:       {mode_info['mode'].upper()}")
                            logger.info(f"  Confidence: {mode_info['confidence']:.0%}")
                            logger.info(f"  Regime:     {analysis['regime_type']}")
                            logger.info(
                                f"  Trend:      {analysis['trend']['strength']} / {analysis['trend']['direction']}"
                            )
                            logger.info(
                                f"  Volatility: {analysis['volatility']['regime']}"
                            )

                        except Exception as e:
                            logger.error(f"[HYBRID] ❌ Analysis failed: {e}")
                            signal_details_base = None

                    # Fallback if hybrid analysis fails
                    if signal_details_base is None:
                        logger.warning(
                            f"[HYBRID] Using fallback signal_details (no market analysis)"
                        )
                        signal_details_base = {
                            "imported": True,
                            "import_time": datetime.now().isoformat(),
                            "aggregator_mode": "unknown",
                            "mode_confidence": 0.5,
                            "regime_analysis": {
                                "regime_type": "unknown",
                                "trend_strength": "unknown",
                                "trend_direction": "unknown",
                                "adx": 20.0,
                                "volatility_regime": "normal",
                                "volatility_ratio": 1.0,
                                "price_clarity": "unknown",
                                "indecision_pct": 0.0,
                                "momentum_aligned": False,
                                "at_key_level": False,
                            },
                            "signal_quality": 0.5,
                            "reasoning": "Position imported from MT5 - market analysis unavailable",
                        }

                    # ============================================================
                    # STEP 3: Get actual account balance
                    # ============================================================
                    try:
                        import MetaTrader5 as mt5

                        account_info = mt5.account_info()
                        account_balance = (
                            account_info.equity
                            if account_info
                            else self.portfolio_manager.current_capital
                        )
                        logger.info(f"[MT5] Account balance: ${account_balance:,.2f}")
                    except:
                        account_balance = self.portfolio_manager.current_capital
                        logger.warning(
                            f"[MT5] Using portfolio capital: ${account_balance:,.2f}"
                        )

                    # ============================================================
                    # STEP 4: Import each position
                    # ============================================================
                    imported_count = 0
                    for pos in mt5_positions:
                        pos_type = (
                            "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                        )

                        logger.info(
                            f"\n  → Importing MT5 {pos_type.upper()}: ticket={pos.ticket}, "
                            f"entry=${pos.price_open:.2f}, current=${pos.price_current:.2f}"
                        )

                        # Check if we can import
                        can_import, reason = self.portfolio_manager.can_open_position(
                            asset, pos_type
                        )
                        if not can_import:
                            logger.warning(f"[SYNC] ⚠️ Cannot import position: {reason}")
                            continue

                        # Add position-specific details
                        signal_details = signal_details_base.copy()
                        signal_details["mt5_ticket"] = pos.ticket
                        signal_details["side"] = pos_type
                        signal_details["entry_price"] = pos.price_open

                        # Import position
                        # The above code is attempting to call the `add_position` method of the
                        # `portfolio_manager` object within the `self` context.
                        success = self.portfolio_manager.add_position(
                            asset=asset,
                            symbol=symbol,
                            side=pos_type,
                            entry_price=pos.price_open,
                            position_size_usd=(
                                pos.volume
                                * pos.price_open
                                * self.symbol_info.trade_contract_size
                            ),
                            stop_loss=pos.sl if pos.sl > 0 else None,
                            take_profit=pos.tp if pos.tp > 0 else None,
                            trailing_stop_pct=self.config["assets"][asset]
                            .get("risk", {})
                            .get("trailing_stop_pct"),
                            mt5_ticket=pos.ticket,
                            ohlc_data=ohlc_data,
                            use_dynamic_management=True,
                            entry_time=datetime.fromtimestamp(pos.time),
                            signal_details=signal_details,
                            # account_balance=account_balance,
                        )

                        if success:
                            imported_count += 1

                            # Verify VTM initialized
                            imported_positions = (
                                self.portfolio_manager.get_asset_positions(asset)
                            )
                            if imported_positions:
                                imported_pos = imported_positions[-1]
                                if imported_pos.trade_manager:
                                    vtm_status = imported_pos.get_vtm_status()
                                    logger.info(
                                        f"[VTM] ✅ ACTIVE with market analysis\n"
                                        f"      Ticket:  {pos.ticket}\n"
                                        f"      Mode:    {signal_details['aggregator_mode'].upper()}\n"
                                        f"      Regime:  {signal_details['regime_analysis']['regime_type']}\n"
                                        f"      Entry:   ${vtm_status['entry_price']:,.2f}\n"
                                        f"      SL:      ${vtm_status['stop_loss']:,.2f} (VTM calculated)\n"
                                        f"      TP:      ${vtm_status.get('take_profit', 0):,.2f} (VTM calculated)"
                                    )
                                else:
                                    logger.error(
                                        f"[VTM] ❌ NOT INITIALIZED for ticket {pos.ticket}"
                                    )
                                    logger.error(
                                        f"      OHLC data: {ohlc_data is not None}"
                                    )
                                    logger.error(
                                        f"      signal_details: {bool(signal_details)}"
                                    )
                                    logger.error(
                                        f"      account_balance: {account_balance}"
                                    )
                        else:
                            logger.error(
                                f"[SYNC] ❌ Failed to import {asset} {pos_type} position"
                            )

                    logger.info(
                        f"\n{'='*80}\n"
                        f"[SYNC] Import complete: {imported_count}/{mt5_count} positions imported\n"
                        f"{'='*80}"
                    )

                    # Verify VTM status after import
                    if imported_count > 0:
                        self._verify_vtm_status_after_sync(asset)

                    return True

                else:
                    logger.warning(
                        f"\n{'='*80}\n"
                        f"[SYNC] ⚠️ IMPORT DISABLED IN CONFIG\n"
                        f"{'='*80}\n"
                        f"Found {mt5_count} MT5 position(s) but import is disabled.\n"
                        f"These positions will NOT be managed by the bot.\n"
                        f"{'='*80}"
                    )

                    for pos in mt5_positions:
                        pos_type = (
                            "LONG" if pos.type == mt5.POSITION_TYPE_BUY else "SHORT"
                        )
                        logger.info(
                            f"  → MT5 {pos_type}: ticket={pos.ticket}, entry=${pos.price_open:.2f}"
                        )

                    return True

            # ================================================================
            # SCENARIO 2: Portfolio has positions, MT5 is empty → CLOSE ALL
            # ================================================================
            if portfolio_count > 0 and mt5_count == 0:
                logger.warning(
                    f"\n{'='*80}\n"
                    f"[SYNC] ⚠️ POSITION MISMATCH\n"
                    f"{'='*80}\n"
                    f"Portfolio shows {portfolio_count} position(s) but MT5 has 0.\n"
                    f"Removing positions from portfolio...\n"
                    f"{'='*80}"
                )

                current_price = self.get_current_price(symbol)
                closed_count = 0

                for position in portfolio_positions:
                    trade_result = self.portfolio_manager.close_position(
                        position_id=position.position_id,
                        exit_price=current_price,
                        reason="sync_missing_mt5",
                    )
                    if trade_result:
                        closed_count += 1
                        logger.info(f"  ✅ Removed position {position.position_id}")

                logger.info(
                    f"\n[SYNC] Cleanup complete: {closed_count}/{portfolio_count} positions removed\n"
                )
                return closed_count == portfolio_count

            # ================================================================
            # SCENARIO 3: Both have positions → VALIDATE
            # ================================================================
            if mt5_count > 0 and portfolio_count > 0:
                logger.info(
                    f"[SYNC] Validating {portfolio_count} portfolio vs {mt5_count} MT5 positions..."
                )

                mt5_by_ticket = {pos.ticket: pos for pos in mt5_positions}
                positions_to_remove = []

                for pos in portfolio_positions:
                    if pos.mt5_ticket and pos.mt5_ticket not in mt5_by_ticket:
                        logger.warning(
                            f"[SYNC] ⚠️ Portfolio position {pos.position_id} (ticket={pos.mt5_ticket}) "
                            f"not found in MT5 → Marking for removal"
                        )
                        positions_to_remove.append(pos)

                if positions_to_remove:
                    current_price = self.get_current_price(symbol)
                    for pos in positions_to_remove:
                        self.portfolio_manager.close_position(
                            position_id=pos.position_id,
                            exit_price=current_price,
                            reason="sync_mt5_ticket_missing",
                        )
                        logger.info(f"  ✅ Removed orphaned position {pos.position_id}")

                remaining_portfolio_count = (
                    self.portfolio_manager.get_asset_position_count(asset)
                )

                if remaining_portfolio_count == mt5_count:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"[SYNC] ✅ {asset} positions in sync ({remaining_portfolio_count} positions)\n"
                        f"{'='*80}"
                    )
                    self._verify_vtm_status_after_sync(asset)
                    return True
                else:
                    logger.warning(
                        f"[SYNC] ⚠️ Count mismatch after sync: Portfolio={remaining_portfolio_count}, MT5={mt5_count}"
                    )
                    return False

            # ================================================================
            # SCENARIO 4: Both empty
            # ================================================================
            logger.info(f"[SYNC] ✅ No positions for {asset} in MT5 or portfolio")
            return True

        except Exception as e:
            logger.error(f"[SYNC] ❌ Error syncing MT5 positions: {e}", exc_info=True)
            return False

    def _verify_vtm_status_after_sync(self, asset: str):
        """Verify VTM is working after position sync"""
        try:
            positions = self.portfolio_manager.get_asset_positions(asset)

            if not positions:
                return

            logger.info(f"\n{'='*80}")
            logger.info(f"[VTM VERIFICATION] Checking {len(positions)} position(s)...")
            logger.info(f"{'='*80}")

            vtm_active_count = 0
            vtm_missing_count = 0

            for pos in positions:
                if pos.trade_manager:
                    vtm_active_count += 1
                    try:
                        status = pos.get_vtm_status()
                        logger.info(
                            f"\n✅ {pos.position_id}: VTM ACTIVE\n"
                            f"   Ticket:  {pos.mt5_ticket}\n"
                            f"   Entry:   ${status['entry_price']:,.2f}\n"
                            f"   Current: ${status['current_price']:,.2f}\n"
                            f"   P&L:     {status['pnl_pct']:+.2f}%\n"
                            f"   SL:      ${status['stop_loss']:,.2f}\n"
                            f"   TP:      ${status.get('take_profit', 0):,.2f}\n"
                            f"   Locked:  {'Yes' if status['profit_locked'] else 'No'}"
                        )
                    except Exception as e:
                        logger.error(f"   Error getting VTM status: {e}")
                else:
                    vtm_missing_count += 1
                    logger.warning(
                        f"\n⚠️ {pos.position_id}: VTM NOT ACTIVE\n"
                        f"   Ticket:  {pos.mt5_ticket}\n"
                        f"   Entry:   ${pos.entry_price:,.2f}\n"
                        f"   → Using static SL/TP instead"
                    )

            logger.info(
                f"\n{'='*80}\n"
                f"[VTM VERIFICATION] Summary:\n"
                f"  Active:  {vtm_active_count}/{len(positions)}\n"
                f"  Missing: {vtm_missing_count}/{len(positions)}\n"
                f"{'='*80}\n"
            )

        except Exception as e:
            logger.error(f"[VTM VERIFICATION] ❌ Error: {e}")
