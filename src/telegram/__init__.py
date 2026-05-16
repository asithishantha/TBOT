#!/usr/bin/env python3
"""
 Telegram Bot Interface for Trading Bot
Provides notifications and remote control capabilities
✨ ENHANCED: Added /brain command for Asymmetric Trading Visualization
"""

import logging
import asyncio
import io
import sys
import html
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from functools import wraps
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict, deque
from telegram.request import HTTPXRequest
import threading
import time
import httpcore
import httpx
import uuid

matplotlib.use("Agg")  # Non-interactive backend

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.constants import ParseMode
from telegram.error import NetworkError, TimedOut, RetryAfter, TelegramError

logger = logging.getLogger(__name__)


# ... [KEEP EXISTING SignalMonitoringIntegration and admin_only classes UNCHANGED] ...
class SignalMonitoringIntegration:
    """
    Signal monitoring features for Telegram bot
    Integrates with PerformanceWeightedAggregator to track signals in real-time
    """

    def __init__(self, max_history: int = 100):
        self.signal_history: Dict[str, List[Dict]] = defaultdict(list)
        self.max_history = max_history
        self.regime_tracking: Dict[str, Dict] = defaultdict(
            lambda: {"current": None, "changes": [], "change_count": 0}
        )
        self.override_tracking: Dict[str, List[Dict]] = defaultdict(list)
        logger.info(
            f"SignalMonitoringIntegration initialized (max_history={max_history})"
        )

    def record_signal(
        self,
        asset: str,
        signal: int,
        details: Dict,
        price: float,
        timestamp: datetime = None,
    ):
        """Record a signal for monitoring"""
        if timestamp is None:
            timestamp = datetime.now()

        entry = {
            "timestamp": timestamp,
            "signal": signal,
            "original_signal": details.get("original_signal", signal),
            "trade_type": details.get("trade_type", "TREND"),
            "price": price,
            "regime": details.get("regime"),
            "is_bull": details.get("is_bull_market"),
            "quality": details.get("signal_quality", 0),
            "reasoning": details.get("reasoning"),
            "mr_signal": details.get("mean_reversion_signal"),
            "mr_conf": details.get("mean_reversion_confidence", 0),
            "tf_signal": details.get("trend_following_signal"),
            "tf_conf": details.get("trend_following_confidence", 0),
            "ema_signal": details.get("ema_signal"),
            "ema_conf": details.get("ema_confidence", 0),
            "regime_changed": details.get("regime_changed", False),
            # Hybrid mode additions
            "aggregator_mode": details.get("aggregator_mode"),
            "council_score": details.get("total_score"),
            "council_decision": details.get("decision_type"),
            # NEW: Regime details from main.py
            "regime_score": details.get("regime_score"),
            "regime_is_bullish": details.get("regime_is_bullish"),
            "regime_is_bearish": details.get("regime_is_bearish"),
            "regime_reasoning": details.get("regime_reasoning"),
        }

        self.signal_history[asset].append(entry)

        if len(self.signal_history[asset]) > self.max_history:
            self.signal_history[asset].pop(0)

        if entry["regime_changed"]:
            self.regime_tracking[asset]["changes"].append(
                {
                    "timestamp": timestamp,
                    "regime": entry["regime"],
                    "price": price,
                    "ema_conf": entry["ema_conf"],
                }
            )
            self.regime_tracking[asset]["change_count"] += 1

        # Gracefully handle None for reasoning
        reasoning = (
            entry.get("reasoning", "").lower()
            if entry.get("reasoning") is not None
            else ""
        )
        if "override" in reasoning:
            self.override_tracking[asset].append(
                {
                    "timestamp": timestamp,
                    "price": price,
                    "ema_conf": entry["ema_conf"],
                    "quality": entry["quality"],
                }
            )

    def get_last_signals(self, asset: str, n: int = 5) -> List[Dict]:
        """Get last N signals for an asset"""
        return self.signal_history[asset][-n:] if asset in self.signal_history else []

    def get_signal_statistics(self, asset: str) -> Dict:
        """Get signal statistics for an asset"""
        if asset not in self.signal_history or not self.signal_history[asset]:
            return {}

        signals = self.signal_history[asset]
        buy_count = sum(1 for s in signals if s["signal"] == 1)
        sell_count = sum(1 for s in signals if s["signal"] == -1)
        hold_count = sum(1 for s in signals if s["signal"] == 0)

        qualities = [s["quality"] for s in signals if s["signal"] != 0]

        return {
            "total_signals": len(signals),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "hold_signals": hold_count,
            "buy_pct": (buy_count / len(signals) * 100) if signals else 0,
            "sell_pct": (sell_count / len(signals) * 100) if signals else 0,
            "hold_pct": (hold_count / len(signals) * 100) if signals else 0,
            "avg_quality": np.mean(qualities) if qualities else 0,
            "high_quality_count": sum(1 for q in qualities if q >= 0.65),
        }

    def get_regime_info(self, asset: str) -> Dict:
        """Get regime information for an asset"""
        if asset not in self.regime_tracking:
            return {}

        tracking = self.regime_tracking[asset]

        return {
            "change_count": tracking["change_count"],
            "last_changes": tracking["changes"][-5:],
        }

    def get_override_info(self, asset: str) -> Dict:
        """Get override event information"""
        if asset not in self.override_tracking:
            return {"total": 0, "last_events": []}

        events = self.override_tracking[asset]
        avg_quality = np.mean([e["quality"] for e in events]) if events else 0

        return {
            "total": len(events),
            "avg_quality": avg_quality,
            "last_events": events[-5:],
        }


def admin_only(func):
    """Decorator to restrict commands to admin users only"""

    @wraps(func)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in self.admin_ids:
            await update.message.reply_text(
                "🚫 *Access Denied*\n\n"
                "This command is restricted to authorized users only.",
                parse_mode=ParseMode.MARKDOWN,
            )
            logger.warning(f"Unauthorized access attempt by user {user_id}")
            return
        return await func(self, update, context)

    return wrapper


class TradingTelegramBot:
    """
    Telegram bot interface for trading bot
    Handles notifications and user commands properly
    """

    def __init__(
        self,
        token: str,
        admin_ids: List[int],
        trading_bot,
        signal_monitor: SignalMonitoringIntegration,
    ):
        self.instance_id = uuid.uuid4()
        logger.info(f"Creating new TradingTelegramBot instance with ID: {self.instance_id}")
        self.token = token
        self.admin_ids = admin_ids
        self.trading_bot = trading_bot
        self.application = None
        self.is_running = False

        # Signal monitoring (keep existing)
        self.signal_monitor = signal_monitor

        self._shutdown_event = None
        # The asyncio loop for this bot instance will be managed by the calling thread
        self._current_loop = None # Stores the loop that is running this bot's operations

        # State tracking
        self._is_ready = False
        self._initialization_lock = threading.Lock()
        self._message_queue = []

        # Error handling attributes
        self._network_error_count = 0
        self._last_network_error = None
        self._max_consecutive_errors = 5  # Max errors before forceful reconnect
        self._reconnect_delay = 10  # Initial reconnect delay in seconds


        logger.info(f"TelegramBot initialized - Admins: {admin_ids}")





    def _run_in_loop(self, coro, timeout: float = 30.0):
        """
        Run coroutine in the bot's currently active event loop with timeout.
        Safe to call from any thread (assumes loop is set for this bot instance).
        """
        if self._current_loop is None or self._current_loop.is_closed():
            raise RuntimeError("Telegram event loop not available or closed for this bot instance.")

        future = asyncio.run_coroutine_threadsafe(coro, self._current_loop)

        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.error(f"[TELEGRAM] Operation timeout after {timeout}s")
            future.cancel()
            raise
        except Exception as e:
            logger.error(f"[TELEGRAM] Operation failed: {e}")
            raise

    async def _keepalive_task(self):
        """
        ✅ NEW: Keepalive task to prevent connection timeout
        Runs in background and pings Telegram API every 5 minutes
        """
        logger.info("[TELEGRAM] Keepalive task started")

        while self.is_running and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # 5 minutes

                if self._is_ready and self.application:
                    # Send a lightweight API call to keep connection alive
                    try:
                        await self.application.bot.get_me()
                        logger.debug("[TELEGRAM] Keepalive ping successful")
                    except Exception as e:
                        logger.warning(f"[TELEGRAM] Keepalive ping failed: {e}")

            except asyncio.CancelledError:
                logger.info("[TELEGRAM] Keepalive task cancelled")
                break
            except Exception as e:
                logger.error(f"[TELEGRAM] Keepalive error: {e}")
                await asyncio.sleep(60)

        logger.info("[TELEGRAM] Keepalive task stopped")

    def _register_handlers(self):
        """Register all command and message handlers."""
        # Information Commands
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("brain", self.cmd_brain))
        self.application.add_handler(CommandHandler("positions", self.cmd_positions))
        self.application.add_handler(CommandHandler("modes", self.cmd_aggregator_modes))
        self.application.add_handler(CommandHandler("history", self.cmd_history))
        self.application.add_handler(CommandHandler("performance", self.cmd_performance))
        self.application.add_handler(CommandHandler("presets", self.cmd_presets))
        self.application.add_handler(CommandHandler("signals", self.cmd_signals))
        self.application.add_handler(CommandHandler("vtm", self.cmd_VTM_status))
        self.application.add_handler(CommandHandler("vtm_status", self.cmd_vtm_status_detail))
        self.application.add_handler(CommandHandler("set_sl", self.cmd_set_sl))
        self.application.add_handler(CommandHandler("set_tp", self.cmd_set_tp))
        self.application.add_handler(CommandHandler("reset_equity", self.cmd_reset_equity))
        self.application.add_handler(CommandHandler("stats", self.cmd_signal_stats))
        self.application.add_handler(CommandHandler("regimes", self.cmd_regimes))
        self.application.add_handler(CommandHandler("overrides", self.cmd_overrides))
        self.application.add_handler(CommandHandler("chart", self.cmd_chart))
        self.application.add_handler(CommandHandler("lastdecision", self.cmd_last_decision))
        self.application.add_handler(CommandHandler("modedetails", self.cmd_mode_details))
        self.application.add_handler(CommandHandler("preset_history", self.cmd_preset_history))
        # /debug_positions removed — no implementation existed, caused AttributeError on use
        self.application.add_handler(CommandHandler("test_viz", self.cmd_test_viz))


        # Admin Commands
        self.application.add_handler(CommandHandler("stop_trading", self.cmd_stop_trading))
        self.application.add_handler(CommandHandler("start_trading", self.cmd_start_trading))
        self.application.add_handler(CommandHandler("resume", self.cmd_resume_trading))
        self.application.add_handler(CommandHandler("close_all", self.cmd_close_all))
        self.application.add_handler(CommandHandler("close", self.cmd_close_asset))

        # Callback Query Handler
        self.application.add_handler(CallbackQueryHandler(self.button_callback))

    def initialize(self):
        """
        Initialize bot components (Application, handlers).
        This method is synchronous and should be called before run_polling.
        """
        with self._initialization_lock:
            if self.application:
                logger.info("[TELEGRAM] Already initialized.")
                return

            try:
                logger.info("[TELEGRAM] Initializing bot components...")

                from telegram.request import HTTPXRequest
                request = HTTPXRequest(
                    http_version="1.1",
                    connection_pool_size=50,
                    read_timeout=30,
                    write_timeout=30,
                    connect_timeout=30,
                    pool_timeout=30,
                )

                self.application = (
                    Application.builder().token(self.token).request(request).build()
                )

                self._register_handlers()
                self._is_ready = True
                logger.info("[TELEGRAM] ✅ Bot components initialized.")

            except Exception as e:
                logger.error(f"[TELEGRAM] Initialization of components failed: {e}", exc_info=True)
                self._is_ready = False
                self.application = None

    def run_polling(self):
        """
        Run the bot's polling loop. This is a blocking call.
        It sets up its own asyncio event loop and runs until shutdown.
        """
        if not self._is_ready or not self.application:
            self.initialize()
            if not self._is_ready:
                logger.error("[TELEGRAM] Cannot start polling, initialization failed.")
                return

        self.is_running = True
        
        async def main():
            self._shutdown_event = asyncio.Event()
            self._current_loop = asyncio.get_running_loop()

            await self.application.initialize()
            await self.application.start()
            
            logger.info("[TELEGRAM] Starting keepalive task...")
            asyncio.create_task(self._keepalive_task())

            logger.info("[TELEGRAM] Starting polling...")
            await self.application.updater.start_polling(
                poll_interval=1.0,
                timeout=30,
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES,
                bootstrap_retries=-1,
            )
            
            await self._shutdown_event.wait()
            
            await self.application.updater.stop()
            await self.application.stop()
            logger.info("[TELEGRAM] Polling stopped.")

        try:
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            asyncio.run(main())

        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("[TELEGRAM] Polling loop interrupted.")
        except Exception as e:
            logger.error(f"[TELEGRAM] Unhandled error in run_polling: {e}", exc_info=True)
        finally:
            self.is_running = False
            self._is_ready = False
            self._current_loop = None
            logger.info("[TELEGRAM] Polling loop finished.")

    async def _keepalive_task(self):
        """
        ✅ NEW: Keepalive task to prevent connection timeout
        Runs in background and pings Telegram API every 5 minutes
        """
        logger.info("[TELEGRAM] Keepalive task started")
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)
                if self._is_ready and self.application:
                    try:
                        await self.application.bot.get_me()
                        logger.debug("[TELEGRAM] Keepalive ping successful")
                    except Exception as e:
                        logger.warning(f"[TELEGRAM] Keepalive ping failed: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[TELEGRAM] Keepalive error: {e}")
                await asyncio.sleep(60)
        logger.info("[TELEGRAM] Keepalive task stopped")

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("[TELEGRAM] Shutdown initiated...")
        self._shutdown_event.set()

    # ==================== COMMAND HANDLERS ====================

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        user_id = update.effective_user.id
        is_admin = user_id in self.admin_ids

        help_text = (
            "📋 <b>TBOT Command Reference</b>\n\n"

            "📊 <b>Market &amp; Signals</b>\n"
            "/status — Bot health, portfolio value, market status\n"
            "/brain — MTF regime engine &amp; governor per asset\n"
            "/signals — Last 5 signals per asset with quality scores\n"
            "/stats — Signal distribution (BUY/SELL/HOLD %) + quality\n"
            "/regimes — Regime change log per asset\n"
            "/overrides — Signal override events log\n"
            "/lastdecision — Most recent decision + filter that fired\n"
            "/chart [ASSET] — Live AI decision chart (all assets if no arg)\n\n"

            "📈 <b>Positions &amp; Trades</b>\n"
            "/positions — Open positions with P&amp;L, SL, TP, VTM state\n"
            "/vtm — VTM live stop/target levels per position\n"
            "/vtm_status — Full VTM override breakdown with hints\n"
            "/set_sl ASSET PRICE — Move stop loss on a live position\n"
            "/set_tp ASSET PRICE [TIER] — Move take profit (tier optional)\n"
            "/history — Last 10 closed trades with P&amp;L and hold time\n"
            "/performance — Win rate, profit factor, drawdown, equity curve\n\n"

            "⚙️ <b>Configuration &amp; Engine</b>\n"
            "/modes — Current aggregator mode per asset (council vs perf)\n"
            "/modedetails ASSET — Full mode switch history for one asset\n"
            "/presets — Current aggregator preset per asset\n"
            "/preset_history — Preset change log\n"
        )

        if is_admin:
            help_text += (
                "\n🎮 <b>Admin Controls</b>\n"
                "/start_trading — Resume signal processing\n"
                "/stop_trading — Pause (keep positions open)\n"
                "/resume — Override a tripped circuit breaker (clears at midnight)\n"
                "/reset_equity — Re-baseline equity to clear phantom drawdown\n"
                "/close_all — Emergency close all positions\n"
                "/close ASSET [#] — Close all or specific position for asset\n"
                "\n⚠️ <i>Admin commands are restricted to authorized users.</i>"
            )

        await update.message.reply_text(help_text, parse_mode="HTML")

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        is_admin = user_id in self.admin_ids

        welcome_msg = (
            "🤖 *Welcome to Trading Bot Control*\n\n"
            f"👤 User: @{username}\n"
            f"🆔 ID: `{user_id}`\n"
            f"🔐 Access: {'✅ Admin' if is_admin else '❌ Guest'}\n\n"
        )

        if is_admin:
            welcome_msg += (
                "You have full access to all bot commands.\n"
                "Use /help to see available commands."
            )
        else:
            welcome_msg += (
                "⚠️ You are not authorized to control this bot.\n"
                "Contact the bot administrator for access."
            )

        await update.message.reply_text(welcome_msg, parse_mode=ParseMode.MARKDOWN)
        logger.info(f"User {user_id} ({username}) started bot - Admin: {is_admin}")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Bot status — full dashboard: engine, portfolio, open trades, asset signals"""
        try:
            portfolio = self.trading_bot.portfolio_manager.get_portfolio_status()
            is_running = getattr(self.trading_bot, "is_running", False)

            # ── Engine & circuit breaker ─────────────────────────────────
            run_icon = "🟢 RUNNING" if is_running else "🔴 STOPPED"
            cb = getattr(self.trading_bot, "circuit_breaker", None) or {}
            if hasattr(cb, "__dict__"):
                cb = cb.__dict__
            cb_active = cb.get("is_active", False) or cb.get("triggered", False)
            cb_str = "🚨 TRIGGERED" if cb_active else "✅ OK"

            # ── Portfolio numbers ────────────────────────────────────────
            total_value  = portfolio.get("total_value", 0)
            cash         = portfolio.get("cash", 0)
            open_pos     = portfolio.get("open_positions", 0)
            daily_pnl    = portfolio.get("daily_pnl", 0)
            unrealized   = portfolio.get("total_unrealized_pnl", 0)
            dpnl_icon    = "🟢" if daily_pnl >= 0 else "🔴"
            dpnl_sign    = "+" if daily_pnl >= 0 else ""
            unreal_icon  = "🟢" if unrealized >= 0 else "🔴"
            unreal_sign  = "+" if unrealized >= 0 else ""

            # Total return vs initial capital
            initial_cap = getattr(self.trading_bot.portfolio_manager, "initial_capital", None)
            total_ret_str = ""
            if initial_cap and initial_cap > 0:
                total_ret = (total_value - initial_cap) / initial_cap * 100
                ret_icon  = "🟢" if total_ret >= 0 else "🔴"
                total_ret_str = f"  Return   : {ret_icon} <code>{total_ret:+.2f}%</code>  vs ${initial_cap:,.0f} init\n"

            # ── Today's closed trades ────────────────────────────────────
            closed = getattr(self.trading_bot.portfolio_manager, "closed_positions", [])
            today  = datetime.now().date()
            today_trades = [t for t in closed if t.get("exit_time") and t["exit_time"].date() == today]
            wins_today   = sum(1 for t in today_trades if t.get("pnl", 0) > 0)
            loss_today   = sum(1 for t in today_trades if t.get("pnl", 0) < 0)
            today_str    = f"{len(today_trades)} trades"
            if today_trades:
                today_str += f"  {wins_today}W / {loss_today}L"

            msg = (
                f"<b>🤖 TBOT STATUS</b>\n"
                f"{'─' * 30}\n"
                f"Engine   : {run_icon}\n"
                f"Circuit  : {cb_str}\n\n"

                f"<b>💼 Portfolio</b>\n"
                f"  Value    : <code>${total_value:,.2f}</code>\n"
                f"  Cash     : <code>${cash:,.2f}</code>\n"
                f"{total_ret_str}"
                f"  Daily P&amp;L : {dpnl_icon} <code>{dpnl_sign}${daily_pnl:,.2f}</code>\n"
                f"  Unrealised: {unreal_icon} <code>{unreal_sign}${unrealized:,.2f}</code>\n"
                f"  Today    : {today_str}\n\n"
            )

            # ── Open positions summary ───────────────────────────────────
            positions = self.trading_bot.portfolio_manager.positions
            if positions:
                msg += f"<b>📍 Open Positions ({open_pos})</b>\n"
                for pos_id, position in positions.items():
                    asset    = position.asset
                    side     = position.side.upper()
                    side_icon = "🟢" if side == "LONG" else "🔴"
                    entry    = position.entry_price

                    # Try to get live price for P&L
                    asset_cfg = self.trading_bot.config["assets"].get(asset, {})
                    exchange  = asset_cfg.get("exchange", "binance")
                    symbol    = self.trading_bot._resolve_symbol(asset)
                    handler   = (self.trading_bot.binance_handler if exchange == "binance"
                                 else self.trading_bot.mt5_handler)
                    current = entry
                    try:
                        if handler and symbol:
                            p = handler.get_current_price(symbol=symbol, force_live=True)
                            if p:
                                current = p
                    except Exception:
                        pass

                    qty   = position.quantity
                    pnl   = (current - entry) * qty if side == "LONG" else (entry - current) * qty
                    pnl_pct = (pnl / (entry * qty) * 100) if entry > 0 and qty > 0 else 0
                    pi    = "🟢" if pnl >= 0 else "🔴"
                    ps    = "+" if pnl >= 0 else ""

                    # SL from VTM if available
                    vtm = getattr(position, "trade_manager", None)
                    sl  = vtm.current_stop_loss if vtm else getattr(position, "stop_loss", None)
                    sl_str = f"  SL <code>${sl:,.2f}</code>" if sl else ""

                    msg += (
                        f"  {side_icon} <b>{html.escape(asset)}</b> {side} "
                        f"{pi} <code>{ps}{pnl_pct:.2f}%</code>{sl_str}\n"
                    )
                msg += "\n"
            else:
                msg += f"<b>📍 No open positions</b>\n\n"

            # ── Per-asset signal & market status ────────────────────────
            msg += "<b>🌐 Assets</b>\n"
            agg_modes = {}
            if hasattr(self.trading_bot, "hybrid_selector"):
                try:
                    agg_modes = self.trading_bot.hybrid_selector.get_statistics().get("current_modes", {})
                except Exception:
                    pass

            for asset_name, asset_cfg in self.trading_bot.config["assets"].items():
                if not asset_cfg.get("enabled", False):
                    continue

                emoji = "₿" if "BTC" in asset_name.upper() else "🥇" if "GOLD" in asset_name.upper() else "📈"

                # Market open/closed
                if "BTC" in asset_name.upper():
                    mstatus = "✅"
                elif hasattr(self.trading_bot, "check_market_hours"):
                    mstatus = "✅" if self.trading_bot.check_market_hours(asset_name) else "🔴"
                else:
                    mstatus = "❓"

                # Aggregator mode label
                mode = agg_modes.get(asset_name, "")
                mode_tag = f"[{mode[:4].upper()}] " if mode else ""

                # Last signal — direction + quality + time
                last_sigs = self.signal_monitor.get_last_signals(asset_name, n=1)
                sig_str = "<i>—</i>"
                if last_sigs:
                    s      = last_sigs[0]
                    val    = s["signal"]
                    ts     = s["timestamp"].strftime("%H:%M")
                    q      = s.get("quality", 0) or 0
                    sicon  = "📈" if val == 1 else "📉" if val == -1 else "⚪"
                    slabel = "BUY" if val == 1 else "SELL" if val == -1 else "HOLD"
                    sig_str = f"{sicon} {slabel} {q:.0%} <i>{ts}</i>"

                msg += f"  {emoji} <b>{html.escape(asset_name)}</b> {mstatus} {mode_tag}{sig_str}\n"

            msg += f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            keyboard = [
                [InlineKeyboardButton("📊 Positions", callback_data="positions"),
                 InlineKeyboardButton("🧠 Brain",     callback_data="brain")],
                [InlineKeyboardButton("📜 History",   callback_data="history"),
                 InlineKeyboardButton("🔄 Refresh",   callback_data="status")],
            ]
            await update.message.reply_text(msg, parse_mode="HTML",
                                            reply_markup=InlineKeyboardMarkup(keyboard))
        except Exception as e:
            logger.error(f"[TG] /status error: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching status")

    async def cmd_last_decision(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /lastdecision command - Show the last trading decision for each asset."""
        try:
            if not self.trading_bot.config["assets"]:
                await update.message.reply_text("❌ No assets configured.")
                return

            msg = "🕰️ <b>LATEST TRADING DECISIONS</b>\n"
            msg += "───────────────────\n\n"
            found_decisions = False

            for asset_name in self.trading_bot.config["assets"].keys():
                if not self.trading_bot.config["assets"][asset_name].get("enabled", False):
                    continue

                last_signal_entry = self.signal_monitor.get_last_signals(asset_name, n=1)

                if last_signal_entry:
                    found_decisions = True
                    entry = last_signal_entry[0]
                    
                    # Icons and status
                    original = entry.get("original_signal", entry["signal"])
                    final = entry["signal"]
                    
                    orig_str = {1: "BUY 🟢", -1: "SELL 🔴", 0: "HOLD ⚪"}.get(original, "UNKNOWN")
                    final_str = {1: "BUY 🟢", -1: "SELL 🔴", 0: "HOLD ⚪"}.get(final, "UNKNOWN")
                    
                    # Determine result status
                    status_icon = "✅" if original == final and final != 0 else "⚠️" if original != final and final == 0 else "⚪"
                    if original == 0 and final == 0:
                        status_icon = "💤"

                    timestamp = entry["timestamp"].strftime("%H:%M:%S")
                    
                    msg += f"<b>{html.escape(asset_name)}</b> <code>[{timestamp}]</code>\n"
                    msg += f"  ├ Original: {orig_str}\n"
                    msg += f"  ├ Final:    {final_str} {status_icon}\n"
                    
                    # --- Institutional Filter Details ---
                    if original != final and final == 0:
                        reason = str(entry.get('reasoning', 'Unknown')).lower()
                        
                        # 1. Trap Filter specific reporting
                        if "trap" in reason:
                            msg += f"  ├ 🪤 <b>Trap Filter:</b> <code>VETOED (Bad Structure)</code>\n"
                        
                        # 2. Gatekeeper specific reporting (Only if it was the reason)
                        elif "gatekeeper" in reason:
                            r_score = entry.get("regime_score", 0)
                            r_bias = "BULLISH" if r_score > 0 else "BEARISH" if r_score < 0 else "NEUTRAL"
                            msg += f"  ├ 🛡️ <b>Gatekeeper:</b> <code>BLOCKED ({r_bias} @ {r_score:.2f})</code>\n"
                        
                        # 3. Momentum / ATR Gate reporting
                        elif "expansion" in reason or "atr" in reason:
                            msg += f"  ├ 🚀 <b>Momentum:</b> <code>VETOED (Low Vol Expansion)</code>\n"
                        
                        # 4. Volatility Gate
                        elif "volatility" in reason:
                            msg += f"  ├ 📉 <b>Volatility:</b> <code>BLOCKED (Dead Market)</code>\n"

                        # 5. Sniper / Pattern rejection
                        elif "sniper" in reason or "pattern" in reason:
                            msg += f"  ├ 🎯 <b>Sniper:</b> <code>REJECTED (No Edge)</code>\n"
                        
                        # General fallback if not specific
                        else:
                            msg += f"  ├ ⚠️ <b>Veto:</b> <code>{html.escape(str(entry.get('reasoning', 'N/A')))}</code>\n"
                    
                    elif final != 0:
                        msg += f"  ├ <b>Status:</b> <code>{html.escape(str(entry.get('reasoning', 'N/A')))}</code>\n"
                    
                    # Add trade type and regime context
                    regime_name = entry.get('regime', 'NEUTRAL')
                    msg += f"  └ Context:  <code>{entry.get('trade_type', 'TREND')} | {regime_name}</code>\n\n"
                else:
                    msg += f"<b>{html.escape(asset_name)}</b>: No recent decision.\n\n"

            if not found_decisions:
                msg = "🤷 No trading decisions recorded yet."

            await self._send_chunked(
                update.message.reply_text, msg, parse_mode=ParseMode.HTML
            )

        except Exception as e:
            logger.error(f"Error in cmd_last_decision: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching last decisions.")


    # ====================================================================
    # ✨ NEW: The "Brain" Visualizer Command
    # ====================================================================

    @staticmethod
    async def _send_chunked(send_method, msg: str, chunk_size: int = 4000, **kwargs):
        """
        Fix #13 — Split messages exceeding Telegram's 4096-char limit.
        Splits on newlines to avoid breaking HTML tags mid-element.
        Falls back to truncating a single oversized line.
        """
        if len(msg) <= chunk_size:
            await send_method(msg, **kwargs)
            return

        chunks = []
        current = ""
        for line in msg.split("\n"):
            candidate = (current + "\n" + line) if current else line
            if len(candidate) > chunk_size:
                if current:
                    chunks.append(current)
                # If a single line exceeds limit, hard-cut it
                current = line[:chunk_size]
            else:
                current = candidate
        if current:
            chunks.append(current)

        for i, chunk in enumerate(chunks):
            try:
                # Only attach reply_markup to the last chunk
                chunk_kwargs = dict(kwargs)
                if i < len(chunks) - 1:
                    chunk_kwargs.pop("reply_markup", None)
                await send_method(chunk, **chunk_kwargs)
            except Exception as chunk_err:
                logger.error(f"[TG] Chunk {i+1}/{len(chunks)} send error: {chunk_err}")

    async def cmd_brain(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /brain command - Visualizes the MTF Governor and Asymmetric Engine State
        """
        await self._send_brain_message(update.message.reply_text)

    async def _send_brain_message(self, send_method, is_query=False):
        """Generates and sends the Asymmetric Engine visualization — enhanced with ADX, confidence, and signal quality"""
        try:
            if (
                not hasattr(self.trading_bot, "_current_regime_data")
                or not self.trading_bot._current_regime_data
            ):
                msg = "❌ <b>Brain Offline</b>\nNo MTF Regime Data available yet. Waiting for first cycle."
                await send_method(msg, parse_mode="HTML")
                return

            msg = (
                "🧠 <b>THE BRAIN — Asymmetric Engine</b>\n"
                "<i>Macro Trend Governor &amp; Risk Management</i>\n"
                f"{'─' * 30}\n\n"
            )

            for asset in list(self.trading_bot.config["assets"].keys()):
                if not self.trading_bot.config["assets"].get(asset, {}).get("enabled", False):
                    continue

                regime_data_obj = self.trading_bot._current_regime_data.get(asset)
                if not regime_data_obj:
                    continue

                regime_data = regime_data_obj.to_dict() if hasattr(regime_data_obj, "to_dict") else regime_data_obj
                emoji = "₿" if "BTC" in asset.upper() else "🥇" if "GOLD" in asset.upper() else "📈"

                # ── Regime & confidence ─────────────────────────────────────
                current_regime = str(regime_data.get("consensus_regime", "NEUTRAL")).replace("_", " ")
                bull_score = regime_data.get("bullish_score", 0) or 0
                bear_score = regime_data.get("bearish_score", 0) or 0
                total_score_rd = bull_score + bear_score
                confidence = (max(bull_score, bear_score) / total_score_rd * 100) if total_score_rd > 0 else 0

                if "NEUTRAL" in current_regime:
                    trend_icon = "⚖️"
                elif bull_score >= bear_score:
                    trend_icon = "📈"
                else:
                    trend_icon = "📉"

                bar_filled = int(confidence / 10)
                conf_bar = "█" * bar_filled + "░" * (10 - bar_filled)

                # ── ADX (from hybrid_selector mode history) ─────────────────
                adx_str = "—"
                try:
                    if hasattr(self.trading_bot, "hybrid_selector"):
                        hist = self.trading_bot.hybrid_selector.get_mode_history(asset, n=1)
                        if hist:
                            adx_val = hist[-1].get("trend", {}).get("adx", None)
                            if adx_val is not None:
                                if adx_val >= 40:
                                    adx_label = "💥 Strong"
                                elif adx_val >= 25:
                                    adx_label = "📊 Trending"
                                elif adx_val >= 20:
                                    adx_label = "〰️ Developing"
                                else:
                                    adx_label = "😴 Weak"
                                adx_str = f"{adx_val:.1f} — {adx_label}"
                except Exception:
                    pass

                # ── Trade mode / risk ────────────────────────────────────────
                trade_type = regime_data.get("trade_type")
                if trade_type == "TREND":
                    type_icon, risk_str = "📈", "2.0% risk  (1.33× mult)"
                elif trade_type == "SCALP":
                    type_icon, risk_str = "⚡", "1.0% risk  (0.67× mult)"
                elif trade_type and "V_SHAPE" in trade_type:
                    type_icon, risk_str = "🚀", "1.5% risk  (1.0× mult)"
                else:
                    type_icon, risk_str = "🛑", "0% risk — no trade"
                    trade_type = trade_type or "UNKNOWN"

                # ── Last signal + quality + council score ────────────────────
                sig_line = "<i>no signal yet</i>"
                quality_bar_line = ""
                engine_line = ""
                try:
                    last_sigs = self.signal_monitor.get_last_signals(asset, n=1)
                    if last_sigs:
                        sig = last_sigs[0]
                        ts_str = sig["timestamp"].strftime("%H:%M")
                        val = sig["signal"]
                        sig_icon = "📈 BUY" if val == 1 else "📉 SELL" if val == -1 else "⚪ HOLD"
                        sig_price = sig["price"]
                        sig_line = f"{sig_icon} @ <code>${sig_price:,.2f}</code>  <i>[{ts_str}]</i>"

                        quality = sig.get("quality", 0) or 0
                        q_filled = int(quality * 10)
                        q_bar = "█" * q_filled + "░" * (10 - q_filled)
                        quality_bar_line = f"  Quality   : <code>{q_bar}</code> <b>{quality:.1%}</b>\n"

                        mode = sig.get("aggregator_mode", "performance")
                        if mode == "council":
                            c_score = sig.get("council_score") or 0
                            c_decision = sig.get("council_decision", "N/A")
                            engine_line = (
                                f"  Engine    : <code>COUNCIL</code>  score <b>{c_score:.2f}/5.0</b>"
                                f"  <i>{html.escape(str(c_decision))}</i>\n"
                            )
                        else:
                            engine_line = f"  Engine    : <code>PERF-WEIGHTED</code>\n"
                except Exception:
                    pass

                msg += (
                    f"{emoji} <b>{html.escape(asset)}</b>\n"
                    f"  Regime    : {trend_icon} <code>{html.escape(current_regime)}</code>\n"
                    f"  Confidence: <code>{conf_bar}</code> <b>{confidence:.0f}%</b>\n"
                    f"  ADX       : {html.escape(adx_str)}\n"
                    f"  Mode      : {type_icon} <b>{html.escape(str(trade_type))}</b>  |  {risk_str}\n"
                    f"  ─────────────────────────\n"
                    f"  Last Sig  : {sig_line}\n"
                    f"{quality_bar_line}"
                    f"{engine_line}"
                    f"\n"
                )

            msg += f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            keyboard = [
                [
                    InlineKeyboardButton("📊 Positions", callback_data="positions"),
                    InlineKeyboardButton("🔄 Refresh", callback_data="brain"),
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await self._send_chunked(
                send_method, msg, parse_mode="HTML", reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in _send_brain_message: {e}", exc_info=True)
            await send_method("❌ Error fetching Brain status")

    # ====================================================================
    # UPDATE BUTTON CALLBACK TO INCLUDE BRAIN
    # ====================================================================

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks"""
        query = update.callback_query
        await query.answer()

        callback_data = query.data

        try:
            if callback_data == "status":
                await self._send_status_message(query)
            elif callback_data == "brain":  # ✨ NEW: Handle Brain Refresh Button
                await self._send_brain_message(query.edit_message_text, is_query=True)
            elif callback_data == "positions":
                await self._send_positions_message(query)
            # ... [KEEP REST OF CALLBACKS UNCHANGED] ...
            elif callback_data == "modes":
                await self._send_modes_message(query)
            elif callback_data == "history":
                await self._send_history_message(query)
            elif callback_data == "presets":
                await self._send_presets_message(query)
            elif callback_data == "signals":
                await self._send_signals_message(query)
            elif callback_data == "stats":
                await self._send_stats_message(query)
            elif callback_data == "regimes":
                await self._send_regimes_message(query)
            elif callback_data == "overrides":
                await self._send_overrides_message(query)
            elif callback_data == "preset_history":
                await self._send_preset_history_message(query)
            else:
                await query.edit_message_text(f"⚠️ Unknown command: {callback_data}")
        except Exception as e:
            logger.error(f"Button callback error: {e}", exc_info=True)
            try:
                await query.edit_message_text("❌ Error processing request")
            except:
                pass

    # ... [KEEP THE REST OF THE FILE UNCHANGED] ...
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Open positions with full P&L, SL/TP, and VTM state"""
        try:
            positions = self.trading_bot.portfolio_manager.positions
            if not positions:
                await update.message.reply_text("📭 <b>No open positions</b>", parse_mode="HTML")
                return

            msg = "📊 <b>OPEN POSITIONS</b>\n\n"

            for pos_id, position in positions.items():
                asset = position.asset
                side = position.side.upper()
                side_icon = "🟢" if side == "LONG" else "🔴"
                entry = position.entry_price
                qty = position.quantity

                # Live price
                asset_cfg = self.trading_bot.config["assets"].get(asset, {})
                exchange = asset_cfg.get("exchange", "binance")
                symbol = self.trading_bot._resolve_symbol(asset)
                handler = (self.trading_bot.binance_handler if exchange == "binance"
                           else self.trading_bot.mt5_handler)
                current = entry
                if handler and symbol:
                    try:
                        p = handler.get_current_price(symbol=symbol, force_live=True)
                        if p:
                            current = p
                    except Exception:
                        pass

                pnl = (current - entry) * qty if side == "LONG" else (entry - current) * qty
                pnl_pct = (pnl / (qty * entry) * 100) if entry > 0 else 0
                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                pnl_sign = "+" if pnl >= 0 else ""

                msg += f"{side_icon} <b>{html.escape(asset)} {side}</b>\n"
                msg += f"  Entry   : <code>${entry:,.2f}</code>\n"
                msg += f"  Current : <code>${current:,.2f}</code>\n"
                msg += f"  P&amp;L    : {pnl_icon} <code>{pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)</code>\n"
                msg += f"  Qty     : {qty:.6g}\n"

                # VTM state
                vtm = getattr(position, "trade_manager", None)
                if vtm:
                    sl = vtm.current_stop_loss or 0
                    sl_pct = abs(current - sl) / current * 100 if current > 0 else 0
                    # Next unfilled TP
                    remaining_tps = [vtm.take_profit_levels[i]
                                     for i in range(len(vtm.take_profit_levels))
                                     if i not in vtm.partials_hit]
                    tp = remaining_tps[0] if remaining_tps else None
                    tp_pct = abs(tp - current) / current * 100 if tp and current > 0 else 0

                    lock_icon = "🔒" if vtm.profit_locked else "🔓"
                    runner_icon = "🏃" if vtm.runner_activated else "—"

                    msg += f"  SL      : <code>${sl:,.2f}</code> ({sl_pct:.2f}% away)\n"
                    if tp:
                        msg += f"  TP      : <code>${tp:,.2f}</code> ({tp_pct:.2f}% away)\n"
                    else:
                        msg += f"  TP      : all targets hit — runner active\n"
                    msg += f"  Lock    : {lock_icon} | Runner: {runner_icon} | Bars: {vtm.bars_in_trade}\n"
                    tps_hit = len(vtm.partials_hit)
                    total_tps = len(vtm.take_profit_levels)
                    msg += f"  Partials: {tps_hit}/{total_tps} hit | Rem: {vtm.remaining_position:.0%}\n"
                else:
                    sl_val = getattr(position, "stop_loss", None)
                    tp_val = getattr(position, "take_profit", None)
                    if sl_val:
                        msg += f"  SL      : <code>${sl_val:,.2f}</code>\n"
                    if tp_val:
                        msg += f"  TP      : <code>${tp_val:,.2f}</code>\n"
                    msg += f"  VTM     : static management\n"

                msg += "\n"

            msg += f"🕐 {datetime.now().strftime('%H:%M:%S')}  |  💡 /set_sl /set_tp to adjust"
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="positions"),
                         InlineKeyboardButton("🎯 VTM", callback_data="status")]]
            await self._send_chunked(
                update.message.reply_text, msg,
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
        except Exception as e:
            logger.error(f"[TG] /positions error: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching positions")

    async def cmd_VTM_status(self, update, context):
        """Show Dynamic Trade Manager status -"""
        try:
            if not self.trading_bot:
                await update.message.reply_text("⚠️ Trading bot not connected")
                return

            positions = self.trading_bot.portfolio_manager.positions

            if not positions:
                await update.message.reply_text("📊 No open positions")
                return

            msg = "🎯 <b>VETERAN TRADE MANAGER STATUS</b>\n\n"

            # Update exchange profits first to ensure real-time accuracy
            if not self.trading_bot.portfolio_manager.is_paper_mode:
                try:
                    self.trading_bot.portfolio_manager.update_mt5_positions_profit()
                    self.trading_bot.portfolio_manager.update_binance_positions_profit()
                except Exception as e:
                    logger.debug(f"Failed to update exchange profits for VTM status: {e}")

            for position_id, position in positions.items():
                # 1. Get the correct handler for the asset
                asset_cfg = self.trading_bot.config['assets'].get(position.asset, {})
                exchange = asset_cfg.get('exchange', 'binance')
                symbol = self.trading_bot._resolve_symbol(position.asset)
                handler = self.trading_bot.binance_handler if exchange == 'binance' else self.trading_bot.mt5_handler

                # 2. Fetch the live price
                live_price = None
                if handler and symbol:
                    try:
                        live_price = handler.get_current_price(symbol=symbol, force_live=True)
                    except Exception as e:
                        logger.warning(f"Could not fetch live price for {position.asset}: {e}")

                # 3. Pass live price to get VTM status
                vtm_status = position.get_vtm_status(live_price=live_price)

                if vtm_status:
                    side_emoji = "🟢" if vtm_status["side"] == "long" else "🔴"
                    pnl_emoji = "💰" if vtm_status["pnl_pct"] > 0 else "📉"
                    lock_emoji = "🔒" if vtm_status["profit_locked"] else "🔓"

                    # Format P&L string to include absolute and percentage
                    pnl_abs_val = vtm_status.get('pnl_abs', 0.0)
                    pnl_sign = "+" if pnl_abs_val >= 0 else ""
                    pnl_string = f"<b>{pnl_sign}${pnl_abs_val:,.2f} ({vtm_status['pnl_pct']:+.3f}%)</b>"

                    msg += f"{side_emoji} <b>{position.asset} {vtm_status['side'].upper()}</b>\n"
                    msg += f"{pnl_emoji} P&L: {pnl_string}\n"
                    msg += f"💵 Entry: ${(vtm_status.get('entry_price') or 0.0):,.2f}\n"
                    msg += f"📍 Current: ${(vtm_status.get('current_price') or 0.0):,.2f}\n"
                    _sl_price = vtm_status.get('stop_loss') or 0.0
                    _sl_dist  = vtm_status.get('distance_to_sl_pct') or 0.0
                    msg += f"🛑 SL: ${_sl_price:,.2f} ({_sl_dist:+.2f}%)\n"
                    _tp_price = vtm_status.get('take_profit') or 0.0
                    _tp_dist  = vtm_status.get('distance_to_tp_pct') or 0.0
                    _tp_label = f"${_tp_price:,.2f} ({_tp_dist:+.2f}%)" if _tp_price else "none set"
                    msg += f"🎯 TP: {_tp_label}\n"
                    msg += f"{lock_emoji} Profit Lock: {'ON' if vtm_status['profit_locked'] else 'OFF'}\n"
                    
                    # Display dynamic VTM parameters
                    if vtm_status.get("early_lock_atr_multiplier") is not None:
                        msg += f"🔗 Early Lock: Dynamic ({vtm_status['early_lock_atr_multiplier']}x ATR)\n"
                        msg += f"   └─ Threshold: {vtm_status.get('current_early_lock_threshold_pct', 0):.2%}\n"
                    
                    if vtm_status.get("runner_trail_atr_multiplier") is not None:
                        msg += f"🏃 Runner Trail: Dynamic ({vtm_status['runner_trail_atr_multiplier']}x ATR)\n"
                        msg += f"   └─ Current Trail: {vtm_status.get('current_runner_trail_pct', 0):.2%}\n"
                    else:
                        msg += f"🏃 Runner Trail: Fixed ({vtm_status.get('current_runner_trail_pct', 0):.2%})\n"

                    msg += f"🔄 Updates: {vtm_status['update_count']}\n\n"
                else:
                    msg += f"📊 <b>{position.asset}</b>: Static management (no VTM)\n\n"

            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            logger.error(f"VTM status command error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    # ─────────────────────────────────────────────────────────────────────────
    # T3.2 — VTM Manual Override Commands
    # /vtm_status  — rich per-position breakdown from VTM.get_override_status()
    # /set_sl <asset> <price> — move stop loss on a live position
    # /set_tp <asset> <price> [tier] — move take profit on a live position
    # ─────────────────────────────────────────────────────────────────────────

    async def cmd_vtm_status_detail(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        /vtm_status — detailed VTM breakdown for all open positions.
        Shows remaining TPs, bar count, runner state, and override hints.
        """
        try:
            if not self.trading_bot:
                await update.message.reply_text("⚠️ Trading bot not connected")
                return

            positions = self.trading_bot.portfolio_manager.positions
            if not positions:
                await update.message.reply_text("📊 No open positions")
                return

            lines = ["🎯 <b>VTM OVERRIDE STATUS</b>\n"]
            for pos_id, position in positions.items():
                if not position.trade_manager:
                    lines.append(f"• <b>{position.asset}</b>: static management (no VTM)\n")
                    continue

                s = position.trade_manager.get_override_status()
                side_emoji = "🟢" if s["side"] == "LONG" else "🔴"
                pnl_sign = "+" if s["pnl_pct"] >= 0 else ""
                tps_str = " → ".join(f"{tp:.5g}" for tp in s["remaining_tps"]) or "none"
                runner_str = "✅ active" if s["runner_active"] else "—"

                lines.append(
                    f"{side_emoji} <b>{s['asset']} {s['side']}</b>  |  "
                    f"type={s['trade_type']}\n"
                    f"  Entry   : {s['entry_price']:.5g}\n"
                    f"  Current : {s['current_price']:.5g}\n"
                    f"  SL      : {s['stop_loss']:.5g}  (init: {s['initial_sl']:.5g})\n"
                    f"  TPs left: {tps_str}  (hit: {s['tps_hit']})\n"
                    f"  P&L     : {pnl_sign}{s['pnl_pct']:.3f}%  |  "
                    f"Size rem: {s['remaining_pct']:.0f}%\n"
                    f"  Bars    : {s['bars_in_trade']}  |  Runner: {runner_str}\n\n"
                    f"  💡 <i>/set_sl {s['asset']} &lt;price&gt;</i>\n"
                    f"  💡 <i>/set_tp {s['asset']} &lt;price&gt; [tier]</i>\n"
                )

            await update.message.reply_text("".join(lines), parse_mode="HTML")

        except Exception as e:
            logger.error(f"[TG] /vtm_status error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {e}")

    async def cmd_set_sl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        /set_sl <asset> <new_stop_loss_price>

        Example: /set_sl BTC 93500
        Moves the stop loss on the BTC position to 93500.
        Validates direction (long SL must be below entry, short SL above).
        """
        try:
            if not self.trading_bot:
                await update.message.reply_text("⚠️ Trading bot not connected")
                return

            # Parse args
            args = context.args
            if not args or len(args) < 2:
                await update.message.reply_text(
                    "ℹ️ Usage: /set_sl &lt;asset&gt; &lt;price&gt;\n"
                    "Example: /set_sl BTC 93500",
                    parse_mode="HTML",
                )
                return

            asset_arg = args[0].upper()
            try:
                new_sl = float(args[1].replace(",", ""))
            except ValueError:
                await update.message.reply_text(f"❌ Invalid price: {args[1]}")
                return

            # Find matching position
            positions = self.trading_bot.portfolio_manager.positions
            matched = [
                p for p in positions.values()
                if p.asset.upper() == asset_arg and p.trade_manager
            ]

            if not matched:
                await update.message.reply_text(
                    f"⚠️ No active VTM position found for <b>{asset_arg}</b>.\n"
                    f"Use /vtm_status to see open positions.",
                    parse_mode="HTML",
                )
                return

            position = matched[0]
            result = position.trade_manager.override_stop_loss(new_sl)

            # Sync portfolio_manager's cached stop loss so it stays consistent
            if "✅" in result:
                position.stop_loss = new_sl

            await update.message.reply_text(
                f"<pre>{html.escape(result)}</pre>", parse_mode="HTML"
            )

        except Exception as e:
            logger.error(f"[TG] /set_sl error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {e}")

    async def cmd_set_tp(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        /set_tp <asset> <new_tp_price> [tier_index]

        Examples:
          /set_tp BTC 98000        — update the nearest unfilled TP to 98000
          /set_tp BTC 99000 2      — update TP tier 2 (1-indexed) to 99000
        Validates direction (long TP must be above entry, short below).
        """
        try:
            if not self.trading_bot:
                await update.message.reply_text("⚠️ Trading bot not connected")
                return

            args = context.args
            if not args or len(args) < 2:
                await update.message.reply_text(
                    "ℹ️ Usage: /set_tp &lt;asset&gt; &lt;price&gt; [tier]\n"
                    "Example: /set_tp BTC 98000\n"
                    "Example: /set_tp BTC 99000 2",
                    parse_mode="HTML",
                )
                return

            asset_arg = args[0].upper()
            try:
                new_tp = float(args[1].replace(",", ""))
            except ValueError:
                await update.message.reply_text(f"❌ Invalid price: {args[1]}")
                return

            tier = 0  # default: nearest unfilled TP
            if len(args) >= 3:
                try:
                    tier = max(0, int(args[2]) - 1)  # convert 1-indexed to 0-indexed
                except ValueError:
                    await update.message.reply_text(f"❌ Invalid tier: {args[2]}")
                    return

            positions = self.trading_bot.portfolio_manager.positions
            matched = [
                p for p in positions.values()
                if p.asset.upper() == asset_arg and p.trade_manager
            ]

            if not matched:
                await update.message.reply_text(
                    f"⚠️ No active VTM position found for <b>{asset_arg}</b>.\n"
                    f"Use /vtm_status to see open positions.",
                    parse_mode="HTML",
                )
                return

            position = matched[0]
            result = position.trade_manager.override_take_profit(new_tp, target_index=tier)

            # Sync portfolio_manager's cached take profit (tier-0 = primary TP)
            if "✅" in result and tier == 0:
                position.take_profit = new_tp

            await update.message.reply_text(
                f"<pre>{html.escape(result)}</pre>", parse_mode="HTML"
            )

        except Exception as e:
            logger.error(f"[TG] /set_tp error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {e}")

    async def cmd_reset_equity(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        /reset_equity — Re-baseline peak_equity and session_start_equity to the
        current live broker balance.  Use this after switching broker accounts
        (e.g. Binance demo → MT5 live) to clear a phantom drawdown that would
        otherwise keep the circuit-breaker permanently halted.
        """
        try:
            if not self.trading_bot:
                await update.message.reply_text("⚠️ Trading bot not connected")
                return

            pm = self.trading_bot.portfolio_manager
            if pm.is_paper_mode:
                await update.message.reply_text("ℹ️ Paper mode — equity baseline is virtual, nothing to reset.")
                return

            # Force-fetch the real balance from the broker
            ok = pm.refresh_capital(force=True)
            if not ok or not pm.equity:
                await update.message.reply_text("❌ Could not fetch live balance — check broker connection.")
                return

            old_peak = pm.peak_equity
            pm.peak_equity          = pm.equity
            pm.session_start_equity = pm.equity
            pm.session_start_capital= pm.equity
            pm.initial_capital      = pm.equity

            logger.info(
                f"[TG] /reset_equity: peak reset ${old_peak:,.2f} → ${pm.equity:,.2f} "
                f"by {update.effective_user.username or update.effective_user.id}"
            )
            await update.message.reply_text(
                f"✅ <b>Equity baseline reset</b>\n\n"
                f"Old peak:  <s>${old_peak:,.2f}</s>\n"
                f"New baseline: <b>${pm.equity:,.2f}</b>\n\n"
                f"Circuit-breaker has been cleared. Trading will resume on the next cycle.",
                parse_mode="HTML",
            )

        except Exception as e:
            logger.error(f"[TG] /reset_equity error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {e}")

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Last 10 closed trades with P&L, entry/exit prices, hold duration, exit reason"""
        import asyncio

        try:
            def _prepare_history():
                closed_positions = self.trading_bot.portfolio_manager.closed_positions
                if not closed_positions:
                    return "📭 <b>No Trade History</b>\n\nNo completed trades yet."

                recent_trades = closed_positions[-10:]
                msg = "📜 <b>RECENT TRADE HISTORY</b>  (last 10)\n\n"

                wins = sum(1 for t in recent_trades if t["pnl"] > 0)
                losses = sum(1 for t in recent_trades if t["pnl"] < 0)
                total_pnl = sum(t["pnl"] for t in recent_trades)
                pnl_sign = "+" if total_pnl >= 0 else ""
                msg += (f"Recent {len(recent_trades)}: {wins}W / {losses}L  |  "
                        f"P&amp;L {pnl_sign}${total_pnl:,.2f}\n"
                        f"{'─' * 30}\n\n")

                for trade in reversed(recent_trades):
                    asset   = trade["asset"]
                    side    = trade["side"].upper()
                    pnl     = trade["pnl"]
                    pnl_pct = trade["pnl_pct"] * 100
                    entry   = trade.get("entry_price", 0)
                    exit_p  = trade.get("exit_price", 0)
                    reason  = str(trade.get("reason", "unknown")).replace("_", " ").title()

                    # Hold duration
                    entry_time = trade.get("entry_time")
                    exit_time  = trade["exit_time"]
                    if entry_time:
                        hold_delta = exit_time - entry_time
                        hold_hrs   = int(hold_delta.total_seconds() / 3600)
                        hold_mins  = int((hold_delta.total_seconds() % 3600) / 60)
                        hold_str   = f"{hold_hrs}h {hold_mins}m"
                    else:
                        hold_str = "—"

                    pnl_icon = "🟢" if pnl >= 0 else "🔴"
                    pnl_sign = "+" if pnl >= 0 else ""
                    side_icon = "📈" if side == "LONG" else "📉"

                    msg += (
                        f"{pnl_icon} <b>{html.escape(asset)}</b> {side_icon} {side}\n"
                        f"  Entry : <code>${entry:,.4g}</code>  →  Exit: <code>${exit_p:,.4g}</code>\n"
                        f"  P&amp;L  : <b>{pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)</b>\n"
                        f"  Hold  : {hold_str}  |  <i>{html.escape(reason)}</i>\n"
                        f"  Closed: {exit_time.strftime('%m/%d %H:%M')}\n\n"
                    )

                return msg

            history_msg = await asyncio.to_thread(_prepare_history)
            await update.message.reply_text(history_msg, parse_mode="HTML")

        except Exception as e:
            logger.error(f"[TG] /history error: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching history.")

    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Full performance metrics — P&L, win rate, profit factor, drawdown, streaks"""
        try:
            closed_positions = self.trading_bot.portfolio_manager.closed_positions
            portfolio_status = self.trading_bot.portfolio_manager.get_portfolio_status()

            if not closed_positions:
                await update.message.reply_text(
                    "📊 <b>No Performance Data</b>\n\nNot enough trades yet.", parse_mode="HTML")
                return

            total_trades  = len(closed_positions)
            winning       = [t for t in closed_positions if t["pnl"] > 0]
            losing        = [t for t in closed_positions if t["pnl"] < 0]
            win_count     = len(winning)
            loss_count    = len(losing)
            win_rate      = (win_count / total_trades * 100) if total_trades > 0 else 0

            total_pnl     = sum(t["pnl"] for t in closed_positions)
            gross_profit  = sum(t["pnl"] for t in winning) if winning else 0
            gross_loss    = abs(sum(t["pnl"] for t in losing)) if losing else 0
            avg_win       = gross_profit / win_count if win_count else 0
            avg_loss      = (gross_loss / loss_count) if loss_count else 0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

            # Best / worst single trade
            best_trade  = max(closed_positions, key=lambda t: t["pnl"])
            worst_trade = min(closed_positions, key=lambda t: t["pnl"])

            # Current win/loss streak
            streak_val  = 0
            streak_type = "—"
            for t in reversed(closed_positions):
                is_win = t["pnl"] > 0
                if streak_val == 0:
                    streak_val  = 1
                    streak_type = "W" if is_win else "L"
                elif (is_win and streak_type == "W") or (not is_win and streak_type == "L"):
                    streak_val += 1
                else:
                    break

            # Max drawdown — peak-to-trough on running equity
            initial_capital = self.trading_bot.portfolio_manager.initial_capital
            equity_curve    = [initial_capital]
            for t in closed_positions:
                equity_curve.append(equity_curve[-1] + t["pnl"])
            peak      = initial_capital
            max_dd    = 0.0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            current_equity = portfolio_status.get("equity", equity_curve[-1])
            total_return   = (current_equity - initial_capital) / initial_capital * 100
            perf_icon      = "🟢" if total_return >= 0 else "🔴"

            msg = (
                f"<b>📈 PERFORMANCE METRICS</b>\n"
                f"{'─' * 30}\n\n"

                f"<b>📊 Trade Summary</b>\n"
                f"  Total Trades  : {total_trades}\n"
                f"  Win / Loss    : {win_count}W / {loss_count}L\n"
                f"  Win Rate      : <b>{win_rate:.1f}%</b>\n"
                f"  Current Streak: <b>{streak_val} {streak_type}</b>\n\n"

                f"<b>💰 P&amp;L Breakdown</b>\n"
                f"  Total P&amp;L     : <code>{'+' if total_pnl >= 0 else ''}${total_pnl:,.2f}</code>\n"
                f"  Avg Win       : <code>+${avg_win:,.2f}</code>\n"
                f"  Avg Loss      : <code>-${avg_loss:,.2f}</code>\n"
                f"  Profit Factor : <b>{profit_factor:.2f}</b>\n\n"

                f"<b>🏆 Best / Worst</b>\n"
                f"  Best Trade    : <code>+${best_trade['pnl']:,.2f}</code>  ({best_trade['asset']} {best_trade['side'].upper()})\n"
                f"  Worst Trade   : <code>${worst_trade['pnl']:,.2f}</code>  ({worst_trade['asset']} {worst_trade['side'].upper()})\n\n"

                f"<b>📉 Risk</b>\n"
                f"  Max Drawdown  : <b>{max_dd:.2f}%</b>\n\n"

                f"<b>💼 Equity</b>\n"
                f"  Initial       : <code>${initial_capital:,.2f}</code>\n"
                f"  Current       : <code>${current_equity:,.2f}</code>\n"
                f"  Total Return  : {perf_icon} <b>{total_return:+.2f}%</b>\n"
            )

            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            logger.error(f"[TG] /performance error: {e}", exc_info=True)
            await update.message.reply_text("❌ Error calculating performance.")

    @admin_only
    async def cmd_start_trading(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /start_trading command"""
        if self.trading_bot.is_running:
            await update.message.reply_text("ℹ️ Trading is already running.")
        else:
            self.trading_bot.is_running = True
            await update.message.reply_text(
                "🟢 *Trading Resumed*\n\n" "Bot will now process trading signals.",
                parse_mode=ParseMode.MARKDOWN,
            )
            logger.info("Trading resumed via Telegram command")

    @admin_only
    async def cmd_resume_trading(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """
        /resume — Override a tripped circuit breaker and allow new trades.

        This bypasses the daily-loss / drawdown / loss-streak gates until
        midnight (when the next trading session resets the override).
        Existing equity and peak values are left untouched — use /reset_equity
        if you also want to re-baseline those.
        """
        if not self.trading_bot:
            await update.message.reply_text("⚠️ Trading bot not connected.")
            return

        pm = self.trading_bot.portfolio_manager

        # Diagnose current breaker state before overriding
        halted, reason = pm.check_circuit_breaker()

        # Set override flag
        pm._circuit_breaker_override = True

        # Also make sure the bot loop is actually running
        was_stopped = not self.trading_bot.is_running
        if was_stopped:
            self.trading_bot.is_running = True

        lines = [
            "🟡 <b>Circuit Breaker Overridden</b>",
            "",
        ]
        if halted:
            lines.append(f"Was halted because: <i>{reason}</i>")
        else:
            lines.append("ℹ️ Circuit breaker was not currently active — override set anyway.")

        if was_stopped:
            lines.append("▶️ Bot loop was stopped — restarted.")

        lines += [
            "",
            "⚠️ <b>New positions may now be opened.</b>",
            "Override clears automatically at the next session start (midnight).",
            "Use /stop_trading to re-engage the halt manually.",
        ]

        user = update.effective_user
        logger.warning(
            f"[TG] /resume: circuit-breaker override set by "
            f"{user.username or user.id}. Prior reason: {reason or 'none'}"
        )
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    @admin_only
    async def cmd_stop_trading(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /stop_trading command"""
        if not self.trading_bot.is_running:
            await update.message.reply_text("ℹ️ Trading is already stopped.")
        else:
            self.trading_bot.is_running = False
            # Also clear any active /resume override so the circuit breaker re-engages
            pm = getattr(self.trading_bot, "portfolio_manager", None)
            if pm and getattr(pm, "_circuit_breaker_override", False):
                pm._circuit_breaker_override = False
            await update.message.reply_text(
                "🔴 *Trading Stopped*\n\n"
                "Bot will not open new positions.\n"
                "Existing positions remain open.\n\n"
                "Use /close\\_all to close all positions.",
                parse_mode=ParseMode.MARKDOWN,
            )
            logger.info("Trading stopped via Telegram command")

    @admin_only
    async def cmd_close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /close_all command (Non-Blocking)"""
        import asyncio

        status_msg = await update.message.reply_text(
            "🚨 *EMERGENCY CLOSE INITIATED*\nContacting exchanges in background...",
            parse_mode=ParseMode.MARKDOWN,
        )

        try:

            def _close_everything():
                # Get current prices directly from exchanges for all enabled assets
                prices = {}
                for asset_name, asset_cfg in self.trading_bot.config["assets"].items():
                    if not asset_cfg.get("enabled", False):
                        continue
                    
                    exchange = asset_cfg.get("exchange", "binance")
                    symbol = self.trading_bot._resolve_symbol(asset_name)

                    handler = (
                        self.trading_bot.binance_handler
                        if exchange == "binance"
                        else self.trading_bot.mt5_handler
                    )
                    
                    if handler and symbol:
                        price = handler.get_current_price(symbol=symbol)
                        if price:
                            prices[asset_name] = price

                # Close all positions
                self.trading_bot.portfolio_manager.close_all_positions(prices)

            # ✅ Offload the heavy multi-exchange API calls to a thread
            await asyncio.to_thread(_close_everything)

            await status_msg.edit_text(
                "✅ *All Positions Closed*\n\n"
                "All open positions have been liquidated successfully.\n"
                "Check /history for trade results.",
                parse_mode=ParseMode.MARKDOWN,
            )

            logger.info("All positions closed via Telegram command")

        except Exception as e:
            logger.error(f"Error closing all positions: {e}", exc_info=True)
            await status_msg.edit_text(
                "❌ Error closing positions. Check bot logs immediately."
            )

    @admin_only
    async def cmd_close_asset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /close <asset> [position_number] command (Non-Blocking)
        """
        import asyncio

        try:
            if not context.args:
                await update.message.reply_text(
                    "⚠️ *Usage:*\n"
                    "`/close BTC` - Close ALL BTC positions\n"
                    "`/close BTC 2` - Close 2nd BTC position\n"
                    "`/close GOLD` - Close ALL GOLD positions",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            asset = context.args[0].upper()

            if asset not in list(self.trading_bot.config["assets"].keys()):
                await update.message.reply_text(f"⚠️ Invalid asset. Available: {list(self.trading_bot.config['assets'].keys())}")
                return

            # Check if market is open for this asset
            asset_cfg = self.trading_bot.config["assets"].get(asset, {})
            exchange = asset_cfg.get("exchange", "binance")
            symbol = self.trading_bot._resolve_symbol(asset)

            if exchange == "mt5":
                mt5_handler = self.trading_bot.mt5_handler
                if mt5_handler and symbol:
                    is_open, market_msg = mt5_handler._is_market_open_for_closing(
                        symbol
                    )

                    if not is_open:
                        await update.message.reply_text(
                            f"⏰ *{asset} Market Closed*\n\n"
                            f"Cannot close positions:\n{market_msg}\n\n"
                            f"Please wait until the market reopens.\n"
                            f"Positions will remain open and safe.",
                            parse_mode=ParseMode.MARKDOWN,
                        )
                        return

            # ================================================================
            # CASE 1: Close Specific Position (Index provided)
            # ================================================================
            if len(context.args) > 1:
                try:
                    position_index = int(context.args[1]) - 1  # Convert to 0-indexed
                    if position_index < 0:
                        raise ValueError
                except ValueError:
                    await update.message.reply_text(
                        "⚠️ Invalid position number. Use: `/close BTC 2`",
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    return

                positions = self.trading_bot.portfolio_manager.get_asset_positions(
                    asset
                )

                if not positions or position_index >= len(positions):
                    await update.message.reply_text(
                        f"⚠️ Position #{position_index + 1} not found for {asset}"
                    )
                    return

                status_msg = await update.message.reply_text(
                    f"⏳ Contacting exchange to close {asset} position #{position_index + 1}..."
                )

                def _close_single():
                    # Get exit price
                    price = None
                    handler = (
                        self.trading_bot.binance_handler
                        if exchange == "binance"
                        else self.trading_bot.mt5_handler
                    )
                    if handler and symbol:
                        price = handler.get_current_price(symbol=symbol)

                    return self.trading_bot.portfolio_manager.close_position(
                        position_id=positions[position_index].position_id,
                        exit_price=price or positions[position_index].entry_price,
                        reason="manual_telegram_specific",
                    )

                # ✅ Offload single close to thread
                result = await asyncio.to_thread(_close_single)

                if result:
                    pnl_icon = "🟢" if result["pnl"] >= 0 else "🔴"
                    await status_msg.edit_text(
                        f"{pnl_icon} *{asset} Position #{position_index + 1} Closed*\n\n"
                        f"P&L: ${result['pnl']:,.2f} ({result['pnl_pct']*100:+.2f}%)\n"
                        f"Reason: Manual close via Telegram",
                        parse_mode=ParseMode.MARKDOWN,
                    )
                else:
                    await status_msg.edit_text(
                        f"❌ *Failed to close {asset} position #{position_index + 1}*\n\n"
                        f"Possible reasons:\n"
                        f"• Market is closed\n"
                        f"• Exchange connection issue\n"
                        f"• Position no longer exists on exchange\n\n"
                        f"Position remains in portfolio. Check logs for details.",
                        parse_mode=ParseMode.MARKDOWN,
                    )

            # ================================================================
            # CASE 2: Close ALL Positions for Asset (No index provided)
            # ================================================================
            else:
                status_msg = await update.message.reply_text(
                    f"⏳ Contacting exchange to close ALL {asset} positions..."
                )

                def _close_all_asset():
                    # Get exit price
                    price = None
                    handler = (
                        self.trading_bot.binance_handler
                        if exchange == "binance"
                        else self.trading_bot.mt5_handler
                    )
                    if handler and symbol:
                        price = handler.get_current_price(symbol=symbol)

                    return self.trading_bot.portfolio_manager.close_all_positions_for_asset(
                        asset=asset, exit_price=price, reason="manual_telegram_all"
                    )

                # ✅ Offload batch close to thread
                results = await asyncio.to_thread(_close_all_asset)

                if not results:
                    # Tracked positions existed but every close attempt failed
                    await status_msg.edit_text(
                        f"❌ *Failed to close {asset} positions*\n\n"
                        f"Possible reasons:\n"
                        f"• Market is currently closed\n"
                        f"• Exchange connection issue\n\n"
                        f"Check the bot logs for details.",
                        parse_mode=ParseMode.MARKDOWN,
                    )
                elif results and results[0].get("already_closed"):
                    # Sentinel: position was gone before this command ran
                    await status_msg.edit_text(
                        f"ℹ️ *{asset} position already closed*\n\n"
                        f"No open positions were found in the bot's tracker or on the exchange.\n"
                        f"The position was likely closed manually or by a stop-loss.",
                        parse_mode=ParseMode.MARKDOWN,
                    )
                else:
                    # Filter out sentinel entries before summing P&L
                    real_results = [r for r in results if not r.get("already_closed")]
                    total_pnl = sum(r["pnl"] for r in real_results)
                    pnl_icon = "🟢" if total_pnl >= 0 else "🔴"

                    msg = f"✅ *Closed ALL {asset} Positions*\n\n"
                    msg += f"Trades Closed: {len(real_results)}\n"
                    msg += f"{pnl_icon} Total P&L: ${total_pnl:,.2f}\n\n"

                    for i, result in enumerate(real_results[:5], 1):
                        pnl = result["pnl"]
                        pnl_pct = result.get("pnl_pct", 0.0) * 100
                        side = result.get("side", "?").upper()
                        orphan = " *(manual position)*" if result.get("orphan_close") else ""
                        icon = "🟢" if pnl >= 0 else "🔴"
                        msg += f"{icon} #{i} {side}: ${pnl:,.2f} ({pnl_pct:+.2f}%){orphan}\n"

                    if len(real_results) > 5:
                        msg += f"\n... and {len(real_results) - 5} more"

                    await status_msg.edit_text(msg, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"Error in cmd_close_asset: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ *Error Processing Close Command*\n\n"
                f"Details: {str(e)[:200]}\n\n"
                f"Check bot logs for more information.",
                parse_mode=ParseMode.MARKDOWN,
            )

    async def cmd_aggregator_modes(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /modes command - Show current and recent aggregator modes (Non-Blocking)"""
        import asyncio

        try:
            if not hasattr(self.trading_bot, "hybrid_selector"):
                await update.message.reply_text(
                    "❌ Hybrid aggregator selector not available."
                )
                return

            def _prepare_modes():
                selector = self.trading_bot.hybrid_selector
                msg = "🔀 *AGGREGATOR MODE STATUS*\n\n"

                stats = selector.get_statistics()
                current_modes = stats.get("current_modes", {})

                if not current_modes:
                    msg += "ℹ️ No mode information available yet.\n"
                else:
                    msg += "*Current Modes:*\n"
                    for asset, mode in current_modes.items():
                        emoji = "₿" if asset == "BTC" else "🥇"
                        mode_emoji = "🏛️" if mode == "council" else "📊"
                        msg += f"{emoji} {asset}: {mode_emoji} `{mode.upper()}`\n"

                    msg += f"\n*Total Switches:* {stats['total_switches']}\n"
                    msg += f"  • Council Signals: {stats['council_signals']}\n"
                    msg += (
                        f"  • Performance Signals: {stats['performance_signals']}\n\n"
                    )

                # Heavy operation: Iterating and sorting history
                for asset in list(self.trading_bot.config["assets"].keys()):
                    if asset not in self.trading_bot.config[
                        "assets"
                    ] or not self.trading_bot.config["assets"][asset].get(
                        "enabled", False
                    ):
                        continue

                    history = selector.get_mode_history(asset, n=3)
                    emoji = "₿" if asset == "BTC" else "🥇"
                    msg += f"\n{emoji} *{asset} - Recent Switches:*\n"

                    if not history:
                        msg += "  No switches recorded yet\n"
                    else:
                        for switch in reversed(history):
                            ts = switch["timestamp"].strftime("%m/%d %H:%M")
                            old = switch["old_mode"] or "None"
                            new = switch["new_mode"]
                            conf = switch["confidence"]
                            reg = switch["regime_type"].replace("_", " ").title()

                            old_e = (
                                "🏛️"
                                if old == "council"
                                else "📊" if old == "performance" else "❓"
                            )
                            new_e = "🏛️" if new == "council" else "📊"

                            msg += f"\n  *{ts}*\n"
                            msg += (
                                f"  {old_e} `{old.upper()}` → {new_e} `{new.upper()}`\n"
                            )
                            msg += f"  Confidence: {conf:.0%}\n"
                            msg += f"  Regime: {reg}\n"

                msg += f"\n🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

                keyboard = [
                    [
                        InlineKeyboardButton("📊 Status", callback_data="status"),
                        InlineKeyboardButton("📡 Signals", callback_data="signals"),
                    ],
                    [InlineKeyboardButton("🔄 Refresh", callback_data="modes")],
                ]
                return msg, InlineKeyboardMarkup(keyboard)

            # ✅ Offload to thread
            msg, reply_markup = await asyncio.to_thread(_prepare_modes)

            await self._send_chunked(
                update.message.reply_text, msg,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
            )

        except Exception as e:
            logger.error(f"Error in cmd_aggregator_modes: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching mode information")

    async def cmd_mode_details(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /modedetails command (Non-Blocking)"""
        import asyncio

        try:
            if not hasattr(self.trading_bot, "hybrid_selector"):
                await update.message.reply_text("❌ Hybrid mode not available")
                return

            if not context.args or len(context.args) < 1:
                await update.message.reply_text("⚠️ Usage: /modedetails <asset>")
                return

            asset = context.args[0].upper()
            if asset not in list(self.trading_bot.config["assets"].keys()):
                await update.message.reply_text(f"⚠️ Invalid asset. Available: {list(self.trading_bot.config['assets'].keys())}")
                return

            def _prepare_mode_details():
                selector = self.trading_bot.hybrid_selector
                history = selector.get_mode_history(asset, n=5)

                if not history:
                    return f"ℹ️ No mode history available for {asset}"

                emoji = "₿" if asset == "BTC" else "🥇"
                msg = f"{emoji} *{asset} MODE HISTORY (Last 5)*\n\n"

                for i, switch in enumerate(reversed(history), 1):
                    ts = switch["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                    old = switch["old_mode"] or "None"
                    new = switch["new_mode"]
                    conf = switch["confidence"]
                    reg = switch["regime_type"].replace("_", " ").title()
                    reason = switch["reasoning"]

                    old_e = (
                        "🏛️"
                        if old == "council"
                        else "📊" if old == "performance" else "❓"
                    )
                    new_e = "🏛️" if new == "council" else "📊"

                    msg += f"*Switch #{i}* - {ts}\n"
                    msg += f"{old_e} `{old.upper()}` → {new_e} `{new.upper()}`\n"
                    msg += f"*Confidence:* {conf:.0%}\n"
                    msg += f"*Regime:* {reg}\n"
                    msg += f"*Reasoning:* {reason}\n\n"

                    # Add market details
                    trend = switch["trend"]
                    vol = switch["volatility"]
                    pa = switch["price_action"]

                    msg += f"*Market Snapshot:*\n"
                    msg += f"  Trend: {trend['strength'].title()} {trend['direction'].title()} (ADX: {trend['adx']:.1f})\n"
                    msg += (
                        f"  Volatility: {vol['regime'].title()} ({vol['ratio']:.2f}x)\n"
                    )
                    msg += f"  Price Action: {pa['clarity'].title()} ({pa['indecision_pct']:.0f}% indecision)\n"
                    msg += "\n" + "-" * 40 + "\n\n"

                return msg

            # ✅ Offload history processing to thread
            msg = await asyncio.to_thread(_prepare_mode_details)

            # Send message (with pagination handling)
            if len(msg) > 4000:
                chunks = [msg[i : i + 4000] for i in range(0, len(msg), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(
                        chunk, parse_mode=ParseMode.MARKDOWN
                    )
            else:
                await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"Error in cmd_mode_details: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def _send_modes_message(self, query):
        """
        ✅ NEW: Send modes message (for callback button)
        """
        try:
            if not hasattr(self.trading_bot, "hybrid_selector"):
                await query.edit_message_text("❌ Hybrid mode not available")
                return

            selector = self.trading_bot.hybrid_selector

            msg = "🔀 *AGGREGATOR MODE STATUS*\n\n"

            stats = selector.get_statistics()
            current_modes = stats.get("current_modes", {})

            if not current_modes:
                msg += "ℹ️ No mode information available yet.\n"
            else:
                msg += "*Current Modes:*\n"
                for asset, mode in current_modes.items():
                    emoji = "₿" if asset == "BTC" else "🥇"
                    mode_emoji = "🏛️" if mode == "council" else "📊"
                    msg += f"{emoji} {asset}: {mode_emoji} `{mode.upper()}`\n"

                msg += f"\n*Statistics:*\n"
                msg += f"  Total Switches: {stats['total_switches']}\n"
                msg += f"  Council: {stats['council_signals']}\n"
                msg += f"  Performance: {stats['performance_signals']}\n\n"

            # Recent switches
            for asset in list(self.trading_bot.config["assets"].keys()):
                if asset not in self.trading_bot.config["assets"]:
                    continue

                if not self.trading_bot.config["assets"][asset].get("enabled", False):
                    continue

                history = selector.get_mode_history(asset, n=2)

                emoji = "₿" if asset == "BTC" else "🥇"
                msg += f"\n{emoji} *{asset} Recent:*\n"

                if not history:
                    msg += "  No switches yet\n"
                else:
                    for switch in reversed(history):
                        ts = switch["timestamp"].strftime("%m/%d %H:%M")
                        old = switch["old_mode"] or "None"
                        new = switch["new_mode"]

                        old_emoji = (
                            "🏛️"
                            if old == "council"
                            else "📊" if old == "performance" else "❓"
                        )
                        new_emoji = "🏛️" if new == "council" else "📊"

                        msg += f"  {ts}: {old_emoji} → {new_emoji} ({switch['confidence']:.0%})\n"

            msg += f"\n🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            keyboard = [
                [
                    InlineKeyboardButton("📊 Status", callback_data="status"),
                    InlineKeyboardButton("📡 Signals", callback_data="signals"),
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="modes")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in _send_modes_message: {e}")

    async def cmd_preset_history(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """
        Handle /preset_history command - Show recent preset changes
        """
        try:
            if not hasattr(self.trading_bot, "dynamic_selector"):
                await update.message.reply_text(
                    "❌ Dynamic preset selector not available"
                )
                return

            stats = self.trading_bot.dynamic_selector.get_statistics()

            msg = "📊 *PRESET CHANGE HISTORY*\n\n"

            # Total changes
            total = stats.get("total_changes", 0)
            msg += f"Total Changes: {total}\n\n"

            # Per-asset breakdown
            for asset, count in stats.get("changes_by_asset", {}).items():
                history = self.trading_bot.dynamic_selector.preset_history.get(
                    asset, []
                )

                emoji = "₿" if asset == "BTC" else "🥇"
                current = stats.get("current_presets", {}).get(asset, "N/A")

                msg += f"{emoji} *{asset}*\n"
                msg += f"  Current: `{current.upper()}`\n"
                msg += f"  Changes: {count}\n"

                # Show last 3 changes or a message if no history
                if history:
                    msg += f"  Recent:\n"
                    for change in history[-3:]:
                        ts = change["timestamp"].strftime("%m/%d %H:%M")
                        old = change["old_preset"] or "None"
                        new = change["new_preset"]
                        msg += f"    • {ts}: {old.upper()} → {new.upper()}\n"
                else:
                    msg += f"  No change history available.\n"

                msg += "\n"

            msg += f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            keyboard = [
                [
                    InlineKeyboardButton("⚙️ Current Presets", callback_data="presets"),
                    InlineKeyboardButton("📊 Status", callback_data="status"),
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="preset_history")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await self._send_chunked(
                update.message.reply_text, msg,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup,
            )

        except Exception as e:
            logger.error(f"Error in cmd_preset_history: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching preset history")

    async def _send_preset_history_message(self, query):
        """Send preset history message (for callback)"""
        try:
            if not hasattr(self.trading_bot, "dynamic_selector"):
                await query.edit_message_text(
                    "❌ Dynamic preset selector not available"
                )
                return

            stats = self.trading_bot.dynamic_selector.get_statistics()

            msg = "📊 *PRESET CHANGE HISTORY*\n\n"

            total = stats.get("total_changes", 0)
            msg += f"Total Changes: {total}\n\n"

            for asset, count in stats.get("changes_by_asset", {}).items():
                history = self.trading_bot.dynamic_selector.preset_history.get(
                    asset, []
                )

                emoji = "₿" if asset == "BTC" else "🥇"
                current = stats.get("current_presets", {}).get(asset, "N/A")

                msg += f"{emoji} *{asset}*\n"
                msg += f"  Current: `{current.upper()}`\n"
                msg += f"  Changes: {count}\n"

                if history:
                    msg += f"  Recent:\n"
                    for change in history[-3:]:
                        ts = change["timestamp"].strftime("%m/%d %H:%M")
                        old = change["old_preset"] or "None"
                        new = change["new_preset"]
                        msg += f"    • {ts}: {old.upper()} → {new.upper()}\n"

                msg += "\n"

            msg += f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            keyboard = [
                [
                    InlineKeyboardButton("⚙️ Current Presets", callback_data="presets"),
                    InlineKeyboardButton("📊 Status", callback_data="status"),
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="preset_history")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in _send_preset_history_message: {e}")

    async def cmd_presets(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /presets command - Show current aggregator presets (Non-Blocking)"""
        import asyncio

        try:
            if not hasattr(self.trading_bot, "selected_presets"):
                await update.message.reply_text(
                    "❌ Preset information not available.\nPresets are determined during bot startup."
                )
                return

            def _prepare_presets():
                presets = self.trading_bot.selected_presets

                if not presets:
                    return (
                        "ℹ️ No presets configured yet.\nBot may still be initializing.",
                        None,
                    )

                aggregator_cfg = self.trading_bot.config.get("aggregator_settings", {})
                preset_mode = aggregator_cfg.get("preset", "auto")

                msg = "⚙️ *Aggregator Preset Configuration*\n\n"

                if preset_mode == "auto":
                    msg += "🤖 *Mode:* AUTO-SELECT\n"
                    msg += "Presets are automatically selected based on current market conditions.\n\n"
                else:
                    msg += f"🔧 *Mode:* MANUAL ({preset_mode.upper()})\n"
                    msg += "All assets use the same preset.\n\n"

                msg += "*Current Presets:*\n"

                for asset, preset in presets.items():
                    emoji = "₿" if "BTC" in asset.upper() else "🥇" if "GOLD" in asset.upper() else "📈"
                    msg += f"\n{emoji} *{asset}:* `{preset.upper()}`\n"
                    msg += self._get_preset_description(preset)

                msg += f"\n\n🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

                keyboard = [
                    [
                        InlineKeyboardButton("📊 Status", callback_data="status"),
                        InlineKeyboardButton("📡 Signals", callback_data="signals"),
                    ],
                    [InlineKeyboardButton("🔄 Refresh", callback_data="presets")],
                ]
                return msg, InlineKeyboardMarkup(keyboard)

            # ✅ Offload data retrieval and string concatenation
            msg, reply_markup = await asyncio.to_thread(_prepare_presets)

            if reply_markup is None:
                await update.message.reply_text(msg)
            else:
                await update.message.reply_text(
                    msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
                )

        except Exception as e:
            logger.error(f"Error in cmd_presets: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching preset information")

    def _get_preset_description(self, preset: str) -> str:
        """Get description for a preset"""
        descriptions = {
            "conservative": "  • Low risk, high thresholds\n  • Best for stable markets",
            "balanced": "  • Moderate risk/reward\n  • Default for most conditions",
            "aggressive": "  • Higher frequency trading\n  • Best for trending markets",
            "scalper": "  • Maximum activity\n  • Best for high volatility",
        }
        return descriptions.get(preset, "  • Unknown preset")

    # ==================== DIRECT COMMAND WRAPPERS ====================
    # These commands previously only had _send_* callback versions.
    # Without a cmd_ method the CommandHandler registration above raises
    # AttributeError on the first use of /signals, /stats, /regimes, /overrides.

    async def cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals — latest trading signals for all assets"""
        try:
            msg = "📡 <b>LATEST TRADING SIGNALS</b>\n\n"
            any_signals = False

            for asset in list(self.trading_bot.config["assets"].keys()):
                if not self.trading_bot.config["assets"][asset].get("enabled", False):
                    continue

                emoji = "₿" if "BTC" in asset.upper() else "🥇" if "GOLD" in asset.upper() else "📈"
                signals = self.signal_monitor.get_last_signals(asset, n=5)

                msg += f"{emoji} <b>{html.escape(asset)}</b> — last 5 signals\n"

                if not signals:
                    msg += "  <i>No signals recorded yet.</i>\n\n"
                    continue

                any_signals = True
                for sig in reversed(signals):
                    ts = sig["timestamp"].strftime("%H:%M:%S")
                    val = sig["signal"]
                    price = sig["price"]
                    sig_icon = "📈 BUY" if val == 1 else "📉 SELL" if val == -1 else "⚪ HOLD"
                    quality = sig.get("quality", 0) or 0
                    mode = sig.get("aggregator_mode", "performance")
                    regime = sig.get("regime", "N/A") or "N/A"

                    msg += f"  <code>[{ts}]</code> {sig_icon} @ <code>${price:,.2f}</code>\n"

                    if mode == "council":
                        score = sig.get("council_score") or 0
                        decision = sig.get("council_decision", "N/A")
                        msg += f"    Mode: COUNCIL | Score: <b>{score:.2f}</b> | {decision}\n"
                    else:
                        reasoning = sig.get("reasoning") or "N/A"
                        msg += f"    Mode: PERF | Quality: <b>{quality:.1%}</b>\n"
                        if reasoning and reasoning != "N/A":
                            msg += f"    Reason: <i>{html.escape(str(reasoning))}</i>\n"

                    msg += f"    Regime: <code>{html.escape(str(regime))}</code>\n"

                msg += "\n"

            if not any_signals:
                msg += "<i>Bot hasn't produced any signals yet. Check back after the first cycle.</i>"

            msg += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}"
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="signals")]]
            await self._send_chunked(
                update.message.reply_text, msg,
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception as e:
            logger.error(f"[TG] /signals error: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching signals")

    async def cmd_signal_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats — signal distribution and quality statistics"""
        try:
            msg = "📊 <b>SIGNAL STATISTICS</b>\n\n"
            any_data = False

            for asset in list(self.trading_bot.config["assets"].keys()):
                if not self.trading_bot.config["assets"][asset].get("enabled", False):
                    continue

                stats = self.signal_monitor.get_signal_statistics(asset)
                if not stats:
                    continue

                any_data = True
                emoji = "₿" if "BTC" in asset.upper() else "🥇" if "GOLD" in asset.upper() else "📈"
                total = stats["total_signals"]
                buy = stats["buy_signals"]
                sell = stats["sell_signals"]
                hold = stats["hold_signals"]
                avg_q = stats["avg_quality"]
                hq = stats["high_quality_count"]

                # Simple bar visualisation for signal distribution
                buy_bar = "█" * int(stats["buy_pct"] / 10)
                sell_bar = "█" * int(stats["sell_pct"] / 10)
                hold_bar = "█" * int(stats["hold_pct"] / 10)

                msg += f"{emoji} <b>{html.escape(asset)}</b>  ({total} signals tracked)\n"
                msg += f"  📈 BUY  {buy_bar:<10} {buy:>3} ({stats['buy_pct']:.1f}%)\n"
                msg += f"  📉 SELL {sell_bar:<10} {sell:>3} ({stats['sell_pct']:.1f}%)\n"
                msg += f"  ⚪ HOLD {hold_bar:<10} {hold:>3} ({stats['hold_pct']:.1f}%)\n"
                msg += f"  ⭐ Avg Quality: <b>{avg_q:.2f}</b>  |  High-Quality (≥65%): {hq}\n\n"

            if not any_data:
                msg += "<i>No signal data yet — check back after the first cycle.</i>"

            msg += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}"
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="stats")]]
            await update.message.reply_text(msg, parse_mode="HTML",
                                            reply_markup=InlineKeyboardMarkup(keyboard))
        except Exception as e:
            logger.error(f"[TG] /stats error: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching signal stats")

    async def cmd_regimes(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /regimes — market regime tracking per asset"""
        try:
            msg = "🔄 <b>MARKET REGIME TRACKING</b>\n\n"

            for asset in list(self.trading_bot.config["assets"].keys()):
                if not self.trading_bot.config["assets"][asset].get("enabled", False):
                    continue

                emoji = "₿" if "BTC" in asset.upper() else "🥇" if "GOLD" in asset.upper() else "📈"
                info = self.signal_monitor.get_regime_info(asset)

                # Also pull live regime from regime_data if available
                live_regime = "—"
                live_score = None
                if hasattr(self.trading_bot, "_current_regime_data"):
                    rd = self.trading_bot._current_regime_data.get(asset)
                    if rd:
                        rd = rd.to_dict() if hasattr(rd, "to_dict") else rd
                        live_regime = rd.get("consensus_regime", "NEUTRAL")
                        bull = rd.get("bullish_score", 0)
                        bear = rd.get("bearish_score", 0)
                        live_score = f"+{bull:.2f} / -{bear:.2f}"

                msg += f"{emoji} <b>{html.escape(asset)}</b>\n"
                msg += f"  Current Regime : <code>{html.escape(str(live_regime))}</code>\n"
                if live_score:
                    msg += f"  Bull/Bear Score: <code>{live_score}</code>\n"
                msg += f"  Regime Changes : {info.get('change_count', 0)}\n"

                last_changes = info.get("last_changes", [])
                if last_changes:
                    msg += "  Recent Changes :\n"
                    for change in last_changes[-3:]:
                        ts = change["timestamp"].strftime("%m/%d %H:%M")
                        regime = str(change.get("regime", "?"))
                        regime_icon = "🚀" if "BULL" in regime else "🐻" if "BEAR" in regime else "⚖️"
                        msg += f"    {regime_icon} {ts} — <code>{html.escape(regime)}</code> @ ${change['price']:,.2f}\n"
                msg += "\n"

            msg += f"🕐 {datetime.now().strftime('%H:%M:%S')}"
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="regimes")]]
            await update.message.reply_text(msg, parse_mode="HTML",
                                            reply_markup=InlineKeyboardMarkup(keyboard))
        except Exception as e:
            logger.error(f"[TG] /regimes error: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching regime info")

    async def cmd_overrides(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /overrides — signal override events (Golden Cross etc.)"""
        try:
            msg = "🔒 <b>SIGNAL OVERRIDE EVENTS</b>\n"
            msg += "<i>Fires when a strong trend override vetoes the base signal</i>\n\n"

            any_data = False
            for asset in list(self.trading_bot.config["assets"].keys()):
                if not self.trading_bot.config["assets"][asset].get("enabled", False):
                    continue

                info = self.signal_monitor.get_override_info(asset)
                emoji = "₿" if "BTC" in asset.upper() else "🥇" if "GOLD" in asset.upper() else "📈"

                msg += f"{emoji} <b>{html.escape(asset)}</b>\n"
                msg += f"  Total Overrides: {info['total']}\n"

                if info["total"] > 0:
                    any_data = True
                    msg += f"  Avg Quality at Override: {info['avg_quality']:.2f}\n"
                    last_events = info.get("last_events", [])
                    if last_events:
                        msg += "  Recent:\n"
                        for ev in last_events[-3:]:
                            ts = ev["timestamp"].strftime("%m/%d %H:%M")
                            msg += f"    • {ts} — quality={ev['quality']:.2f} @ ${ev['price']:,.2f}\n"
                msg += "\n"

            if not any_data:
                msg += "<i>No override events recorded yet.</i>\n"

            msg += f"\n🕐 {datetime.now().strftime('%H:%M:%S')}"
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="overrides")]]
            await update.message.reply_text(msg, parse_mode="HTML",
                                            reply_markup=InlineKeyboardMarkup(keyboard))
        except Exception as e:
            logger.error(f"[TG] /overrides error: {e}", exc_info=True)
            await update.message.reply_text("❌ Error fetching override info")

    # ==================== NOTIFICATION METHODS ====================

    async def _send_presets_message(self, query):
        """Send presets message (for callback)"""
        try:
            if not hasattr(self.trading_bot, "selected_presets"):
                await query.edit_message_text(
                    "❌ Preset information not available.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            presets = self.trading_bot.selected_presets

            if not presets:
                await query.edit_message_text(
                    "ℹ️ No presets configured yet.", parse_mode=ParseMode.MARKDOWN
                )
                return

            aggregator_cfg = self.trading_bot.config.get("aggregator_settings", {})
            preset_mode = aggregator_cfg.get("preset", "auto")

            msg = "⚙️ *Aggregator Preset Configuration*\n\n"

            if preset_mode == "auto":
                msg += "🤖 *Mode:* AUTO-SELECT\n\n"
            else:
                msg += f"🔧 *Mode:* MANUAL ({preset_mode.upper()})\n\n"

            msg += "*Current Presets:*\n"

            for asset, preset in presets.items():
                if not self.trading_bot.config["assets"].get(asset, {}).get("enabled", False):
                    continue
                emoji = "₿" if "BTC" in asset.upper() else "🥇" if "GOLD" in asset.upper() else "📈"
                msg += f"\n{emoji} *{asset}:* `{preset.upper()}`\n"
                msg += self._get_preset_description(preset)

            msg += f"\n\n🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"

            keyboard = [
                [
                    InlineKeyboardButton("📊 Status", callback_data="status"),
                    InlineKeyboardButton("📡 Signals", callback_data="signals"),
                ],
                [InlineKeyboardButton("🔄 Refresh", callback_data="presets")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in _send_presets_message: {e}")

    async def _send_regimes_message(self, query):
        """Send regime information (for callback)"""
        try:
            msg = "🔄 *Market Regimes*\n\n"
            for asset in self.trading_bot.config["assets"].keys():
                if not self.trading_bot.config["assets"][asset].get("enabled", False):
                    continue
                info = self.signal_monitor.get_regime_info(asset)
                emoji = "₿" if "BTC" in asset.upper() else "🥇" if "GOLD" in asset.upper() else "📈"
                msg += f"{emoji} *{asset}*\n"
                msg += f"Changes: {info.get('change_count', 0)}\n"
                if info.get("last_changes"):
                    msg += "Recent:\n"
                    for change in info["last_changes"][-3:]:
                        ts = change["timestamp"].strftime("%H:%M")
                        regime_icon = "🚀" if "BULL" in change["regime"] else "🐻"
                        msg += f"  {ts}: {regime_icon} @ ${change['price']:,.2f}\n"
                msg += "\n"

            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="regimes")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )
        except Exception as e:
            logger.error(f"Error in _send_regimes_message: {e}")

    async def _send_overrides_message(self, query):
        """Send override information (for callback)"""
        try:
            msg = "🔒 *Golden Cross Overrides*\n\n"
            for asset in self.trading_bot.config["assets"].keys():
                if not self.trading_bot.config["assets"][asset].get("enabled", False):
                    continue
                info = self.signal_monitor.get_override_info(asset)
                emoji = "₿" if "BTC" in asset.upper() else "🥇" if "GOLD" in asset.upper() else "📈"
                msg += f"{emoji} *{asset}*\n"
                msg += f"Total: {info['total']}\n"
                if info["total"] > 0:
                    msg += f"Avg Quality: {info['avg_quality']:.2f}\n"
                msg += "\n"

            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="overrides")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )
        except Exception as e:
            logger.error(f"Error in _send_overrides_message: {e}")

    async def cmd_test_viz(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Test AI visualization (admin only) - Non-Blocking Version"""
        import asyncio

        if update.effective_user.id not in self.admin_ids:
            await update.message.reply_text("❌ Admin only")
            return

        status_msg = await update.message.reply_text(
            "🎨 Fetching data and generating test chart in background..."
        )

        def _generate_test_viz(asset):
            df_15min = self.trading_bot._fetch_current_data(asset)
            df_4h = self.trading_bot._fetch_4h_data(asset)
            aggregator = self.trading_bot.aggregators.get(asset)
            signal, details = aggregator.get_aggregated_signal(df_15min)
            
            asset_cfg = self.trading_bot.config['assets'].get(asset, {})
            exchange = asset_cfg.get('exchange', 'binance')
            symbol = self.trading_bot._resolve_symbol(asset)
            
            handler = (
                self.trading_bot.binance_handler
                if exchange == "binance"
                else self.trading_bot.mt5_handler
            )
            
            current_price = None
            if handler and symbol:
                current_price = handler.get_current_price(symbol=symbol)
            
            if current_price is None:
                current_price = df_15min["close"].iloc[-1]
                
            return df_15min, df_4h, signal, details, current_price

        try:
            for asset_name in list(self.trading_bot.config["assets"].keys()):
                if not self.trading_bot.config["assets"][asset_name].get("enabled"):
                    continue

                await status_msg.edit_text(f"⏳ Processing {asset_name}...")

                # ✅ OFFLOAD HEAVY WORK TO THREAD
                df_15, df_4, sig, det, price = await asyncio.to_thread(
                    _generate_test_viz, asset_name
                )

                if self.trading_bot.chart_sender:
                    await self.trading_bot.chart_sender.send_decision_chart(
                        asset_name=asset_name,
                        df_15min=df_15,
                        df_4h=df_4,
                        signal=sig,
                        details=det,
                        current_price=price,
                    )

            await status_msg.edit_text("✅ Test charts sent successfully")

        except Exception as e:
            logger.error(f"Test viz error: {e}", exc_info=True)
            await status_msg.edit_text(f"❌ Error: {str(e)[:100]}")

    async def cmd_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        ✅ FIXED: Chart command that anchors to active loop
        """
        try:
            user_id = update.effective_user.id
            if user_id not in self.admin_ids:
                await update.message.reply_text("❌ Admin only")
                return

            # Get requested asset
            requested_asset = None
            if context.args and len(context.args) > 0:
                requested_asset = context.args[0].upper()
                if requested_asset not in list(self.trading_bot.config["assets"].keys()):
                    await update.message.reply_text(
                        f"⚠️ Invalid asset. Use: /chart {list(self.trading_bot.config['assets'].keys())}"
                    )
                    return

            # ✅ FIX: Use current loop (guaranteed to be alive)
            loop = asyncio.get_running_loop()

            # Send initial status
            status_message = await update.message.reply_text(
                "🎨 *Generating AI decision charts...*\nFetching live data...",
                parse_mode=ParseMode.MARKDOWN,
            )

            # Get assets to chart
            assets_to_chart = []
            for asset_name in list(self.trading_bot.config["assets"].keys()):
                if not self.trading_bot.config["assets"][asset_name].get(
                    "enabled", False
                ):
                    continue
                if requested_asset and asset_name != requested_asset:
                    continue
                assets_to_chart.append(asset_name)

            if not assets_to_chart:
                await status_message.edit_text("❌ No enabled assets to chart")
                return

            charts_sent = 0
            for asset_name in assets_to_chart:
                try:
                    await status_message.edit_text(
                        f"🧠 Running AI Inference for {asset_name}..."
                    )

                    # ✅ FIX: Offload heavy work to thread executor
                    def _generate_chart_data():
                        df_15 = self.trading_bot._fetch_current_data(asset_name)
                        df_4 = self.trading_bot._fetch_4h_data(asset_name)

                        agg = self.trading_bot.aggregators.get(asset_name)
                        if agg is None:
                            # Aggregator never initialised (failed at startup or not yet set).
                            # Mirror main-loop behaviour: skip with a clear message.
                            raise ValueError(
                                f"No aggregator available for {asset_name} — "
                                "it may have failed to initialise at startup. "
                                "Check the bot log for [ERROR] messages during startup."
                            )
                        elif isinstance(agg, dict) and agg.get("mode") == "hybrid":
                            # Hybrid mode: dict carries both aggregator objects
                            sig, det = (
                                self.trading_bot.get_aggregated_signal_hybrid_dynamic(
                                    asset_name=asset_name,
                                    df=df_15.copy(),
                                    aggregators=agg,
                                    hybrid_selector=self.trading_bot.hybrid_selector,
                                )
                            )
                        else:
                            # Single-mode aggregator (council or performance object)
                            mtf_regime = {}
                            if (
                                hasattr(self.trading_bot, "_current_regime_data")
                                and asset_name in self.trading_bot._current_regime_data
                            ):
                                mtf_regime = self.trading_bot._current_regime_data[asset_name]
                            sig, det = agg.get_aggregated_signal(
                                df_15.copy(),
                                current_regime=mtf_regime.get("regime", "NEUTRAL"),
                                is_bull_market=mtf_regime.get("is_bull", False),
                                governor_data=mtf_regime,
                            )

                        exchange = self.trading_bot.config["assets"].get(asset_name, {}).get("exchange", "binance")
                        symbol = self.trading_bot._resolve_symbol(asset_name)
                        handler = (
                            self.trading_bot.binance_handler
                            if exchange == "binance"
                            else self.trading_bot.mt5_handler
                        )
                        price = None
                        if handler and symbol:
                            price = handler.get_current_price(symbol=symbol)
                        
                        if price is None:
                            price = df_15["close"].iloc[-1]

                        return df_15, df_4, sig, det, price

                    # ✅ Run in thread pool to avoid blocking
                    df_15min, df_4h, signal, details, current_price = (
                        await loop.run_in_executor(None, _generate_chart_data)
                    )

                    await status_message.edit_text(
                        f"🎨 Rendering {asset_name} chart..."
                    )

                    # Send chart
                    if self.trading_bot.chart_sender:
                        await self.trading_bot.chart_sender.send_decision_chart(
                            asset_name=asset_name,
                            df_15min=df_15min,
                            df_4h=df_4h,
                            signal=signal,
                            details=details,
                            current_price=current_price,
                        )
                        charts_sent += 1

                except Exception as e:
                    logger.error(
                        f"[CHART CMD] Error for {asset_name}: {e}", exc_info=True
                    )
                    await update.message.reply_text(
                        f"❌ Error generating {asset_name} chart: {str(e)[:100]}"
                    )

            # Final status
            if charts_sent > 0:
                await status_message.edit_text(
                    f"✅ Sent {charts_sent} chart(s) successfully."
                )
            else:
                await status_message.edit_text("❌ Failed to generate any charts.")

        except Exception as e:
            logger.error(f"[CHART CMD] Critical error: {e}", exc_info=True)

            # ✅ Better error recovery
            try:
                await update.message.reply_text(
                    "❌ Command failed. Bot is recovering..."
                )
            except:
                pass

    async def _send_stats_message(self, query):
        """Send signal statistics (for callback)"""
        try:
            msg = "📊 *Signal Statistics*\n\n"
            
            for asset in list(self.trading_bot.config["assets"].keys()):
                if not self.trading_bot.config["assets"][asset].get("enabled", False):
                    continue
                
                stats = self.signal_monitor.get_signal_statistics(asset)
                if not stats:
                    continue
                    
                emoji = "₿" if asset == "BTC" else "🥇" if "GOLD" in asset.upper() else "📈"
                msg += f"*{emoji} {asset}*\n"
                msg += f"Total: {stats['total_signals']}\n"
                msg += f"🟢 BUY: {stats['buy_signals']} ({stats['buy_pct']:.1f}%)\n"
                msg += f"🔴 SELL: {stats['sell_signals']} ({stats['sell_pct']:.1f}%)\n"
                msg += f"⚪ HOLD: {stats['hold_signals']} ({stats['hold_pct']:.1f}%)\n"
                msg += f"⭐ Avg Quality: {stats['avg_quality']:.2f}\n\n"

            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="stats")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )
        except Exception as e:
            logger.error(f"Error in _send_stats_message: {e}")

    async def _send_signals_message(self, query):
        """Send signals message (for callback)"""
        try:
            msg = "📡 *Latest Trading Signals*\n\n"
            for asset in list(self.trading_bot.config["assets"].keys()):
                if not self.trading_bot.config["assets"][asset].get("enabled", False):
                    continue

                emoji = "₿" if asset == "BTC" else "🥇"
                msg += f"{emoji} *{asset} Signals (last 5)*\n"

                signals = self.signal_monitor.get_last_signals(asset, n=5)
                if not signals:
                    msg += "_No signals recorded yet._\n\n"
                    continue

                for sig in reversed(signals):
                    ts = sig["timestamp"].strftime("%H:%M:%S")
                    signal_val = sig["signal"]
                    price = sig["price"]
                    sig_icon = (
                        "📈 BUY"
                        if signal_val == 1
                        else "📉 SELL" if signal_val == -1 else " HOLD"
                    )

                    msg += f"\n*{ts}* - *{sig_icon}* @ `${price:,.2f}`\n"

                    # Hybrid-aware output
                    mode = sig.get("aggregator_mode", "performance")
                    msg += f"  Mode: `{mode.upper()}`\n"

                    if mode == "council":
                        score = sig.get("council_score") or 0
                        decision = sig.get("council_decision", "N/A")
                        msg += f"  Score: *{score:.2f}* | Decision: {decision}\n"
                    else:  # Default to performance
                        quality = sig.get("quality", 0) or 0
                        reasoning = sig.get("reasoning", "N/A")
                        msg += f"  Quality: *{quality:.1%}*\n"
                        msg += f"  Reasoning: _{reasoning}_\n"
                msg += "\n"

            reply_markup = InlineKeyboardMarkup(
                [[InlineKeyboardButton("🔄 Refresh", callback_data="signals")]]
            )
            await query.edit_message_text(
                msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in _send_signals_message: {e}", exc_info=True)
            # Try to send a simple error message if editing fails
            try:
                await query.edit_message_text("❌ Error fetching signals.")
            except:
                pass

    async def send_notification(self, message: str, disable_preview: bool = True, parse_mode: str = None):
        """
        Send notification with proper error handling and retry logic.
        parse_mode: "HTML" | "Markdown" | None (defaults to ParseMode.MARKDOWN)
        """
        if not self._is_ready or not self.application:
            logger.debug("[TELEGRAM] Not ready, queuing message")
            self._message_queue.append(message)
            return

        if not self.is_running:
            logger.warning("[TELEGRAM] Bot not running")
            return

        effective_parse_mode = parse_mode if parse_mode else ParseMode.MARKDOWN
        success_count = 0

        for admin_id in self.admin_ids:
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    # ✅ Use asyncio.wait_for to enforce timeout
                    await asyncio.wait_for(
                        self.application.bot.send_message(
                            chat_id=admin_id,
                            text=message,
                            parse_mode=effective_parse_mode,
                            disable_web_page_preview=disable_preview,
                        ),
                        timeout=15.0,  # ✅ 15 second timeout
                    )

                    success_count += 1
                    logger.debug(f"[TELEGRAM] Message sent to {admin_id}")
                    break  # Success

                except asyncio.TimeoutError:
                    logger.warning(
                        f"[TELEGRAM] Timeout sending to {admin_id} (attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                    else:
                        logger.error(
                            f"[TELEGRAM] Failed to send to {admin_id} after {max_retries} attempts"
                        )

                except (NetworkError, TimedOut) as e:
                    logger.warning(f"[TELEGRAM] Network error to {admin_id}: {e}")
                    if attempt < max_retries - 1:
                        # ✅ Try to reconnect by getting bot info
                        try:
                            await self.application.bot.get_me()
                            logger.info("[TELEGRAM] Reconnected after network error")
                        except:
                            pass
                        await asyncio.sleep(2)

                except RuntimeError as e:
                    if "Event loop is closed" in str(e):
                        logger.warning(f"[TELEGRAM] Skipping send to {admin_id}: Event loop is closed.")
                        # This admin_id cannot be reached right now, no need to retry.
                        break
                    else:
                        logger.error(f"[TELEGRAM] Runtime error sending to {admin_id}: {e}")
                        break
                except Exception as e:
                    logger.error(f"[TELEGRAM] Error sending to {admin_id}: {e}")
                    break

        if success_count > 0:
            logger.info(
                f"[TELEGRAM] Notification sent to {success_count}/{len(self.admin_ids)} admins"
            )
        else:
            logger.error("[TELEGRAM] Failed to send to any admin")

    async def notify_trade_opened(
        self,
        asset: str,
        side: str,
        price: float,
        size: float,
        sl: float,
        tp: float,
        leverage: int = 1,
        margin_type: str = "SPOT",
        is_futures: bool = False,
        vtm_is_active: bool = False,
    ):
        """
        ✅ FIXED: Notify when a trade is opened with correct futures/spot detection

        Args:
            asset: Asset name (e.g., "BTC", "GOLD")
            side: "long" or "short"
            price: Entry price
            size: Position size in USD
            sl: Stop loss price
            tp: Take profit price
            leverage: Leverage multiplier (default 1)
            margin_type: "SPOT", "CROSSED", "ISOLATED"
            is_futures: True if futures/margin trading
            vtm_is_active: True if VTM is managing the trade
        """
        try:
            # Determine icons and labels
            side_icon = "🟢" if side.lower() == "long" else "🔴"

            # ✅ FIX: Better type detection and formatting
            if is_futures:
                if leverage > 1:
                    type_str = f"⚡ Futures {leverage}x ({margin_type})"
                else:
                    type_str = f"⚡ Margin ({margin_type})"
            else:
                type_str = "💰 Spot"

            # Calculate risk metrics
            if sl and sl > 0:
                if side.lower() == "long":
                    sl_distance_pct = ((price - sl) / price) * 100
                else:
                    sl_distance_pct = ((sl - price) / price) * 100
                sl_risk_usd = size * (sl_distance_pct / 100)
            else:
                sl_distance_pct = 0
                sl_risk_usd = 0

            if tp and tp > 0:
                if side.lower() == "long":
                    tp_distance_pct = ((tp - price) / price) * 100
                else:
                    tp_distance_pct = ((price - tp) / price) * 100
                tp_profit_usd = size * (tp_distance_pct / 100)
            else:
                tp_distance_pct = 0
                tp_profit_usd = 0

            # Build message
            msg = (
                f"{side_icon} *Trade Opened: {asset}*\n\n"
                f"⚙️ Type: {type_str}\n"
                f"📊 Side: {side.upper()}\n"
                f"💵 Entry: ${price:,.2f}\n"
                f"💰 Size: ${size:,.2f}\n\n"
            )

            # Add SL info if available
            if sl and sl > 0:
                msg += (
                    f"🛑 Stop Loss: ${sl:,.2f}\n"
                    f"   └─ Risk: {sl_distance_pct:.2f}% (${sl_risk_usd:.2f})\n\n"
                )
            else:
                msg += "🛑 Stop Loss: VTM Dynamic\n\n"

            # Add TP info if available
            if tp and tp > 0:
                msg += (
                    f"🎯 Take Profit: ${tp:,.2f}\n"
                    f"   └─ Target: {tp_distance_pct:.2f}% (${tp_profit_usd:.2f})\n\n"
                )
            else:
                msg += "🎯 Take Profit: VTM Dynamic\n\n"

            # Add risk-reward ratio if both SL and TP exist
            if sl and tp and sl > 0 and tp > 0:
                rr_ratio = (
                    tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0
                )
                msg += f"📈 Risk/Reward: 1:{rr_ratio:.2f}\n\n"
            
            # ✅ NEW: Add VTM warning if applicable
            if vtm_is_active:
                msg += "⚠ *SL Management: VTM Only (No Exchange SL)*\n\n"

            # Add timestamp
            msg += f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            # Send notification
            await self.send_notification(msg)

            logger.info(f"[TELEGRAM] Trade opened notification sent for {asset}")

        except Exception as e:
            logger.error(
                f"[TELEGRAM] Error sending trade opened notification: {e}",
                exc_info=True,
            )

    async def notify_signal_blocked(
        self,
        asset: str,
        signal: int,
        block_source: str,
        block_reason: str,
        details: dict = None,
        price: float = None,
    ):
        """
        Automatic alert when a BUY/SELL signal is generated but blocked/not executed.
        Sent to all admins immediately when the bot vetoes a trade.
        """
        try:
            import html as html_lib
            details = details or {}

            direction = "BUY" if signal == 1 else "SELL"
            dir_icon  = "🟢" if signal == 1 else "🔴"

            # Source → icon map
            source_icons = {
                "AI Validation":    "🤖",
                "MTF Counter-Trend":"🔄",
                "MTF Max Positions":"📊",
                "Trading Limits":   "⏸",
                "Cooldown":         "⏱",
                "System Health":    "🏥",
                "Circuit Breaker":  "⚡",
                "Quality Gate":     "📉",
                "Economy Calendar": "📅",
                "NY Open Block":    "🗽",
            }
            src_icon = source_icons.get(block_source, "🚫")

            # Key signal metrics
            quality    = details.get("signal_quality", 0)
            regime     = details.get("regime", "")
            trade_type = details.get("trade_type", "")
            agg_mode   = details.get("aggregator_mode", "")
            engine_tag = (
                "[COUNCIL]" if agg_mode == "council"
                else "[HYBRID]" if agg_mode == "hybrid"
                else "[PERF]"
            )

            quality_bar_filled = int(quality * 10)
            quality_bar = "█" * quality_bar_filled + "░" * (10 - quality_bar_filled)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            msg = (
                f"🚫 <b>Signal Blocked — {html_lib.escape(asset)}</b>\n\n"
                f"{dir_icon} <b>Direction:</b> {direction}\n"
                f"{src_icon} <b>Blocked by:</b> {html_lib.escape(block_source)}\n"
                f"📋 <b>Reason:</b> <i>{html_lib.escape(block_reason)}</i>\n\n"
            )

            if price:
                msg += f"💵 <b>Price:</b> ${price:,.4f}\n"

            msg += (
                f"📊 <b>Quality:</b> {quality:.1%}  <code>{quality_bar}</code>\n"
            )

            if regime:
                msg += f"🌐 <b>Regime:</b> {html_lib.escape(regime)}\n"
            if trade_type:
                msg += f"⚙️ <b>Mode:</b> {html_lib.escape(trade_type)}\n"
            if agg_mode:
                msg += f"🧠 <b>Engine:</b> <code>{engine_tag}</code>\n"

            # Council-specific: show score that was rejected
            if agg_mode == "council":
                total_score = details.get("total_score")
                decision    = details.get("decision_type", "")
                if total_score is not None:
                    msg += f"🏛️ <b>Council:</b> {total_score:.2f}/5.0  [{decision}]\n"

            # Performance-specific: AI rejection detail
            ai_rej = details.get("ai_rejection_reason") or ""
            ai_reasons = details.get("rejection_reasons") or []
            if ai_rej and not ai_reasons:
                ai_reasons = [ai_rej.replace("_", " ").title()]
            if ai_reasons:
                msg += "\n<b>AI Rejection detail:</b>\n"
                for r in ai_reasons[:3]:
                    msg += f"  • {html_lib.escape(str(r))}\n"

            msg += f"\n🕐 {ts}"

            await self.send_notification(msg, parse_mode="HTML")
            logger.info(f"[TELEGRAM] Signal-blocked notification sent for {asset} ({direction})")

        except Exception as e:
            logger.error(f"[TELEGRAM] notify_signal_blocked failed: {e}", exc_info=True)

    # ── Exit reason registry ─────────────────────────────────────────────────
    # Maps every VTM exit reason string to (emoji, short label, one-line context).
    # Prefix "vtm_" is stripped before lookup so "VTM_stop_loss" → "stop_loss".
    _EXIT_REASON_MAP = {
        # Mechanical exits
        "stop_loss":           ("🛑", "Stop Loss Hit",        "Price hit the hard stop."),
        "trailing_stop":       ("🏃", "Trailing Stop",        "Trailing stop caught the pullback."),
        "break_even":          ("⚖️",  "Break Even",           "Stopped out at entry — no loss."),
        "take_profit_1":       ("💰", "TP1 — Partial (45%)",  "First target reached. Runner active."),
        "take_profit_2":       ("💰", "TP2 — Partial (30%)",  "Second target reached. Runner continues."),
        "take_profit_3":       ("💰", "TP3 — Final exit",     "All targets hit. Position closed."),
        "time_stop":           ("⏰", "Time Stop",            "Position held too long without resolution."),
        "early_scale":         ("⚡", "Early Scale (20%)",    "Quick profit locked in first bars."),
        # Smart market-condition exits
        "volatility_spike":    ("🌪️",  "Volatility Spike",    "ATR doubled — original risk model no longer valid. 75% closed, runner kept."),
        "reversal_candle":     ("🕯️",  "Reversal Candle",     "Strong engulfing bar against position. 50% closed, SL tightened to entry."),
        "trend_invalidation":  ("❌",  "Trend Invalidated",   "3 bars against + ADX < 20. Market turned flat — full close."),
        "momentum_exhaustion": ("📉",  "Momentum Exhaustion", "RSI extreme + MACD dying + ADX falling. Move spent — 50% closed, SL at break-even."),
        # Manual / other
        "manual":              ("🖐️",  "Manual Close",        "Closed via Telegram command."),
        "manual_close_asset":  ("🖐️",  "Manual Close",        "Closed via Telegram command."),
        "manual_telegram_all": ("🖐️",  "Manual Close (All)",  "All positions closed via Telegram."),
    }

    async def notify_trade_closed(
        self, asset: str, side: str, pnl: float, pnl_pct: float, reason: str,
        partial: bool = False, partial_pct: float = None
    ):
        """Notify when a trade is closed (full or partial)."""
        pnl_icon = "🟢" if pnl >= 0 else "🔴"
        pnl_sign = "+" if pnl >= 0 else ""

        # Normalise reason key: strip vtm_ prefix, lowercase
        key = reason.lower().lstrip("vtm_").lstrip("vtm ")
        # Handle compound prefixes like "vtm_take_profit_1"
        if reason.lower().startswith("vtm_"):
            key = reason[4:].lower()

        entry = self._EXIT_REASON_MAP.get(key)
        if entry:
            r_emoji, r_label, r_context = entry
        else:
            r_emoji  = "📋"
            r_label  = reason.replace("_", " ").title()
            r_context = ""

        close_type = "Partial Close" if partial else "Trade Closed"
        size_note  = f" ({partial_pct:.0f}% of position)" if partial and partial_pct else ""

        msg = (
            f"{pnl_icon} *{close_type}: {asset}*{size_note}\n\n"
            f"Side    : {side.upper()}\n"
            f"P&L     : {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
            f"Exit    : {r_emoji} {r_label}\n"
        )
        if r_context:
            msg += f"Context : _{r_context}_\n"
        msg += f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        await self.send_notification(msg)

    async def notify_error(self, error_msg: str):
        """Notify about errors"""
        msg = (
            f"⚠️ *Error Alert*\n\n"
            f"{error_msg}\n\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        await self.send_notification(msg)

    async def send_daily_summary(self):
        """Send end-of-day performance summary - PROPERLY ASYNC"""
        try:
            portfolio_status = self.trading_bot.portfolio_manager.get_portfolio_status()

            daily_pnl = portfolio_status.get("daily_pnl", 0)
            total_value = portfolio_status.get("total_value", 0)
            open_positions = portfolio_status.get("open_positions", 0)

            # Get today's trades
            closed_positions = self.trading_bot.portfolio_manager.closed_positions
            today = datetime.now().date()
            today_trades = [
                t for t in closed_positions if t["exit_time"].date() == today
            ]

            winning_today = sum(1 for t in today_trades if t["pnl"] > 0)
            losing_today = sum(1 for t in today_trades if t["pnl"] < 0)

            pnl_icon = "🟢" if daily_pnl >= 0 else "🔴"
            pnl_sign = "+" if daily_pnl >= 0 else ""

            msg = (
                f"📊 *Daily Summary - {today.strftime('%Y-%m-%d')}*\n\n"
                # f"{pnl_icon} Daily P&L: {pnl_sign}${daily_pnl:,.2f}\n"
                f"💰 Portfolio Value: ${total_value:,.2f}\n"
                f"📈 Open Positions: {open_positions}\n\n"
                f"📊 Today's Trades: {len(today_trades)}\n"
                f"🟢 Winning: {winning_today}\n"
                f"🔴 Losing: {losing_today}\n"
            )

            await self.send_notification(msg)
            self.last_daily_summary = datetime.now()
            logger.info("[TELEGRAM] Daily summary sent successfully")

        except Exception as e:
            logger.error(f"Error sending daily summary: {e}", exc_info=True)

    # ==================== HELPER METHODS ====================

    def _get_gold_market_status(self) -> str:
        """Get GOLD market status from trading bot"""
        try:
            from src.utils.market_hours import MarketHours, should_trade_gold

            is_open = should_trade_gold()
            status, message = MarketHours.get_market_status("gold")

            if is_open:
                return "✅ Open"
            else:
                return f"🔴 Closed - {message.split('-')[1].strip() if '-' in message else 'Weekend'}"
        except:
            return "❓ Unknown"

    async def _send_status_message(self, query):
        """Send status message (for Refresh callback) — mirrors cmd_status"""
        try:
            portfolio  = self.trading_bot.portfolio_manager.get_portfolio_status()
            is_running = getattr(self.trading_bot, "is_running", False)

            run_icon = "🟢 RUNNING" if is_running else "🔴 STOPPED"
            cb = getattr(self.trading_bot, "circuit_breaker", None) or {}
            if hasattr(cb, "__dict__"):
                cb = cb.__dict__
            cb_active = cb.get("is_active", False) or cb.get("triggered", False)
            cb_str = "🚨 TRIGGERED" if cb_active else "✅ OK"

            total_value = portfolio.get("total_value", 0)
            cash        = portfolio.get("cash", 0)
            open_pos    = portfolio.get("open_positions", 0)
            daily_pnl   = portfolio.get("daily_pnl", 0)
            unrealized  = portfolio.get("total_unrealized_pnl", 0)
            dpnl_icon   = "🟢" if daily_pnl >= 0 else "🔴"
            dpnl_sign   = "+" if daily_pnl >= 0 else ""
            ui          = "🟢" if unrealized >= 0 else "🔴"
            us          = "+" if unrealized >= 0 else ""

            initial_cap = getattr(self.trading_bot.portfolio_manager, "initial_capital", None)
            total_ret_str = ""
            if initial_cap and initial_cap > 0:
                total_ret = (total_value - initial_cap) / initial_cap * 100
                ret_icon  = "🟢" if total_ret >= 0 else "🔴"
                total_ret_str = f"  Return   : {ret_icon} <code>{total_ret:+.2f}%</code>\n"

            closed = getattr(self.trading_bot.portfolio_manager, "closed_positions", [])
            today  = datetime.now().date()
            today_trades = [t for t in closed if t.get("exit_time") and t["exit_time"].date() == today]
            wins_today   = sum(1 for t in today_trades if t.get("pnl", 0) > 0)
            loss_today   = sum(1 for t in today_trades if t.get("pnl", 0) < 0)
            today_str    = f"{len(today_trades)} trades"
            if today_trades:
                today_str += f"  {wins_today}W / {loss_today}L"

            msg = (
                f"<b>🤖 TBOT STATUS</b>\n"
                f"{'─' * 30}\n"
                f"Engine   : {run_icon}\n"
                f"Circuit  : {cb_str}\n\n"
                f"<b>💼 Portfolio</b>\n"
                f"  Value    : <code>${total_value:,.2f}</code>\n"
                f"  Cash     : <code>${cash:,.2f}</code>\n"
                f"{total_ret_str}"
                f"  Daily P&amp;L : {dpnl_icon} <code>{dpnl_sign}${daily_pnl:,.2f}</code>\n"
                f"  Unrealised: {ui} <code>{us}${unrealized:,.2f}</code>\n"
                f"  Today    : {today_str}\n\n"
            )

            # Open positions summary
            positions = self.trading_bot.portfolio_manager.positions
            if positions:
                msg += f"<b>📍 Open Positions ({open_pos})</b>\n"
                for pos_id, position in positions.items():
                    asset = position.asset
                    side  = position.side.upper()
                    side_icon = "🟢" if side == "LONG" else "🔴"
                    entry = position.entry_price
                    asset_cfg = self.trading_bot.config["assets"].get(asset, {})
                    exchange  = asset_cfg.get("exchange", "binance")
                    symbol    = self.trading_bot._resolve_symbol(asset)
                    handler   = (self.trading_bot.binance_handler if exchange == "binance"
                                 else self.trading_bot.mt5_handler)
                    current = entry
                    try:
                        if handler and symbol:
                            p = handler.get_current_price(symbol=symbol, force_live=True)
                            if p:
                                current = p
                    except Exception:
                        pass
                    qty  = position.quantity
                    pnl  = (current - entry) * qty if side == "LONG" else (entry - current) * qty
                    pnl_pct = (pnl / (entry * qty) * 100) if entry > 0 and qty > 0 else 0
                    pi   = "🟢" if pnl >= 0 else "🔴"
                    ps   = "+" if pnl >= 0 else ""
                    vtm  = getattr(position, "trade_manager", None)
                    sl   = vtm.current_stop_loss if vtm else getattr(position, "stop_loss", None)
                    sl_str = f"  SL <code>${sl:,.2f}</code>" if sl else ""
                    msg += (
                        f"  {side_icon} <b>{html.escape(asset)}</b> {side} "
                        f"{pi} <code>{ps}{pnl_pct:.2f}%</code>{sl_str}\n"
                    )
                msg += "\n"
            else:
                msg += "<b>📍 No open positions</b>\n\n"

            # Per-asset signal summary
            msg += "<b>🌐 Assets</b>\n"
            agg_modes = {}
            if hasattr(self.trading_bot, "hybrid_selector"):
                try:
                    agg_modes = self.trading_bot.hybrid_selector.get_statistics().get("current_modes", {})
                except Exception:
                    pass

            for asset_name, asset_cfg in self.trading_bot.config["assets"].items():
                if not asset_cfg.get("enabled", False):
                    continue
                emoji = "₿" if "BTC" in asset_name.upper() else "🥇" if "GOLD" in asset_name.upper() else "📈"
                if "BTC" in asset_name.upper():
                    mstatus = "✅"
                elif hasattr(self.trading_bot, "check_market_hours"):
                    mstatus = "✅" if self.trading_bot.check_market_hours(asset_name) else "🔴"
                else:
                    mstatus = "❓"
                mode = agg_modes.get(asset_name, "")
                mode_tag = f"[{mode[:4].upper()}] " if mode else ""
                last_sigs = self.signal_monitor.get_last_signals(asset_name, n=1)
                sig_str = "<i>—</i>"
                if last_sigs:
                    s     = last_sigs[0]
                    val   = s["signal"]
                    ts    = s["timestamp"].strftime("%H:%M")
                    q     = s.get("quality", 0) or 0
                    sicon = "📈" if val == 1 else "📉" if val == -1 else "⚪"
                    slbl  = "BUY" if val == 1 else "SELL" if val == -1 else "HOLD"
                    sig_str = f"{sicon} {slbl} {q:.0%} <i>{ts}</i>"
                msg += f"  {emoji} <b>{html.escape(asset_name)}</b> {mstatus} {mode_tag}{sig_str}\n"

            msg += f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            keyboard = [
                [InlineKeyboardButton("📊 Positions", callback_data="positions"),
                 InlineKeyboardButton("🧠 Brain",     callback_data="brain")],
                [InlineKeyboardButton("📜 History",   callback_data="history"),
                 InlineKeyboardButton("🔄 Refresh",   callback_data="status")],
            ]
            await query.edit_message_text(msg, parse_mode="HTML",
                                          reply_markup=InlineKeyboardMarkup(keyboard))

        except Exception as e:
            logger.error(f"Error in _send_status_message: {e}", exc_info=True)

    async def _send_positions_message(self, query):
        """Send positions message (for callback) -  VERSION"""
        try:
            # Fetch current prices
            current_prices = {}
            for asset_name, asset_cfg in self.trading_bot.config["assets"].items():
                if not asset_cfg.get("enabled", False):
                    continue

                exchange = asset_cfg.get("exchange", "binance")
                handler = (
                    self.trading_bot.binance_handler
                    if exchange == "binance"
                    else self.trading_bot.mt5_handler
                )

                if handler:
                    try:
                        symbol = self.trading_bot._resolve_symbol(asset_name)
                        price = handler.get_current_price(symbol=symbol)
                        if price and price > 0:
                            current_prices[asset_name] = price
                    except Exception as e:
                        logger.debug(f"Failed to get {asset_name} price: {e}")

            # Update positions with latest prices
            self.trading_bot.portfolio_manager.update_positions(current_prices)

            # Get updated status
            portfolio_status = self.trading_bot.portfolio_manager.get_portfolio_status(
                current_prices
            )
            positions = portfolio_status.get("positions", {})

            if not positions:
                await query.edit_message_text(
                    "📭 *No Open Positions*\n\nCurrently no active trades.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            positions_msg = "📊 *Open Positions*\n\n"

            for asset, pos in positions.items():
                side = pos["side"].upper()
                side_icon = "🟢" if side == "LONG" else "🔴"

                entry_price = pos.get("entry_price", 0)
                current_price = pos.get("current_price", 0)
                quantity = pos.get("quantity", 0)
                current_value = pos.get("current_value", 0)
                pnl = pos.get("pnl", 0)
                pnl_pct = pos.get("pnl_pct", 0) * 100

                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                pnl_sign = "+" if pnl >= 0 else ""

                positions_msg += (
                    f"{side_icon} *{asset} - {side}*\n"
                    f"📍 Entry: ${entry_price:,.2f}\n"
                    f"💹 Current: ${current_price:,.2f}\n"
                    f"📦 Qty: {quantity:.6f}\n"
                    f"💰 Value: ${current_value:,.2f}\n"
                    f"{pnl_icon} P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
                )

                if pos.get("stop_loss"):
                    positions_msg += f"🛑 SL: ${pos['stop_loss']:,.2f}\n"
                if pos.get("take_profit"):
                    positions_msg += f"🎯 TP: ${pos['take_profit']:,.2f}\n"

                positions_msg += "\n"

            # Add summary
            total_unrealized = portfolio_status.get("total_unrealized_pnl", 0)
            unrealized_icon = "🟢" if total_unrealized >= 0 else "🔴"
            unrealized_sign = "+" if total_unrealized >= 0 else ""

            positions_msg += (
                f"{unrealized_icon} Total Unrealized: {unrealized_sign}${total_unrealized:,.2f}\n"
                f"🕐 Updated: {datetime.now().strftime('%H:%M:%S')}"
            )

            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data="positions"),
                    InlineKeyboardButton("📊 Status", callback_data="status"),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                positions_msg, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Error in _send_positions_message: {e}", exc_info=True)

    async def _send_history_message(self, query):
        """Send history message (for callback)"""
        try:
            closed_positions = self.trading_bot.portfolio_manager.closed_positions

            if not closed_positions:
                await query.edit_message_text(
                    "📭 *No Trade History*\n\nNo completed trades yet.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            recent_trades = closed_positions[-10:]

            history_msg = "📜 *Recent Trade History*\n\n"

            for trade in reversed(recent_trades):
                asset = trade["asset"]
                side = trade["side"].upper()
                pnl = trade["pnl"]
                pnl_pct = trade["pnl_pct"] * 100

                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                pnl_sign = "+" if pnl >= 0 else ""

                exit_time = trade["exit_time"].strftime("%m/%d %H:%M")
                reason = trade["reason"].replace("_", " ").title()

                history_msg += (
                    f"{pnl_icon} *{asset} {side}*\n"
                    f"P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
                    f"Exit: {exit_time} | {reason}\n\n"
                )

            await query.edit_message_text(history_msg, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"Error in _send_history_message: {e}")

    def _format_signal_entry(self, signal_entry: Dict) -> str:
        """Format a signal entry for display"""
        timestamp = signal_entry["timestamp"].strftime("%H:%M:%S")
        price = signal_entry["price"]
        signal = signal_entry["signal"]
        regime = signal_entry["regime"]
        quality = signal_entry["quality"]
        reasoning = signal_entry["reasoning"].replace("_", " ").title()

        # Signal icon
        if signal == 1:
            signal_icon = "\U0001f7e2 BUY"
        elif signal == -1:
            signal_icon = "\U0001f534 SELL"
        else:
            signal_icon = "\u26aa HOLD"

        # Quality indicator
        quality_icon = "\u2605" if quality >= 0.65 else "\u2022"

        entry = (
            f"  {timestamp} | {signal_icon} | ${price:,.2f}\n"
            f"    {regime} | Quality: {quality_icon} {quality:.2f}\n"
            f"    {reasoning}\n"
        )

        return entry
