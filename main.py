#!/usr/bin/env python3
"""
Main Trading Bot -  STABILITY VERSION
Enhanced error handling, network resilience, and Telegram thread management
"""


import subprocess
import json
import logging
import sys
import time
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
import schedule
import io
import signal
import threading
from threading import Thread, Event
from typing import Optional, Tuple, Dict
from types import SimpleNamespace


from dotenv import load_dotenv
import os

# Load environment variables from .env file at startup
load_dotenv()

# Windows encoding fix
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
    except:
        pass

from src.data.data_manager import DataManager
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.ema_strategy import EMAStrategy
from src.execution.signal_aggregator import PerformanceWeightedAggregator
from src.execution.binance_handler import BinanceExecutionHandler
from src.execution.mt5_handler import MT5ExecutionHandler
from src.portfolio.portfolio_manager import PortfolioManager
from src.utils.market_hours import MarketHours, should_trade_btc, should_trade_gold
from src.execution.auto_preset_selector import DynamicPresetSelector
from src.execution.hybrid_aggregator_selector import HybridAggregatorSelector
from src.ai import (
    DynamicAnalyst,
    OHLCSniper,
    HybridSignalValidator,
    AIValidatorMonitor,
    AIValidatorTuner,
)
from src.database.database_manager import (
    TradingDatabaseManager,
    calculate_daily_summary_from_trades,
)
from src.ai.visualization import (
    AIVisualizationGenerator,
    TelegramChartSender,
    create_visualization_system,
    should_send_chart,
)
from src.telegram.telegram_data_manager import ThreadSafeBotDataManager
from src.update.historical_updater import HistoricalDataUpdater
from src.utils.trade_logger import log_trade_event
from src.utils.calendar_updater import CalendarUpdater
from src.audit_logger.audit_logger import log_trade
from src.monitoring.health_monitor import HealthMonitor
from src.portfolio.hedging_support import (
    enable_hedging_for_portfolio,
    log_hedging_status,
)


import pickle

# Import Telegram bot
from src.telegram import TradingTelegramBot, SignalMonitoringIntegration
from telegram_config import TELEGRAM_CONFIG
from src.global_error_handler import GlobalErrorHandler, ErrorSeverity, handle_errors
from src.execution.mtf_integration import MTFRegimeIntegration
from src.training.autotrainer import ContinuousLearningPipeline
from src.execution.cvd_consumer import CVDConsumer
from src.execution.council_aggregator import InstitutionalCouncilAggregator
from src.execution.shadow_trader import ShadowTradingEngine  # T3.1


def setup_logging(config):
    """Setup logging with proper encoding and rotation"""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_file = log_config.get("file", "logs/trading_bot.log")

    Path(log_file).parent.mkdir(exist_ok=True)

    # ✨  Add log rotation
    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        log_file, encoding="utf-8", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot with  stability and error recovery"""

    def __init__(self, config_path: str = "config/config.json"):
        logger.info("=" * 70)
        logger.info("INITIALIZING TRADING BOT")
        logger.info("=" * 70)

        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        # Override config with environment variables for security
        if os.getenv("SUPABASE_URL"):
            self.config.setdefault("database", {})["supabase_url"] = os.getenv("SUPABASE_URL")
        if os.getenv("SUPABASE_KEY"):
            self.config.setdefault("database", {})["supabase_key"] = os.getenv("SUPABASE_KEY")
        
        # Override Trading Mode
        if os.getenv("TRADING_MODE"):
            self.config.setdefault("trading", {})["mode"] = os.getenv("TRADING_MODE").lower()
            logger.info(f"[CONFIG] Overriding trading mode from .env: {os.getenv('TRADING_MODE')}")

        # Override Telegram config
        if os.getenv("TELEGRAM_BOT_TOKEN"):
            self.config.setdefault("telegram", {})["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN")
            # If token is provided in env, assume enabled unless explicitly disabled
            self.config.setdefault("telegram", {})["enabled"] = True
            
        if os.getenv("TELEGRAM_ENABLED"):
            enabled_str = os.getenv("TELEGRAM_ENABLED").lower()
            self.config.setdefault("telegram", {})["enabled"] = enabled_str in ("true", "1", "yes")

        if os.getenv("TELEGRAM_ADMIN_IDS"):
            # Convert comma-separated string to list of ints
            try:
                ids_str = os.getenv("TELEGRAM_ADMIN_IDS")
                admin_ids = [int(i.strip()) for i in ids_str.split(",") if i.strip()]
                self.config.setdefault("telegram", {})["admin_ids"] = admin_ids
            except Exception as e:
                logger.error(f"Failed to parse TELEGRAM_ADMIN_IDS: {e}")

        # Override Exchange Credentials in DataManager config
        if os.getenv("BINANCE_API_KEY"):
            self.config.setdefault("api", {}).setdefault("binance_futures", {})["api_key"] = os.getenv("BINANCE_API_KEY")
            self.config.setdefault("api", {}).setdefault("binance", {})["api_key"] = os.getenv("BINANCE_API_KEY")
        if os.getenv("BINANCE_API_SECRET"):
            self.config.setdefault("api", {}).setdefault("binance_futures", {})["api_secret"] = os.getenv("BINANCE_API_SECRET")
            self.config.setdefault("api", {}).setdefault("binance", {})["api_secret"] = os.getenv("BINANCE_API_SECRET")
            
        # Determine if we should use testnet
        is_testnet = self.config.get("api", {}).get("binance_futures", {}).get("testnet", False)
        if os.getenv("BINANCE_TESTNET"):
            is_testnet = os.getenv("BINANCE_TESTNET").lower() in ("true", "1", "yes")
        elif self.config.get("trading", {}).get("mode") == "paper":
            is_testnet = True # Default to testnet in paper mode
            
        self.config.setdefault("api", {}).setdefault("binance_futures", {})["testnet"] = is_testnet
        self.config.setdefault("api", {}).setdefault("binance", {})["testnet"] = is_testnet
        
        if is_testnet:
            logger.info("[CONFIG] Binance API will use TESTNET endpoints")
        else:
            logger.info("[CONFIG] Binance API will use LIVE endpoints")
            
        if os.getenv("MT5_LOGIN"):
            self.config.setdefault("api", {}).setdefault("mt5", {})["login"] = int(os.getenv("MT5_LOGIN"))
        if os.getenv("MT5_PASSWORD"):
            self.config.setdefault("api", {}).setdefault("mt5", {})["password"] = os.getenv("MT5_PASSWORD")
        if os.getenv("MT5_SERVER"):
            self.config.setdefault("api", {}).setdefault("mt5", {})["server"] = os.getenv("MT5_SERVER")

        # ============================================================================
        # ✨ NEW: CONFIGURATION VALIDATION
        # ============================================================================
        try:
            portfolio_cfg = self.config.get("portfolio", {})
            risk_per_trade = portfolio_cfg.get("target_risk_per_trade", 0.015)
            max_drawdown = portfolio_cfg.get("max_drawdown", 0.20)
            
            # 1. Validate Base Risk
            if not (0 < risk_per_trade < 0.10):
                raise ValueError(f"Invalid target_risk_per_trade: {risk_per_trade}. Must be between 0 and 0.10 (10%)")
            
            # 2. Validate Max Drawdown
            if not (0 < max_drawdown < 1.0):
                raise ValueError(f"Invalid max_drawdown: {max_drawdown}. Must be between 0 and 1.0 (100%)")
            
            # 3. Validate Fixed Risk USD for each asset
            assets_cfg = self.config.get("assets", {})
            for asset, cfg in assets_cfg.items():
                fixed_risk = cfg.get("fixed_risk_usd", {})
                if isinstance(fixed_risk, dict):
                    for r_type, val in fixed_risk.items():
                        if val <= 0:
                            raise ValueError(f"Invalid fixed_risk_usd for {asset} ({r_type}): {val}. Must be > 0.")
            
            logger.info("[CONFIG] ✓ All risk parameters validated and safe.")
            
        except Exception as e:
            logger.error(f"[CONFIG] ❌ FATAL: Invalid configuration detected: {e}")
            raise RuntimeError(f"Startup aborted due to unsafe configuration: {e}")

        setup_logging(self.config)

        # ✨  Initialize AI components as None FIRST
        self.analyst = None
        self.sniper = None
        self.ai_validator = None
        self.ai_monitor = None
        self.ai_tuner = None
        # Visualization system (will be initialized after AI layer)
        self.chart_sender = None

        self.params = SimpleNamespace(
            use_ai_validation=True,
            ai_sr_threshold=0.020,
            ai_pattern_confidence=0.50,
            ai_enable_adaptive=True,
            ai_strong_signal_bypass=0.55,  # Lowered from 0.70 — high-conviction signals were still being blocked
        )
        self.detailed_logging = True

        # Core components
        self.data_manager = DataManager(self.config)
        self.portfolio_manager = None
        self.data_manager_telegram = ThreadSafeBotDataManager(max_cache_age=10)
        self.db_manager = None  # ✨ Initialize BEFORE portfolio
        self.signal_monitor = SignalMonitoringIntegration(max_history=100) # MOVED HERE

        # Handler instances
        self.binance_handler = None
        self.mt5_handler = None

        # Strategy storage
        self.strategies = {}

        # Trading state
        self.is_running = False
        self.trade_count_today = 0
        self.daily_loss = 0.0
        self.last_trade_date = None
        self.last_trade_times = {}
        self.last_market_status_log = {}  # Per-asset logging dictionary

        # Signal aggregators
        self.aggregators = {}
        self.selected_presets = {}

        # Telegram thread management
        self.telegram_bot = None
        self.telegram_thread = None
        self.vtm_thread = None

        # Main bot state
        self._shutdown_requested = False
        self._main_loop_running = False
        self._last_successful_cycle = None
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5

        # ✨ NEW: Portfolio snapshot tracking
        self._last_snapshot_time = None
        self._snapshot_interval = self.config.get("database", {}).get(
            "snapshot_interval_seconds", 300
        )
        _db_cfg = self.config.get("database", {})
        self._log_all_signals = _db_cfg.get("log_all_signals", True)
        self._log_system_events = _db_cfg.get("log_system_events", True)
        self._df_4h_cache = {}

        # T3.5: BTC funding rate Z-score state (fetched every 8 hours)
        self.current_funding_rate = 0.0
        self.funding_rate_zscore = 0.0
        self._last_funding_fetch = None

        # T3.1: Shadow trading engine — tracks blocked signals' outcomes for ML
        self.shadow_trader: Optional[ShadowTradingEngine] = None

        # F.4: CVD WebSocket consumer for BTC order flow
        self.cvd_consumer: Optional[CVDConsumer] = None
        self.cvd_thread: Optional[threading.Thread] = None

        # Initialize components in CORRECT order
        self._initialize_telegram()
        self._initialize_strategies()
        self.mtf_integration = None
        self._current_regime_data = {}

        self.dynamic_selector = None
        self.hybrid_selector = None

        # ✨ NEW: System Health Tracking
        self.health_monitor = HealthMonitor()

        self.error_handler = GlobalErrorHandler(
            telegram_bot=self.telegram_bot,
            db_manager=self.db_manager,
            health_monitor=self.health_monitor,
            config={
                "error_window_seconds": 300,  # 5 minutes
                "max_duplicate_notifications": 3,
                "database": self.config.get("database", {}),
            },
        )

        logger.info("[ERROR HANDLER] Global error handler initialized")

        self.historical_updater = HistoricalDataUpdater(
            data_manager=self.data_manager, config=self.config
        )
        self._last_history_update = None
        self.autotrainer = None
        self.calendar_updater = None

    def initialize_exchanges(self):
        """
        ✅ FIXED: Initialize exchanges and link handlers to portfolio
        """
        logger.info("\n" + "-" * 70)
        logger.info("STEP 1: Initializing Exchange Connections")
        logger.info("-" * 70)

        mt5_initialized = False
        binance_initialized = False

        # Get assets by exchange
        assets_by_exchange = {"binance": [], "mt5": []}
        for asset, cfg in self.config["assets"].items():
            if cfg.get("enabled", False):
                exchange = cfg.get("exchange", "binance").lower()
                if exchange in assets_by_exchange:
                    assets_by_exchange[exchange].append(asset)

        # ============================================================
        # Connect to MT5
        # ============================================================
        if assets_by_exchange["mt5"]:
            try:
                if self.data_manager.initialize_mt5():
                    logger.info(f"[OK] MT5 connection established for: {', '.join(assets_by_exchange['mt5'])}")
                    mt5_initialized = True
                else:
                    logger.error("[FAIL] Failed to initialize MT5")
                    # Disable all MT5 assets if connection fails
                    for a in assets_by_exchange["mt5"]:
                        self.config["assets"][a]["enabled"] = False
            except Exception as e:
                logger.error(f"[FAIL] MT5 initialization error: {e}")
                for a in assets_by_exchange["mt5"]:
                    self.config["assets"][a]["enabled"] = False

        # ============================================================
        # Connect to Binance
        # ============================================================
        if assets_by_exchange["binance"]:
            try:
                if self.data_manager.initialize_binance():
                    logger.info(f"[OK] Binance connection established for: {', '.join(assets_by_exchange['binance'])}")
                    binance_initialized = True
                else:
                    logger.error("[FAIL] Failed to initialize Binance")
                    for a in assets_by_exchange["binance"]:
                        self.config["assets"][a]["enabled"] = False
            except Exception as e:
                logger.error(f"[FAIL] Binance initialization error: {e}")
                for a in assets_by_exchange["binance"]:
                    self.config["assets"][a]["enabled"] = False

        # ============================================================
        # STEP 1.5: Initialize Database BEFORE Portfolio
        # ============================================================
        logger.info("\n" + "-" * 70)
        logger.info("STEP 1.5: Initializing Database Connection")
        logger.info("-" * 70)

        if self.config.get("database", {}).get("enabled", False):
            try:
                db_config = self.config["database"]
                self.db_manager = TradingDatabaseManager(
                    supabase_url=db_config["supabase_url"],
                    supabase_key=db_config["supabase_key"],
                )
                logger.info("[DB] ✓ Connected to Supabase")
            except Exception as e:
                logger.error(f"[DB] Failed to initialize: {e}")
                logger.warning("[DB] Continuing without database logging")
                self.db_manager = None
        else:
            logger.info("[DB] Database disabled in config")
            self.db_manager = None

        # ============================================================
        # STEP 2: Initialize Portfolio Manager
        # ============================================================
        logger.info("\n" + "-" * 70)
        logger.info("STEP 2: Initializing Portfolio Manager")
        logger.info("-" * 70)

        try:
            import MetaTrader5 as mt5

            mt5_handler = mt5 if mt5_initialized else None
        except ImportError:
            mt5_handler = None
            logger.warning("[WARN] MetaTrader5 not available")

        try:
            self.portfolio_manager = PortfolioManager(
                config=self.config,
                mt5_handler=mt5_initialized,
                binance_client=self.data_manager.futures_client,  # ✅ FIXED: Use Futures Client
                db_manager=self.db_manager,
                telegram_bot=self.telegram_bot,
            )

            # ✨ NEW: Enable hedging support
            hedging_enabled = self.config.get("trading", {}).get(
                "allow_simultaneous_long_short", True
            )
            if hedging_enabled:
                max_hedge_ratio = self.config.get("portfolio", {}).get(
                    "max_hedge_ratio", 1.0
                )
                enable_hedging_for_portfolio(self.portfolio_manager, max_hedge_ratio)
                logger.info(
                    f"[HEDGING] ✅ Enabled with max ratio {max_hedge_ratio:.0%}"
                )

            logger.info(
                f"[OK] Portfolio Manager initialized (Mode: {self.portfolio_manager.mode.upper()})"
            )
            logger.info(f"     Capital: ${self.portfolio_manager.current_capital:,.2f}")

            # Log system startup to database
            if self.db_manager:
                self.db_manager.log_system_event(
                    event_type="startup",
                    severity="info",
                    message="Trading bot started",
                    component="main",
                    metadata={
                        "mode": self.portfolio_manager.mode,
                        "initial_capital": self.portfolio_manager.initial_capital,
                        "assets_enabled": [
                            asset
                            for asset, cfg in self.config["assets"].items()
                            if cfg.get("enabled", False)
                        ],
                    },
                )

        except Exception as e:
            logger.error(f"[FAIL] Portfolio Manager initialization failed: {e}")
            raise

        # ============================================================
        # STEP 2.5: Load portfolio state BEFORE initializing handlers
        # ============================================================
        try:
            if not self.portfolio_manager.is_paper_mode:
                logger.info("\n" + "-" * 70)
                logger.info("STEP 2.5: Loading Saved Portfolio State")
                logger.info("-" * 70)
                self.portfolio_manager.load_portfolio_state(self.data_manager)
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
            
        logger.info("-" * 70)

        # ============================================================
        # STEP 3: Initialize Execution Handlers (with auto-sync)
        # ============================================================
        logger.info("\n" + "-" * 70)
        logger.info("STEP 3: Initializing Execution Handlers (with auto-sync)")
        logger.info("-" * 70)

        # ✅ BINANCE HANDLER
        if assets_by_exchange["binance"] and binance_initialized:
            try:
                # Let the handler run its internal auto-sync on startup
                self.binance_handler = BinanceExecutionHandler(
                    config=self.config,
                    client=self.data_manager.get_futures_client(),
                    portfolio_manager=self.portfolio_manager,
                    data_manager=self.data_manager,
                )

                self.binance_handler.trading_bot = self
                if self.binance_handler:
                    self.binance_handler.error_handler = self.error_handler
                    self.binance_handler.trading_bot = self

                # Link database to handler
                if self.db_manager:
                    self.binance_handler.db_manager = self.db_manager
                    logger.info("[DB] ✓ Database linked to Binance handler")

                logger.info("[OK] Binance Execution Handler initialized")

            except Exception as e:
                logger.error(f"[FAIL] Binance handler: {e}")
                self.binance_handler = None
                for a in assets_by_exchange["binance"]:
                    self.config["assets"][a]["enabled"] = False

        # ✅ MT5 HANDLER
        if assets_by_exchange["mt5"] and mt5_initialized:
            try:
                # MT5 handler also runs its sync on init
                self.mt5_handler = MT5ExecutionHandler(
                    config=self.config,
                    portfolio_manager=self.portfolio_manager,
                    data_manager=self.data_manager,
                )

                self.mt5_handler.trading_bot = self

                if self.mt5_handler:
                    self.mt5_handler.error_handler = self.error_handler
                    self.mt5_handler.trading_bot = self

                # Link database to handler
                if self.db_manager:
                    self.mt5_handler.db_manager = self.db_manager
                    logger.info("[DB] ✓ Database linked to MT5 handler")

                logger.info("[OK] MT5 Execution Handler initialized")

            except Exception as e:
                logger.error(f"[FAIL] MT5 handler: {e}")
                self.mt5_handler = None
                for a in assets_by_exchange["mt5"]:
                    self.config["assets"][a]["enabled"] = False

        if not self.binance_handler and not self.mt5_handler:
            raise RuntimeError("No execution handlers available!")

        # ============================================================
        # ✅ STEP 4: LINK HANDLERS TO PORTFOLIO MANAGER
        # ============================================================
        logger.info("\n" + "-" * 70)
        logger.info("STEP 4: Linking Execution Handlers to Portfolio")
        logger.info("-" * 70)

        # Create execution_handlers dict
        execution_handlers = {}

        if self.binance_handler:
            execution_handlers["binance"] = self.binance_handler
            logger.info("[LINK] ✓ Binance handler linked")

        if self.mt5_handler:
            execution_handlers["mt5"] = self.mt5_handler
            logger.info("[LINK] ✓ MT5 handler linked")

        # ✅ NEW: Pass handlers to Portfolio Manager
        self.portfolio_manager.execution_handlers = execution_handlers

        logger.info("[OK] Portfolio can now close positions on exchanges")

        # ============================================================
        # STEP 4.5: Reconcile DB open positions against live brokers
        # ============================================================
        # Positions closed manually while the bot was offline remain
        # stuck as status="open" in Supabase. Fix them now before the
        # bot loop starts so the dashboard only shows truly live trades.
        if self.db_manager:
            try:
                logger.info("\n" + "-" * 70)
                logger.info("STEP 4.5: Reconciling open DB positions against brokers")
                logger.info("-" * 70)
                corrected = self.db_manager.reconcile_open_positions(
                    mt5_handler=self.mt5_handler,
                    binance_handler=self.binance_handler,
                )
                if corrected:
                    logger.info(
                        f"[RECONCILE] ✅ {corrected} offline-closed position(s) "
                        f"cleaned up from database"
                    )
            except Exception as _re:
                logger.warning(f"[RECONCILE] Startup reconcile failed (non-fatal): {_re}")
        # STEP 4.6: Restore cooldown clock from DB
        # ============================================================
        # last_trade_times is in-memory only; a restart resets it to {},
        # which makes check_min_time_between_trades() always pass on the
        # first cycle even if a trade was opened seconds before the crash.
        # Fix: seed it from the most-recent entry_time per asset in the DB.
        if self.db_manager:
            try:
                logger.info("\n" + "-" * 70)
                logger.info("STEP 4.6: Restoring cooldown clock from trade history")
                logger.info("-" * 70)
                # Look back far enough to cover the longest possible cooldown
                max_cooldown_h = max(
                    self.config.get("trading", {}).get(
                        "min_time_between_trades_minutes", 480
                    ),
                    120,          # hard floor: never look back less than 2 hours
                ) / 60.0 * 1.2   # 20 % buffer
                restored = self.db_manager.get_last_trade_times(
                    lookback_hours=int(max_cooldown_h) + 1
                )
                if restored:
                    self.last_trade_times.update(restored)
                    for asset, ts in restored.items():
                        elapsed_min = (datetime.now() - ts).total_seconds() / 60
                        logger.info(
                            f"[COOLDOWN] {asset}: last trade {elapsed_min:.0f} min ago "
                            f"(restored from DB)"
                        )
                else:
                    logger.info("[COOLDOWN] No recent trades found — cooldown clock starts fresh")
            except Exception as _ce:
                logger.warning(f"[COOLDOWN] Cooldown restore failed (non-fatal): {_ce}")

    def _initialize_telegram(self):
        """Initialize Telegram bot"""
        if hasattr(self, 'telegram_bot') and self.telegram_bot:
            logger.warning("[TELEGRAM] Bot already initialized, skipping.")
            return

        try:
            tg_config = self.config.get("telegram", {})
            if not tg_config.get("enabled", False):
                logger.info("[TELEGRAM] Disabled in config")
                return

            token = tg_config.get("bot_token")
            admin_ids = tg_config.get("admin_ids", [])

            if not token or not admin_ids or token == "your_token_here":
                logger.warning("[TELEGRAM] Missing or placeholder config")
                return

            self.telegram_bot = TradingTelegramBot(
                token=token, 
                admin_ids=admin_ids, 
                trading_bot=self,
                signal_monitor=self.signal_monitor
            )
            
            logger.info(f"[TELEGRAM] Initialized for {len(admin_ids)} admin(s) (Loop will be started by main thread)")

        except Exception as e:
            logger.warning(f"[TELEGRAM] Initialization failed: {e}")
            self.telegram_bot = None

    def _initialize_strategies(self):
        """Initialize all strategies with extreme safety"""
        logger.info("\n" + "-" * 70)
        logger.info("Initializing Strategies (MR + TF + EMA)")
        logger.info("-" * 70)

        if not hasattr(self, 'strategies') or self.strategies is None:
            self.strategies = {}

        strategy_cfgs = self.config.get("strategy_configs", {})

        for asset_name, asset_config in self.config["assets"].items():
            if not asset_config.get("enabled", False):
                logger.debug(f"[SKIP] {asset_name}: Disabled")
                continue

            # Ensure asset dictionary exists
            self.strategies.setdefault(asset_name, {})
            
            strategies_cfg = asset_config.get("strategies", {})

            # 1. Mean Reversion
            if strategies_cfg.get("mean_reversion", {}).get("enabled", False):
                try:
                    cfg = strategy_cfgs.get("mean_reversion", {}).get(asset_name, {})
                    cfg["asset"] = asset_name  # Fix #11: ensure MR knows which asset it trades
                    self.strategies[asset_name]["mean_reversion"] = MeanReversionStrategy(cfg)
                    logger.info(f"[OK] {asset_name}: Mean Reversion")
                except Exception as e:
                    logger.error(f"[FAIL] {asset_name} Mean Reversion: {e}")

            # 2. Trend Following
            if strategies_cfg.get("trend_following", {}).get("enabled", False):
                try:
                    cfg = strategy_cfgs.get("trend_following", {}).get(asset_name, {})
                    self.strategies[asset_name]["trend_following"] = TrendFollowingStrategy(cfg)
                    logger.info(f"[OK] {asset_name}: Trend Following")
                except Exception as e:
                    logger.error(f"[FAIL] {asset_name} Trend Following: {e}")

            # 3. EMA Strategy
            if strategies_cfg.get("exponential_moving_averages", {}).get("enabled", False):
                try:
                    cfg = strategy_cfgs.get("exponential_moving_averages", {}).get(asset_name, {})
                    self.strategies[asset_name]["ema_strategy"] = EMAStrategy(cfg)
                    logger.info(f"[OK] {asset_name}: EMA Strategy")
                except Exception as e:
                    logger.error(f"[FAIL] {asset_name} EMA Strategy: {e}")

            # Safe length check
            enabled_strats = self.strategies.get(asset_name, {})
            enabled_count = len(enabled_strats)
            
            if enabled_count == 0:
                logger.warning(f"[!] {asset_name}: NO strategies enabled")
            else:
                strat_names = ", ".join(enabled_strats.keys())
                logger.info(f"[OK] {asset_name}: {enabled_count}/3 strategies -> {strat_names}")

    def initialize_ai_layer(self):
        """
        ✨  Safe AI initialization with proper error handling
        """
        try:
            logger.info("=" * 70)
            logger.info("Initializing AI Layer...")
            logger.info("=" * 70)

            models_dir = Path("models/ai")

            # Check if model files exist
            model_path = models_dir / "sniper_dual_timeframe_v1.weights.h5"
            mapping_path = models_dir / "sniper_dual_timeframe_v1_mapping.pkl"
            config_path = models_dir / "sniper_dual_timeframe_v1_config.pkl"

            if not model_path.exists():
                logger.error(f"[AI] Model not found: {model_path}")
                logger.error("[AI] Please run: python train_dual_timeframe.py")
                logger.warning("[AI] AI layer will be DISABLED")
                return False

            # Load pattern mapping
            try:
                with open(mapping_path, "rb") as f:
                    pattern_map = pickle.load(f)

                logger.info(f"[AI] Loaded {len(pattern_map)} patterns")

                # Ensure noise class exists
                if "Noise" not in pattern_map:
                    logger.warning("[AI] Adding missing 'Noise' class")
                    pattern_map["Noise"] = 0
                    with open(mapping_path, "wb") as f:
                        pickle.dump(pattern_map, f)

            except Exception as e:
                logger.error(f"[AI] Pattern mapping error: {e}")
                return False

            # Load config
            try:
                with open(config_path, "rb") as f:
                    config = pickle.load(f)

                logger.info(f"[AI] Model: {config.get('model_version', 'unknown')}")
                logger.info(f"[AI] Accuracy: {config.get('val_accuracy', 0):.2%}")

            except Exception as e:
                logger.warning(f"[AI] Config warning: {e}")
                config = {"num_classes": len(pattern_map)}

            # Initialize Analyst (4H)
            try:
                self.analyst = DynamicAnalyst(atr_multiplier=1.5, min_samples=5)
                logger.info("[AI] ✓ Analyst (4H S/R)")

            except Exception as e:
                logger.error(f"[AI] Analyst failed: {e}")
                return False

            # Initialize Sniper (15min)
            try:
                num_classes = config.get("num_classes", len(pattern_map))

                self.sniper = OHLCSniper(
                    input_shape=(15, 4), num_classes=num_classes, dropout_rate=0.3
                )

                logger.info(f"[AI] Sniper created ({num_classes} classes)")

                # Load weights
                self.sniper.load_model(str(model_path))
                logger.info("[AI] ✓ Weights loaded")

            except ValueError as e:
                if "shape" in str(e).lower():
                    logger.error(f"[AI] ✗ ARCHITECTURE MISMATCH!")
                    logger.error(f"     {e}")
                    logger.error("[AI] Solution: Retrain model")
                    logger.error("     python train_dual_timeframe.py")
                    return False
                raise

            except Exception as e:
                logger.error(f"[AI] Sniper failed: {e}")
                return False

            # Initialize Validator
            try:
                self.ai_validator = HybridSignalValidator(
                    analyst=self.analyst,
                    sniper=self.sniper,
                    pattern_id_map=pattern_map,
                    sr_threshold_pct=self.params.ai_sr_threshold,
                    pattern_confidence_min=self.params.ai_pattern_confidence,
                    enable_adaptive_thresholds=self.params.ai_enable_adaptive,
                    strong_signal_bypass_threshold=self.params.ai_strong_signal_bypass,
                    use_ai_validation=self.params.use_ai_validation,
                )

                logger.info("[AI] ✓ Validator initialized")

            except Exception as e:
                logger.error(f"[AI] Validator failed: {e}")
                return False

            # Initialize monitoring (optional)
            try:
                self.ai_monitor = AIValidatorMonitor(self.ai_validator)
                self.ai_tuner = AIValidatorTuner(self.ai_validator)

                schedule.every(1).hours.do(self.ai_monitor.log_periodic_report)

                logger.info("[AI] ✓ Monitoring enabled")

            except Exception as e:
                logger.warning(f"[AI] Monitoring warning: {e}")

            logger.info("=" * 70)
            logger.info("✅ AI Layer READY")
            logger.info("  Analyst:   4H S/R detection")
            logger.info("  Sniper:    15min patterns")
            logger.info(
                f"  Status:    {'ENABLED' if self.params.use_ai_validation else 'DISABLED'}"
            )
            logger.info("=" * 70)

            return True

        except Exception as e:
            logger.error(f"[AI] Initialization failed: {e}", exc_info=True)
            logger.error("[AI] AI layer DISABLED")

            # Reset all AI components
            self.analyst = None
            self.sniper = None
            self.ai_validator = None
            self.ai_monitor = None
            self.ai_tuner = None

            return False

    def run_mtf_regime_analysis(self):
        """
        Run multi-timeframe regime analysis for all enabled assets.
        Primary use is for initial startup and periodic background logging.
        Actual trading signals now fetch fresh regime data on-demand (5min cache).
        """
        try:
            if not self.mtf_integration:
                logger.debug("[MTF] Not initialized, skipping analysis")
                return

            logger.info("\n" + "=" * 70)
            logger.info(f"[MTF] Running Multi-Timeframe Regime Analysis")
            logger.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)

            enabled_assets = [
                name
                for name, cfg in self.config["assets"].items()
                if cfg.get("enabled", False)
            ]

            for asset_name in enabled_assets:
                try:
                    asset_cfg = self.config["assets"][asset_name]
                    symbol = asset_cfg.get("symbol")
                    exchange = asset_cfg.get("exchange", "binance")

                    # Force refresh to get latest data
                    regime_data = self.mtf_integration.get_regime_for_trading(
                        asset_name=asset_name, symbol=symbol, exchange=exchange
                    )

                    # Store in cache for aggregators and trading logic
                    self._current_regime_data[asset_name] = regime_data

                    # Log summary
                    logger.info(f"\n[MTF] {asset_name} Analysis:")
                    logger.info(f"  Regime:           {regime_data['regime'].upper()}")
                    logger.info(
                        f"  Direction:        {'BULL' if regime_data['is_bullish'] else 'BEAR'}"
                    )
                    logger.info(f"  Confidence:       {regime_data['confidence']:.2%}")
                    logger.info(
                        f"  TF Agreement:     {regime_data['timeframe_agreement']:.2%}"
                    )
                    logger.info(
                        f"  Recommended Mode: {regime_data['recommended_mode'].upper()}"
                    )
                    logger.info(
                        f"  Risk Level:       {regime_data['risk_level'].upper()}"
                    )
                    logger.info(
                        f"  Volatility:       {regime_data['volatility'].upper()}"
                    )
                    logger.info(
                        f"  Counter-Trend:    {'✓ Allowed' if regime_data['allow_counter_trend'] else '✗ Blocked'}"
                    )
                    logger.info(f"  Max Positions:    {regime_data['max_positions']}")

                except Exception as e:
                    logger.error(f"[MTF] Error analyzing {asset_name}: {e}")

            logger.info("=" * 70 + "\n")

        except Exception as e:
            logger.error(f"[MTF] Analysis error: {e}", exc_info=True)

    def initialize_mtf_regime_detection(self):
        """
        Initialize multi-timeframe regime detection
        Should be called AFTER AI and DB initialization
        """
        try:
            logger.info("\n" + "=" * 70)
            logger.info("[MTF] Initializing Multi-Timeframe Regime Detection")
            logger.info("=" * 70)

            self.mtf_integration = MTFRegimeIntegration(
                data_manager=self.data_manager,
                db_manager=self.db_manager,
                ai_validator=self.ai_validator,
                telegram_bot=self.telegram_bot,
            )

            logger.info("[MTF] ✅ Multi-Timeframe Regime Detection Ready")
            logger.info("=" * 70 + "\n")

            return True

        except Exception as e:
            logger.error(f"[MTF] Initialization failed: {e}", exc_info=True)
            self.mtf_integration = None
            return False

    def _initialize_aggregators(self):
        """
        Initialize signal aggregators with clear mode selection

        Modes:
        - 'performance' (default): Your existing advanced aggregator
        - 'council': New institutional-style weighted council
        - 'hybrid': Both aggregators for comparison
        """

        logger.info("\n" + "=" * 70)
        logger.info("INITIALIZING SIGNAL AGGREGATORS")
        logger.info("=" * 70)

        # ================================================================
        # STEP 1: Load Configuration
        # ================================================================
        aggregator_cfg = self.config.get("aggregator_settings", {})
        mode = aggregator_cfg.get("mode", "performance").lower()
        preset = aggregator_cfg.get("preset", "auto")

        logger.info(f"\nMode:   {mode.upper()}")
        logger.info(f"Preset: {preset.upper()}")

        # Validate mode
        valid_modes = ["performance", "council", "hybrid"]
        if mode not in valid_modes:
            logger.warning(f"Invalid mode '{mode}', defaulting to 'performance'")
            mode = "performance"

        # ================================================================
        # STEP 2: Define Preset Configurations
        # ================================================================
        # These presets work for BOTH aggregator types
        AGGREGATOR_PRESETS = {}
        try:
            CONFIG_PATH = Path("config/aggregator_presets.json")
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, "r") as f:
                    data = json.load(f)
                    AGGREGATOR_PRESETS = data.get("AGGREGATOR_PRESETS", {})
                logger.info("[INIT] Loaded aggregator presets from file")
            else:
                logger.warning("[INIT] aggregator_presets.json not found")
        except Exception as e:
            logger.error(f"[INIT] Error loading presets: {e}")

        # ================================================================
        # STEP 3: Auto-Preset Selection (if enabled)
        # ================================================================
        if preset == "auto":
            logger.info("\n🎯 AUTO-PRESET MODE ACTIVE")
            logger.info("Analyzing market conditions for each asset...")

            asset_presets = {}

            # Get preset for each enabled asset
            for asset_name in self.strategies.keys():
                if self.config["assets"][asset_name].get("enabled", False):
                    # Use the preset already calculated during init
                    selected_preset = self.dynamic_selector.current_presets.get(
                        asset_name, "balanced"
                    )
                    asset_presets[asset_name] = selected_preset

            logger.info("\n📊 CURRENT PRESETS:")
            for asset, selected_preset in asset_presets.items():
                logger.info(f"  {asset:6} → {selected_preset.upper()}")

        elif preset in ["conservative", "balanced", "aggressive", "scalper"]:
            logger.info(f"\nUsing manual preset: {preset.upper()}")
            asset_presets = {name: preset for name in self.strategies.keys()}

        else:
            logger.warning(f"Unknown preset '{preset}', using 'balanced'")
            asset_presets = {name: "balanced" for name in self.strategies.keys()}

        # Store selected presets
        self.selected_presets = asset_presets.copy()

        # ================================================================
        # STEP 4: Get AI Validator (if available)
        # ================================================================
        ai_validator = None
        if hasattr(self, "ai_validator") and self.ai_validator is not None:
            ai_validator = self.ai_validator
            logger.info("\n✅ AI Validator available")
        else:
            logger.info("\n⚠️  AI Validator not available")

        # ================================================================
        # STEP 5: Initialize Aggregators for Each Asset
        # ================================================================
        logger.info("\n" + "-" * 70)
        logger.info("CREATING AGGREGATORS")
        logger.info("-" * 70)

        # Ensure aggregators dict exists
        if not hasattr(self, "aggregators") or self.aggregators is None:
            self.aggregators = {}

        # Extract global filter flags
        agg_settings = self.config.get("aggregator_settings", {})
        use_macro_gov = agg_settings.get("use_macro_governor", True)
        use_gatekeeper = agg_settings.get("use_gatekeeper", True)
        trend_threshold = agg_settings.get("trend_aligned_threshold", 3.0)
        counter_threshold = agg_settings.get("counter_trend_threshold", 3.5)

        for asset_name, strategies in self.strategies.items():
            # Skip disabled assets
            if not self.config["assets"][asset_name].get("enabled", False):
                logger.info(f"\n[SKIP] {asset_name}: Asset disabled")
                continue

            # Check if we have strategies
            strategy_count = len(strategies)
            if strategy_count == 0:
                logger.warning(f"\n[SKIP] {asset_name}: No strategies available")
                continue

            # Get preset config for this asset
            selected_preset = asset_presets.get(asset_name, "balanced")

            # Map auto-preset strategy names → valid AGGREGATOR_PRESETS keys.
            # auto_preset_selector may return strategy-style names like "mean_reversion",
            # "trend_following", "scalping" — these must be mapped to the four valid
            # preset buckets so BTC always gets a working aggregator config.
            PRESET_NAME_MAP = {
                # Short-form aliases returned by auto_preset_selector
                "mr": "conservative",           # sideways/chop mean-reversion mode
                "mean_reversion": "conservative",
                "mean_reversion_forced": "conservative",
                "tf": "balanced",               # trend-following shorthand
                "trend_following": "balanced",
                "trend": "balanced",
                "momentum": "aggressive",
                "scalping": "scalper",
                "scalper": "scalper",
                # Pass-through for the four valid preset names
                "conservative": "conservative",
                "balanced": "balanced",
                "aggressive": "aggressive",
            }
            selected_preset = PRESET_NAME_MAP.get(selected_preset, selected_preset)

            # Handle asset key mapping (BTCUSDT -> BTC, everything else defaults to GOLD presets for now)
            config_key = "BTC" if "BTC" in asset_name.upper() else "GOLD"
            preset_config = AGGREGATOR_PRESETS.get(config_key, {}).get(selected_preset)

            if preset_config is None:
                logger.error(
                    f"\n[ERROR] {asset_name}: No config for preset '{selected_preset}'"
                )
                continue

            logger.info(f"\n{asset_name}:")
            logger.info(f"  Strategies: {strategy_count}")
            logger.info(f"  Preset:     {selected_preset}")

            # ============================================================
            # MODE SELECTION
            # ============================================================

            if mode == "performance":
                # --------------------------------------------------------
                # PERFORMANCE MODE
                # --------------------------------------------------------
                try:
                    self.aggregators[asset_name] = PerformanceWeightedAggregator(
                        mean_reversion_strategy=strategies.get("mean_reversion"),
                        trend_following_strategy=strategies.get("trend_following"),
                        ema_strategy=strategies.get("ema_strategy"),
                        asset_type=asset_name,
                        config=preset_config,
                        ai_validator=(
                            ai_validator if self.params.use_ai_validation else None
                        ),
                        mtf_integration=self.mtf_integration,  # Pass MTF for Governor
                        enable_world_class_filters=True,  # Enable filters
                        enable_ai_circuit_breaker=True,
                        enable_detailed_logging=getattr(
                            self, "detailed_logging", False
                        ),
                        strong_signal_bypass_threshold=getattr(
                            self.params, "ai_strong_signal_bypass", 0.70
                        ),
                        use_macro_governor=use_macro_gov,
                        use_gatekeeper=use_gatekeeper
                    )

                    logger.info(f"  Type:       Performance Aggregator")
                    logger.info(
                        f"  AI:         {'Enabled' if ai_validator else 'Disabled'}"
                    )

                except Exception as e:
                    logger.error(
                        f"  [ERROR] Failed to create Performance aggregator: {e}"
                    )
                    continue

            elif mode == "council":
                # --------------------------------------------------------
                # COUNCIL MODE (New institutional aggregator)
                # --------------------------------------------------------
                try:
                    self.aggregators[asset_name] = InstitutionalCouncilAggregator(
                        mean_reversion_strategy=strategies.get("mean_reversion"),
                        trend_following_strategy=strategies.get("trend_following"),
                        ema_strategy=strategies.get("ema_strategy"),
                        asset_type=asset_name,
                        ai_validator=(
                            ai_validator if self.params.use_ai_validation else None
                        ),
                        enable_detailed_logging=getattr(
                            self, "detailed_logging", False
                        ),
                        # Council-specific settings
                        config=preset_config,  # ✅ CORRECT: Pass config for dynamic thresholds
                        trend_aligned_threshold=trend_threshold,
                        counter_trend_threshold=counter_threshold,
                        weight_structure=1.0,
                        weight_momentum=1.5,
                        performance_tracker=self.portfolio_manager.performance_tracker,
                        use_macro_governor=use_macro_gov,
                        use_gatekeeper=use_gatekeeper
                    )

                    logger.info(f"  Type:       Council Aggregator")
                    logger.info(
                        f"  AI:         {'Enabled' if ai_validator else 'Disabled'}"
                    )
                    # Log the actual active thresholds
                    thresh_trend = preset_config.get("council_trend_aligned", 3.5)
                    thresh_count = preset_config.get("council_counter_trend", 4.0)
                    logger.info(
                        f"  Thresholds: {thresh_trend} (trend) / {thresh_count} (counter)"
                    )

                except Exception as e:
                    logger.error(f"  [ERROR] Failed to create Council aggregator: {e}")
                    continue

            elif mode == "hybrid":
                # --------------------------------------------------------
                # HYBRID MODE (Both aggregators for comparison)
                # --------------------------------------------------------
                try:
                    # Create Performance Aggregator
                    perf_agg = PerformanceWeightedAggregator(
                        mean_reversion_strategy=strategies.get("mean_reversion"),
                        trend_following_strategy=strategies.get("trend_following"),
                        ema_strategy=strategies.get("ema_strategy"),
                        asset_type=asset_name,
                        config=preset_config,
                        ai_validator=(
                            ai_validator if self.params.use_ai_validation else None
                        ),
                        mtf_integration=self.mtf_integration,  # Pass MTF for Governor
                        enable_world_class_filters=True,  # Enable filters
                        enable_ai_circuit_breaker=True,
                        enable_detailed_logging=False,  # Reduce noise in hybrid mode
                        strong_signal_bypass_threshold=getattr(
                            self.params, "ai_strong_signal_bypass", 0.70
                        ),
                    )

                    # Create Council Aggregator
                    council_agg = InstitutionalCouncilAggregator(
                        mean_reversion_strategy=strategies.get("mean_reversion"),
                        trend_following_strategy=strategies.get("trend_following"),
                        ema_strategy=strategies.get("ema_strategy"),
                        asset_type=asset_name,
                        ai_validator=(
                            ai_validator if self.params.use_ai_validation else None
                        ),
                        enable_detailed_logging=False,
                        config=preset_config,  # ✅ CORRECT: Pass config for dynamic thresholds
                        trend_aligned_threshold=trend_threshold,
                        counter_trend_threshold=counter_threshold,
                        weight_structure=1.0,
                        weight_momentum=1.5,
                        use_macro_governor=use_macro_gov,
                        use_gatekeeper=use_gatekeeper
                    )

                    # Store both in a dict
                    self.aggregators[asset_name] = {
                        "performance": perf_agg,
                        "council": council_agg,
                        "mode": "hybrid",
                    }

                    logger.info(f"  Type:       Hybrid (Both aggregators)")
                    logger.info(
                        f"  AI:         {'Enabled' if ai_validator else 'Disabled'}"
                    )
                    logger.info(f"  Note:       Signals require consensus")

                except Exception as e:
                    logger.error(f"  [ERROR] Failed to create Hybrid aggregators: {e}")
                    continue

        # ================================================================
        # STEP 6: Summary
        # ================================================================
        logger.info("\n" + "=" * 70)
        logger.info("AGGREGATOR INITIALIZATION COMPLETE")
        logger.info("=" * 70)

        successful = len([a for a in self.aggregators.values() if a is not None])
        total = len([a for a in self.config["assets"].values() if a.get("enabled")])

        logger.info(f"\nStatus: {successful}/{total} aggregators ready")
        logger.info(f"Mode:   {mode.upper()}")

        if mode == "hybrid":
            logger.info(
                "\n⚠️  HYBRID MODE: Signals require consensus from both aggregators"
            )
            logger.info(f"  Type:       Hybrid (Dynamic Mode Selection)")
            logger.info(f"  AI:         {'Enabled' if ai_validator else 'Disabled'}")
            logger.info(f"  Note:       Intelligent mode switching enabled")

        if preset == "auto":
            logger.info(f"\n🎯 AUTO PRESET: Active (Dynamic adjustment enabled)")
            logger.info(
                f"  Cooldown:     {self.dynamic_selector.min_switch_interval} minutes"
            )
            logger.info(f"  Presets Used: {set(asset_presets.values())}")

    def _format_ai_validation_direct(self, asset_name: str, signal: int, df: pd.DataFrame) -> dict:
        """
        Direct AI validation formatting (fallback when aggregator doesn't have the method)
        MATCHES the full implementation from _format_ai_validation_for_viz

        Args:
            asset_name: The name of the asset being processed (e.g., 'BTC', 'GOLD').
            signal: Trading signal
            df: Market dataframe

        Returns:
            Formatted AI validation dict with top3 patterns
        """
        try:
            if not self.ai_validator:
                return {
                    "pattern_detected": False,
                    "validation_passed": False,
                    "pattern_name": "N/A",
                    "pattern_confidence": 0.0,
                    "action": "ai_disabled",
                    "error": "AI validator not initialized",
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
                }

            current_price = float(df["close"].iloc[-1])

            # Get S/R analysis
            sr_result = self.ai_validator._check_support_resistance_fixed(
                asset=asset_name,
                df=df,
                current_price=current_price,
                signal=signal,
                threshold=self.ai_validator.current_sr_threshold,
            )

            # Get pattern detection
            pattern_result = self.ai_validator._check_pattern(
                df=df,
                signal=signal,
                min_confidence=self.ai_validator.current_pattern_threshold,
            )

            # ✅ FIX: Get top 3 patterns (was missing!)
            top3_patterns = []
            top3_confidences = []

            if hasattr(self.ai_validator, "sniper") and self.ai_validator.sniper:
                try:
                    # Get last 15 candles for pattern detection
                    snippet = df[["open", "high", "low", "close"]].iloc[-15:].values
                    first_open = snippet[0, 0]

                    if first_open > 0:
                        snippet_norm = snippet / first_open - 1
                        snippet_input = snippet_norm.reshape(1, 15, 4)

                        # Get predictions
                        predictions = self.ai_validator.sniper.model.predict(
                            snippet_input, verbose=0
                        )[0]

                        # Get top 3
                        top3_indices = predictions.argsort()[-3:][::-1]
                        top3_confidences = predictions[top3_indices].tolist()

                        # Map to pattern names
                        for idx in top3_indices:
                            pattern_name = self.ai_validator.reverse_pattern_map.get(
                                idx, f"Pattern_{idx}"
                            )
                            top3_patterns.append(pattern_name)

                except Exception as e:
                    logger.debug(f"[AI DIRECT] Top3 patterns failed: {e}")

            # Build result
            return {
                "pattern_detected": pattern_result.get("pattern_confirmed", False),
                "validation_passed": signal != 0,  # If signal survived, it passed
                "pattern_name": pattern_result.get("pattern_name", "None"),
                "pattern_id": pattern_result.get("pattern_id"),
                "pattern_confidence": pattern_result.get("confidence", 0.0),
                "top3_patterns": top3_patterns,
                "top3_confidences": top3_confidences,
                "sr_analysis": {
                    "near_sr_level": sr_result.get("near_level", False),
                    "level_type": sr_result.get("level_type", "none"),
                    "nearest_level": sr_result.get("nearest_level"),
                    "distance_pct": sr_result.get("distance_pct"),
                    "levels": sr_result.get("all_levels", [])[:5],
                    "total_levels_found": len(sr_result.get("all_levels", [])),
                },
                "action": "direct_validation",
                "rejection_reasons": [],
                "error": None,
            }

        except Exception as e:
            logger.error(f"[AI DIRECT] Validation failed: {e}", exc_info=True)
            return {
                "pattern_detected": False,
                "validation_passed": False,
                "pattern_name": "ERROR",
                "pattern_confidence": 0.0,
                "action": "error",
                "error": str(e),
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
            }

    def _validate_ai_details_structure(
        self, ai_validation: dict, context: str = ""
    ) -> bool:
        """
        ✅ ENHANCED: Validate AI validation dict with numpy type handling (NumPy 2.0 Safe)
        """
        import numpy as np  # Ensure numpy is imported

        if not ai_validation or not isinstance(ai_validation, dict):
            logger.error(
                f"[AI VIZ {context}] ❌ ai_validation is not a dict: {type(ai_validation)}"
            )
            return False

        required_fields = {
            "pattern_detected": bool,
            "pattern_name": str,
            "pattern_confidence": (int, float),
            "pattern_id": (int, type(None)),
            "top3_patterns": list,
            "top3_confidences": list,
            "sr_analysis": dict,
            "validation_passed": bool,
            "action": str,
        }

        sr_required_fields = {
            "near_sr_level": bool,
            "level_type": str,
            "nearest_level": (int, float, type(None)),
            "distance_pct": (int, float, type(None)),
            "levels": list,
            "total_levels_found": int,
        }

        all_valid = True

        # Check top-level fields
        for field, expected_type in required_fields.items():
            if field not in ai_validation:
                logger.error(f"[AI VIZ {context}] ❌ Missing field: {field}")
                all_valid = False
                continue

            value = ai_validation[field]

            # ✅ FIX: Robust NumPy conversion using duck typing
            # Standard Python types (bool, int, float) DO NOT have .item()
            # NumPy scalars DO have .item()
            if hasattr(value, "item"):
                try:
                    value = value.item()
                    ai_validation[field] = value  # Update in place
                except (ValueError, TypeError):
                    pass  # Ignore if conversion fails

            if not isinstance(value, expected_type):
                # Allow fallback for confidence if it's int/float compatible
                if field == "pattern_confidence" and isinstance(value, (int, float)):
                    pass
                else:
                    logger.error(
                        f"[AI VIZ {context}] ❌ {field} wrong type: "
                        f"expected {expected_type}, got {type(value)}"
                    )
                    all_valid = False

        # Check sr_analysis sub-fields
        if "sr_analysis" in ai_validation:
            sr_analysis = ai_validation["sr_analysis"]

            if not isinstance(sr_analysis, dict):
                logger.error(f"[AI VIZ {context}] ❌ sr_analysis is not a dict")
                all_valid = False
            else:
                for field, expected_type in sr_required_fields.items():
                    if field not in sr_analysis:
                        # Optional fields logic could go here, but for now log missing
                        # logger.error(f"[AI VIZ {context}] ❌ sr_analysis missing: {field}")
                        # all_valid = False
                        pass  # Be lenient on sub-fields to prevent crashes
                    else:
                        value = sr_analysis[field]

                        # ✅ FIX: Handle numpy types in sr_analysis using duck typing
                        if hasattr(value, "item"):
                            try:
                                value = value.item()
                                sr_analysis[field] = value
                            except (ValueError, TypeError):
                                pass

                        if not isinstance(value, expected_type):
                            logger.error(
                                f"[AI VIZ {context}] ❌ sr_analysis.{field} wrong type: "
                                f"expected {expected_type}, got {type(value)}"
                            )
                            all_valid = False

        if all_valid:
            logger.info(f"[AI VIZ {context}] ✅ All fields valid")
        else:
            logger.error(f"[AI VIZ {context}] ❌ Validation FAILED")

        return all_valid

    def _detect_regime(
        self, df: pd.DataFrame, asset_name: str = "BTC"
    ) -> Tuple[bool, float]:
        """
        ✅ ENHANCED: Use MTF regime detection if available, fallback to original logic
        """
        try:
            # PRIORITY 1: Use MTF Regime Detection if available
            if self.mtf_integration and hasattr(self, "_current_regime_data"):
                mtf_regime = self._current_regime_data.get(asset_name)

                if mtf_regime:
                    is_bull = mtf_regime["is_bull"]
                    confidence = mtf_regime["confidence"]

                    logger.debug(f"[REGIME] {asset_name}: Using MTF regime")
                    logger.debug(f"  Direction: {'BULL' if is_bull else 'BEAR'}")
                    logger.debug(f"  Confidence: {confidence:.2%}")

                    # Update previous regime for continuity
                    self.previous_regime = is_bull

                    return is_bull, confidence

            # FALLBACK: Your existing single-timeframe detection
            logger.debug(f"[REGIME] {asset_name}: Using fallback detection")

            # 1. Get strategy for indicators
            from src.strategies.ema_strategy import EMAStrategy
            ema_cfg = self.config.get("strategy_configs", {}).get("exponential_moving_averages", {}).get(asset_name, {})
            ema_strat = EMAStrategy(ema_cfg)
            
            # 2. Generate features
            features_df = ema_strat.generate_features(df)
            if features_df.empty:
                return False, 0.5
                
            latest = features_df.iloc[-1]
            ema_diff = latest.get("ema_diff_pct", 0.0)
            
            # 3. Decision (Hysteresis)
            is_bull = ema_diff > 0
            confidence = min(1.0, 0.5 + abs(ema_diff) / 10.0)
            
            self.previous_regime = is_bull
            return is_bull, confidence

        except Exception as e:
            logger.error(f"[REGIME] Detection failed: {e}", exc_info=True)
            # Emergency fallback
            return False, 0.5

    def _log_ai_validation_summary(self, asset_name: str, details: dict):
        """
        ✅ NEW: Log comprehensive AI validation summary
        Call this before sending charts
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"[AI VIZ SUMMARY] {asset_name}")
        logger.info(f"{'='*70}")

        ai_validation = details.get("ai_validation")

        if not ai_validation:
            logger.error(f"❌ ai_validation is missing from details")
            logger.error(f"Available keys: {list(details.keys())}")
            return

        if not isinstance(ai_validation, dict):
            logger.error(f"❌ ai_validation is not a dict: {type(ai_validation)}")
            return

        # Pattern info
        pattern_name = ai_validation.get("pattern_name", "N/A")
        pattern_conf = ai_validation.get("pattern_confidence", 0)
        pattern_detected = ai_validation.get("pattern_detected", False)

        logger.info(f"Pattern:")
        logger.info(f"  Name:       {pattern_name}")
        logger.info(f"  Confidence: {pattern_conf:.2%}")
        logger.info(f"  Detected:   {pattern_detected}")

        # Top 3 patterns
        top3 = ai_validation.get("top3_patterns", [])
        top3_conf = ai_validation.get("top3_confidences", [])

        logger.info(f"Top 3 Patterns:")
        if top3:
            for i, (name, conf) in enumerate(zip(top3, top3_conf), 1):
                logger.info(f"  {i}. {name}: {conf:.2%}")
        else:
            logger.warning(f"  ⚠️ No top3 patterns available")

        # S/R Analysis
        sr_analysis = ai_validation.get("sr_analysis", {})

        nearest = sr_analysis.get("nearest_level")
        distance = sr_analysis.get("distance_pct")

        logger.info("S/R Analysis:")
        logger.info(f"  Near Level: {sr_analysis.get('near_sr_level', False)}")
        logger.info(f"  Type:       {sr_analysis.get('level_type', 'N/A')}")
        logger.info(
            f"  Nearest:    ${nearest:,.2f}"
            if isinstance(nearest, (int, float))
            else "  Nearest:    N/A"
        )
        logger.info(
            f"  Distance:   {distance:.2f}%"
            if isinstance(distance, (int, float))
            else "  Distance:   N/A"
        )
        logger.info(f"  Total Levels: {sr_analysis.get('total_levels_found', 0)}")

        # Validation status
        validation_passed = ai_validation.get("validation_passed", False)
        action = ai_validation.get("action", "unknown")

        logger.info(f"Validation:")
        logger.info(f"  Passed: {validation_passed}")
        logger.info(f"  Action: {action}")

        # Rejection reasons (if any)
        rejection_reasons = ai_validation.get("rejection_reasons", [])
        if rejection_reasons:
            logger.info(f"Rejection Reasons:")
            for reason in rejection_reasons:
                logger.info(f"  - {reason}")

        # Error (if any)
        error = ai_validation.get("error")
        if error:
            logger.error(f"❌ Error: {error}")

        # Validate structure
        is_valid = self._validate_ai_details_structure(ai_validation, asset_name)

        logger.info(f"{'='*70}\n")

        return is_valid

    def get_aggregated_signal_hybrid_dynamic(
        self,
        asset_name: str,
        df: pd.DataFrame,
        aggregators: Dict,
        hybrid_selector,
        live_price: Optional[float] = None  # ✨ NEW: Pass through for staleness check
    ) -> Tuple[int, Dict]:
        """
        ✅ FIXED: Ensures AI validation details are ALWAYS populated
        ✅ FIXED: Injects MTF Governor data into Council Aggregator
        """

        # ================================================================
        # STEP 1: Determine optimal aggregator mode
        # ================================================================
        mode_info = hybrid_selector.get_optimal_mode(asset_name, df)

        selected_mode = mode_info["mode"]
        confidence = mode_info["confidence"]
        switch_occurred = mode_info["switch_occurred"]
        analysis = mode_info["analysis"]

        # Log mode selection
        if switch_occurred:
            logger.info(f"\n{'='*70}")
            logger.info(f"[HYBRID] {asset_name}: MODE SWITCH → {selected_mode.upper()}")
            logger.info(f"{'='*70}")
        else:
            logger.debug(f"[HYBRID] {asset_name}: Using {selected_mode.upper()} mode")

        # ================================================================
        # STEP 2: Get signal from selected aggregator
        # ================================================================
        if selected_mode == "council":
            aggregator = aggregators["council"]

            # ✅ NEW: Fetch the latest MTF regime data for the Council
            mtf_regime = {}
            if (
                hasattr(self, "_current_regime_data")
                and asset_name in self._current_regime_data
            ):
                mtf_regime = self._current_regime_data[asset_name].copy()

            # Error 9 fix: inject T3.5/T3.6 enrichments into the council hybrid fork
            if asset_name in ("BTC", "BTCUSDT"):
                mtf_regime["funding_rate_zscore"] = getattr(self, "funding_rate_zscore", 0.0)
                # F.4: BTC CVD order flow + F.6: L2 order book
                if self.cvd_consumer:
                    mtf_regime["cvd_trend"] = self.cvd_consumer.get_trend()
                    mtf_regime["cvd_stale"] = self.cvd_consumer.is_stale()
                    mtf_regime["order_book_imbalance"] = self.cvd_consumer.get_order_book_imbalance()
                    mtf_regime["order_book_wall_detected"] = self.cvd_consumer.is_wall_detected()
                    mtf_regime["last_trade_price"] = self.cvd_consumer.get_last_price()
                    # Section 2.3 Step 3: depth snapshot for TransitionDetector
                    mtf_regime["depth_data"] = self.cvd_consumer.get_depth()
            if hasattr(self, "_dxy_falling"):
                mtf_regime["dxy_falling"] = self._dxy_falling
            # df_4h injection: CompositeState, TransitionDetector momentum, MR/TF strategies,
            # and 4H slope alignment all read governor_data.get('df_4h'). Without this the
            # cached 4H data was fetched but never reached the aggregator.
            mtf_regime["df_4h"] = self._df_4h_cache.get(asset_name)

            # ✅ FIXED: Pass full market context to the Institutional Council
            signal, details = aggregator.get_aggregated_signal(
                df,
                current_regime=mtf_regime.get("regime", "NEUTRAL"),
                is_bull_market=mtf_regime.get("is_bull", False),
                governor_data=mtf_regime,  # This contains the trade_type needed for Asymmetry
                live_price=live_price      # ✨ NEW: For accurate staleness check
            )

            logger.info(
                f"[COUNCIL] Total Score: {details.get('total_score', 0):.2f}/5.0"
            )
            logger.info(f"[COUNCIL] Decision: {details.get('decision_type', 'N/A')}")

        else:  # performance mode
            aggregator = aggregators["performance"]
            # Fetch the latest MTF regime data to pass to the aggregator
            mtf_regime = {}
            if (
                hasattr(self, "_current_regime_data")
                and asset_name in self._current_regime_data
            ):
                mtf_regime = self._current_regime_data[asset_name].copy()

            # Error 9 fix: inject T3.5/T3.6 enrichments into the performance hybrid fork
            if asset_name in ("BTC", "BTCUSDT"):
                mtf_regime["funding_rate_zscore"] = getattr(self, "funding_rate_zscore", 0.0)
                # F.4: BTC CVD order flow + F.6: L2 order book
                if self.cvd_consumer:
                    mtf_regime["cvd_trend"] = self.cvd_consumer.get_trend()
                    mtf_regime["cvd_stale"] = self.cvd_consumer.is_stale()
                    mtf_regime["order_book_imbalance"] = self.cvd_consumer.get_order_book_imbalance()
                    mtf_regime["order_book_wall_detected"] = self.cvd_consumer.is_wall_detected()
                    mtf_regime["last_trade_price"] = self.cvd_consumer.get_last_price()
                    # Section 2.3 Step 3: depth snapshot for TransitionDetector
                    mtf_regime["depth_data"] = self.cvd_consumer.get_depth()
            if hasattr(self, "_dxy_falling"):
                mtf_regime["dxy_falling"] = self._dxy_falling
            # df_4h injection: CompositeState, TransitionDetector momentum, MR/TF strategies,
            # and 4H slope alignment all read governor_data.get('df_4h'). Without this the
            # cached 4H data was fetched but never reached the aggregator.
            mtf_regime["df_4h"] = self._df_4h_cache.get(asset_name)

            signal, details = aggregator.get_aggregated_signal(
                df,
                current_regime=mtf_regime.get("regime", "NEUTRAL"),
                is_bull_market=mtf_regime.get("is_bull", False),
                governor_data=mtf_regime,
                live_price=live_price      # ✨ NEW: For accurate staleness check
            )

            logger.info(
                f"[PERFORMANCE] Signal Quality: {details.get('signal_quality', 0):.2%}"
            )
            logger.info(f"[PERFORMANCE] Reasoning: {details.get('reasoning', 'N/A')}")

        # ================================================================
        # ✅ FIX: ALWAYS format AI validation (don't rely on aggregator)
        # ================================================================
        ai_validation = details.get("ai_validation")

        if ai_validation is None or not isinstance(ai_validation, dict):
            logger.warning(
                f"[HYBRID] ⚠️ No AI validation from {selected_mode} aggregator, "
                f"generating manually..."
            )

            # Get the actual aggregator instance
            actual_aggregator = aggregators.get(selected_mode)

            # Try aggregator's method first
            if actual_aggregator and hasattr(
                actual_aggregator, "_format_ai_validation_for_viz"
            ):
                try:
                    ai_validation = actual_aggregator._format_ai_validation_for_viz(
                        final_signal=signal, details=details.copy(), df=df
                    )
                    logger.info(
                        f"[HYBRID] ✅ AI validation from {selected_mode} aggregator"
                    )
                except Exception as e:
                    logger.error(
                        f"[HYBRID] Aggregator method failed: {e}, using fallback"
                    )
                    ai_validation = None

            # Fallback: Use direct AI validation
            if ai_validation is None:
                logger.info(f"[HYBRID] Using direct AI validation fallback")
                ai_validation = self._format_ai_validation_direct(asset_name, signal, df)

            # Store in details
            details["ai_validation"] = ai_validation

        else:
            logger.info(
                f"[HYBRID] ✅ AI validation present from {selected_mode} aggregator"
            )
            logger.debug(
                f"[HYBRID] Pattern: {ai_validation.get('pattern_name', 'N/A')}, "
                f"Confidence: {ai_validation.get('pattern_confidence', 0):.2%}"
            )

        # ================================================================
        # STEP 3: Verify AI validation has all required fields
        # ================================================================
        required_fields = [
            "pattern_detected",
            "pattern_name",
            "pattern_confidence",
            "top3_patterns",
            "top3_confidences",
            "sr_analysis",
            "validation_passed",
            "action",
        ]

        missing_fields = [
            field for field in required_fields if field not in ai_validation
        ]

        if missing_fields:
            logger.warning(f"[HYBRID] ⚠️ AI validation missing fields: {missing_fields}")
            logger.warning(f"[HYBRID] Regenerating complete AI validation...")

            # Regenerate completely
            ai_validation = self._format_ai_validation_direct(asset_name, signal, df)
            details["ai_validation"] = ai_validation

        # ================================================================
        # STEP 4: Get current price
        # ================================================================
        try:
            current_price = float(df["close"].iloc[-1])
        except:
            current_price = 0.0

        # ================================================================
        # STEP 5: Calculate adaptive TP/SL if signal is not HOLD
        # ================================================================
        tp_sl_info = None

        if signal != 0:
            try:
                tp_sl_info = hybrid_selector.calculate_tp_sl(
                    asset_name=asset_name,
                    entry_price=current_price,
                    signal=signal,
                    df=df,
                    mode=selected_mode,
                    confidence=confidence,
                )

                logger.info(
                    f"\n[TP/SL] Adaptive Levels ({selected_mode.upper()} mode):"
                )
                logger.info(f"  Entry:         ${current_price:,.2f}")
                logger.info(f"  Stop Loss:     ${tp_sl_info['stop_loss']:,.2f}")
                logger.info(f"  Take Profit:   ${tp_sl_info['take_profit']:,.2f}")
                logger.info(f"  Risk/Reward:   {tp_sl_info['risk_reward_ratio']:.2f}:1")

            except Exception as e:
                logger.error(f"[TP/SL] Calculation failed: {e}")

        # ================================================================
        # STEP 6: Build merged details
        # ================================================================
        merged_details = details.copy()

        # Add hybrid-specific metadata
        hybrid_metadata = {
            "aggregator_mode": selected_mode,
            "mode_confidence": confidence,
            "mode_switched": switch_occurred,
            "regime_analysis": {
                "regime_type": analysis["regime_type"],
                "trend_strength": analysis["trend"]["strength"],
                "trend_direction": analysis["trend"]["direction"],
                "adx": analysis["trend"]["adx"],
                "volatility_regime": analysis["volatility"]["regime"],
                "volatility_ratio": analysis["volatility"]["ratio"],
                "price_clarity": analysis["price_action"]["clarity"],
                "momentum_aligned": analysis.get("momentum_aligned", False),
                "at_key_level": analysis.get("at_key_level", False),
            },
            "adaptive_tpsl": tp_sl_info,
            "signal_quality": max(details.get("signal_quality", 0), confidence * 0.8),
        }

        merged_details.update(hybrid_metadata)

        # ================================================================
        # ✅ CRITICAL: Verify ai_validation is in merged_details
        # ================================================================
        if "ai_validation" not in merged_details:
            logger.error(f"[HYBRID] ❌ CRITICAL: ai_validation lost during merge!")
            # Last resort: add placeholder
            merged_details["ai_validation"] = {
                "pattern_detected": False,
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
                "validation_passed": False,
                "action": "error_lost_validation",
                "error": "AI validation lost during hybrid merge",
            }

        # ================================================================
        # STEP 7: Final validation check
        # ================================================================
        final_ai_validation = merged_details.get("ai_validation")

        if not final_ai_validation or not isinstance(final_ai_validation, dict):
            logger.error(f"[HYBRID] ❌ FINAL CHECK FAILED: ai_validation is invalid")
        else:
            logger.info(f"[HYBRID] ✅ Final AI validation verified:")
            logger.info(f"  Pattern: {final_ai_validation.get('pattern_name', 'N/A')}")
            logger.info(
                f"  Confidence: {final_ai_validation.get('pattern_confidence', 0):.2%}"
            )
            logger.info(f"  Action: {final_ai_validation.get('action', 'N/A')}")
            logger.info(
                f"  Top3: {len(final_ai_validation.get('top3_patterns', []))} patterns"
            )

        # ================================================================
        # STEP 8: Apply mode-specific quality filters
        # ================================================================
        if selected_mode == "council":
            min_score = merged_details.get("required_score", 3.5)
            actual_score = merged_details.get("total_score", 0)

            if signal != 0 and actual_score < min_score:
                logger.info(
                    f"[COUNCIL] Signal filtered: {actual_score:.2f} < {min_score:.2f}"
                )
                signal = 0
                merged_details["reasoning"] = (
                    f"Council score too low ({actual_score:.2f}/{min_score:.2f})"
                )

        elif selected_mode == "performance":
            min_quality = 0.28
            actual_quality = merged_details.get("signal_quality", 0)

            if signal != 0 and actual_quality < min_quality:
                logger.info(
                    f"[PERFORMANCE] Signal filtered: {actual_quality:.2%} < {min_quality:.2%}"
                )
                signal = 0
                merged_details["reasoning"] = (
                    f"Signal quality too low ({actual_quality:.2%})"
                )

        # Update signal in details
        merged_details["signal"] = signal

        return signal, merged_details

    def _signal_str(self, signal: int) -> str:
        """Convert signal to readable string"""
        return {1: "BUY", -1: "SELL", 0: "HOLD"}.get(signal, "UNKNOWN")

    def load_models(self):
        """
        ✨  Load models with safe AI initialization
        """
        logger.info("\n" + "-" * 70)
        logger.info("Loading Trained Models")
        logger.info("-" * 70)

        loaded = 0
        expected = 0

        # Load strategy models
        for asset_name, strategies in self.strategies.items():
            for strategy_name, strategy in strategies.items():
                expected += 1

                model_filename = f"{strategy_name}_{asset_name.lower()}.pkl"
                model_path = f"models/{model_filename}"

                if Path(model_path).exists():
                    try:
                        if strategy.load_model(model_path):
                            logger.info(f"[OK] {model_path}")
                            loaded += 1
                        else:
                            logger.error(f"[FAIL] {model_path}")
                    except Exception as e:
                        logger.error(f"[FAIL] {model_path}: {e}")
                else:
                    logger.error(f"[FAIL] Not found: {model_path}")

        if loaded == 0:
            logger.error("=" * 70)
            logger.error("NO MODELS LOADED! Run: python train.py")
            logger.error("=" * 70)
            sys.exit(1)

        logger.info(f"\n[OK] Loaded {loaded}/{expected} strategy models")

        # ✅ NEW: Set initial presets using DynamicPresetSelector
        if self.config.get("aggregator_settings", {}).get("preset") == "auto":
            selector = DynamicPresetSelector(self.data_manager, self.config)

            # Get preset for EACH enabled asset
            self.selected_presets = {}
            for asset_name in self.strategies.keys():
                preset = selector.get_preset_for_asset(asset_name)
                self.selected_presets[asset_name] = preset
                self.dynamic_selector.current_presets[asset_name] = preset

        else:
            # Manual preset
            preset = self.config.get("aggregator_settings", {}).get(
                "preset", "balanced"
            )
            self.selected_presets = {name: preset for name in self.strategies.keys()}

            for asset in self.selected_presets:
                self.dynamic_selector.current_presets[asset] = preset

        # ✨  Try to initialize AI (non-fatal)
        ai_success = False
        try:
            ai_success = self.initialize_ai_layer()
        except Exception as e:
            logger.error(f"[AI] Initialization error: {e}")
            ai_success = False

        if not ai_success:
            logger.warning("[AI] Continuing WITHOUT AI validation")
            # Ensure all AI components are None
            self.analyst = None
            self.sniper = None
            self.ai_validator = None

        # Initialize aggregators
        self._initialize_aggregators()

        # ✨  Only set logging if AI validator exists
        if self.ai_validator:
            try:
                self.ai_validator.detailed_logging = True
                logger.info("[AI] Detailed logging enabled")
            except Exception as e:
                logger.warning(f"[AI] Logging config failed: {e}")

        if self.ai_validator and self.telegram_bot and self.analyst and self.sniper:
            try:
                logger.info("[VIZ] Initializing AI visualization system...")
                self.chart_sender = create_visualization_system(
                    telegram_bot=self.telegram_bot,
                    analyst=self.analyst,
                    sniper=self.sniper,
                    ai_validator=self.ai_validator,
                )

                if self.chart_sender:
                    logger.info("[VIZ] ✅ Visualization system ready")
                else:
                    logger.warning("[VIZ] ⚠️ Visualization system failed")

            except Exception as e:
                logger.error(f"[VIZ] Initialization error: {e}")
                self.chart_sender = None
        else:
            logger.info("[VIZ] Skipping visualization (AI or Telegram not available)")
            self.chart_sender = None

    def initialize_autotrainer(self):
        """Initializes and starts the continuous learning pipeline."""
        if self.config.get("ml", {}).get("enable_autotrain", False):
            logger.info("\n" + "=" * 70)
            logger.info("INITIALIZING AUTO-TRAINER")
            logger.info("=" * 70)
            try:
                self.autotrainer = ContinuousLearningPipeline(
                    config=self.config,
                    trading_bot=self,
                    telegram_bot=self.telegram_bot
                )
                self.autotrainer.start()
            except Exception as e:
                logger.error(f"[AUTO-TRAIN] Failed to initialize: {e}", exc_info=True)
                self.autotrainer = None
        else:
            logger.info("[AUTO-TRAIN] Disabled in config.")

    def _run_telegram_loop(self):
        """
        Target function for the dedicated Telegram thread.
        This function blocks the thread, keeping it alive for continuous operation.
        """
        try:
            if self.telegram_bot:
                logger.info("[TELEGRAM] Starting Telegram polling loop in dedicated thread.")
                self.telegram_bot.run_polling()
                logger.info("[TELEGRAM] Telegram polling loop has finished.")
        except Exception as e:
            logger.error(f"[TELEGRAM] Critical error in dedicated Telegram thread: {e}", exc_info=True)

    def _send_telegram_notification(self, coro):
        """
        ✅ FIXED: Send notification safely from main thread to Telegram's event loop
        """
        if not self.telegram_bot or not self.telegram_bot._is_ready:
            logger.debug("[TELEGRAM] Bot not ready, queueing notification")
            if hasattr(self.telegram_bot, '_message_queue'):
                self.telegram_bot._message_queue.append(str(coro))
            return

        try:
            loop = self.telegram_bot._current_loop

            if not loop or loop.is_closed():
                logger.warning("[TELEGRAM] Event loop is closed or unavailable, cannot send notification.")
                return

            # ✅ Submit coroutine to the bot's event loop
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            
            # Wait for completion with timeout
            try:
                future.result(timeout=15.0)
                logger.debug("[TELEGRAM] Notification sent successfully")
            except TimeoutError:
                logger.warning("[TELEGRAM] Notification timeout (15s)")
                future.cancel()
            except Exception as e:
                logger.error(f"[TELEGRAM] Notification error: {e}")

        except Exception as e:
            logger.error(f"[TELEGRAM] Failed to send notification: {e}", exc_info=True)

    # ------------------------------------------------------------------ #
    #  Shadow-trade helper — call at every gate that blocks a real signal #
    # ------------------------------------------------------------------ #
    def _shadow_open_blocked(
        self,
        asset_name: str,
        signal: int,
        details: dict,
        df,
        current_price: float,
        gate_label: str,
        asset_cfg: dict,
    ):
        """
        Open a shadow (virtual) position for any signal that was blocked
        before reaching the execution layer.  Safe to call from every gate;
        no-ops when shadow_trader is absent or signal is 0.
        """
        try:
            if not self.shadow_trader or signal == 0 or current_price <= 0:
                return
            _side = "long" if signal > 0 else "short"
            # VTM-style regime-adaptive ATR (same logic used in the main shadow block)
            _atr = None
            try:
                import numpy as _np_s
                if df is not None and len(df) >= 30:
                    def _ratr(n):
                        hi = df["high"].values
                        lo = df["low"].values
                        cl = df["close"].values
                        tr = _np_s.maximum(
                            hi[-n-1:] - lo[-n-1:],
                            _np_s.abs(hi[-n-1:] - _np_s.roll(cl, 1)[-n-1:]),
                            _np_s.abs(lo[-n-1:]  - _np_s.roll(cl, 1)[-n-1:]),
                        )
                        return float(_np_s.nanmean(tr[-n:]))
                    _a7, _a14, _a28 = _ratr(7), _ratr(14), _ratr(28)
                    if _a28 > 0:
                        _r = _a7 / _a28
                        _atr = _a7 if _r > 1.30 else (_a28 if _r < 0.70 else _a14)
                    else:
                        _atr = _a14
            except Exception:
                _atr = None
            _risk_cfg = asset_cfg.get("risk_management", asset_cfg)
            _atr_mult = float(_risk_cfg.get("atr_multiplier", 1.8))
            _tp_mults = _risk_cfg.get("partial_targets", [2.5, 4.0, 6.0])
            _src = details.get("aggregator_mode", "PERF").upper()
            # J2.1: Pass CompositeState snapshot at entry
            _aggregator = self.aggregators.get(asset_name)
            _comp_state_dict = {}
            if _aggregator and hasattr(_aggregator, '_cached_composite') and \
               _aggregator._cached_composite is not None:
                try:
                    _comp_state_dict = _aggregator._cached_composite.to_dict()
                except Exception:
                    pass
            self.shadow_trader.open_position(
                asset=asset_name,
                side=_side,
                entry_price=current_price,
                strategy_source=_src,
                gate_blocked_by=gate_label[:60],
                signal_details=details,
                atr=float(_atr) if _atr else None,
                atr_multiplier=_atr_mult,
                tp_multiples=_tp_mults,
                composite_state=_comp_state_dict,
            )
            logger.debug(f"[SHADOW] Opened {_side} for {asset_name} (gate={gate_label})")
        except Exception as _e:
            logger.debug(f"[SHADOW] _shadow_open_blocked failed: {_e}")

    def _notify_blocked(
        self,
        asset: str,
        signal: int,
        block_source: str,
        block_reason: str,
        details: dict = None,
        price: float = None,
    ):
        """
        Fire-and-forget helper: sends a signal-blocked Telegram alert from the
        main (non-async) thread.  Only sends when there is a real directional
        signal (BUY=1 / SELL=-1) — HOLD=0 is silently skipped.
        """
        if signal == 0:
            return  # genuine hold, nothing to report
        if not self.telegram_bot or not self.telegram_bot._is_ready:
            return
        try:
            self._send_telegram_notification(
                self.telegram_bot.notify_signal_blocked(
                    asset=asset,
                    signal=signal,
                    block_source=block_source,
                    block_reason=block_reason,
                    details=details or {},
                    price=price,
                )
            )
        except Exception as e:
            logger.error(f"[TELEGRAM] _notify_blocked failed for {asset}: {e}")

    def _reinitialize_aggregator(self, asset_name: str, preset: str):
        """
        Reinitialize aggregator with new preset
        ✅ ENHANCED: Better logging for auto preset changes
        """
        try:
            # Load Presets
            CONFIG_PATH = Path("config/aggregator_presets.json")
            with open(CONFIG_PATH, "r") as f:
                AGGREGATOR_PRESETS = json.load(f)["AGGREGATOR_PRESETS"]
            # Get strategies for this asset
            strategies = self.strategies.get(asset_name, {})
            if not strategies:
                logger.warning(f"[AUTO PRESET] No strategies for {asset_name}")
                return

            # Get preset config
            # Fix #9: _preset_key selects which preset bucket to use (BTC vs GOLD/FX),
            # but asset_type must be the real asset name so aggregators log correctly.
            # Previously asset_type was set to "GOLD" for ALL non-BTC assets, meaning
            # GBPAUD, EURUSD, USTEC etc. were all identified as "GOLD" internally.
            _preset_key = "BTC" if "BTC" in asset_name.upper() else "GOLD"
            preset_config = AGGREGATOR_PRESETS.get(_preset_key, {}).get(preset)
            asset_type = asset_name  # Pass actual asset name to aggregator

            if not preset_config:
                logger.error(f"[AUTO PRESET] No config for {asset_name} {preset}")
                return

            # Get AI validator if available
            ai_validator = None
            if hasattr(self, "ai_validator") and self.ai_validator:
                ai_validator = self.ai_validator

            # Extract global filter flags
            agg_settings = self.config.get("aggregator_settings", {})
            use_macro_gov = agg_settings.get("use_macro_governor", True)
            use_gatekeeper = agg_settings.get("use_gatekeeper", True)
            trend_threshold = agg_settings.get("trend_aligned_threshold", 3.0)
            counter_threshold = agg_settings.get("counter_trend_threshold", 3.5)

            # Determine aggregator mode from config
            global_mode = (
                self.config.get("aggregator_settings", {})
                .get("mode", "performance")
                .lower()
            )

            # Check current aggregator state
            current_aggregator = self.aggregators.get(asset_name)
            is_hybrid_state = (
                isinstance(current_aggregator, dict)
                and current_aggregator.get("mode") == "hybrid"
            )

            # Force hybrid if config says so OR state says so
            if global_mode == "hybrid" or is_hybrid_state:
                mode_to_init = "hybrid"
            elif global_mode == "council":
                mode_to_init = "council"
            else:
                mode_to_init = "performance"

            logger.info(
                f"[AUTO PRESET] Reinitializing {asset_name}\n"
                f"  Preset: {preset.upper()}\n"
                f"  Mode:   {mode_to_init.upper()}"
            )

            # ================================================================
            # RE-INITIALIZE BASED ON MODE
            # ================================================================

            if mode_to_init == "hybrid":
                # HYBRID MODE: Recreate both
                perf_agg = PerformanceWeightedAggregator(
                    mean_reversion_strategy=strategies.get("mean_reversion"),
                    trend_following_strategy=strategies.get("trend_following"),
                    ema_strategy=strategies.get("ema_strategy"),
                    asset_type=asset_type,
                    config=preset_config,
                    ai_validator=ai_validator,
                    mtf_integration=self.mtf_integration,  # Pass MTF for Governor
                    enable_world_class_filters=True,  # Enable filters
                    enable_ai_circuit_breaker=True,
                    enable_detailed_logging=getattr(self, "detailed_logging", False),
                    strong_signal_bypass_threshold=getattr(
                        self.params, "ai_strong_signal_bypass", 0.70
                    ),
                    use_macro_governor=use_macro_gov,
                    use_gatekeeper=use_gatekeeper
                )
                council_agg = InstitutionalCouncilAggregator(
                    mean_reversion_strategy=strategies.get("mean_reversion"),
                    trend_following_strategy=strategies.get("trend_following"),
                    ema_strategy=strategies.get("ema_strategy"),
                    asset_type=asset_type,
                    ai_validator=ai_validator,
                    enable_detailed_logging=False,
                    config=preset_config,
                    trend_aligned_threshold=trend_threshold,
                    counter_trend_threshold=counter_threshold,
                    weight_structure=1.0,
                    weight_momentum=1.5,
                    use_macro_governor=use_macro_gov,
                    use_gatekeeper=use_gatekeeper
                )

                self.aggregators[asset_name] = {
                    "performance": perf_agg,
                    "council": council_agg,
                    "mode": "hybrid",
                }
                logger.info(
                    f"[AUTO PRESET] ✓ Hybrid aggregators refreshed for {asset_name}"
                )

            elif mode_to_init == "council":
                # COUNCIL MODE
                new_aggregator = InstitutionalCouncilAggregator(
                    mean_reversion_strategy=strategies.get("mean_reversion"),
                    trend_following_strategy=strategies.get("trend_following"),
                    ema_strategy=strategies.get("ema_strategy"),
                    asset_type=asset_type,
                    ai_validator=ai_validator,
                    enable_detailed_logging=getattr(self, "detailed_logging", False),
                    config=preset_config,
                    trend_aligned_threshold=trend_threshold,
                    counter_trend_threshold=counter_threshold,
                    use_macro_governor=use_macro_gov,
                    use_gatekeeper=use_gatekeeper
                )
                self.aggregators[asset_name] = new_aggregator
                logger.info(
                    f"[AUTO PRESET] ✓ Council aggregator refreshed for {asset_name}"
                )

            else:
                # PERFORMANCE MODE
                new_aggregator = PerformanceWeightedAggregator(
                    mean_reversion_strategy=strategies.get("mean_reversion"),
                    trend_following_strategy=strategies.get("trend_following"),
                    ema_strategy=strategies.get("ema_strategy"),
                    asset_type=asset_type,
                    config=preset_config,
                    ai_validator=ai_validator,
                    mtf_integration=self.mtf_integration,  # Pass MTF for Governor
                    enable_world_class_filters=True,  # Enable filters
                    enable_ai_circuit_breaker=True,
                    enable_detailed_logging=getattr(self, "detailed_logging", False),
                    strong_signal_bypass_threshold=getattr(
                        self.params, "ai_strong_signal_bypass", 0.70
                    ),
                    use_macro_governor=use_macro_gov,
                    use_gatekeeper=use_gatekeeper
                )
                self.aggregators[asset_name] = new_aggregator
                logger.info(
                    f"[AUTO PRESET] ✓ Performance aggregator refreshed for {asset_name}"
                )

        except Exception as e:
            logger.error(f"[AUTO PRESET] Aggregator reinit error: {e}", exc_info=True)

    def _update_dynamic_presets(self):
        """
        Check market conditions and update presets if regime changed
        """
        try:
            logger.info("\n[REGIME CHECK] Analyzing market conditions...")

            enabled_assets = [
                name
                for name, cfg in self.config["assets"].items()
                if cfg.get("enabled", False)
            ]

            preset_changed = False
            changes = []
            for asset_name in enabled_assets:
                # Get optimal preset for current market conditions
                # Fetch the latest MTF regime data to pass to the selector for Asset-DNA Gating
                regime_data = None
                if (
                    hasattr(self, "_current_regime_data")
                    and asset_name in self._current_regime_data
                ):
                    regime_data = self._current_regime_data[asset_name]

                new_preset = self.dynamic_selector.get_preset_for_asset(asset_name, regime_data=regime_data)

                if new_preset:
                    old_preset = self.selected_presets.get(asset_name)

                    # Fix #12: Skip costly reinit when the preset hasn't actually changed.
                    # The dynamic selector can return the same preset on consecutive cycles.
                    # Reinitialising resets DynamicThresholds history, discards the cold-start
                    # warm-up period we need for min_samples=5 to activate.
                    if not hasattr(self, '_last_applied_preset'):
                        self._last_applied_preset = {}
                    if self._last_applied_preset.get(asset_name) == new_preset:
                        logger.debug(f"[AUTO PRESET] {asset_name}: no change ({new_preset}), skipping reinit")
                        continue

                    # If preset changed, log and mark as changed
                    if old_preset != new_preset:
                        logger.info(
                            f"[AUTO PRESET] {asset_name}: {old_preset.upper()} → {new_preset.upper()}"
                        )
                        self.selected_presets[asset_name] = new_preset
                        preset_changed = True
                        changes.append(f"{asset_name}: {old_preset} → {new_preset}")

                    self._last_applied_preset[asset_name] = new_preset
                    self._reinitialize_aggregator(asset_name, new_preset)

                    if preset_changed:
                        logger.info(f"[AUTO PRESET] ✓ Updated {len(changes)} preset(s)")
                        for change in changes:
                            logger.info(f"  • {change}")

                        # Log statistics
                        stats = self.dynamic_selector.get_statistics()
                        logger.info(
                            f"[AUTO PRESET] Total preset changes: {stats['total_changes']}"
                        )
                        logger.info(
                            f"[AUTO PRESET] Distribution: {stats['preset_distribution']}"
                        )
                    else:
                        logger.debug("[AUTO PRESET] No preset changes needed")

        except Exception as e:
            logger.error(f"[REGIME] Update error: {e}", exc_info=True)

    # ✨  Trading cycle with better error handling
    @handle_errors(
        component="main_trading_loop",
        severity=ErrorSeverity.ERROR,
        notify=True,
        reraise=False,
        default_return=None,
    )
    def run_trading_cycle(self):
        """Execute one complete trading cycle with VTM support"""
        try:
            # Global trading toggle check
            if not self.config.get("trading", {}).get("enabled", True):
                logger.info("[CYCLE] ⏸ Trading is GLOBALLY DISABLED. Skipping cycle.")
                return

            # T3.5: Refresh BTC funding rate (self-throttles to every 8 hours)
            try:
                self._update_funding_rate()
            except Exception as _fe:
                logger.debug(f"[FUNDING] Skipped: {_fe}")

            # ✨ Record heartbeat and check health
            if hasattr(self, 'health_monitor') and self.health_monitor:
                logger.info("[HEARTBEAT] System running...")
                self.health_monitor.heartbeat()
                if not self.health_monitor.is_healthy():
                    logger.critical("[HEALTH] ⚠️ System is UNHEALTHY! Triggering emergency shutdown.")
                    
                    # 🚨 EMERGENCY: Close all trades
                    self.portfolio_manager.emergency_close_all()
                    
                    # Optionally notify via Telegram
                    if self.telegram_bot:
                        self._send_telegram_notification(
                            self.telegram_bot.notify_error("🚨 *EMERGENCY HALT*\nSystem is UNHEALTHY. All positions have been closed for safety!")
                        )
                    
                    # Stop the bot
                    self.stop()
                    return

            logger.info("\n" + "=" * 70)
            logger.info(f"[CYCLE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)

            # ✨ NEW: Update historical data every hour (or every 12 cycles if running every 5 min)
            current_time = datetime.now()
            
            # Debug log to track history update frequency
            time_since_last = (current_time - self._last_history_update).total_seconds() if self._last_history_update else "N/A"
            logger.info(f"[HISTORY] Last update: {self._last_history_update} ({time_since_last}s ago)")

            if (
                self._last_history_update is None
                or (current_time - self._last_history_update).total_seconds() > 3600
            ):  # 1 hour

                logger.info("[HISTORY] Updating historical CSV files...")
                try:
                    self.historical_updater.update_all_enabled_assets()
                    self._last_history_update = current_time
                except Exception as e:
                    logger.error(f"[HISTORY] Update failed: {e}")

            preset_mode = self.config.get("aggregator_settings", {}).get(
                "preset", "balanced"
            )
            if preset_mode == "auto":
                self._update_dynamic_presets()
            # Refresh capital if live
            if not self.portfolio_manager.is_paper_mode:
                try:
                    self.portfolio_manager.refresh_capital()
                except Exception as e:
                    logger.error(f"[ERROR] Failed to refresh capital: {e}")

            self.reset_daily_counters()
            self._consecutive_errors = 0
            self._last_successful_cycle = datetime.now()

            enabled = [
                name
                for name, cfg in self.config["assets"].items()
                if cfg.get("enabled", False)
            ]

            logger.info(f"[SIGNALS] Updating signals for: {', '.join(enabled)}")

            for asset_name in enabled:
                try:
                    # Fix #14: Skip signal evaluation for MT5 assets when market is closed.
                    # Execution is already skipped by check_market_hours() downstream, but
                    # the evaluation still runs, generating stale price warnings and
                    # consuming CPU on closed-market weekends. BTC is 24/7 so always runs.
                    _is_mt5_asset = "BTC" not in asset_name.upper()
                    if _is_mt5_asset and not self.check_market_hours(asset_name):
                        logger.debug(f"[SIGNALS] Skipping {asset_name}: market closed")
                        continue
                    self._update_asset_signal(asset_name)
                except Exception as e:
                    logger.error(f"[SIGNAL] Error updating {asset_name}: {e}")

            # Get current prices for all assets

            current_prices = {}
            enabled = [
                name
                for name, cfg in self.config["assets"].items()
                if cfg.get("enabled", False)
            ]

            for asset_name in enabled:
                try:
                    asset_cfg = self.config["assets"][asset_name]
                    exchange = asset_cfg.get("exchange", "binance")
                    handler = (
                        self.binance_handler
                        if exchange == "binance"
                        else self.mt5_handler
                    )

                    if handler:
                        symbol = asset_cfg.get("symbol")
                        current_prices[asset_name] = handler.get_current_price(symbol=symbol)
                except Exception as e:
                    logger.error(f"Failed to get {asset_name} price: {e}")

            # T3.1: Shadow candle-tier update — every trading cycle (~5min)
            try:
                if self.shadow_trader and current_prices:
                    self.shadow_trader.candle_update_all(current_prices)
                    self.shadow_trader.tick_update_all(current_prices)
                    logger.debug(f"[SHADOW] {self.shadow_trader.summary}")
                    # Persist snapshot for dashboard
                    import os as _os
                    _shadow_path = _os.path.join(
                        _os.path.dirname(_os.path.abspath(__file__)),
                        "logs", "shadow_state.json"
                    )
                    self.shadow_trader.dump_state(_shadow_path)
            except Exception as _sle:
                logger.debug(f"[SHADOW] Update failed: {_sle}")

            # ✨ NEW: Update positions with OHLC data for VTM
            try:
                ohlc_data_dict = {}
                for asset_name in enabled:
                    # Only update if position exists
                    if self.portfolio_manager.has_position(asset_name):
                        handler = (
                            self.binance_handler
                            if self.config["assets"][asset_name].get("exchange")
                            == "binance"
                            else self.mt5_handler
                        )

                        if handler:
                            try:
                                # Fetch latest OHLC
                                end_time = datetime.now(timezone.utc)
                                start_time = end_time - timedelta(hours=24)

                                if handler == self.binance_handler:
                                    df = self.data_manager.fetch_binance_data(
                                        symbol=self.config["assets"][asset_name][
                                            "symbol"
                                        ],
                                        interval=self.config["assets"][asset_name].get(
                                            "interval", "1h"
                                        ),
                                        start_date=start_time.strftime("%Y-%m-%d"),
                                        end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    )
                                else:
                                    df = self.data_manager.fetch_mt5_data(
                                        symbol=self.config["assets"][asset_name][
                                            "symbol"
                                        ],
                                        timeframe=self.config["assets"][asset_name].get(
                                            "timeframe", "H1"
                                        ),
                                        start_date=start_time.strftime("%Y-%m-%d"),
                                        end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    )

                                if len(df) > 0:
                                    latest = df.iloc[-1]
                                    ohlc_data_dict[asset_name] = {
                                        "high": latest["high"],
                                        "low": latest["low"],
                                        "close": latest["close"],
                                    }
                            except Exception as e:
                                logger.debug(
                                    f"[VTM] Failed to fetch OHLC for {asset_name}: {e}"
                                )

                # Update all positions with VTM
                if ohlc_data_dict:
                    closed_count = self.portfolio_manager.update_positions_with_ohlc(
                        ohlc_data_dict
                    )
                    if closed_count > 0:
                        logger.info(
                            f"[VTM] Closed {closed_count} position(s) via dynamic management"
                        )

            except Exception as e:
                logger.error(f"[VTM] Error updating positions: {e}")

            # Update positions with current prices (traditional method)
            try:
                self.portfolio_manager.update_positions(current_prices)
            except Exception as e:
                logger.error(f"[ERROR] Failed to update positions: {e}")

            logger.info(f"[ASSETS] Enabled: {', '.join(enabled)}")

            if hasattr(self, "data_manager_telegram"):
                self.data_manager_telegram.update_snapshot(self)
                self.data_manager_telegram.process_queued_commands(self)

            # Trade each asset
            for asset_name in enabled:
                try:
                    self.trade_asset(asset_name)
                    time.sleep(2)
                except Exception as e:
                    logger.error(
                        f"[ERROR] {asset_name} trade failed: {e}", exc_info=True
                    )

                    # ✨ Log error to database
                    if self.db_manager:
                        self.db_manager.log_system_event(
                            event_type="error",
                            severity="error",
                            message=f"{asset_name} trading error: {str(e)}",
                            component="trade_execution",
                        )

            # Get portfolio status
            try:
                status = self.portfolio_manager.get_portfolio_status(current_prices)
                self._log_portfolio_status(status)
            except Exception as e:
                logger.error(f"[ERROR] Portfolio status: {e}")

            # Reset error counter
            self._consecutive_errors = 0
            self._last_successful_cycle = datetime.now()

            # ✨ Take periodic snapshot
            if self.db_manager:
                self._maybe_take_portfolio_snapshot()

            # ✅ NEW: Log hybrid statistics periodically
            # ✅ NEW: Log hybrid statistics periodically
            if hasattr(self, "hybrid_selector"):
                stats = self.hybrid_selector.get_statistics()

                # ✅ FIXED: Actively query the current modes for the dashboard/logs
                active_modes = {}
                for asset in self.config["assets"]:
                    if self.config["assets"][asset].get("enabled", False):
                        # Get mode without logging the switch to avoid log spam
                        mode_info = self.hybrid_selector.get_optimal_mode(
                            asset, pd.DataFrame()
                        )
                        active_modes[asset] = mode_info["mode"]

                logger.info(f"\n[HYBRID STATS]")
                logger.info(f"  Total Switches:      {stats['total_switches']}")
                logger.info(f"  Council Signals:     {stats['council_signals']}")
                logger.info(f"  Performance Signals: {stats['performance_signals']}")
                logger.info(
                    f"  Current Modes:       {active_modes}"
                )  # <-- Now uses active_modes

                logger.info("[OK] Trading cycle complete")
                logger.info("=" * 70)

        except Exception as e:
            logger.error(f"[ERROR] Cycle failed: {e}", exc_info=True)
            self._consecutive_errors += 1

            # Send critical alert if too many failures
            if self._consecutive_errors >= self._max_consecutive_errors:
                if self.error_handler:
                    self.error_handler.handle_error(
                        exception=e,
                        component="main_trading_loop",
                        severity=ErrorSeverity.CRITICAL,
                        additional_info={
                            "consecutive_errors": self._consecutive_errors,
                            "last_successful": self._last_successful_cycle,
                        },
                        notify=True,
                    )

            if self.db_manager:
                self.db_manager.log_system_event(
                    event_type="error",
                    severity="critical",
                    message=f"Trading cycle error: {str(e)}",
                    component="main",
                )
                time.sleep(300)

    def _vtm_management_loop(self):
        """
        A dedicated loop for updating VTM positions at a high frequency.
        """
        logger.info("[VTM LOOP] Starting high-frequency VTM update loop.")
        
        # Get the VTM update interval from config, default to 5 seconds
        update_interval = self.config["trading"].get("vtm_update_interval_seconds", 5)
        logger.info(f"[VTM LOOP] Update interval set to {update_interval} seconds.")

        while self.is_running:
            try:
                # Check if there are any positions to manage to avoid unnecessary work
                if self.portfolio_manager and self.portfolio_manager.get_open_positions_count() > 0:
                    self._check_VTM_positions()
                
                # Sleep until the next update
                time.sleep(update_interval)

            except Exception as e:
                logger.error(f"[VTM LOOP] Error in VTM management loop: {e}", exc_info=True)
                # Sleep longer on error to prevent spamming logs
                time.sleep(60)
        
        logger.info("[VTM LOOP] VTM management loop has stopped.")

    def _check_VTM_positions(self):
        """
        ✅ FIXED: Iterate through all open positions to ensure every VTM is updated.
        Now attempts to re-initialize missing VTMs for synced/imported trades.
        """
        try:
            # Iterate over a copy of the dictionary's items
            for position_id, position in list(self.portfolio_manager.positions.items()):
                asset_name = position.asset
                
                if not self.config["assets"].get(asset_name, {}).get("enabled", False):
                    continue

                # --------------------------------------------------------
                # ✅ NEW: Attempt to re-initialize VTM if missing
                # (Common for synced/imported positions or reloads)
                # --------------------------------------------------------
                if not position.trade_manager:
                    # Get 4H data from cache (or try to fetch if not present)
                    df_4h = self._df_4h_cache.get(asset_name)
                    
                    if df_4h is not None and not df_4h.empty:
                        logger.info(f"[VTM LOOP] Attempting to re-initialize missing VTM for {position_id}...")
                        try:
                            # Use Position object's logic via PortfolioManager helper (if it existed)
                            # For now, we manually re-initialize if we have OHLC data
                            asset_cfg = self.config["assets"].get(asset_name, {})
                            risk_cfg = asset_cfg.get("risk", {})
                            
                            # Prepare ohlc_data for the Position class logic
                            ohlc_data = {
                                "high": df_4h["high"].values,
                                "low": df_4h["low"].values,
                                "close": df_4h["close"].values,
                                "volume": df_4h["volume"].values if "volume" in df_4h else None
                            }
                            
                            # We can re-call the initialization logic or just create the VTM here
                            from src.execution.veteran_trade_manager import VeteranTradeManager
                            
                            position.trade_manager = VeteranTradeManager(
                                entry_price=position.entry_price,
                                side=position.side,
                                asset=position.asset,
                                risk_config=risk_cfg,
                                high=ohlc_data["high"],
                                low=ohlc_data["low"],
                                close=ohlc_data["close"],
                                volume=ohlc_data["volume"],
                                quantity=position.quantity,
                                signal_details=getattr(position, 'signal_details', {}),
                                trade_type=getattr(position, 'signal_details', {}).get("trade_type", "TREND"),
                            )
                            logger.info(f"[VTM LOOP] ✅ Successfully re-initialized VTM for {position_id}")
                        except Exception as e:
                            logger.error(f"[VTM LOOP] Failed to auto-initialize VTM for {position_id}: {e}")
                    else:
                        # VTM is missing and we don't have data in cache yet
                        continue

                # Get the appropriate handler for the asset
                exchange = self.config["assets"][asset_name].get("exchange", "binance")
                handler = self.binance_handler if exchange == "binance" else self.mt5_handler

                if not handler:
                    continue

                # Get 4H data from the cache
                df_4h = self._df_4h_cache.get(asset_name)

                # Check VTM with real-time updates
                try:
                    vtm_result = None
                    if exchange == "binance":
                        vtm_result = handler.check_and_update_positions_VTM(asset_name, df_4h=df_4h)
                    else:
                        vtm_result = handler.check_and_update_positions_VTM(asset_name, df_4h=df_4h)

                    # ✅ Handle Pyramid Requests
                    if isinstance(vtm_result, dict) and "pyramid_requests" in vtm_result:
                        for req in vtm_result["pyramid_requests"]:
                            logger.info(f"[VTM LOOP] 🗼 Executing PYRAMID for {asset_name} ({req['side'].upper()})")
                            
                            # ✅ Standardized Log
                            log_trade_event("PYRAMID", {
                                "symbol": self.config["assets"].get(asset_name, {}).get("symbol"),
                                "asset": asset_name,
                                "side": req["side"],
                                "trade_type": "TREND_PYRAMID",
                                "position_id": f"PYR_{req['original_position_id']}"
                            })

                            # Convert side to signal
                            pyramid_signal = 1 if req["side"] == "long" else -1
                            
                            # Add pyramiding flag to signal details
                            sig_details = req["signal_details"].copy()
                            sig_details["is_pyramid_scale_in"] = True
                            sig_details["parent_position_id"] = req["original_position_id"]
                            
                            # Execute the new trade
                            handler.execute_signal(
                                signal=pyramid_signal,
                                asset_name=asset_name,
                                signal_details=sig_details
                            )

                except Exception as e:
                    logger.error(f"[VTM LOOP] Error checking {asset_name} (Position: {position_id}): {e}")

        except Exception as e:
            logger.error(f"[VTM LOOP] Position check error: {e}", exc_info=True)

    def _log_VTM_status(self):
        """Log Veteran Trade Manager status for all positions"""
        try:
            has_VTM = False

            for asset, position in self.portfolio_manager.positions.items():
                if position.trade_manager:
                    has_VTM = True
                    VTM_status = position.get_vtm_status()

                    logger.info(f"\n{'-' * 70}")
                    logger.info(f"[VTM STATUS] {asset} {VTM_status['side'].upper()}")
                    logger.info(f"{'-' * 70}")
                    logger.info(f"Entry Price:      ${VTM_status['entry_price']:,.2f}")
                    logger.info(
                        f"Current Price:    ${VTM_status['current_price']:,.2f}"
                    )
                    # Display both absolute P&L and percentage P&L
                    pnl_color = "+" if VTM_status['pnl_abs'] >= 0 else ""
                    logger.info(f"P&L:              {pnl_color}${VTM_status['pnl_abs']:,.2f} ({VTM_status['pnl_pct']:+.2f}%)")
                    logger.info(f"")
                    logger.info(
                        f"Stop Loss:        ${VTM_status['stop_loss']:,.2f} ({VTM_status['distance_to_sl_pct']:+.2f}% away)"
                    )
                    logger.info(
                        f"Take Profit:      ${VTM_status['take_profit']:,.2f} ({VTM_status['distance_to_tp_pct']:+.2f}% away)"
                    )
                    logger.info(f"")
                    logger.info(
                        f"Profit Locked:    {'✓ YES' if VTM_status['profit_locked'] else '✗ NO'}"
                    )
                    logger.info(f"Updates Count:    {VTM_status['update_count']}")
                    logger.info(f"Last Update:      {VTM_status['last_update']}")
                    logger.info(f"{'-' * 70}")

            if not has_VTM and len(self.portfolio_manager.positions) > 0:
                logger.debug("[VTM] No positions using dynamic management")

        except Exception as e:
            logger.error(
                f"Error logging VTM status: {e}"
            )  # Wait 5 minutes before next cycle # Wait 5 minutes before next cycle

    def _log_portfolio_status(self, status):
        """
        ✅  Enhanced logging with per-asset position breakdown
        """
        logger.info(f"\n{'-' * 70}")
        logger.info("[PORTFOLIO STATUS]")
        logger.info(f"{'-' * 70}")
        logger.info(f"Mode:           {status.get('mode', 'N/A').upper()}")
        logger.info(f"Total Value:    ${status.get('total_value', 0):,.2f}")
        logger.info(f"Cash:           ${status.get('capital', 0):,.2f}")
        logger.info(f"Exposure:       ${status.get('total_exposure', 0):,.2f}")

        # ✅ NEW: Per-asset position counts
        asset_counts = status.get("asset_position_counts", {})
        asset_details = status.get("asset_positions_detail", {})
        max_per_asset = status.get("max_positions_per_asset", 3)

        logger.info(f"\n[POSITION COUNTS] (Max: {max_per_asset} per asset per side)")
        for asset, counts in asset_counts.items():
            long_count = counts["long"]
            short_count = counts["short"]
            total_count = counts["total"]

            # Get position IDs for this asset
            details = asset_details.get(asset, {})
            long_ids = details.get("long_ids", [])
            short_ids = details.get("short_ids", [])
            long_tickets = details.get("long_tickets", [])
            short_tickets = details.get("short_tickets", [])

            logger.info(f"\n{asset}:")
            logger.info(f"  LONG:  {long_count}/{max_per_asset}")
            if long_ids:
                logger.info(f"    IDs:     {', '.join(long_ids)}")
            if long_tickets:
                logger.info(f"    Tickets: {', '.join(map(str, long_tickets))}")

            logger.info(f"  SHORT: {short_count}/{max_per_asset}")
            if short_ids:
                logger.info(f"    IDs:     {', '.join(short_ids)}")
            if short_tickets:
                logger.info(f"    Tickets: {', '.join(map(str, short_tickets))}")

            logger.info(f"  TOTAL: {total_count}/{max_per_asset * 2}")

        # Daily P&L
        daily_pnl = status.get("daily_pnl", 0)
        daily_pnl_color = "+" if daily_pnl >= 0 else ""
        logger.info(f"\n[P&L]")
        logger.info(f"Daily P&L:      {daily_pnl_color}${daily_pnl:,.2f}")

        realized_pnl = status.get("realized_pnl_today", 0)
        realized_color = "+" if realized_pnl >= 0 else ""
        logger.info(f"Realized P&L:   {realized_color}${realized_pnl:,.2f}")

        # Individual position P&L
        positions = status.get("positions", {})
        if positions:
            logger.info(f"\n{'-' * 70}")
            logger.info("[INDIVIDUAL POSITIONS]")
            logger.info(f"{'-' * 70}")

            for position_id, pos_data in positions.items():
                asset = pos_data.get("asset", "N/A")
                side = pos_data.get("side", "N/A").upper()
                entry = pos_data.get("entry_price", 0)
                current = pos_data.get("current_price", 0)
                pnl = pos_data.get("pnl", 0)
                pnl_pct = pos_data.get("pnl_pct", 0) * 100

                pnl_color = "+" if pnl >= 0 else ""

                logger.info(f"\n{position_id} ({asset} {side}):")
                logger.info(f"  Entry:   ${entry:,.2f}")
                logger.info(f"  Current: ${current:,.2f}")
                logger.info(
                    f"  P&L:     {pnl_color}${pnl:,.2f} ({pnl_color}{pnl_pct:.2f}%)"
                )

                if pos_data.get("mt5_ticket"):
                    logger.info(f"  MT5:     Ticket {pos_data['mt5_ticket']}")

        logger.info(f"\n{'-' * 70}")

    def reset_daily_counters(self):
        """Reset daily trading counters"""
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.trade_count_today = 0
            self.daily_loss = 0.0
            self.last_trade_date = current_date
            logger.info(f"[RESET] Daily counters reset for {current_date}")
            self.portfolio_manager.start_trading_session()
            logger.info(f"[SESSION] Trading session started")

            if not self.portfolio_manager.is_paper_mode:
                logger.info("[REFRESH] Fetching updated capital...")
                try:
                    self.portfolio_manager.refresh_capital()
                    self.portfolio_manager.reset_daily_pnl()
                except Exception as e:
                    logger.error(f"[ERROR] Failed to refresh capital: {e}")

            # Send daily summary via Telegram
            if self.telegram_bot and self.telegram_bot._is_ready:
                try:
                    self._send_telegram_notification(
                        self.telegram_bot.send_daily_summary()
                    )
                except Exception as e:
                    logger.debug(f"[TELEGRAM] Daily summary error: {e}")

    def check_trading_limits(self) -> bool:
        """Check if trading limits are reached.

        Side-effect: when a limit is hit, populates self._last_limit_reason with a
        specific human-readable reason (e.g. 'Daily trade cap reached (30/30)').
        This is consumed by the Telegram block-notification so users can tell
        WHICH limit fired instead of seeing the generic 'count or loss' string.
        """
        risk_cfg = self.config.get("risk_management", {})
        self._last_limit_reason = None

        max_daily_trades = risk_cfg.get("max_daily_trades", 30)
        if self.trade_count_today >= max_daily_trades:
            self._last_limit_reason = (
                f"Daily trade cap reached ({self.trade_count_today}/{max_daily_trades}). "
                f"Resets at next UTC day rollover."
            )
            logger.warning(
                f"[LIMIT] Daily trades ({self.trade_count_today}/{max_daily_trades})"
            )
            return False

        max_daily_loss = risk_cfg.get("max_daily_loss_pct", 0.05)
        if self.daily_loss >= max_daily_loss:
            self._last_limit_reason = (
                f"Daily loss limit reached ({self.daily_loss:.2%} ≥ {max_daily_loss:.2%})"
            )
            logger.warning(f"[LIMIT] Daily loss ({self.daily_loss:.2%})")
            return False

        circuit_breaker = risk_cfg.get("circuit_breaker_loss_pct", 0.10)
        loss_pct = (
            self.daily_loss / self.portfolio_manager.initial_capital
            if self.portfolio_manager.initial_capital > 0
            else 0
        )
        if loss_pct >= circuit_breaker:
            self._last_limit_reason = (
                f"Circuit breaker tripped: drawdown {loss_pct:.2%} ≥ {circuit_breaker:.2%}"
            )
            logger.error(f"[BREAKER] CIRCUIT BREAKER! Loss: {loss_pct:.2%}")
            if self.telegram_bot and self._telegram_ready.is_set():
                try:
                    self._send_telegram_notification(
                        self.telegram_bot.notify_error(
                            f"🚨 CIRCUIT BREAKER!\nLoss: {loss_pct:.2%}"
                        )
                    )
                except:
                    pass
            return False

        trading_cfg = self.config.get("trading", {})
        if trading_cfg.get("allow_simultaneous_positions", True):
            max_positions = trading_cfg.get("max_simultaneous_positions", 2)
            current = self.portfolio_manager.get_open_positions_count()
            if current >= max_positions:
                self._last_limit_reason = (
                    f"Max simultaneous positions reached ({current}/{max_positions})"
                )
                logger.info(f"[LIMIT] Max positions ({current}/{max_positions})")
                return False

        return True

    def check_min_time_between_trades(self, asset_name: str) -> bool:
        """Check minimum time between trades.

        Per-asset override wins if set in config (min_time_between_trades_minutes
        under the asset key).  Falls back to the global trading setting (default 60).
        In strongly-trending regimes (BEARISH/BULLISH) a 0.5× multiplier is applied
        so the bot can re-enter faster when the move is still clearly intact.
        """
        global_default = self.config["trading"].get("min_time_between_trades_minutes", 480)

        # Per-asset override (e.g. GOLD: min_time_between_trades_minutes: 30)
        asset_cfg = self.config.get("assets", {}).get(asset_name, {})
        min_minutes = asset_cfg.get("min_time_between_trades_minutes", global_default)

        # Regime-aware reduction: if the MTF regime is strongly directional,
        # halve the cooldown so re-entries are faster after manual closes or TP hits.
        regime_data = getattr(self, "_current_regime_data", {}).get(asset_name, {})
        regime = regime_data.get("regime", "NEUTRAL")
        regime_conf = regime_data.get("confidence", 0.0)
        if regime in ("BULLISH", "BEARISH") and regime_conf >= 0.7:
            min_minutes = max(15, min_minutes * 0.5)

        # Bypass cooldown entirely when there are no open positions for the asset.
        # The cooldown exists to prevent over-trading; with zero exposure there is
        # nothing to protect and blocking here only causes missed entries.
        if asset_name in self.last_trade_times:
            open_positions = self.portfolio_manager.get_asset_positions(asset_name)
            if not open_positions:
                logger.debug(f"[COOLDOWN] {asset_name}: no open positions — cooldown bypassed")
                return True

            elapsed = datetime.now() - self.last_trade_times[asset_name]
            if elapsed.total_seconds() < min_minutes * 60:
                remaining = min_minutes - (elapsed.total_seconds() / 60)
                logger.info(f"[COOLDOWN] {asset_name}: {remaining:.0f}min remaining (limit={min_minutes:.0f}min, regime={regime})")
                return False

    def check_market_hours(self, asset_name: str) -> bool:
        """Check if market is open for the asset"""
        asset_name_upper = asset_name.upper()

        # 1. Crypto is 24/7
        if "BTC" in asset_name_upper:
            return True

        # 2. Check Weekend Block for Institutional Assets (Gold, USOIL, Forex)
        is_open = MarketHours.should_trade(asset_name)
        if not is_open:
            status, message = MarketHours.get_market_status("forex")
            current_hour = datetime.now().hour
            
            # Use per-asset logging
            if self.last_market_status_log.get(asset_name) != current_hour:
                logger.info(f"[MARKET] {asset_name}: {message}")
                self.last_market_status_log[asset_name] = current_hour
            return False

        # 3. Rollover Dead Zone Protection (21:30 - 23:30 UTC)
        # Reason: Spreads explode and liquidity vanishes during this period.
        if MarketHours.is_rollover_dead_zone():
            current_hour = datetime.now().hour
            if self.last_market_status_log.get(asset_name) != current_hour:
                logger.info(f"[MARKET] {asset_name}: Rollover Dead Zone (21:30-23:30 UTC) — Blocking entry.")
                self.last_market_status_log[asset_name] = current_hour
            return False

        return True
    @handle_errors(
        component="trade_asset",
        severity=ErrorSeverity.ERROR,
        notify=True,
        reraise=False,
        default_return=None,
    )
    def trade_asset(self, asset_name: str):
        """
        ✅ FIXED: Execute trading logic with proper MTF filtering for ALL aggregator types
        """
        asset_cfg = self.config["assets"][asset_name]
        if not asset_cfg.get("enabled", False):
            return

        # Check market hours BEFORE trading
        if not self.check_market_hours(asset_name):
            logger.info(f"[SKIP] {asset_name}: Market closed")
            return

        exchange = asset_cfg.get("exchange", "binance")
        symbol = asset_cfg.get("symbol", "BTCUSDT")
        handler = self.binance_handler if exchange == "binance" else self.mt5_handler

        if not handler:
            logger.warning(f"[!] {asset_name}: Handler unavailable")
            return

        try:
            logger.info(f"\n{'-' * 70}")
            logger.info(f"[TRADE ASSET] Processing {asset_name}")
            logger.info(f"{'-' * 70}")

            # ============================================================
            # 0. Emergency Trading Halt (Circuit Breaker)
            # ============================================================
            halted, reason = self.portfolio_manager.check_circuit_breaker()
            if halted:
                logger.warning(f"[CIRCUIT BREAKER] Halted: {reason}")
                return

            # ============================================================
            # 1. Fetch FRESH Data & Signal
            # ============================================================
            end_time = datetime.now(timezone.utc)

            if exchange == "binance":
                interval = asset_cfg.get("interval", "1h")
                lookback = 15 if interval == "1h" else 60
                start_time = end_time - timedelta(days=lookback)

                df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                timeframe = asset_cfg.get("timeframe", "H1")
                lookback = 25 if timeframe == "H1" else 75
                start_time = end_time - timedelta(days=lookback)

                df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )

            df = self.data_manager.clean_data(df)

            # B.1: Drop incomplete (current-hour) candle — signal generation
            # must only use confirmed, closed candle data. The VTM uses
            # live tick prices separately via update_with_current_price().
            if not df.empty:
                import pandas as pd
                _now_floor = pd.Timestamp.now(tz='UTC').floor('h')
                if df.index[-1] >= _now_floor:
                    df = df.iloc[:-1]

            # Fetch and cache 4H data for VTM loop
            self._df_4h_cache[asset_name] = self._fetch_4h_data(asset_name)
            # Propagate df_4h into the persistent regime dict so it's available
            # to the hybrid selector forks even on cycles where MTF wasn't re-run.
            if asset_name in self._current_regime_data:
                self._current_regime_data[asset_name]["df_4h"] = self._df_4h_cache.get(asset_name)

            if len(df) < 250:
                logger.warning(
                    f"[SKIP] {asset_name}: Insufficient data ({len(df)}/250)"
                )
                return

            aggregator = self.aggregators.get(asset_name)
            if not aggregator:
                logger.warning(f"[SKIP] {asset_name}: No aggregator available")
                return

            # ============================================================
            # 2. Get Current Price & Generate Signal
            # ============================================================
            # Fetch live price BEFORE aggregator so it can use it for staleness check
            try:
                # ✅ CRITICAL: Force a live price fetch ONLY at the moment of execution
                current_price = handler.get_current_price(symbol, force_live=True)
            except Exception as e:
                # Log stale data details for easier debugging
                last_ts = df.index[-1] if not df.empty else "N/A"
                last_c = df["close"].iloc[-1] if not df.empty else "N/A"
                logger.warning(
                    f"[TRADE ASSET] Live price fetch failed ({e}). "
                    f"Falling back to stale CSV close from {last_ts}: ${last_c}"
                )
                current_price = float(df["close"].iloc[-1]) if not df.empty else 0.0

            # ✅ RE-CALCULATE REGIME (5min Cache): Ensure we are using fresh regime data
            # before making any directional decisions. This prevents staying in a stale 
            # regime for up to 30 mins after a trend flip.
            mtf_regime = {}
            if self.mtf_integration:
                try:
                    mtf_regime = self.mtf_integration.get_regime_for_trading(
                        asset_name=asset_name, symbol=symbol, exchange=exchange
                    )
                    self._current_regime_data[asset_name] = mtf_regime
                except Exception as e:
                    logger.error(f"[MTF] Failed to update regime for {asset_name}: {e}")
                    mtf_regime = self._current_regime_data.get(asset_name, {})
            elif (
                hasattr(self, "_current_regime_data")
                and asset_name in self._current_regime_data
            ):
                mtf_regime = self._current_regime_data[asset_name]

            if isinstance(aggregator, dict) and aggregator.get("mode") == "hybrid":
                signal, details = self.get_aggregated_signal_hybrid_dynamic(
                    asset_name,
                    df,
                    aggregators=aggregator,
                    hybrid_selector=self.hybrid_selector,
                    live_price=current_price
                )
            else:
                # SINGLE AGGREGATOR MODE:
                # Both Council and Performance aggregators now accept full context
                signal, details = aggregator.get_aggregated_signal(
                    df,
                    current_regime=mtf_regime.get("regime", "NEUTRAL"),
                    is_bull_market=mtf_regime.get("is_bull", False),
                    governor_data=mtf_regime,
                    live_price=current_price
                )

            details["price"] = current_price

            # Log Signal Quality
            logger.info(
                f"[SIGNAL] {asset_name} Signal: {signal} "
                f"(Quality: {details.get('signal_quality', 0):.2f})"
            )

            if details.get("aggregator_mode"):
                logger.info(
                    f"  Mode: {details['aggregator_mode'].upper()} | "
                    f"Conf: {details.get('mode_confidence', 0):.2%}"
                )

            # ============================================================
            # ✅ FIX: Apply MTF Filtering AFTER signal generation
            # This applies to ALL aggregator types (hybrid, council, performance)
            # ============================================================
            if (
                signal != 0
                and hasattr(self, "_current_regime_data")
                and asset_name in self._current_regime_data
            ):
                mtf_regime = self._current_regime_data[asset_name]

                # Determine regime string for logging
                regime_str = "NEUTRAL"
                if mtf_regime.get('is_bullish'):
                    regime_str = "BULLISH"
                elif mtf_regime.get('is_bearish'):
                    regime_str = "BEARISH"

                logger.info(f"\n[MTF FILTER] Checking regime filters:")
                logger.info(f"  Current Regime: {regime_str}")
                logger.info(
                    f"  Direction:      {'BULL' if mtf_regime.get('is_bullish') else 'BEARISH' if mtf_regime.get('is_bearish') else 'NEUTRAL'}"
                )
                logger.info(f"  Confidence:     {mtf_regime.get('confidence', 0):.2%}")
                logger.info(
                    f"  Recommended:    {mtf_regime.get('recommended_mode', 'N/A').upper()}"
                )
                logger.info(f"  Risk Level:     {mtf_regime.get('risk_level', 'N/A').upper()}")

                # --------------------------------------------------------
                # Filter 1: Counter-trend blocking
                # --------------------------------------------------------
                if not mtf_regime.get("allow_counter_trend", True):
                    is_counter_trend = (signal == 1 and not mtf_regime.get("is_bullish")) or \
                                     (signal == -1 and mtf_regime.get("is_bullish"))

                    if is_counter_trend:
                        logger.warning(f"[MTF FILTER] ✗ BLOCKED: Counter-trend trade")
                        logger.info(
                            f"  Signal Direction: {'LONG' if signal == 1 else 'SHORT'}"
                        )
                        logger.info(
                            f"  MTF Regime:       {'BULL' if mtf_regime.get('is_bullish') else 'BEARISH'}"
                        )
                        logger.info(
                            f"  Reason:           MTF confidence {mtf_regime.get('confidence', 0):.2%} "
                            f"blocks counter-trend in {regime_str} regime."
                        )
                        self._notify_blocked(
                            asset=asset_name,
                            signal=signal,
                            block_source="MTF Counter-Trend",
                            block_reason=(
                                f"{'LONG' if signal == 1 else 'SHORT'} rejected in {regime_str} regime "
                                f"(MTF confidence {mtf_regime.get('confidence', 0):.0%})"
                            ),
                            details=details,
                            price=details.get("price"),
                        )
                        self._shadow_open_blocked(
                            asset_name, signal, details, df, current_price,
                            "mtf_counter_trend", asset_cfg,
                        )
                        return  # ← Block the trade

                # --------------------------------------------------------
                # Filter 2: Max positions limit
                # --------------------------------------------------------
                max_positions = mtf_regime.get("max_positions", 3)
                current_positions = len(
                    self.portfolio_manager.get_asset_positions(asset_name)
                )

                if current_positions >= max_positions:
                    logger.warning(f"[MTF FILTER] ✗ BLOCKED: Max positions reached")
                    logger.info(
                        f"  Current: {current_positions}, Max: {max_positions} "
                        f"(MTF Risk: {mtf_regime['risk_level'].upper()})"
                    )
                    self._notify_blocked(
                        asset=asset_name,
                        signal=signal,
                        block_source="MTF Max Positions",
                        block_reason=(
                            f"Already at max {max_positions} open position(s) "
                            f"(MTF risk level: {mtf_regime.get('risk_level','').upper()})"
                        ),
                        details=details,
                        price=details.get("price"),
                    )
                    self._shadow_open_blocked(
                        asset_name, signal, details, df, current_price,
                        "mtf_max_positions", asset_cfg,
                    )
                    return  # ← Block the trade

                # --------------------------------------------------------
                # Filter 3: High risk adjustment
                # --------------------------------------------------------
                if mtf_regime.get("risk_level") == "high":
                    logger.info(
                        f"[MTF FILTER] ⚠️  High risk regime - position size reduced to 70%"
                    )
                    details["mtf_risk_multiplier"] = 0.7
                elif mtf_regime.get("risk_level") == "low":
                    logger.info(
                        f"[MTF FILTER] ✅ Low risk regime - position size increased to 120%"
                    )
                    details["mtf_risk_multiplier"] = 1.2
                else:
                    details["mtf_risk_multiplier"] = 1.0

                # --------------------------------------------------------
                # Filter 4: Volatility adjustment
                # --------------------------------------------------------
                if mtf_regime.get("volatility") == "high":
                    logger.info(
                        f"[MTF FILTER] ⚠️  High volatility - wider stops recommended"
                    )
                    details["mtf_volatility_adjustment"] = 1.5
                else:
                    details["mtf_volatility_adjustment"] = 1.0

                # Add complete MTF data to signal details
                details["mtf_regime"] = mtf_regime

                logger.info(f"[MTF FILTER] ✓ All filters passed")

            elif signal != 0:
                logger.debug(
                    "[MTF FILTER] No MTF data available, skipping regime filters"
                )

            # ============================================================
            # 3. Check HOLD Signal
            # ============================================================
            if signal == 0:
                # Distinguish aggregator/AI rejection from a natural HOLD
                original_sig = details.get("original_signal", 0)
                reasoning = details.get("reasoning", "")
                ai_details = details.get("ai_validation", {})
                
                # If we had an original signal that was zeroed out, or if reasoning indicates a block
                is_blocked = (original_sig != 0) or (reasoning and "hold" not in reasoning.lower())
                
                if is_blocked:
                    # Determine block source and reason
                    block_source = "Signal Aggregator"
                    
                    # If it was specifically an AI validation rejection
                    if ai_details.get("action") == "rejected" and ai_details.get("rejection_reasons"):
                        block_source = "AI Validation"
                        block_reason = ", ".join(ai_details.get("rejection_reasons"))
                    else:
                        # General aggregator block (volatility, governor, etc)
                        # Clean up technical reasoning strings for humans
                        raw_reason = reasoning or "Signal blocked by aggregator filters"
                        block_reason = raw_reason.replace("_", " ").title()
                        
                        # Special handling for common technical reasons
                        if "low_volatility" in raw_reason:
                            block_reason = "Market volatility below minimum threshold"
                        elif "no_sniper_confirmation" in raw_reason:
                            block_reason = "No institutional sniper confirmation"
                        elif "blocked_by_governor" in raw_reason:
                            block_reason = "Blocked by Macro Governor (Daily Trend)"
                        elif "blocked_by_trap_filter" in raw_reason:
                            block_reason = "Candle structure indicates a potential trap"
                        elif "insufficient_trend_strength" in raw_reason:
                            block_reason = "ADX indicates insufficient trend strength"

                    logger.info(f"[HOLD] {asset_name}: Signal BLOCKED by {block_source} ({block_reason})")
                    self._notify_blocked(
                        asset=asset_name,
                        signal=original_sig or 1, # fallback if original_sig is missing
                        block_source=block_source,
                        block_reason=block_reason,
                        details=details,
                        price=details.get("price"),
                    )
                else:
                    logger.info(f"[HOLD] {asset_name}: No action taken")
                return

            # ============================================================
            # 4. Check Trading Limits & Cooldowns
            # ============================================================
            if not self.check_trading_limits():
                logger.info(f"[LIMIT] Trading limits reached")
                self._notify_blocked(
                    asset=asset_name,
                    signal=signal,
                    block_source="Trading Limits",
                    block_reason=(
                        getattr(self, "_last_limit_reason", None)
                        or "Daily trade count or loss limit reached — trading paused"
                    ),
                    details=details,
                    price=details.get("price"),
                )
                self._shadow_open_blocked(
                    asset_name, signal, details, df, current_price,
                    "trading_limits", asset_cfg,
                )
                return

            if not self.check_min_time_between_trades(asset_name):
                logger.info(f"[COOLDOWN] {asset_name} is in cooldown")
                self._notify_blocked(
                    asset=asset_name,
                    signal=signal,
                    block_source="Cooldown",
                    block_reason=f"Minimum time between trades for {asset_name} not yet elapsed",
                    details=details,
                    price=details.get("price"),
                )
                # Log to DB so dashboard shows "⏸ Cooldown" instead of "AI not validated"
                if self.db_manager and self._log_all_signals:
                    details["reasoning"] = "cooldown_block"
                    self.db_manager.insert_signal_smart(
                        asset=asset_name,
                        signal=signal,
                        signal_quality=details.get("signal_quality", 0),
                        regime=details.get("regime", "UNKNOWN"),
                        regime_confidence=details.get("regime_confidence", 0),
                        mr_signal=details.get("mr_signal", 0),
                        mr_confidence=details.get("mr_confidence", 0),
                        tf_signal=details.get("tf_signal", 0),
                        tf_confidence=details.get("tf_confidence", 0),
                        ema_signal=details.get("ema_signal"),
                        ema_confidence=details.get("ema_confidence"),
                        buy_score=details.get("buy_score"),
                        sell_score=details.get("sell_score"),
                        reasoning="cooldown_block",
                        price=current_price,
                        ai_validated=details.get("ai_validated", False),
                        ai_modified=details.get("ai_modified", False),
                        ai_details=details.get("ai_validation"),
                        executed=False,
                        force_insert=True,
                    )
                # Shadow-track so the engine can measure cooldown opportunity cost
                self._shadow_open_blocked(
                    asset_name, signal, details, df, current_price,
                    "cooldown_block", asset_cfg,
                )
                return

            # Store BEFORE state
            positions_before = self.portfolio_manager.get_asset_positions(asset_name)
            position_ids_before = {p.position_id for p in positions_before}

            # ============================================================
            # 5. Final Pre-Flight Validation Checkpoint
            # ============================================================
            # A. System Health Check
            if hasattr(self, 'health_monitor') and not self.health_monitor.is_healthy():
                logger.error(f"[SAFETY] ✗ EXECUTION VETOED for {asset_name}: System is unhealthy.")
                self._notify_blocked(
                    asset=asset_name,
                    signal=signal,
                    block_source="System Health",
                    block_reason="Bot health monitor reports system is unhealthy — execution vetoed",
                    details=details,
                    price=details.get("price"),
                )
                self._shadow_open_blocked(
                    asset_name, signal, details, df, current_price,
                    "system_health_veto", asset_cfg,
                )
                return

            # B. Circuit Breaker Check (Last Second)
            halted, reason = self.portfolio_manager.check_circuit_breaker()
            if halted:
                logger.error(f"[SAFETY] ✗ EXECUTION VETOED for {asset_name}: Circuit breaker triggered: {reason}")
                self._notify_blocked(
                    asset=asset_name,
                    signal=signal,
                    block_source="Circuit Breaker",
                    block_reason=str(reason),
                    details=details,
                    price=details.get("price"),
                )
                self._shadow_open_blocked(
                    asset_name, signal, details, df, current_price,
                    "circuit_breaker", asset_cfg,
                )
                return

            # C. Confidence/Quality Check
            min_quality = self.config.get("trading", {}).get("min_signal_quality", 0.40)
            signal_quality = details.get("signal_quality", 0)
            if signal_quality < min_quality:
                logger.warning(f"[SAFETY] ✗ EXECUTION VETOED for {asset_name}: Quality {signal_quality:.2f} < {min_quality:.2f}")
                self._notify_blocked(
                    asset=asset_name,
                    signal=signal,
                    block_source="Quality Gate",
                    block_reason=(
                        f"Signal quality {signal_quality:.1%} below minimum threshold {min_quality:.1%}"
                    ),
                    details=details,
                    price=details.get("price"),
                )
                self._shadow_open_blocked(
                    asset_name, signal, details, df, current_price,
                    "quality_gate", asset_cfg,
                )
                return

            logger.info(f"[SAFETY] ✓ Final validation passed for {asset_name}. Executing...")

            # ============================================================
            # 6. Execute Trade
            # ============================================================
            success = False
            try:
                if exchange == "binance":
                    success = self.binance_handler.execute_signal(
                        signal=signal,
                        current_price=current_price,
                        asset_name=asset_name,
                        confidence_score=details.get("signal_quality", 0.5),
                        market_condition=(
                            "bull" if details.get("regime") == "🚀 BULL" else "bear"
                        ),
                        signal_details=details,
                    )
                    self.binance_handler.check_and_update_positions(asset_name)
                else:
                    success = self.mt5_handler.execute_signal(
                        signal=signal,
                        symbol=symbol,
                        asset_name=asset_name,
                        confidence_score=details.get("signal_quality", 0.5),
                        market_condition=(
                            "bull" if details.get("regime") == "🚀 BULL" else "bear"
                        ),
                        signal_details=details,
                    )
                    self.mt5_handler.check_and_update_positions(asset_name)
            except Exception as e:
                logger.error(f"[ERROR] Failed to execute signal for {asset_name}: {e}")
                return

            # ============================================================
            # 7. Handle Success & DB Logging
            # ============================================
            if success:
                positions_after = self.portfolio_manager.get_asset_positions(asset_name)
                position_ids_after = {p.position_id for p in positions_after}
                new_position_ids = position_ids_after - position_ids_before
                closed_position_ids = position_ids_before - position_ids_after

                # Update internal counters
                self.trade_count_today += 1
                self.last_trade_times[asset_name] = datetime.now()

                logger.info(
                    f"[SUCCESS] {asset_name} Trade Executed "
                    f"(Daily count: {self.trade_count_today})"
                )

                # Send Telegram Notifications (New Positions)
                if new_position_ids:
                    for position_id in new_position_ids:
                        new_pos = next(
                            (
                                p
                                for p in positions_after
                                if p.position_id == position_id
                            ),
                            None,
                        )
                        if (
                            new_pos
                            and self.telegram_bot
                            and self.telegram_bot._is_ready
                        ):
                            try:
                                leverage = getattr(new_pos, "leverage", 1)
                                margin_type = getattr(new_pos, "margin_type", "FUTURES")
                                is_futures = getattr(new_pos, "is_futures", False)

                                sl = new_pos.stop_loss if new_pos.stop_loss else 0.0
                                tp = new_pos.take_profit if new_pos.take_profit else 0.0

                                logger.info(
                                    f"[TELEGRAM] Sending notification for {asset_name}:\n"
                                    f"  Leverage:    {leverage}\n"
                                    f"  Margin Type: {margin_type}\n"
                                    f"  Is Futures:  {is_futures}\n"
                                    f"  Stop Loss:   ${sl:,.2f}\n"
                                    f"  Take Profit: ${tp:,.2f}"
                                )

                                vtm_is_active = new_pos.trade_manager is not None

                                self._send_telegram_notification(
                                    self.telegram_bot.notify_trade_opened(
                                        asset=asset_name,
                                        side=new_pos.side,
                                        price=new_pos.entry_price,
                                        size=new_pos.quantity * new_pos.entry_price,
                                        sl=sl,
                                        tp=tp,
                                        leverage=leverage,
                                        margin_type=margin_type,
                                        is_futures=is_futures,
                                        vtm_is_active=vtm_is_active,
                                    )
                                )

                                logger.info(
                                    f"[TELEGRAM] ✓ Trade opened notification sent"
                                )

                            except Exception as e:
                                logger.error(
                                    f"[TELEGRAM] Notification failed: {e}",
                                    exc_info=True,
                                )

                # Send Visualization Chart
                if new_position_ids and self.chart_sender:
                    try:
                        logger.info(f"[VIZ] Preparing chart for {asset_name}...")

                        df_4h = self._fetch_4h_data(asset_name)

                        logger.info(f"[VIZ] Validating AI details structure...")
                        is_valid = self._log_ai_validation_summary(asset_name, details)

                        if not is_valid:
                            logger.error(
                                f"[VIZ] ⚠️ AI validation structure invalid, "
                                f"attempting repair..."
                            )

                            if not details.get("ai_validation") or not isinstance(
                                details["ai_validation"], dict
                            ):
                                logger.info(
                                    f"[VIZ] Regenerating AI validation from scratch..."
                                )
                                details["ai_validation"] = (
                                    self._format_ai_validation_direct(asset_name, signal, df)
                                )

                                is_valid = self._validate_ai_details_structure(
                                    details["ai_validation"], asset_name
                                )

                                if is_valid:
                                    logger.info(f"[VIZ] ✅ Repair successful")
                                else:
                                    logger.error(
                                        f"[VIZ] ❌ Repair failed, chart may be incomplete"
                                    )

                        logger.info(f"[VIZ] Sending chart to Telegram...")
                        self._send_telegram_notification(
                            self.chart_sender.send_decision_chart(
                                asset_name=asset_name,
                                df_15min=df,
                                df_4h=df_4h,
                                signal=signal,
                                details=details,
                                current_price=current_price,
                            )
                        )

                        logger.info(f"[VIZ] ✅ Chart sent successfully")

                    except Exception as e:
                        logger.error(
                            f"[VIZ] Chart generation error: {e}", exc_info=True
                        )

                # ✅ DATABASE LOGGING (Only on Success)
                if self.db_manager:
                    try:
                        logger.info(
                            f"[DB] Logging successful trade execution for {asset_name}"
                        )

                        signal_id, is_new = self.db_manager.insert_signal_smart(
                            asset=asset_name,
                            signal=signal,
                            signal_quality=details.get("signal_quality", 0),
                            regime=details.get("regime", "UNKNOWN"),
                            regime_confidence=details.get("regime_confidence", 0),
                            mr_signal=details.get("mr_signal", 0),
                            mr_confidence=details.get("mr_confidence", 0),
                            tf_signal=details.get("tf_signal", 0),
                            tf_confidence=details.get("tf_confidence", 0),
                            ema_signal=details.get("ema_signal"),
                            ema_confidence=details.get("ema_confidence"),
                            buy_score=details.get("buy_score"),
                            sell_score=details.get("sell_score"),
                            reasoning=details.get("reasoning"),
                            price=current_price,
                            ai_validated=details.get("ai_validated", False),
                            ai_modified=details.get("ai_modified", False),
                            ai_details=details.get("ai_validation"),
                            executed=True,
                        )

                        # Link signal to trade ID if available
                        if new_position_ids and signal_id is not None:
                            new_pos_id = list(new_position_ids)[0]
                            new_pos = next(
                                (
                                    p
                                    for p in positions_after
                                    if p.position_id == new_pos_id
                                ),
                                None,
                            )
                            if (
                                new_pos
                                and hasattr(new_pos, "db_trade_id")
                                and new_pos.db_trade_id
                            ):
                                self.db_manager.update_signal_execution(
                                    signal_id=signal_id,
                                    executed=True,
                                    trade_id=new_pos.db_trade_id,
                                )

                    except Exception as e:
                        logger.error(f"[DB] Failed to log execution: {e}")

                # Log trade to local file
                if self.config.get("logging", {}).get("save_trades", True):
                    self._log_trade(asset_name, signal, details, current_price)

            else:
                logger.warning(
                    f"[SKIP] {asset_name}: Execution returned False "
                    f"(limits/cooldowns/handler failure)"
                )

        except Exception as e:
            logger.error(f"[ERROR] {asset_name} trading error: {e}", exc_info=True)
            if self.db_manager:
                self.db_manager.log_system_event(
                    event_type="error",
                    severity="error",
                    message=f"{asset_name} trading error: {str(e)}",
                    component="trade_execution",
                )

            if self.telegram_bot and self._telegram_ready.is_set():
                try:
                    self._send_telegram_notification(
                        self.telegram_bot.notify_error(
                            f"Error in {asset_name}:\n{str(e)[:200]}"
                        )
                    )
                except:
                    pass

    def _fetch_4h_data(self, asset_name: str) -> pd.DataFrame:
        """
        Helper method to fetch 4H data for S/R analysis

        """
        try:
            asset_cfg = self.config["assets"][asset_name]
            symbol = asset_cfg.get("symbol")
            exchange = asset_cfg.get("exchange", "binance")

            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=30)

            if exchange == "binance":
                df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval="4h",
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe="H4",
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )

            df_4h = self.data_manager.clean_data(df)

            # B.1: Drop incomplete current 4H candle — use only confirmed bars
            if not df_4h.empty:
                _now_floor_4h = pd.Timestamp.now(tz='UTC').floor('4h')
                if df_4h.index[-1] >= _now_floor_4h:
                    df_4h = df_4h.iloc[:-1]

            return df_4h

        except Exception as e:
            logger.error(f"[VIZ] Failed to fetch 4H data: {e}")
            # Return empty dataframe as fallback
            return pd.DataFrame()

    def _update_funding_rate(self):
        """
        T3.5: Fetch BTC perpetual funding rate every 8 hours and compute Z-score.

        Z-score approach (vs static threshold): A fixed threshold like 0.03% fails
        during sustained bull runs where funding stays elevated for weeks.
        Z-score adapts to the current 14-day baseline automatically.

        Z ≥ +2.0 → market is over-leveraged long → MR shorts are high probability
        Z ≤ -2.0 → market is over-leveraged short → MR longs are high probability
        """
        try:
            from datetime import timezone as _tz
            import numpy as _np_fr

            # Only refresh every 8 hours
            _now = datetime.now(_tz.utc)
            if (
                self._last_funding_fetch is not None
                and (_now - self._last_funding_fetch).total_seconds() < 8 * 3600
            ):
                return

            futures_client = self.data_manager.get_futures_client()
            if not futures_client:
                logger.debug("[FUNDING] No futures client available, skipping")
                return

            # Fetch last 42 funding rate records (~14 days at 8h intervals)
            rates = futures_client.futures_funding_rate(symbol="BTCUSDT", limit=42)
            rate_values = [float(r["fundingRate"]) for r in rates]

            self.current_funding_rate = rate_values[-1] if rate_values else 0.0
            self.funding_rate_zscore = 0.0

            if len(rate_values) >= 10:
                mean = _np_fr.mean(rate_values)
                std = _np_fr.std(rate_values)
                if std > 0:
                    self.funding_rate_zscore = (self.current_funding_rate - mean) / std

            self._last_funding_fetch = _now
            logger.info(
                f"[FUNDING] Rate: {self.current_funding_rate:.4%}, "
                f"Z-score: {self.funding_rate_zscore:+.2f} "
                f"({'EXTREME' if abs(self.funding_rate_zscore) >= 2.0 else 'normal'})"
            )

        except Exception as e:
            logger.debug(f"[FUNDING] Fetch failed: {e}")
            self.funding_rate_zscore = 0.0

    def _update_asset_signal(self, asset_name: str):
        """
        Update signal for an asset (handles all aggregator modes)
        ✅ FIXED: Now handles hybrid mode correctly
        """
        try:
            asset_cfg = self.config["assets"][asset_name]
            exchange = asset_cfg.get("exchange", "binance")
            symbol = asset_cfg.get("symbol")

            # Fetch latest data
            end_time = datetime.now(timezone.utc)

            if exchange == "binance":
                interval = asset_cfg.get("interval", "1h")
                lookback = 15 if interval == "1h" else 60
                start_time = end_time - timedelta(days=lookback)

                df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                timeframe = asset_cfg.get("timeframe", "H1")
                lookback = 25 if timeframe == "H1" else 75
                start_time = end_time - timedelta(days=lookback)

                df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )

            df = self.data_manager.clean_data(df)

            # B.1: Drop incomplete (current-hour) candle — signal generation
            # must only use confirmed, closed candle data. The VTM uses
            # live tick prices separately via update_with_current_price().
            if not df.empty:
                import pandas as pd
                _now_floor = pd.Timestamp.now(tz='UTC').floor('h')
                if df.index[-1] >= _now_floor:
                    df = df.iloc[:-1]

            if len(df) < 250:
                logger.debug(
                    f"[SIGNAL] {asset_name}: Insufficient data ({len(df)}/250)"
                )
                return

            # Get handler for current price
            handler = (
                self.binance_handler if exchange == "binance" else self.mt5_handler
            )
            if not handler:
                logger.debug(f"[SIGNAL] {asset_name}: No handler available")
                return

            try:
                current_price = handler.get_current_price(symbol)
            except:
                current_price = df["close"].iloc[-1]

            # Get aggregator
            aggregator = self.aggregators.get(asset_name)
            if not aggregator:
                logger.debug(f"[SIGNAL] {asset_name}: No aggregator")
                return

            # ================================================================
            # ✅ FIX: Handle hybrid vs single aggregator mode
            # ================================================================
            # ================================================================
            # ✅ FIX: Handle hybrid vs single aggregator mode with context
            # ================================================================
            # Get latest MTF data
            mtf_regime = {}
            if (
                hasattr(self, "_current_regime_data")
                and asset_name in self._current_regime_data
            ):
                mtf_regime = self._current_regime_data[asset_name].copy()

            # T3.5: Inject BTC funding rate Z-score into governor_data
            # signal_aggregator reads this to boost MR confidence at extremes.
            if asset_name in ("BTC", "BTCUSDT"):
                mtf_regime["funding_rate_zscore"] = getattr(self, "funding_rate_zscore", 0.0)
                # F.4: Inject BTC CVD order flow + F.6: L2 order book
                if self.cvd_consumer:
                    mtf_regime["cvd_trend"] = self.cvd_consumer.get_trend()
                    mtf_regime["cvd_stale"] = self.cvd_consumer.is_stale()
                    mtf_regime["order_book_imbalance"] = self.cvd_consumer.get_order_book_imbalance()
                    mtf_regime["order_book_wall_detected"] = self.cvd_consumer.is_wall_detected()

            # T3.6: Inject DXY proxy (computed below) into governor_data
            # Computed from EUR/USD 20-SMA vs current close. Zero API cost.
            try:
                _eurusd_assets = [a for a in self.config.get("assets", {})
                                  if "EURUSD" in a.upper() or "EUR_USD" in a.upper()]
                if _eurusd_assets:
                    _eu_sym = self.config["assets"][_eurusd_assets[0]].get("symbol", "EURUSD")
                    _eu_df = None
                    try:
                        _eu_df = self.data_manager.fetch_mt5_data(
                            symbol=_eu_sym, timeframe="H1",
                            start_date=(datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d"),
                            end_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        )
                    except Exception:
                        pass
                    if _eu_df is not None and len(_eu_df) >= 20:
                        _eu_sma = _eu_df["close"].iloc[-20:].mean()
                        _eu_now = _eu_df["close"].iloc[-1]
                        _dxy_val = bool(_eu_now > _eu_sma)
                        mtf_regime["dxy_falling"] = _dxy_val
                        # Error 9 fix: cache globally so hybrid path can inject it too
                        self._dxy_falling = _dxy_val
            except Exception:
                pass  # DXY proxy is a bonus — never block execution

            # F.7: Inject MT5 spread for spread velocity detection
            try:
                if hasattr(self, 'mt5_handler') and self.mt5_handler:
                    _spread_data = getattr(self.mt5_handler, '_last_spread', {})
                    _sym = self.config.get('assets', {}).get(asset_name, {}).get('symbol', '')
                    if _sym and _sym in _spread_data:
                        mtf_regime['current_spread'] = _spread_data[_sym]
            except Exception:
                pass  # Spread capture is a bonus — never block execution

            if isinstance(aggregator, dict) and aggregator.get("mode") == "hybrid":
                # HYBRID MODE: Use dynamic selector
                signal, details = self.get_aggregated_signal_hybrid_dynamic(
                    asset_name=asset_name,
                    df=df,
                    aggregators=aggregator,
                    hybrid_selector=self.hybrid_selector,
                    live_price=current_price
                )
            else:
                # SINGLE AGGREGATOR MODE:
                if isinstance(aggregator, InstitutionalCouncilAggregator):
                    signal, details = aggregator.get_aggregated_signal(
                        df,
                        current_regime=mtf_regime.get("regime", "NEUTRAL"),
                        is_bull_market=mtf_regime.get("is_bull", False),
                        governor_data=mtf_regime,
                        live_price=current_price
                    )
                    # Stamp the engine label so charts + Telegram always know the source
                    details["aggregator_mode"] = "council"
                else:
                    # Performance mode — pass full MTF regime context so regime/confidence
                    # display correctly on the dashboard and gatekeeper has accurate data.
                    signal, details = aggregator.get_aggregated_signal(
                        df,
                        current_regime=mtf_regime.get("regime", "NEUTRAL"),
                        is_bull_market=mtf_regime.get("is_bull", False),
                        governor_data=mtf_regime,
                        live_price=current_price
                    )
                    # Stamp the engine label so charts + Telegram always know the source
                    details["aggregator_mode"] = "performance"
            # T3.1: Shadow trade — open virtual position for every blocked signal
            # so we can measure what gates are costing us in real P&L terms.
            try:
                if self.shadow_trader and signal == 0 and current_price > 0:
                    _reasoning = details.get("reasoning", "")
                    _raw_mr = details.get("mr_signal_raw", details.get("mr_signal", 0))
                    _raw_tf = details.get("tf_signal_raw", details.get("tf_signal", 0))
                    _raw_ema = details.get("ema_signal", 0)
                    _intended = _raw_tf or _raw_ema or _raw_mr
                    if _intended != 0 and _reasoning and "hold (no strategy" not in _reasoning:
                        _side = "long" if _intended > 0 else "short"
                        _src = ("TF" if _raw_tf != 0 else
                                "EMA" if _raw_ema != 0 else "MR")
                        # Compute VTM-style regime-adaptive ATR from df.
                        # Mirrors VTM._calculate_atr(): ATR7/14/28 ratio decides which to use.
                        _atr = None
                        try:
                            import numpy as _np_atr
                            if df is not None and len(df) >= 30:
                                def _rolling_atr(n):
                                    tr = _np_atr.maximum(
                                        df["high"].values[-n-1:] - df["low"].values[-n-1:],
                                        _np_atr.abs(df["high"].values[-n-1:] - df["close"].shift(1).values[-n-1:]),
                                        _np_atr.abs(df["low"].values[-n-1:]  - df["close"].shift(1).values[-n-1:]),
                                    )
                                    return float(_np_atr.nanmean(tr[-n:]))
                                _atr7  = _rolling_atr(7)
                                _atr14 = _rolling_atr(14)
                                _atr28 = _rolling_atr(28)
                                if _atr28 > 0:
                                    _ratio = _atr7 / _atr28
                                    _atr = _atr7 if _ratio > 1.30 else (_atr28 if _ratio < 0.70 else _atr14)
                                else:
                                    _atr = _atr14
                        except Exception:
                            _atr = None

                        # Read VTM multipliers from asset config risk block
                        _risk_cfg   = asset_cfg.get("risk_management", asset_cfg)
                        _atr_mult   = float(_risk_cfg.get("atr_multiplier", 1.8))
                        _tp_mults   = _risk_cfg.get("partial_targets", [2.5, 4.0, 6.0])

                        # J2.1: Pass CompositeState snapshot at entry
                        _aggregator2 = self.aggregators.get(asset_name)
                        _comp_state_dict2 = {}
                        if _aggregator2 and hasattr(_aggregator2, '_cached_composite') and \
                           _aggregator2._cached_composite is not None:
                            try:
                                _comp_state_dict2 = _aggregator2._cached_composite.to_dict()
                            except Exception:
                                pass
                        self.shadow_trader.open_position(
                            asset=asset_name,
                            side=_side,
                            entry_price=current_price,
                            strategy_source=_src,
                            gate_blocked_by=_reasoning[:60],
                            signal_details=details,
                            atr=float(_atr) if _atr else None,
                            atr_multiplier=_atr_mult,
                            tp_multiples=_tp_mults,
                            composite_state=_comp_state_dict2,
                        )
            except Exception as _se:
                logger.debug(f"[SHADOW] Open failed: {_se}")

            # Log signal to database (gated by log_all_signals config flag)
            if self.db_manager and self._log_all_signals:
                signal_id, is_new = self.db_manager.insert_signal_smart(
                    asset=asset_name,
                    signal=signal,
                    signal_quality=details.get("signal_quality", 0),
                    regime=details.get("regime", "UNKNOWN"),
                    regime_confidence=details.get("regime_confidence", 0),
                    mr_signal=details.get("mr_signal", 0),
                    mr_confidence=details.get("mr_confidence", 0),
                    tf_signal=details.get("tf_signal", 0),
                    tf_confidence=details.get("tf_confidence", 0),
                    ema_signal=details.get("ema_signal"),
                    ema_confidence=details.get("ema_confidence"),
                    buy_score=details.get("buy_score"),
                    sell_score=details.get("sell_score"),
                    reasoning=details.get("reasoning"),
                    price=current_price,
                    ai_validated=details.get("ai_validated", False),
                    ai_modified=details.get("ai_modified", False),
                    ai_details=details.get("ai_validation"),
                    executed=False,
                )

                if is_new:
                    logger.info(
                        f"[SIGNAL] {asset_name}: {signal:+2d} "
                        f"(Q={details.get('signal_quality', 0):.2f})"
                    )

            # Update Telegram monitor (if available)
            if self.telegram_bot:
                # Add regime details to the 'details' dictionary for SignalMonitoringIntegration
                details["regime_score"] = mtf_regime.get("regime_score")
                details["regime_reasoning"] = mtf_regime.get("reasoning")
                
                self.telegram_bot.signal_monitor.record_signal(
                    asset=asset_name,
                    signal=signal,
                    details=details,
                    price=current_price,
                    timestamp=datetime.now(),
                )

        except Exception as e:
            logger.error(f"[SIGNAL] {asset_name} update error: {e}", exc_info=True)

    def _maybe_take_portfolio_snapshot(self):
        """Take periodic portfolio snapshots"""
        try:
            now = datetime.now()

            # Check if snapshot is due
            if (
                self._last_snapshot_time is None
                or (now - self._last_snapshot_time).total_seconds()
                >= self._snapshot_interval
            ):

                # Get current prices
                current_prices = {}
                for asset_name in self.config["assets"].keys():
                    if not self.config["assets"][asset_name].get("enabled", False):
                        continue

                    exchange = self.config["assets"][asset_name].get("exchange", "binance")
                    handler = (
                        self.binance_handler
                        if exchange == "binance"
                        else self.mt5_handler
                    )
                    if handler:
                        try:
                            # Resolve symbol
                            symbol = self.config["assets"][asset_name].get("symbol")
                            current_prices[asset_name] = handler.get_current_price(symbol=symbol)
                        except:
                            pass

                # Get portfolio status
                status = self.portfolio_manager.get_portfolio_status(current_prices)

                # Insert snapshot
                if self.db_manager:
                    self.db_manager.insert_portfolio_snapshot(
                        total_value=status["total_value"],
                        cash=status["capital"],
                        equity=status["equity"],
                        total_exposure=status["total_exposure"],
                        open_positions=status["open_positions"],
                        unrealized_pnl=status["total_unrealized_pnl"],
                        realized_pnl_today=status["realized_pnl_today"],
                        positions_detail=status.get("positions"),
                    )

                    self._last_snapshot_time = now
                    logger.debug(f"[DB] Portfolio snapshot taken")

        except Exception as e:
            logger.error(f"[DB] Snapshot error: {e}")

    def toggle_ai_validation(self, enable: bool):
        """Toggle AI validation on/off"""
        if hasattr(self, "ai_validator") and self.ai_validator is not None:
            self.ai_validator.use_ai_validation = enable
            status = "ENABLED" if enable else "DISABLED"
            logger.info(f"[AI] Validation layer {status}")
            return f"✓ AI Validation {status}"
        else:
            return "✗ AI layer not initialized"

    def _log_trade(self, asset: str, signal: int, details: dict, price: float):
        """Log trade to file"""
        try:
            trade_log = Path("logs/trades.log")
            trade_log.parent.mkdir(exist_ok=True)

            with open(trade_log, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now().isoformat()},{asset},{signal},{price:.2f},"
                    f"{details.get('signal_quality', 0):.3f},"
                    f"{details.get('reasoning', 'N/A')}\n"
                )
        except Exception as e:
            logger.debug(f"Trade log error: {e}")

    def _fetch_current_data(self, asset_name: str) -> pd.DataFrame:
        """
        Helper to fetch current 15min data for chart generation

        """
        try:
            asset_cfg = self.config["assets"][asset_name]
            symbol = asset_cfg.get("symbol")
            exchange = asset_cfg.get("exchange", "binance")

            end_time = datetime.now(timezone.utc)

            # Fetch appropriate data
            if exchange == "binance":
                interval = asset_cfg.get("interval", "15m")
                lookback = 15 if interval == "15m" else 60
                start_time = end_time - timedelta(days=lookback)

                df = self.data_manager.fetch_binance_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                timeframe = asset_cfg.get("timeframe", "M15")
                lookback = 25 if timeframe == "M15" else 75
                start_time = end_time - timedelta(days=lookback)

                df = self.data_manager.fetch_mt5_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_time.strftime("%Y-%m-%d"),
                    end_date=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                )

            return self.data_manager.clean_data(df)

        except Exception as e:
            logger.error(f"[VIZ] Failed to fetch current data for {asset_name}: {e}")
            return pd.DataFrame()

    def log_detailed_pnl_report(self):
        """Log detailed P&L report"""
        try:
            current_prices = {}
            for asset_name, asset_cfg in self.config["assets"].items():
                if not asset_cfg.get("enabled", False):
                    continue

                exchange = asset_cfg.get("exchange", "binance")
                handler = (
                    self.binance_handler if exchange == "binance" else self.mt5_handler
                )

                if handler:
                    try:
                        symbol = asset_cfg.get("symbol")
                        current_prices[asset_name] = handler.get_current_price(symbol=symbol)
                    except:
                        pass

            self.portfolio_manager.update_positions(current_prices)
            status = self.portfolio_manager.get_portfolio_status(current_prices)
            exposure_pct = (
                status["total_exposure"] / status["equity"]
                if status["equity"] > 0
                else 0
            )

            logger.info("\n" + "=" * 70)
            logger.info("DETAILED P&L REPORT")
            logger.info("=" * 70)
            logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Mode: {status['mode'].upper()}")
            logger.info("")

            logger.info("[CAPITAL]")
            logger.info(
                f"  Initial:     ${self.portfolio_manager.initial_capital:,.2f}"
            )
            logger.info(f"  Current:     ${status['capital']:,.2f}")
            logger.info(f"  Total Value: ${status['total_value']:,.2f}")
            logger.info("")

            logger.info("[P&L BREAKDOWN]")
            logger.info(f"  Daily P&L:      ${status['daily_pnl']:+,.2f}")
            logger.info(f"  Realized Today: ${status['realized_pnl_today']:+,.2f}")
            logger.info(f"  Unrealized:     ${status['total_unrealized_pnl']:+,.2f}")
            logger.info("")

            positions = status.get("positions", {})
            if positions:
                logger.info("[POSITION DETAILS]")
                for asset, pos in positions.items():
                    logger.info(f"  {asset} {pos['side'].upper()}:")
                    logger.info(f"    Entry:        ${pos['entry_price']:,.2f}")
                    logger.info(f"    Current:      ${pos['current_price']:,.2f}")
                    logger.info(f"    Quantity:     {pos['quantity']:.6f}")
                    logger.info(f"    Value:        ${pos['current_value']:,.2f}")
                    logger.info(
                        f"    P&L:          ${pos['pnl']:+,.2f} ({pos['pnl_pct']*100:+.2f}%)"
                    )

                    if pos.get("mt5_ticket"):
                        logger.info(f"    MT5 Ticket:   {pos['mt5_ticket']}")
                        logger.info(f"    MT5 Profit:   ${pos['mt5_profit']:+,.2f}")

                    logger.info("")
            else:
                logger.info("[NO OPEN POSITIONS]")
                logger.info("")

            logger.info("[RISK METRICS]")
            logger.info(
                f"  Exposure:     ${status['total_exposure']:,.2f} ({exposure_pct:.1%})"
            )
            # logger.info(f"  Drawdown:     {status['drawdown']:.2%}")
            logger.info(f"  Open Pos:     {status['open_positions']}")
            # logger.info(f"  Total Trades: {status['total_trades']}")
            logger.info("=" * 70 + "\n")

        except Exception as e:
            logger.error(f"Error generating P&L report: {e}", exc_info=True)


    
    def start(self):
        """
        ✨  Start bot with proper Telegram thread management
        """
        logger.info("\n" + "=" * 70)
        logger.info("[START] TRADING BOT INITIALIZING")
        logger.info("=" * 70)

        try:
            # Initialize exchanges and handlers
            self.initialize_exchanges()

            # ── Economic calendar: live refresh ───────────────────────────
            def _reload_all_calendars():
                for _agg in self.aggregators.values():
                    if hasattr(_agg, "reload_calendar"):
                        _agg.reload_calendar()

            self.calendar_updater = CalendarUpdater(
                config=self.config,
                reload_callback=_reload_all_calendars,
            )
            self.calendar_updater.start_background_thread()

            # T3.1: Initialise shadow trading engine after exchanges are ready
            self.shadow_trader = ShadowTradingEngine()  # defaults: max_positions=500, max_closed=10000
            logger.info("[SHADOW] Shadow trading engine started")

            # F.4: Start CVD WebSocket for BTC real-time order flow
            # THREADING FIX: main.py's start() is synchronous and enters a blocking while-loop.
            # asyncio.get_event_loop().create_task() creates a coroutine but the event loop is
            # never driven, so the WebSocket task starves and the CVD feed stays permanently stale.
            # Running asyncio.run() inside a daemon thread gives the coroutine its own event loop.
            try:
                self.cvd_consumer = CVDConsumer()
                self.cvd_thread = threading.Thread(
                    target=lambda: asyncio.run(self.cvd_consumer.start()),
                    daemon=True,
                    name="cvd-websocket",
                )
                self.cvd_thread.start()
                logger.info("[CVD] ✅ BTC order flow WebSocket started in daemon thread")
            except Exception as _cvd_err:
                logger.warning(f"[CVD] Failed to start: {_cvd_err}. BTC order flow disabled.")
                self.cvd_consumer = None
                self.cvd_thread = None

            # ✅ FIXED: Initialize selectors AFTER exchanges are connected
            self.dynamic_selector = DynamicPresetSelector(
                self.data_manager, self.config, telegram_bot=self.telegram_bot
            )
            self.hybrid_selector = HybridAggregatorSelector(
                self.data_manager, self.config, mtf_integration=self.mtf_integration, telegram_bot=self.telegram_bot
            )

            preset_mode = self.config.get("aggregator_settings", {}).get(
                "preset", "balanced"
            )
            if preset_mode == "auto":
                logger.info("\n" + "=" * 70)
                logger.info("🎯 AUTO PRESET MODE ENABLED")
                logger.info("=" * 70)
                logger.info("System will automatically select optimal preset per asset:")
                logger.info("  • CONSERVATIVE: High risk/volatility conditions")
                logger.info("  • BALANCED:     Normal market conditions")
                logger.info("  • AGGRESSIVE:   Strong trending markets")
                logger.info("  • SCALPER:      Ideal low-volatility conditions")
                logger.info("=" * 70 + "\n")

                # Set initial presets for all enabled assets
                logger.info(
                    "[AUTO PRESET] Analyzing market conditions for initial setup..."
                )
                for asset_name in self.strategies.keys():
                    if self.config["assets"][asset_name].get("enabled", False):
                        initial_preset = self.dynamic_selector.get_preset_for_asset(
                            asset_name
                        )
                        self.dynamic_selector.current_presets[asset_name] = initial_preset
                        logger.info(f"  {asset_name:6} → {initial_preset.upper()}")
            else:
                logger.info(f"[PRESET] Using manual preset: {preset_mode.upper()}")

            self.load_models()

            self.initialize_mtf_regime_detection()

            if self.mtf_integration:
                logger.info("[MTF] Running initial regime analysis...")
                self.run_mtf_regime_analysis()

            # Initialize and start the autotrainer
            self.initialize_autotrainer()

            # Start Telegram
            if self.telegram_bot:
                logger.info("\n[TELEGRAM] Starting bot's dedicated thread...")
                self.telegram_thread = Thread(
                    target=self._run_telegram_loop, daemon=True, name="TelegramBot"
                )
                self.telegram_thread.start()
                logger.info("[TELEGRAM] ✅ Telegram thread started.")
            
            # Start dashboard server
            self.dashboard_server = start_dashboard_server()

            # F.4: CVD daily reset at 00:00 UTC
            def _cvd_daily_reset():
                if self.cvd_consumer:
                    self.cvd_consumer.daily_reset()
                    logger.info("[CVD] Daily reset complete")
            schedule.every().day.at("00:00").do(_cvd_daily_reset)

            # Schedule trading cycles
            check_interval = self.config["trading"].get("check_interval_seconds", 300)
            schedule.every(check_interval).seconds.do(self.run_trading_cycle)
            schedule.every(1).hours.do(self.log_detailed_pnl_report)

            self.is_running = True
            self._main_loop_running = True

            # Start high-frequency VTM management thread AFTER is_running=True
            # so the while self.is_running loop can actually run.
            self.vtm_thread = Thread(
                target=self._vtm_management_loop, daemon=True, name="VTMManager"
            )
            self.vtm_thread.start()
            logger.info("[VTM] ✅ VTM management thread started.")

            logger.info(f"\n[OK] Trading bot running")
            logger.info(
                f"[TIME] Cycle interval: {check_interval}s ({check_interval / 60:.1f}min)"
            )
            logger.info(f"[MTF] Regime updates: Every 4 hours")
            logger.info(f"Press Ctrl+C to stop\n")

            # Run initial cycle
            self.run_trading_cycle()

            _restart_flag = str(Path("logs") / "restart.flag")
            while self.is_running:
                try:
                    # ── Control Center restart hook ──────────────────────────
                    if os.path.exists(_restart_flag):
                        os.remove(_restart_flag)
                        logger.info("[CONTROL] 🔄 Restart flag detected — restarting bot now…")
                        self.stop()
                        
                        # ── Windows: task-aware restart with local fallback ──────
                        if sys.platform == "win32":
                            import subprocess as _sp
                            _task_name = self.config.get("trading", {}).get(
                                "task_scheduler_name", "TBOT"
                            )
                            _python  = sys.executable
                            _script  = str(Path(__file__).resolve())
                            _workdir = str(Path(__file__).parent.resolve())
                            
                            # Reverted to structure similar to initial setup but with robust taskkill
                            _ps_cmd = (
                                f"$t = Get-ScheduledTask -TaskName '{_task_name}' -ErrorAction SilentlyContinue; "
                                f"if ($t) {{ "
                                f"  Stop-ScheduledTask -TaskName '{_task_name}' -ErrorAction SilentlyContinue; "
                                f"  Start-Sleep -Seconds 2; "
                                f"  taskkill /F /IM python.exe 2>$null; "
                                f"  Start-Sleep -Seconds 2; "
                                f"  Start-ScheduledTask -TaskName '{_task_name}' "
                                f"}} else {{ "
                                f"  Start-Sleep -Seconds 2; "
                                f"  taskkill /F /IM python.exe 2>$null; "
                                f"  Start-Sleep -Seconds 2; "
                                f"  Set-Location '{_workdir}'; "
                                f"  cmd.exe /c \"start python main.py\" "
                                f"}}"
                            )

                            _sp.Popen(
                                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", _ps_cmd],
                                creationflags=_sp.DETACHED_PROCESS | _sp.CREATE_NEW_PROCESS_GROUP,
                            )
                            logger.info(f"[CONTROL] ✅ Restart scheduled — exiting now")
                            sys.exit(0)
                        else:
                            os.execv(sys.executable, [sys.executable] + sys.argv)

                    schedule.run_pending()
                    time.sleep(1)

                except KeyboardInterrupt:
                    raise

                except Exception as e:
                    logger.error(f"[CRITICAL ERROR] Main loop failure: {e}", exc_info=True)
                    if hasattr(self, 'health_monitor') and self.health_monitor:
                        self.health_monitor.record_error()
                    time.sleep(10) # Safety pause before retry

            logger.info("[STOP] Main loop ended")

        except KeyboardInterrupt:
            logger.info("\n[!] KeyboardInterrupt received")
            self.stop()

        except Exception as e:
            logger.error(f"[FATAL] Fatal error: {e}", exc_info=True)
            self.stop()
            sys.exit(1)



    def stop(self):
        """
        ✨  Graceful shutdown with proper Telegram cleanup
        """
        if hasattr(self, "_shutdown_in_progress") and self._shutdown_in_progress:
            logger.info("[STOP] Shutdown already in progress")
            return

        self._shutdown_in_progress = True

        logger.info("\n" + "=" * 70)
        logger.info("[STOP] SHUTTING DOWN TRADING BOT")
        logger.info("=" * 70)

        # ✨ Finalize database
        if self.db_manager:
            try:
                calculate_daily_summary_from_trades(self.db_manager, datetime.now())

                self.db_manager.log_system_event(
                    event_type="shutdown",
                    severity="info",
                    message="Trading bot stopped",
                    component="main",
                    metadata={
                        "final_capital": self.portfolio_manager.current_capital,
                        "open_positions": self.portfolio_manager.get_open_positions_count(),
                    },
                )

                logger.info("[DB] ✓ Final updates complete")

            except Exception as e:
                logger.error(f"[DB] Shutdown error: {e}")

        # Stop calendar updater background thread
        if self.calendar_updater:
            self.calendar_updater.stop()

        # F.4: Stop CVD WebSocket
        if self.cvd_consumer:
            self.cvd_consumer.stop()

        self.is_running = False
        self._main_loop_running = False

        # NEW: Save portfolio state BEFORE closing positions
        try:
            if self.portfolio_manager:
                logger.info("[SHUTDOWN] Saving portfolio state...")
                self.portfolio_manager.save_portfolio_state()
        except Exception as e:
            logger.error(f"Error saving portfolio state on exit: {e}")

        # Stop the autotrainer
        if self.autotrainer:
            self.autotrainer.stop()

        # Close positions if configured
        if self.config["trading"].get("close_positions_on_shutdown", False):
            logger.info("[STOP] Closing open positions...")
            try:
                self.portfolio_manager.close_all_positions()
                logger.info("[STOP] ✅ Positions closed")
            except Exception as e:
                logger.error(f"[STOP] Error closing positions: {e}")

        # ✨  Shutdown VTM thread
        if self.vtm_thread and self.vtm_thread.is_alive():
            logger.info("[VTM] Shutting down VTM management thread...")
            # The loop will exit on the next iteration since self.is_running is False
            self.vtm_thread.join(timeout=10)
            if self.vtm_thread.is_alive():
                logger.warning("[VTM] VTM thread did not terminate gracefully.")

        # ✨  Shutdown Telegram properly
        if self.telegram_bot and self.telegram_thread:
            logger.info("[TELEGRAM] Shutting down...")
            if hasattr(self.telegram_bot, '_shutdown_event') and self.telegram_bot._current_loop:
                self.telegram_bot._current_loop.call_soon_threadsafe(self.telegram_bot._shutdown_event.set)
            
            self.telegram_thread.join(timeout=10)
            if self.telegram_thread.is_alive():
                logger.warning("[TELEGRAM] Thread did not terminate.")
            else:
                logger.info("[TELEGRAM] Thread terminated.")

        # ✨  Shutdown Dashboard
        if hasattr(self, 'dashboard_server') and self.dashboard_server:
            logger.info("[DASHBOARD] Shutting down...")
            try:
                if sys.platform == "win32":
                    subprocess.call(["taskkill", "/F", "/T", "/PID", str(self.dashboard_server.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    self.dashboard_server.terminate()
                logger.info("[DASHBOARD] Process terminated.")
            except Exception as e:
                logger.error(f"[DASHBOARD] Error during shutdown: {e}")

        # Shutdown data manager
        try:
            self.data_manager.shutdown()
            logger.info("[STOP] ✅ Data manager shutdown")
        except Exception as e:
            logger.error(f"[STOP] Data manager error: {e}")

        logger.info("=" * 70)
        logger.info("[OK] Trading bot stopped")
        logger.info("=" * 70)


import requests


def start_dashboard_server():
    """
    Start dashboard server as a subprocess (most reliable approach).
    Runs server.py with the project root as CWD so .env and imports resolve correctly.
    """
    try:
        logger.info("\n" + "=" * 70)
        logger.info("[DASHBOARD] Starting web dashboard...")
        logger.info("=" * 70)

        # Resolve paths from main.py's location (project root)
        project_root = Path(__file__).resolve().parent
        server_path = project_root / "src" / "dashboard" / "server.py"

        if not server_path.exists():
            logger.error(f"[DASHBOARD] Server file not found: {server_path}")
            return None

        server_process = subprocess.Popen(
            [sys.executable, str(server_path)],
            stdout=None,   # Inherit parent stdout (visible in console)
            stderr=None,   # Inherit parent stderr
            cwd=str(project_root),  # ← project root so .env + imports work
            start_new_session=True if sys.platform != "win32" else False,
        )

        # Give server time to start
        logger.info("[DASHBOARD] Waiting for server to start...")
        time.sleep(3)

        # ✅ FIX 3: Verify server is actually running
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get("http://localhost:5000/api/health", timeout=2)
                if response.status_code == 200:
                    logger.info("[DASHBOARD] ✅ Server is responding")
                    logger.info("[DASHBOARD] 📊 Dashboard: http://localhost:5000")
                    logger.info(
                        "[DASHBOARD] 🔍 Health:    http://localhost:5000/api/health"
                    )
                    logger.info("=" * 70)
                    return server_process
            except requests.exceptions.RequestException:
                if attempt < max_retries:
                    logger.info(
                        f"[DASHBOARD] Waiting for server... ({attempt}/{max_retries})"
                    )
                    time.sleep(2)
                else:
                    logger.warning("[DASHBOARD] ⚠️ Server may not be fully ready")

        # Check if process is still running
        if server_process.poll() is None:
            logger.info(
                "[DASHBOARD] ✅ Web dashboard available at http://localhost:5000"
            )
            logger.info("=" * 70)
            return server_process
        else:
            logger.error("[DASHBOARD] ❌ Server process died immediately")
            return None

    except Exception as e:
        logger.error(f"[DASHBOARD] Error starting server: {e}")
        return None


def start_dashboard_server_threaded():
    """
    Start Flask dashboard in a background thread.
    Uses absolute project-root import so it works regardless of CWD.
    """
    import importlib.util as _ilu
    import traceback as _tb

    def run_flask():
        try:
            # Resolve server.py from the project root (same dir as main.py)
            _root = Path(__file__).resolve().parent
            _server_path = _root / "src" / "dashboard" / "server.py"

            if not _server_path.exists():
                logger.error(f"[DASHBOARD] server.py not found at {_server_path}")
                return

            # Insert project root so server.py's own imports work
            import sys as _sys
            if str(_root) not in _sys.path:
                _sys.path.insert(0, str(_root))

            # Load server module from absolute path
            _spec = _ilu.spec_from_file_location("tbot_dashboard_server", str(_server_path))
            _mod  = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)

            _app = _mod.app
            logger.info("=" * 70)
            logger.info("[DASHBOARD] 🚀 Starting Flask server on port 5000")
            logger.info("[DASHBOARD] 📊 http://localhost:5000")
            logger.info("=" * 70)

            _app.run(
                host="0.0.0.0",
                port=5000,
                debug=False,
                threaded=True,
                use_reloader=False,
            )
        except OSError as e:
            if "address already in use" in str(e).lower() or "10048" in str(e):
                logger.warning("[DASHBOARD] ⚠️ Port 5000 already in use — dashboard may already be running")
            else:
                logger.error(f"[DASHBOARD] OSError: {e}")
                logger.error(_tb.format_exc())
        except Exception as e:
            logger.error(f"[DASHBOARD] Failed to start: {e}")
            logger.error(_tb.format_exc())

    try:
        dashboard_thread = threading.Thread(
            target=run_flask, daemon=True, name="DashboardServer"
        )
        dashboard_thread.start()

        # Give Flask time to bind the port
        time.sleep(4)

        try:
            response = requests.get("http://localhost:5000/api/health", timeout=3)
            if response.status_code == 200:
                logger.info("[DASHBOARD] ✅ Server ready at http://localhost:5000")
            else:
                logger.warning(f"[DASHBOARD] Health check returned {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.error("[DASHBOARD] ❌ Could not connect to dashboard on port 5000 — check logs above")
        except Exception as e:
            logger.warning(f"[DASHBOARD] Health check failed: {e}")

        return dashboard_thread

    except Exception as e:
        logger.error(f"[DASHBOARD] Thread start error: {e}")
        return None


def main():
    """Main entry point"""
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Write PID so the dashboard Control Center can check bot liveness
    import os as _os
    _pid_path = Path("logs") / "bot.pid"
    _pid_path.write_text(str(_os.getpid()))
    logger.info(f"[MAIN] PID {_os.getpid()} written to {_pid_path}")

    try:
        with open("config/config.json", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("[FAIL] config/config.json not found!")
        sys.exit(1)

    # Check required models exist
    required_models = []
    for asset_name, asset_cfg in config["assets"].items():
        if asset_cfg.get("enabled", False):
            strategies = asset_cfg.get("strategies", {})

            if strategies.get("mean_reversion", {}).get("enabled", False):
                required_models.append(
                    f"models/mean_reversion_{asset_name.lower()}.pkl"
                )

            if strategies.get("trend_following", {}).get("enabled", False):
                required_models.append(
                    f"models/mean_reversion_{asset_name.lower()}.pkl"
                )

            if strategies.get("trend_following", {}).get("enabled", False):
                required_models.append(
                    f"models/trend_following_{asset_name.lower()}.pkl"
                )

            if strategies.get("exponential_moving_averages", {}).get("enabled", False):
                required_models.append(f"models/ema_strategy_{asset_name.lower()}.pkl")

    missing = [m for m in required_models if not Path(m).exists()]
    if missing:
        print("=" * 70)
        print("[FAIL] REQUIRED MODELS NOT FOUND")
        print("=" * 70)
        for model in missing:
            print(f"  [X] {model}")
        print("\nRun: python train.py")
        print("=" * 70)
        sys.exit(1)

    ai_model = Path("models/ai/sniper_dual_timeframe_v1.weights.h5")
    if not ai_model.exists():
        print("=" * 70)
        print("\u26a0\ufe0f  AI MODEL NOT FOUND (Optional)")
        print("=" * 70)
        print(f"  Missing: {ai_model}")
        print("\nBot will run without AI validation (pattern detection disabled)")
        print("Train AI: python train_ai.py")
        print("=" * 70)

    # Create and run the trading bot
    try:
        bot = TradingBot("config/config.json")
        bot.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt)")
        print("\n[STOPPED] Bot stopped by user.")
    except Exception as e:
        logger.exception(f"Fatal error in bot: {e}")
        print(f"\n[FATAL] Bot crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
