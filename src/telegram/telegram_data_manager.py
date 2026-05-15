#!/usr/bin/env python3
"""
Thread-Safe Data Manager for Telegram Bot
Prevents "event loop closed" errors by caching bot data
"""

import asyncio
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class BotDataSnapshot:
    """Thread-safe snapshot of bot data"""
    timestamp: datetime
    portfolio_status: Dict[str, Any]
    positions: Dict[str, Any]
    closed_trades: list
    is_running: bool
    selected_presets: Dict[str, str]
    current_prices: Dict[str, float]
    signal_data: Dict[str, Any] = field(default_factory=dict)
    
    def is_stale(self, max_age_seconds: int = 10) -> bool:
        """Check if snapshot is too old"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > max_age_seconds


class ThreadSafeBotDataManager:
    """
    Manages thread-safe access to bot data for Telegram commands
    Prevents "event loop closed" errors by maintaining a cache
    """
    
    def __init__(self, max_cache_age: int = 10):
        self._lock = threading.RLock()  # Reentrant lock
        self._snapshot: Optional[BotDataSnapshot] = None
        self._max_cache_age = max_cache_age
        
        # Command queue for main thread processing
        self._command_queue = deque(maxlen=100)
        self._command_results: Dict[str, Any] = {}
        
        logger.info(f"[DATA MGR] Initialized (cache_age={max_cache_age}s)")
    
    def update_snapshot(self, trading_bot) -> None:
        """
        Update snapshot from main bot thread
        Call this periodically from main trading loop
        """
        try:
            with self._lock:
                # Safely gather data
                portfolio_status = trading_bot.portfolio_manager.get_portfolio_status()
                
                # Get current prices for ALL enabled assets
                current_prices = {}
                for asset_name, asset_cfg in trading_bot.config["assets"].items():
                    if not asset_cfg.get("enabled", False):
                        continue
                    
                    symbol = asset_cfg.get("symbol")
                    if not symbol:
                        continue

                    handler = (
                        trading_bot.binance_handler if asset_name == "BTC" 
                        else trading_bot.mt5_handler
                    )
                    
                    if handler:
                        try:
                            current_prices[asset_name] = handler.get_current_price(symbol=symbol)
                        except:
                            pass
                
                # Get positions detail
                positions = {}
                for position_id, position in trading_bot.portfolio_manager.positions.items():
                    positions[position_id] = {
                        "asset": position.asset,
                        "side": position.side,
                        "entry_price": position.entry_price,
                        "quantity": position.quantity,
                        "stop_loss": position.stop_loss,
                        "take_profit": position.take_profit,
                        "is_futures": getattr(position, "is_futures", False),
                        "leverage": getattr(position, "leverage", 1),
                        "margin_type": getattr(position, "margin_type", "SPOT"),
                    }
                
                # Create snapshot
                self._snapshot = BotDataSnapshot(
                    timestamp=datetime.now(),
                    portfolio_status=portfolio_status,
                    positions=positions,
                    closed_trades=trading_bot.portfolio_manager.closed_positions[-20:],  # Last 20
                    is_running=trading_bot.is_running,
                    selected_presets=getattr(trading_bot, "selected_presets", {}),
                    current_prices=current_prices,
                )
                
                logger.debug(f"[DATA MGR] Snapshot updated ({len(positions)} positions)")
        
        except Exception as e:
            logger.error(f"[DATA MGR] Snapshot update failed: {e}")
    
    def get_snapshot(self, force_fresh: bool = False) -> Optional[BotDataSnapshot]:
        """
        Get cached snapshot (thread-safe)
        
        Args:
            force_fresh: If True, require snapshot to be recent
        
        Returns:
            BotDataSnapshot or None if stale/unavailable
        """
        with self._lock:
            if self._snapshot is None:
                logger.warning("[DATA MGR] No snapshot available")
                return None
            
            if force_fresh and self._snapshot.is_stale(self._max_cache_age):
                logger.warning(f"[DATA MGR] Snapshot is stale (age={(datetime.now() - self._snapshot.timestamp).total_seconds():.1f}s)")
                return None
            
            return self._snapshot
    
    def get_positions_safe(self) -> Dict[str, Any]:
        """Get positions with fallback"""
        snapshot = self.get_snapshot()
        if snapshot:
            return snapshot.positions
        return {}
    
    def get_portfolio_status_safe(self) -> Dict[str, Any]:
        """Get portfolio status with fallback"""
        snapshot = self.get_snapshot()
        if snapshot:
            return snapshot.portfolio_status
        return {
            "total_value": 0,
            "cash": 0,
            "open_positions": 0,
            "daily_pnl": 0,
        }
    
    def get_current_prices_safe(self) -> Dict[str, float]:
        """Get current prices with fallback"""
        snapshot = self.get_snapshot()
        if snapshot:
            return snapshot.current_prices
        return {}
    
    def queue_command(self, command_id: str, command_type: str, params: Dict[str, Any]) -> None:
        """
        Queue a command for main thread to process
        
        Example:
            data_manager.queue_command(
                command_id="close_btc_123",
                command_type="close_position",
                params={"asset": "BTC", "position_index": 0}
            )
        """
        with self._lock:
            self._command_queue.append({
                "id": command_id,
                "type": command_type,
                "params": params,
                "timestamp": datetime.now(),
            })
            logger.info(f"[DATA MGR] Command queued: {command_type} (id={command_id})")
    
    def process_queued_commands(self, trading_bot) -> int:
        """
        Process queued commands (call from main thread)
        
        Returns:
            Number of commands processed
        """
        processed = 0
        
        with self._lock:
            while self._command_queue:
                cmd = self._command_queue.popleft()
                
                try:
                    result = self._execute_command(cmd, trading_bot)
                    self._command_results[cmd["id"]] = {
                        "success": True,
                        "result": result,
                        "timestamp": datetime.now(),
                    }
                    processed += 1
                    
                except Exception as e:
                    logger.error(f"[DATA MGR] Command execution failed: {e}")
                    self._command_results[cmd["id"]] = {
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now(),
                    }
        
        if processed > 0:
            logger.info(f"[DATA MGR] Processed {processed} commands")
        
        return processed
    
    def get_command_result(self, command_id: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Wait for command result (blocking)
        
        Args:
            command_id: Command ID from queue_command()
            timeout: Max seconds to wait
        
        Returns:
            Result dict or None on timeout
        """
        start = datetime.now()
        
        while (datetime.now() - start).total_seconds() < timeout:
            with self._lock:
                if command_id in self._command_results:
                    result = self._command_results.pop(command_id)
                    return result
            
            asyncio.sleep(0.1)
        
        logger.warning(f"[DATA MGR] Command result timeout: {command_id}")
        return None
    
    def _execute_command(self, cmd: Dict[str, Any], trading_bot) -> Any:
        """Execute a queued command (runs in main thread)"""
        cmd_type = cmd["type"]
        params = cmd["params"]
        
        if cmd_type == "close_position":
            asset = params.get("asset")
            position_index = params.get("position_index")
            
            positions = trading_bot.portfolio_manager.get_asset_positions(asset)
            if position_index < len(positions):
                position_id = positions[position_index].position_id
                
                # Get exit price
                handler = (
                    trading_bot.binance_handler if asset == "BTC"
                    else trading_bot.mt5_handler
                )
                symbol = trading_bot.config["assets"].get(asset, {}).get("symbol")
                exit_price = handler.get_current_price(symbol=symbol) if handler and symbol else None
                
                # Close position
                result = trading_bot.portfolio_manager.close_position(
                    position_id=position_id,
                    exit_price=exit_price,
                    reason="telegram_command"
                )
                return result
            
            raise ValueError(f"Position index {position_index} not found")
        
        elif cmd_type == "close_all_positions":
            asset = params.get("asset")
            
            handler = (
                trading_bot.binance_handler if asset == "BTC"
                else trading_bot.mt5_handler
            )
            symbol = trading_bot.config["assets"].get(asset, {}).get("symbol")
            exit_price = handler.get_current_price(symbol=symbol) if handler and symbol else None
            
            results = trading_bot.portfolio_manager.close_all_positions_for_asset(
                asset=asset,
                exit_price=exit_price,
                reason="telegram_command"
            )
            return results
        
        else:
            raise ValueError(f"Unknown command type: {cmd_type}")


# ============================================================================
# Integration Example: Add to TradingBot.__init__()
# ============================================================================

"""
In main.py TradingBot.__init__():

    self.data_manager_telegram = ThreadSafeBotDataManager(max_cache_age=10)
    logger.info("[INIT] Thread-safe data manager initialized")
"""

# ============================================================================
# Integration Example: Add to TradingBot.run_trading_cycle()
# ============================================================================

"""
In main.py TradingBot.run_trading_cycle(), after updating positions:

    # Update Telegram data snapshot
    if hasattr(self, 'data_manager_telegram'):
        self.data_manager_telegram.update_snapshot(self)
    
    # Process queued commands from Telegram
    if hasattr(self, 'data_manager_telegram'):
        processed = self.data_manager_telegram.process_queued_commands(self)
        if processed > 0:
            logger.info(f"[TELEGRAM] Processed {processed} commands")
"""

# ============================================================================
# Integration Example: Update Telegram Bot Commands
# ============================================================================

"""
In src/telegram.py, update cmd_positions():

async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Use thread-safe snapshot instead of direct access
        if not hasattr(self.trading_bot, 'data_manager_telegram'):
            await update.message.reply_text("❌ Data manager not available")
            return
        
        data_mgr = self.trading_bot.data_manager_telegram
        
        # Get snapshot (non-blocking, thread-safe)
        snapshot = data_mgr.get_snapshot()
        
        if not snapshot:
            await update.message.reply_text("⚠️ Data temporarily unavailable, try again")
            return
        
        # Use snapshot data
        positions = snapshot.positions
        portfolio_status = snapshot.portfolio_status
        current_prices = snapshot.current_prices
        
        # ... rest of command logic using snapshot data ...
        
    except Exception as e:
        logger.error(f"Error in cmd_positions: {e}")
"""

# ============================================================================
# Integration Example: Async Command Execution
# ============================================================================

"""
In src/telegram.py, update cmd_close_asset() for async execution:

@admin_only
async def cmd_close_asset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text("⚠️ Usage: /close BTC")
            return
        
        asset = context.args[0].upper()
        
        valid_assets = ["BTC", "GOLD", "EURUSD", "USTEC", "USOIL", "GBPAUD", "GBPUSD", "USDJPY", "EURJPY"]
        if asset not in valid_assets:
            await update.message.reply_text(f"⚠️ Invalid asset. Valid: {', '.join(valid_assets)}")
            return
        
        data_mgr = self.trading_bot.data_manager_telegram
        
        # Queue command for main thread
        command_id = f"close_{asset}_{datetime.now().timestamp()}"
        data_mgr.queue_command(
            command_id=command_id,
            command_type="close_all_positions",
            params={"asset": asset}
        )
        
        await update.message.reply_text(f"⏳ Closing {asset} positions...")
        
        # Wait for result (blocking but with timeout)
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            data_mgr.get_command_result,
            command_id,
            10.0  # 10 second timeout
        )
        
        if result and result.get("success"):
            trades = result["result"]
            total_pnl = sum(t["pnl"] for t in trades)
            
            await update.message.reply_text(
                f"✅ Closed {len(trades)} {asset} position(s)\n"
                f"Total P&L: ${total_pnl:,.2f}"
            )
        else:
            error = result.get("error", "Unknown error") if result else "Timeout"
            await update.message.reply_text(f"❌ Failed to close positions: {error}")
    
    except Exception as e:
        logger.error(f"Error in cmd_close_asset: {e}")
        await update.message.reply_text("❌ Error processing command")
"""