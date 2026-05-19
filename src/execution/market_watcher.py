"""
market_watcher.py — Real-time protective market monitor

Provides the "human eye" layer that the main 5-minute loop cannot: it polls
prices every 15 seconds, detects rapid adverse moves on open positions, and
suppresses new signals when momentum has flipped against the signal direction.

Key responsibilities:
  1. Adverse-move guard   — tighten SL or emergency-close when a position is
                            being hit by a fast candle (the "crazy candle" problem).
  2. Momentum snapshot    — tracks the last 3 closed 1H candles per asset to know
                            whether price is currently moving WITH or AGAINST the
                            prevailing signal direction.
  3. Signal suppression   — `is_signal_suppressed(asset, direction)` is called by
                            main.py before every execution gate. Returns True when
                            recent momentum disagrees with the proposed direction.
  4. Telegram alerts      — warns the user when a position enters danger territory
                            so they can intervene manually if needed.
"""

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Tuning constants ─────────────────────────────────────────────────────────
_POLL_INTERVAL        = 15     # seconds between watcher ticks
_MOMENTUM_CANDLES     = 3      # how many closed 1H candles to inspect for direction
_ADVERSE_ATR_WARN     = 1.10   # adverse move > 1.1 × ATR  → tighten SL to breakeven
_ADVERSE_ATR_CLOSE    = 2.20   # adverse move > 2.2 × ATR  → emergency SL tighten hard
_SUPPRESS_ATR_BOUNCE  = 0.50   # new-signal suppressed if price bounced > 0.5 × ATR
                                # against the signal direction since the last entry
_ALERT_COOLDOWN_SEC   = 300    # minimum seconds between repeated alerts for same position
# ─────────────────────────────────────────────────────────────────────────────


class MarketWatcher:
    """
    Runs in a daemon thread. Call `start()` once during bot startup,
    and `stop()` on shutdown.

    Thread-safe reads via `is_signal_suppressed()` are used by the main loop.
    """

    def __init__(
        self,
        config:            dict,
        portfolio_manager,          # PortfolioManager
        mt5_handler,                # MT5Handler  (may be None)
        binance_handler,            # BinanceHandler (may be None)
        telegram_bot=None,
        send_telegram_fn=None,      # callable(coro) — same helper used in main.py
    ):
        self.config           = config
        self.portfolio_manager = portfolio_manager
        self.mt5_handler      = mt5_handler
        self.binance_handler  = binance_handler
        self.telegram_bot     = telegram_bot
        self._send_telegram   = send_telegram_fn

        self._running = False
        self._thread  = None

        # {asset: {"direction": int, "candles": deque[dict], "atr": float,
        #          "last_price": float, "updated_at": datetime}}
        self._momentum: Dict[str, dict] = {}

        # {asset: {"suppressed": bool, "reason": str, "direction": int}}
        self._suppression: Dict[str, dict] = {}
        self._suppress_lock = threading.Lock()

        # alert de-dup: {position_id: last_alert_timestamp}
        self._last_alert: Dict[str, float] = {}

        # ATR cache — refreshed each momentum update
        self._atr_cache: Dict[str, float] = {}

        logger.info("[WATCHER] MarketWatcher initialised.")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._watch_loop, daemon=True, name="MarketWatcher"
        )
        self._thread.start()
        logger.info("[WATCHER] Started in background thread.")

    def stop(self):
        self._running = False
        logger.info("[WATCHER] Stopped.")

    def is_signal_suppressed(self, asset: str, direction: int) -> Tuple[bool, str]:
        """
        Called by main.py execute_signal() BEFORE the execution gate.

        Returns (True, reason_str) when this watcher thinks the proposed
        entry should be skipped; (False, "") otherwise.
        """
        with self._suppress_lock:
            entry = self._suppression.get(asset)
            if not entry:
                return False, ""
            if not entry.get("suppressed"):
                return False, ""
            if entry.get("direction") != direction:
                # Suppression is direction-specific: a SELL suppression does
                # not block a BUY, and vice versa.
                return False, ""
            return True, entry.get("reason", "momentum_disagreement")

    def get_momentum_summary(self) -> dict:
        """Return a snapshot of momentum state — used by Telegram /status."""
        with self._suppress_lock:
            return {
                asset: {
                    "direction": d.get("direction", 0),
                    "atr":       round(d.get("atr", 0), 4),
                    "suppressed_sell": self._suppression.get(asset, {}).get("suppressed")
                                       and self._suppression[asset].get("direction") == -1,
                    "suppressed_buy":  self._suppression.get(asset, {}).get("suppressed")
                                       and self._suppression[asset].get("direction") == 1,
                }
                for asset, d in self._momentum.items()
            }

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _watch_loop(self):
        logger.info("[WATCHER] Watch loop running.")
        while self._running:
            try:
                self._update_momentum()
                self._check_open_positions()
            except Exception as exc:
                logger.error(f"[WATCHER] Loop error: {exc}", exc_info=True)
            time.sleep(_POLL_INTERVAL)
        logger.info("[WATCHER] Watch loop exited.")

    # ── Momentum snapshot ─────────────────────────────────────────────────────

    def _update_momentum(self):
        """
        For every enabled asset, fetch the last _MOMENTUM_CANDLES closed 1H bars
        and determine whether momentum is bullish, bearish, or mixed.
        Also update signal suppression state.
        """
        assets = self.config.get("assets", {})
        for asset_name, asset_cfg in assets.items():
            if not asset_cfg.get("enabled", False):
                continue
            try:
                self._refresh_asset_momentum(asset_name, asset_cfg)
            except Exception as exc:
                logger.debug(f"[WATCHER] Momentum update failed for {asset_name}: {exc}")

    def _refresh_asset_momentum(self, asset_name: str, asset_cfg: dict):
        exchange = asset_cfg.get("exchange", "mt5")

        # ── Fetch last N+1 closed candles ────────────────────────────────────
        df = None
        try:
            end   = datetime.now(timezone.utc)
            start = end - timedelta(hours=(_MOMENTUM_CANDLES + 2) * 2)  # buffer
            if exchange == "binance" and self.binance_handler:
                from src.data.data_manager import DataManager
                dm: DataManager = getattr(self.binance_handler, "data_manager", None)
                if dm:
                    symbol = asset_cfg.get("symbol", asset_name)
                    df = dm.fetch_binance_data(
                        symbol=symbol,
                        interval=asset_cfg.get("interval", "1h"),
                        start_date=start.strftime("%Y-%m-%d"),
                        end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                    )
            elif exchange == "mt5" and self.mt5_handler:
                symbol = self.mt5_handler._resolve_symbol(asset_name)
                if symbol:
                    tf = asset_cfg.get("timeframe", "H1")
                    from src.data.data_manager import DataManager
                    dm: DataManager = getattr(self.mt5_handler, "data_manager", None)
                    if dm:
                        df = dm.fetch_mt5_data(
                            symbol=symbol, timeframe=tf,
                            start_date=start.strftime("%Y-%m-%d"),
                            end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                        )
        except Exception as exc:
            logger.debug(f"[WATCHER] Candle fetch failed for {asset_name}: {exc}")

        if df is None or df.empty or len(df) < 3:
            return

        # Normalise columns
        df.columns = [c.lower() for c in df.columns]

        # Drop the still-forming candle (last row)
        closed = df.iloc[-(_MOMENTUM_CANDLES + 1):-1].copy()
        if len(closed) < _MOMENTUM_CANDLES:
            return

        # ── ATR (14-period on full df) ────────────────────────────────────────
        atr_val = 0.0
        try:
            import talib
            full_high  = df["high"].values.astype(float)
            full_low   = df["low"].values.astype(float)
            full_close = df["close"].values.astype(float)
            atr_arr = talib.ATR(full_high, full_low, full_close, timeperiod=14)
            atr_val = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
        except Exception:
            # Fallback: mean true range of closed candles
            tr = np.maximum(
                closed["high"].values - closed["low"].values,
                np.abs(closed["high"].values - closed["close"].shift(1).fillna(closed["close"]).values)
            )
            atr_val = float(np.mean(tr)) if len(tr) > 0 else 0.0

        self._atr_cache[asset_name] = atr_val

        # ── Direction score ───────────────────────────────────────────────────
        # +1 per bullish candle (close > open), -1 per bearish
        candle_dirs = []
        for _, row in closed.iterrows():
            candle_dirs.append(1 if row["close"] >= row["open"] else -1)

        direction_score = sum(candle_dirs)   # range: -N..+N
        # Require all 3 candles unanimous before calling momentum directional.
        # 2/3 is too easy to trigger on a normal shallow pullback inside a trend;
        # we only want to suppress signals when the short-term move is unambiguous.
        if direction_score >= _MOMENTUM_CANDLES:
            momentum_direction = 1    # all candles bullish
        elif direction_score <= -_MOMENTUM_CANDLES:
            momentum_direction = -1   # all candles bearish
        else:
            momentum_direction = 0    # mixed / normal retracement

        # Current price (last close in df)
        last_price = float(df["close"].iloc[-1])

        with self._suppress_lock:
            self._momentum[asset_name] = {
                "direction":  momentum_direction,
                "atr":        atr_val,
                "last_price": last_price,
                "updated_at": datetime.now(timezone.utc),
                "candle_dirs": candle_dirs,
            }

        # ── Signal suppression update ─────────────────────────────────────────
        self._update_suppression(asset_name, momentum_direction, atr_val, last_price)

    def _update_suppression(
        self,
        asset_name:         str,
        momentum_direction: int,
        atr_val:            float,
        current_price:      float,
    ):
        """
        Decide whether to suppress a new SELL or BUY signal for this asset.

        Suppression is triggered when:
          - The proposed signal direction DISAGREES with the last 3-candle momentum
          - AND price has bounced > _SUPPRESS_ATR_BOUNCE × ATR against the
            most recent open position's entry (so we don't just suppress based on a
            single choppy candle — we need actual price displacement too).
        """
        positions = self.portfolio_manager.get_asset_positions(asset_name) \
                    if self.portfolio_manager else []

        suppress_sell = False
        suppress_buy  = False
        suppress_reason = ""

        # ── Case 1a: existing SHORT + candles bullish + price bounced above entry ──
        # Suppress additional SELL signals when an open short is already being
        # hit by adverse price action. Prevents doubling into a losing short.
        short_positions = [p for p in positions if p.side == "short"]
        if short_positions and momentum_direction == 1 and atr_val > 0:
            ref_price = max(p.entry_price for p in short_positions)
            bounce    = current_price - ref_price   # positive = price went UP vs entry
            if bounce > _SUPPRESS_ATR_BOUNCE * atr_val:
                suppress_sell   = True
                suppress_reason = (
                    f"Price bounced +{bounce:.4g} ({bounce/atr_val:.1f}× ATR) "
                    f"above short entry ${ref_price:.4g} while momentum is bullish"
                )

        # ── Case 1b: existing LONG + candles bearish + price dropped below entry ──
        # Symmetric protection: suppress additional BUY signals when an open long
        # is already being hit adversely downward.
        long_positions = [p for p in positions if p.side == "long"]
        if not suppress_buy and long_positions and momentum_direction == -1 and atr_val > 0:
            ref_price = min(p.entry_price for p in long_positions)
            drop      = ref_price - current_price   # positive = price went DOWN vs entry
            if drop > _SUPPRESS_ATR_BOUNCE * atr_val:
                suppress_buy    = True
                suppress_reason = (
                    f"Price dropped -{drop:.4g} ({drop/atr_val:.1f}× ATR) "
                    f"below long entry ${ref_price:.4g} while momentum is bearish"
                )

        # NOTE: Context-free suppression (old Case 2 — blocking any SELL when
        # candles are bullish or any BUY when candles are bearish) has been
        # removed. That logic was killing legitimate pullback re-entries and
        # trend continuation setups where 3 corrective candles form before the
        # real move. The CMR gate in signal_aggregator and council_aggregator
        # already handles momentum-vs-signal disagreement with full regime
        # context. The MarketWatcher's suppression role is position protection
        # only (Cases 1a / 1b above).

        with self._suppress_lock:
            new_state: dict = {"suppressed": False, "reason": "", "direction": 0}

            if suppress_sell:
                new_state = {
                    "suppressed": True,
                    "reason":     suppress_reason,
                    "direction":  -1,   # suppress SELL
                }
            elif suppress_buy:
                new_state = {
                    "suppressed": True,
                    "reason":     suppress_reason,
                    "direction":  1,    # suppress BUY
                }

            old_state = self._suppression.get(asset_name, {})
            self._suppression[asset_name] = new_state

            # Log state changes only
            was_suppressed = old_state.get("suppressed", False)
            now_suppressed = new_state["suppressed"]
            if was_suppressed != now_suppressed:
                if now_suppressed:
                    logger.warning(
                        f"[WATCHER] 🛑 {asset_name} signal SUPPRESSED "
                        f"({'SELL' if new_state['direction'] == -1 else 'BUY'}): "
                        f"{suppress_reason}"
                    )
                else:
                    logger.info(
                        f"[WATCHER] ✅ {asset_name} signal suppression LIFTED — "
                        f"momentum realigned"
                    )

    # ── Open-position protection ───────────────────────────────────────────────

    def _check_open_positions(self):
        """
        For every tracked open position, compute how far price has moved
        adversely since entry and act proportionally.
        """
        if not self.portfolio_manager:
            return

        for position_id, position in list(self.portfolio_manager.positions.items()):
            asset_name = position.asset
            try:
                self._guard_position(position)
            except Exception as exc:
                logger.debug(
                    f"[WATCHER] Position guard error for {asset_name}: {exc}"
                )

    def _guard_position(self, position):
        asset_name = position.asset
        asset_cfg  = self.config.get("assets", {}).get(asset_name, {})
        if not asset_cfg.get("enabled", False):
            return

        exchange = asset_cfg.get("exchange", "mt5")
        handler  = self.mt5_handler if exchange == "mt5" else self.binance_handler
        if not handler:
            return

        # ── Get live price ────────────────────────────────────────────────────
        current_price = None
        try:
            if exchange == "mt5":
                symbol = handler._resolve_symbol(asset_name)
                if symbol:
                    current_price = handler.get_current_price(
                        symbol=symbol, force_live=True
                    )
            else:
                current_price = handler.get_current_price(asset_name)
        except Exception:
            return

        if not current_price or current_price <= 0:
            return

        atr = self._atr_cache.get(asset_name, 0.0)
        if atr <= 0:
            return

        # ── Adverse move from entry ───────────────────────────────────────────
        if position.side == "short":
            adverse = current_price - position.entry_price   # positive = bad for short
        else:
            adverse = position.entry_price - current_price   # positive = bad for long

        adverse_in_atr = adverse / atr

        # ── Determine alert level ────────────────────────────────────────────
        if adverse_in_atr >= _ADVERSE_ATR_CLOSE:
            self._handle_extreme_adverse(position, current_price, adverse, atr, handler, asset_name)
        elif adverse_in_atr >= _ADVERSE_ATR_WARN:
            self._handle_warn_adverse(position, current_price, adverse, atr, asset_name)

    def _handle_warn_adverse(
        self, position, current_price: float, adverse: float, atr: float, asset_name: str
    ):
        """
        Moderate adverse move: tighten SL to breakeven and alert.
        """
        pos_id = position.position_id
        now    = time.time()

        # Rate-limit alerts
        if now - self._last_alert.get(pos_id + "_warn", 0) < _ALERT_COOLDOWN_SEC:
            return

        self._last_alert[pos_id + "_warn"] = now

        # Compute breakeven SL
        entry = position.entry_price
        be_sl = entry * 1.001 if position.side == "short" else entry * 0.999  # tiny buffer

        logger.warning(
            f"[WATCHER] ⚠️  {asset_name} {position.side.upper()} adverse move "
            f"{adverse:.4g} ({adverse/atr:.1f}×ATR). "
            f"Attempting SL → breakeven ${be_sl:.4g}"
        )

        # Push SL to breakeven on the exchange
        self._push_sl(position, be_sl, asset_name)

        # Telegram alert
        self._send_alert(
            f"⚠️ <b>MARKET WATCHER</b>\n"
            f"{asset_name} {position.side.upper()} adverse move: "
            f"<b>{adverse:.4g}</b> ({adverse/atr:.1f}× ATR)\n"
            f"Tightened SL → breakeven <b>${be_sl:.4g}</b>\n"
            f"Entry: ${position.entry_price:.4g} | Now: ${current_price:.4g}"
        )

    def _handle_extreme_adverse(
        self, position, current_price: float, adverse: float, atr: float,
        handler, asset_name: str
    ):
        """
        Extreme adverse move (crazy candle): tighten SL hard to current price
        minus a small buffer. This forces an imminent close if price keeps going.
        """
        pos_id = position.position_id
        now    = time.time()

        if now - self._last_alert.get(pos_id + "_extreme", 0) < _ALERT_COOLDOWN_SEC:
            return

        self._last_alert[pos_id + "_extreme"] = now

        # SL = current price + 0.2 ATR in adverse direction (tight but not immediate fill)
        if position.side == "short":
            emergency_sl = current_price + 0.20 * atr   # stop just above current price
        else:
            emergency_sl = current_price - 0.20 * atr   # stop just below current price

        logger.error(
            f"[WATCHER] 🚨 {asset_name} {position.side.upper()} EXTREME adverse move "
            f"{adverse:.4g} ({adverse/atr:.1f}×ATR)! "
            f"Pushing emergency SL → ${emergency_sl:.4g}"
        )

        self._push_sl(position, emergency_sl, asset_name)

        self._send_alert(
            f"🚨 <b>MARKET WATCHER — EMERGENCY</b>\n"
            f"{asset_name} {position.side.upper()} <b>EXTREME</b> adverse move!\n"
            f"Move: <b>{adverse:.4g}</b> ({adverse/atr:.1f}× ATR)\n"
            f"Emergency SL pushed → <b>${emergency_sl:.4g}</b>\n"
            f"Entry: ${position.entry_price:.4g} | Now: ${current_price:.4g}\n"
            f"⚠️ Consider closing manually if market continues against you."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _push_sl(self, position, new_sl: float, asset_name: str):
        """Push a new SL to the exchange for an MT5 position."""
        if not position.mt5_ticket:
            return
        if not self.mt5_handler:
            return
        try:
            symbol = self.mt5_handler._resolve_symbol(asset_name)
            if symbol:
                self.mt5_handler._push_sl_to_exchange(
                    position.mt5_ticket, symbol, new_sl
                )
                logger.info(
                    f"[WATCHER] SL pushed → ${new_sl:.4g} for "
                    f"{asset_name} ticket #{position.mt5_ticket}"
                )
                # Update position object so VTM doesn't fight us
                if position.trade_manager:
                    position.trade_manager.current_stop_loss = new_sl
        except Exception as exc:
            logger.error(f"[WATCHER] SL push failed for {asset_name}: {exc}")

    def _send_alert(self, message: str):
        """Fire a Telegram alert if available."""
        if not self.telegram_bot or not self._send_telegram:
            return
        try:
            if hasattr(self.telegram_bot, "_is_ready") and not self.telegram_bot._is_ready:
                return
            coro = self.telegram_bot.send_message(text=message, parse_mode="HTML")
            if coro:
                self._send_telegram(coro)
        except Exception as exc:
            logger.debug(f"[WATCHER] Telegram alert failed: {exc}")
