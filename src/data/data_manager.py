"""
✅ ENHANCED DataManager - Hybrid Live/Testnet Strategy
Key features:
1. Uses LIVE API for historical data (full history available)
2. Uses TESTNET for trade execution (safe testing)
3. Automatically switches between contexts
4. No API keys needed for historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from typing import Optional, Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from src.market.price_cache import price_cache

import time as _time_module

logger = logging.getLogger(__name__)


def _sync_time_offset(client: Client) -> None:
    """
    Correct the client's timestamp by comparing local clock to Binance server time.
    Fixes APIError -1021 (timestamp outside recvWindow) caused by clock drift.
    Sets client.timestamp_offset (milliseconds) which python-binance adds to every request.
    """
    try:
        local_before = int(_time_module.time() * 1000)
        server_time  = client.get_server_time()["serverTime"]
        local_after  = int(_time_module.time() * 1000)
        # Use midpoint of the round-trip to reduce network-latency bias
        local_mid = (local_before + local_after) // 2
        offset = server_time - local_mid
        client.timestamp_offset = offset
        logger.info(f"[TIME SYNC] Clock offset corrected: {offset:+d} ms")
    except Exception as e:
        logger.warning(f"[TIME SYNC] Failed to sync Binance time offset: {e}")


CLOUDFRONT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Origin": "https://www.binance.com",
    "Referer": "https://www.binance.com/",
}


class DataManager:
    """Unified data manager with smart Live/Testnet separation"""

    def __init__(self, config: Dict):
        self.config = config
        self.binance_client = None  # Primary client (testnet or live)
        self.live_data_client = None  # Always live for historical data
        self.futures_client = None  # For Futures trading (separate)
        self.mt5_initialized = False

    def initialize_binance(self) -> bool:
        """
        ✅ ENHANCED: Initialize with smart Live/Testnet separation
        ✅ FIXED: Added User-Agent to prevent CloudFront 403 errors.

        Strategy:
        1. Primary client = testnet (for safe trade execution)
        2. Live data client = always live public API (full historical data)
        3. Futures client = separate if provided
        """
        try:
            api_config = self.config["api"]["binance"]
            api_key = api_config.get("api_key", "")
            api_secret = api_config.get("api_secret", "")

            logger.info("=" * 70)
            logger.info("INITIALIZING BINANCE API (HYBRID MODE)")
            logger.info("=" * 70)

            # ============================================================
            # STEP 1: Initialize LIVE DATA CLIENT (for historical data)
            # ============================================================
            logger.info(
                "\n📊 Initializing LIVE data client (for historical analysis)..."
            )
            logger.info("   Purpose: Fetch complete historical OHLCV data")
            logger.info("   Endpoint: https://api.binance.com (LIVE - Public API)")

            # Always create a live client for data (no keys needed = public access)
            self.live_data_client = Client("", "", requests_params={'timeout': 10})
            self.live_data_client.session.headers.update(CLOUDFRONT_HEADERS)
            _sync_time_offset(self.live_data_client)
            logger.info(
                f"   [User-Agent] Live Data Client: {self.live_data_client.session.headers.get('User-Agent')}"
            )

            try:
                self.live_data_client.ping()
                logger.info("✅ Live data API connected successfully")
                logger.info("   Note: This is PUBLIC access (read-only, no trading)")
            except Exception as e:
                logger.warning(f"⚠️  Live API connection issue: {e}")
                logger.warning("   Will fall back to testnet (limited history)")
                self.live_data_client = None

            # ============================================================
            # STEP 2: Initialize PRIMARY CLIENT (testnet or live)
            # ============================================================
            if not api_key or not api_secret or api_key == "YOUR_BINANCE_API_KEY":
                logger.warning("\n⚠️  Binance API keys not configured")
                logger.info("   Using public API only (no trading capability)")
                self.binance_client = Client("", "")
                self.binance_client.session.headers.update(CLOUDFRONT_HEADERS)
                logger.info(
                    f"   [User-Agent] Primary Client (Public): {self.binance_client.session.headers.get('User-Agent')}"
                )
                logger.info("✅ Public Spot API initialized")
                return True

            is_testnet = api_config.get("testnet", True)

            logger.info(
                f"\n🔧 Initializing PRIMARY client ({'TESTNET' if is_testnet else 'LIVE'})..."
            )
            logger.info("   Purpose: Trade execution and account management")

            if is_testnet:
                self.binance_client = Client(api_key, api_secret, testnet=True, requests_params={'timeout': 10})
                self.binance_client.session.headers.update(CLOUDFRONT_HEADERS)
                logger.info(
                    f"   [User-Agent] Primary Client (Testnet): {self.binance_client.session.headers.get('User-Agent')}"
                )
                self.binance_client.API_URL = "https://testnet.binance.vision/api"
                logger.info("   Endpoint: https://testnet.binance.vision/api (testnet)")
                logger.warning("   ⚠️  Testnet has LIMITED historical data (~2 days)")
                logger.info("   💡 Historical analysis will use live API automatically")
            else:
                self.binance_client = Client(api_key, api_secret, requests_params={'timeout': 10})
                self.binance_client.session.headers.update(CLOUDFRONT_HEADERS)
                logger.info(
                    f"   [User-Agent] Primary Client (Live): {self.binance_client.session.headers.get('User-Agent')}"
                )
                logger.info("   Endpoint: https://api.binance.com (LIVE)")
                logger.warning("   ⚠️  WARNING: LIVE TRADING MODE - REAL MONEY AT RISK")
            _sync_time_offset(self.binance_client)

            # Test primary connection
            self.binance_client.ping()
            logger.info("✅ Primary API connected successfully")

            # ============================================================
            # STEP 3: Initialize FUTURES CLIENT (if separate keys exist)
            # ============================================================
            futures_config = self.config.get("api", {}).get("binance_futures")

            if futures_config:
                logger.info("\n🚀 Initializing FUTURES client (separate keys)...")

                futures_key = futures_config.get("api_key", "")
                futures_secret = futures_config.get("api_secret", "")

                if futures_key and futures_secret:
                    if futures_config.get("testnet", True):
                        self.futures_client = Client(
                            futures_key, futures_secret, testnet=True, requests_params={'timeout': 10}
                        )
                        self.futures_client.session.headers.update(CLOUDFRONT_HEADERS)
                        logger.info(
                            f"   [User-Agent] Futures Client (Testnet): {self.futures_client.session.headers.get('User-Agent')}"
                        )
                        self.futures_client.API_URL = (
                            "https://testnet.binancefuture.com"
                        )
                        logger.info(
                            "   Endpoint: https://testnet.binancefuture.com (testnet)"
                        )
                    else:
                        self.futures_client = Client(futures_key, futures_secret, requests_params={'timeout': 10})
                        self.futures_client.session.headers.update(CLOUDFRONT_HEADERS)
                        logger.info(
                            f"   [User-Agent] Futures Client (Live): {self.futures_client.session.headers.get('User-Agent')}"
                        )
                        self.futures_client.API_URL = "https://fapi.binance.com"
                        logger.info("   Endpoint: https://fapi.binance.com (LIVE)")
                        logger.warning("   ⚠️  LIVE FUTURES TRADING - HIGH RISK")

                    _sync_time_offset(self.futures_client)

                    # Test Futures connection
                    try:
                        self.futures_client.futures_ping()
                        logger.info("✅ Futures API connected successfully")
                    except Exception as e:
                        logger.warning(f"⚠️  Futures API test failed: {e}")
                        logger.warning("   Will fall back to Spot keys for Futures")
                        self.futures_client = None
                else:
                    logger.info(
                        "⚠️  Futures keys incomplete, will use Spot keys if needed"
                    )
            else:
                logger.info("\n📝 No separate Futures config found")
                logger.info("   Will use Spot keys for Futures trading (if enabled)")

            # ============================================================
            # SUMMARY
            # ============================================================
            logger.info("\n" + "=" * 70)
            logger.info("BINANCE INITIALIZATION COMPLETE - HYBRID MODE")
            logger.info("=" * 70)
            logger.info(
                f"Live Data Client:  {'✅ Active (full history)' if self.live_data_client else '❌ Inactive'}"
            )
            logger.info(
                f"Primary Client:    {'✅ Active (testnet)' if is_testnet else '✅ Active (LIVE)'}"
            )
            logger.info(
                f"Futures Client:    {'✅ Active (separate keys)' if self.futures_client else '📝 Will use primary'}"
            )
            logger.info("\n📋 OPERATIONAL MODE:")
            logger.info("   • Historical data → Live API (full history)")
            logger.info(
                "   • Trade execution → "
                + ("Testnet (safe)" if is_testnet else "LIVE (real money)")
            )
            logger.info("=" * 70 + "\n")

            return True

        except BinanceAPIException as e:
            logger.error(f"❌ Binance API error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Binance API: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def get_futures_client(self) -> Optional[Client]:
        """
        Get the appropriate client for Futures trading
        Returns separate Futures client if available, otherwise primary client
        """
        if self.futures_client:
            return self.futures_client
        elif self.binance_client:
            logger.debug("Using primary client for Futures (no separate keys)")
            return self.binance_client
        else:
            return None

    def _get_data_client(self, prefer_live: bool = True) -> Client:
        """
        ✅ NEW: Smart client selection for data fetching

        Args:
            prefer_live: If True, use live client for full historical data

        Returns:
            Best available client for data fetching
        """
        if prefer_live and self.live_data_client:
            logger.debug("Using live API for historical data (full history available)")
            return self.live_data_client
        elif self.binance_client:
            logger.debug("Using primary client for data")
            return self.binance_client
        else:
            raise RuntimeError("No Binance client available")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _fetch_klines_with_retry(
        self,
        client: Client,
        symbol: str,
        interval: str,
        startTime: int,
        endTime: int,
        limit: int,
    ):
        """Fetch klines with retry logic"""
        return client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=startTime,
            endTime=endTime,
            limit=limit,
        )

    def fetch_binance_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
        limit: int = 1000,
        use_live_for_history: bool = True,
    ) -> pd.DataFrame:
        """
        ✅ ENHANCED: Fetch historical OHLCV data with smart client selection

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date (format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
            end_date: End date (optional, defaults to now)
            limit: Bars per request (max 1000)
            use_live_for_history: If True, prefer live API for full historical data

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Parse datetime strings
            start_dt = pd.to_datetime(start_date)
            if start_dt.tz is None:
                start_dt = start_dt.tz_localize("UTC")
            else:
                start_dt = start_dt.tz_convert("UTC")

            if end_date:
                end_dt = pd.to_datetime(end_date)
                if end_dt.tz is None:
                    end_dt = end_dt.tz_localize("UTC")
                else:
                    end_dt = end_dt.tz_convert("UTC")
            else:
                end_dt = pd.Timestamp.now(tz="UTC")

            start_ts = int(start_dt.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)

            # ✅ SMART CLIENT SELECTION
            # For historical analysis, prefer live API to ensure real market prices
            # regardless of whether we are in testnet for trading.
            days_requested = (end_dt - start_dt).days
            use_live = use_live_for_history
            
            client = self._get_data_client(prefer_live=use_live)

            if use_live and self.live_data_client:
                logger.info(f"📊 Fetching {symbol} from LIVE API (full history)")
            else:
                logger.info(
                    f"📊 Fetching {symbol} from {'testnet' if self.config['api']['binance'].get('testnet', True) else 'live'} API"
                )

            logger.info(f"   Period: {start_dt} to {end_dt} (UTC)")
            logger.info(f"   Requested: {days_requested} days of data")

            all_klines = []
            current_start = start_ts
            max_iterations = 100
            iteration = 0

            while current_start < end_ts and iteration < max_iterations:
                iteration += 1

                try:
                    klines = self._fetch_klines_with_retry(
                        client=client,
                        symbol=symbol,
                        interval=interval,
                        startTime=current_start,
                        endTime=end_ts,
                        limit=limit,
                    )
                except Exception as e:
                    logger.error(f"Error fetching batch {iteration}: {e}")
                    break

                if not klines:
                    logger.info(f"No more data available at timestamp {current_start}")
                    break

                all_klines.extend(klines)
                last_timestamp = klines[-1][0]

                if last_timestamp <= current_start:
                    logger.warning("Duplicate timestamp detected, stopping")
                    break

                current_start = last_timestamp + 1

                logger.debug(
                    f"Batch {iteration}: Fetched {len(klines)} bars, total: {len(all_klines)}"
                )

                if len(klines) < limit:
                    logger.debug("Reached end of available data")
                    break

                if last_timestamp >= end_ts:
                    logger.debug("Reached requested end timestamp")
                    break

            if iteration >= max_iterations:
                logger.warning(f"Stopped after {max_iterations} iterations")

            if not all_klines:
                logger.error(f"❌ No data received for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(
                all_klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df = df.set_index("timestamp")

            # Remove duplicates
            df = df[~df.index.duplicated(keep="first")]
            df = df.sort_index()

            logger.info(f"✅ Successfully fetched {len(df)} bars for {symbol}")
            logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")

            # Validate data range
            days_received = (df.index[-1] - df.index[0]).days

            if df.index[0] > start_dt:
                missing_days = (df.index[0] - start_dt).days
                if missing_days > 1:
                    logger.warning(
                        f"   ⚠️  Data starts at {df.index[0]}, requested {start_dt}"
                    )
                    logger.warning(f"   Missing {missing_days} days at beginning")

                    # Suggest switching to live API if using testnet
                    if not use_live and self.live_data_client:
                        logger.info(
                            "   💡 TIP: Use fetch_binance_data(..., use_live_for_history=True)"
                        )
                        logger.info("      for complete historical data")

            if df.index[-1] < end_dt:
                missing_hours = (end_dt - df.index[-1]).total_seconds() / 3600
                if missing_hours > 2:
                    logger.warning(
                        f"   ⚠️  Data ends at {df.index[-1]}, requested {end_dt}"
                    )
                    logger.warning(f"   Missing {missing_hours:.1f} hours at end")

            coverage_pct = (days_received / max(days_requested, 1)) * 100
            logger.info(
                f"   Coverage: {days_received}/{days_requested} days ({coverage_pct:.1f}%)"
            )

            # Update the price cache with the latest close
            if not df.empty:
                last_close = df['close'].iloc[-1]
                price_cache.set(symbol, last_close)
                logger.info(f"[CACHE] Price cache updated with last kline close: {last_close}")

            return self.clean_data(df)

        except Exception as e:
            logger.error(f"Error fetching Binance data: {e}", exc_info=True)
            return pd.DataFrame()

    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection with detailed diagnostics"""
        try:
            import MetaTrader5 as mt5

            if "api" not in self.config or "mt5" not in self.config["api"]:
                logger.warning("MT5 configuration not found. Skipping.")
                return False

            mt5_config = self.config["api"]["mt5"]

            path = mt5_config.get("path")
            login = mt5_config.get("login")
            password = mt5_config.get("password")
            server = mt5_config.get("server")

            if path == "null" or path == "None":
                path = None

            if not all([login, password, server]):
                logger.warning("MT5 credentials incomplete. Skipping.")
                return False

            logger.info("=" * 60)
            logger.info("MT5 INITIALIZATION")
            logger.info("=" * 60)
            logger.info(f"Path: {path if path else 'Auto-detect'}")
            logger.info(f"Login: {login}")
            logger.info(f"Server: {server}")

            if not path:
                init_result = mt5.initialize()
            else:
                init_result = mt5.initialize(
                    path=path,
                    login=int(login),
                    password=str(password),
                    server=str(server),
                )

            if not init_result:
                error = mt5.last_error()
                logger.error(f"MT5 initialize() failed: {error}")
                return False

            authorized = mt5.login(
                login=int(login), password=str(password), server=str(server)
            )

            if not authorized:
                error = mt5.last_error()
                logger.error(f"MT5 login failed: {error}")
                mt5.shutdown()
                return False

            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Cannot get account info")
                mt5.shutdown()
                return False

            logger.info(f"✅ MT5 connected: {account_info.login}")
            logger.info(f"   Balance: ${account_info.balance:.2f}")
            logger.info("=" * 60)

            self.mt5_initialized = True
            return True

        except ImportError:
            logger.error("MetaTrader5 module not installed")
            return False
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False

    def fetch_mt5_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str] = None,
        count: int = 10000,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from MT5"""
        if not self.mt5_initialized:
            raise RuntimeError("MT5 not initialized")

        try:
            import MetaTrader5 as mt5

            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }

            tf = timeframe_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)

            start_dt = pd.to_datetime(start_date)
            if end_date:
                end_dt = pd.to_datetime(end_date)
            else:
                end_dt = datetime.now()

            start_dt = (
                start_dt.to_pydatetime()
                if hasattr(start_dt, "to_pydatetime")
                else start_dt
            )
            end_dt = (
                end_dt.to_pydatetime() if hasattr(end_dt, "to_pydatetime") else end_dt
            )

            logger.info(f"Fetching {symbol} from MT5: {start_dt} to {end_dt}")

            rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)

            if rates is None or len(rates) == 0:
                logger.error(f"No MT5 data for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df[["timestamp", "open", "high", "low", "close", "tick_volume"]]
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            df = df.set_index("timestamp")

            logger.info(f"✅ Fetched {len(df)} bars from MT5")
            
            # Update the price cache with the latest close
            if not df.empty:
                last_close = df['close'].iloc[-1]
                price_cache.set(symbol, last_close)
                logger.info(f"[CACHE] Price cache updated with last kline close from MT5: {last_close}")

            return self.clean_data(df)

        except ImportError:
            logger.error("MetaTrader5 module not installed")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"MT5 data fetch error: {e}")
            return pd.DataFrame()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data"""
        if df.empty:
            return df

        df = df.copy()
        initial_len = len(df)

        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        if df.isnull().any().any():
            df = df.ffill().dropna()

        # ✨ ENHANCED VALIDATION: Ensure no corrupted inputs
        # 1. Price must be > 0
        # 2. Volume must be >= 0
        # 3. Structure must be logical (high >= low, etc.)
        invalid_bars = (
            (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
            | (df["volume"] < 0)
            | (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )

        if invalid_bars.any():
            logger.warning(f"Removing {invalid_bars.sum()} invalid bars (corrupted data detected)")
            df = df[~invalid_bars]

        final_len = len(df)
        if initial_len - final_len > 0:
            logger.info(f"Cleaned: removed {initial_len - final_len} bars")

        return df

    def split_train_test(
        self, df: pd.DataFrame, train_pct: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data chronologically"""
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        split_idx = int(len(df) * train_pct)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(f"Train: {len(train_df)} bars ({train_pct*100:.0f}%)")
        logger.info(f"Test: {len(test_df)} bars ({(1-train_pct)*100:.0f}%)")

        return train_df, test_df

    def get_latest_data(
        self, symbol: str, interval: str, lookback_bars: int = 500
    ) -> pd.DataFrame:
        """
        Get latest data for live trading
        Uses primary client (testnet or live) for current data
        """
        try:
            interval_map = {
                "1m": timedelta(minutes=lookback_bars),
                "5m": timedelta(minutes=5 * lookback_bars),
                "15m": timedelta(minutes=15 * lookback_bars),
                "1h": timedelta(hours=lookback_bars),
                "4h": timedelta(hours=4 * lookback_bars),
                "1d": timedelta(days=lookback_bars),
            }

            delta = interval_map.get(interval, timedelta(hours=lookback_bars))
            start_date = (datetime.now(timezone.utc) - delta).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            # For live trading, use primary client (respect testnet setting)
            df = self.fetch_binance_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                use_live_for_history=True,  # Use primary client for current data
            )

            df = self.clean_data(df)

            if len(df) > lookback_bars:
                df = df.iloc[-lookback_bars:]

            return df

        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return pd.DataFrame()

    def shutdown(self):
        """Cleanup connections"""
        if self.mt5_initialized:
            try:
                import MetaTrader5 as mt5

                mt5.shutdown()
                logger.info("MT5 shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down MT5: {e}")

        self.binance_client = None
        self.live_data_client = None
        self.futures_client = None
