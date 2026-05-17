"""
market_hours.py - Market hours detection for multi-asset trading
Supports US Stock Market hours and Forex/Gold trading hours (Exness MT5)
"""

from datetime import datetime, time, timedelta
import pytz
from typing import Tuple, Optional


class MarketHours:
    """Multi-asset market hours helper"""

    # US Stock Market holidays 2025 + 2026
    US_MARKET_HOLIDAYS = [
        "2025-01-01",  # New Year's Day
        "2025-01-20",  # MLK Day
        "2025-02-17",  # Presidents Day
        "2025-04-18",  # Good Friday
        "2025-05-26",  # Memorial Day
        "2025-07-04",  # Independence Day
        "2025-09-01",  # Labor Day
        "2025-11-27",  # Thanksgiving
        "2025-12-25",  # Christmas
        # 2026
        "2026-01-01",  # New Year's Day
        "2026-01-19",  # MLK Day
        "2026-02-16",  # Presidents Day
        "2026-04-03",  # Good Friday
        "2026-05-25",  # Memorial Day
        "2026-07-03",  # Independence Day (observed)
        "2026-09-07",  # Labor Day
        "2026-11-26",  # Thanksgiving
        "2026-12-25",  # Christmas
    ]

    # US Stock Market hours (Eastern Time)
    US_MARKET_OPEN = time(9, 30)
    US_MARKET_CLOSE = time(16, 0)
    US_PRE_MARKET_START = time(4, 0)
    US_AFTER_HOURS_END = time(20, 0)

    # Forex/Gold Market hours (Sunday 22:00 GMT to Friday 22:00 GMT)
    # Most brokers including Exness follow this schedule
    FOREX_WEEK_OPEN_DAY = 6  # Sunday
    FOREX_WEEK_OPEN_HOUR = 22  # 22:00 GMT (Sunday evening)
    FOREX_WEEK_CLOSE_DAY = 4  # Friday
    FOREX_WEEK_CLOSE_HOUR = 22  # 22:00 GMT (Friday evening)

    @staticmethod
    def get_eastern_time() -> datetime:
        """Get current time in US/Eastern timezone"""
        eastern = pytz.timezone("US/Eastern")
        return datetime.now(eastern)

    @staticmethod
    def get_gmt_time() -> datetime:
        """Get current time in GMT/UTC timezone"""
        gmt = pytz.timezone("GMT")
        return datetime.now(gmt)

    @staticmethod
    def is_us_stock_market_day() -> bool:
        """Check if today is a US stock market day"""
        now = MarketHours.get_eastern_time()

        if now.weekday() >= 5:  # Weekend
            return False

        date_str = now.strftime("%Y-%m-%d")
        if date_str in MarketHours.US_MARKET_HOLIDAYS:
            return False

        return True

    @staticmethod
    def is_us_stock_market_open() -> bool:
        """Check if US stock market is currently open"""
        if not MarketHours.is_us_stock_market_day():
            return False

        now = MarketHours.get_eastern_time()
        current_time = now.time()

        return MarketHours.US_MARKET_OPEN <= current_time <= MarketHours.US_MARKET_CLOSE

    @staticmethod
    def is_forex_market_open() -> bool:
        """
        Check if Forex/Gold market is open (Exness schedule)
        Market is open from Sunday 22:00 GMT to Friday 22:00 GMT
        """
        now = MarketHours.get_gmt_time()
        current_day = now.weekday()  # 0=Monday, 6=Sunday
        current_hour = now.hour

        # Saturday is always closed
        if current_day == 5:  # Saturday
            return False

        # Sunday: open from 22:00 onwards
        if current_day == 6:  # Sunday
            return current_hour >= MarketHours.FOREX_WEEK_OPEN_HOUR

        # Monday to Thursday: always open
        if current_day < 4:  # Monday to Thursday
            return True

        # Friday: open until 22:00
        if current_day == 4:  # Friday
            return current_hour < MarketHours.FOREX_WEEK_CLOSE_HOUR

        return False

    @staticmethod
    def is_crypto_market_open() -> bool:
        """Crypto markets are always open 24/7"""
        return True

    @staticmethod
    def get_market_status(asset_type: str = "forex") -> Tuple[str, str]:
        """
        Get detailed market status for specific asset type
        
        Args:
            asset_type: "stocks", "forex", "crypto"
            
        Returns: (status, message)
            status: 'OPEN', 'EXTENDED', 'CLOSED'
        """
        asset_type = asset_type.lower()

        if asset_type == "crypto":
            return "OPEN", "Crypto market is always open (24/7)"

        elif asset_type == "forex" or asset_type == "gold":
            if MarketHours.is_forex_market_open():
                return "OPEN", "Forex/Gold market is OPEN"
            else:
                now = MarketHours.get_gmt_time()
                day_name = now.strftime("%A")
                
                if now.weekday() == 5:  # Saturday
                    return "CLOSED", f"Market closed - Weekend ({day_name})"
                elif now.weekday() == 6 and now.hour < 22:  # Sunday before 22:00
                    hours_until = 22 - now.hour
                    return "CLOSED", f"Market opens Sunday at 22:00 GMT (in ~{hours_until}h)"
                elif now.weekday() == 4 and now.hour >= 22:  # Friday after 22:00
                    return "CLOSED", f"Market closed - Weekend ({day_name})"
                else:
                    return "CLOSED", f"Market closed - Outside trading hours"

        elif asset_type == "stocks":
            now = MarketHours.get_eastern_time()

            if not MarketHours.is_us_stock_market_day():
                day_name = now.strftime("%A")
                if now.weekday() >= 5:
                    return "CLOSED", f"Market closed - Weekend ({day_name})"
                else:
                    return "CLOSED", "Market closed - Holiday"

            if MarketHours.is_us_stock_market_open():
                return "OPEN", "US Stock Market is OPEN"

            current_time = now.time()
            if MarketHours.US_PRE_MARKET_START <= current_time < MarketHours.US_MARKET_OPEN:
                return "EXTENDED", "Pre-market hours (4:00 AM - 9:30 AM ET)"
            elif MarketHours.US_MARKET_CLOSE < current_time <= MarketHours.US_AFTER_HOURS_END:
                return "EXTENDED", "After-hours trading (4:00 PM - 8:00 PM ET)"

            if current_time < MarketHours.US_PRE_MARKET_START:
                return "CLOSED", "Market closed - Before pre-market"
            else:
                return "CLOSED", "Market closed - After extended hours"

        return "CLOSED", "Unknown asset type"

    @staticmethod
    def time_until_market_open(asset_type: str = "forex") -> int:
        """
        Get seconds until next market open
        Returns 0 if market is currently open
        
        Args:
            asset_type: "stocks", "forex", "crypto"
        """
        asset_type = asset_type.lower()

        if asset_type == "crypto":
            return 0  # Always open

        if asset_type == "forex" or asset_type == "gold":
            if MarketHours.is_forex_market_open():
                return 0

            now = MarketHours.get_gmt_time()
            
            # If it's Sunday before 22:00, calculate time until 22:00 today
            if now.weekday() == 6 and now.hour < 22:
                next_open = now.replace(hour=22, minute=0, second=0, microsecond=0)
                return int((next_open - now).total_seconds())
            
            # Otherwise, find next Sunday 22:00
            days_until_sunday = (6 - now.weekday()) % 7
            if days_until_sunday == 0:  # Today is Sunday but after 22:00
                days_until_sunday = 7
            
            next_open = now + timedelta(days=days_until_sunday)
            next_open = next_open.replace(hour=22, minute=0, second=0, microsecond=0)
            
            return int((next_open - now).total_seconds())

        elif asset_type == "stocks":
            if MarketHours.is_us_stock_market_open():
                return 0

            now = MarketHours.get_eastern_time()
            
            # Target tomorrow if after market close
            if now.time() > MarketHours.US_MARKET_CLOSE:
                next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
                next_open += timedelta(days=1)
            else:
                next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

            # Skip weekends
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)

            # Skip holidays
            while next_open.strftime("%Y-%m-%d") in MarketHours.US_MARKET_HOLIDAYS:
                next_open += timedelta(days=1)

            return int((next_open - now).total_seconds())

        return 0

    @staticmethod
    def get_btc_session_quality() -> str:
        """
        ✅ TASK 23: Evaluates current session liquidity for BTC.
        Returns: "HIGH" or "LOW"
        """
        now = MarketHours.get_gmt_time()
        hour = now.hour
        day = now.weekday()

        # Weekend Gate (already implemented in should_trade, but here for safety)
        if day == 5 or (day == 4 and hour >= 22) or (day == 6 and hour < 22):
            return "LOW"

        # Asian Session (00:00 - 07:00 UTC) is generally lower liquidity for BTC institutional moves
        if 0 <= hour < 7:
            return "LOW"
            
        return "HIGH"

    @staticmethod
    def is_rollover_dead_zone() -> bool:
        """
        Detects the high-risk 'Rollover' period (21:30 - 23:30 UTC) on weekdays.
        This is when liquidity is lowest, spreads are highest, and price
        action is often directionless or prone to gaps.

        NOTE: Sunday is explicitly excluded — 22:00 UTC on Sunday is the Forex/Gold
        market OPEN on Exness (and most MT5 brokers), not a rollover period.
        Blocking Sunday 22:00–23:30 would eat the first 90 minutes of the week.
        """
        now = MarketHours.get_gmt_time()
        day = now.weekday()   # 0=Monday … 5=Saturday, 6=Sunday
        hour = now.hour
        minute = now.minute

        # Sunday: market is opening, not rolling over — never block
        if day == 6:
            return False

        # Saturday: market is closed anyway, no need to flag rollover
        if day == 5:
            return False

        # Mon–Fri: block the 21:30–23:30 UTC window
        if hour == 21 and minute >= 30:
            return True
        if hour == 22:
            return True
        if hour == 23 and minute < 30:
            return True

        return False

    @staticmethod
    def should_trade(asset_type: str = "forex") -> bool:
        """
        Simple check: should we be actively trading this asset right now?
        
        Args:
            asset_type: "stocks", "forex", "crypto", "gold", "btc"
        """
        asset_type = asset_type.lower()
        
        # Map common asset names to types
        if asset_type in ["btc", "bitcoin", "crypto", "eth", "ethereum"]:
            # ✅ Crypto is 24/7 - Removed Institutional Weekend Gate
            return True

        if asset_type in ["gold", "xauusd", "forex", "eur", "gbp", "usd", "usoil",
                          "eurusd", "eurjpy", "gbpusd", "gbpaud", "usdjpy"]:
            return MarketHours.is_forex_market_open()

        if asset_type in ["stocks", "spy", "qqq", "aapl", "ustec", "nas100", "us100"]:
            return MarketHours.is_us_stock_market_open()
        
        # Default to forex hours for unknown assets
        return MarketHours.is_forex_market_open()

    @staticmethod
    def get_next_market_open(asset_type: str = "forex") -> Optional[datetime]:
        """Get the next market open datetime for the asset type"""
        seconds = MarketHours.time_until_market_open(asset_type)
        
        if seconds == 0:
            return None  # Market is currently open
        
        if asset_type.lower() in ["forex", "gold"]:
            return MarketHours.get_gmt_time() + timedelta(seconds=seconds)
        else:
            return MarketHours.get_eastern_time() + timedelta(seconds=seconds)


    # ── Preferred entry sessions per asset (UTC hours) ────────────────────
    # Outside these windows liquidity thins, spreads widen, and signals
    # from 1H bars formed during low-volume periods are unreliable.
    # Key: asset name (uppercase). Value: list of (open_utc, close_utc) tuples.
    # An entry is allowed if the current UTC hour falls within ANY listed window.
    #
    # Session reference (UTC):
    #   Tokyo:   00:00 – 09:00
    #   London:  07:00 – 16:00
    #   New York:13:00 – 22:00
    #   Overlap: 13:00 – 16:00  ← highest liquidity for FX
    PREFERRED_SESSIONS: dict = {
        # GOLD: London + NY only. Asian session gaps are dangerous.
        "GOLD":   [(7, 21)],
        # Major FX: avoid dead Asian hours (00:00–06:00 UTC)
        "EURUSD": [(7, 21)],
        "GBPUSD": [(7, 21)],
        "EURJPY": [(7, 21)],
        # JPY pairs — Tokyo is fine, so allow Asian session too
        "USDJPY": [(0, 9), (7, 21)],
        # GBPAUD: best during London/Sydney overlap + London session.
        # Spreads blow out in late NY/Asian hours.
        "GBPAUD": [(0, 17)],
        # Indices and commodities: US session driven
        "USTEC":  [(13, 21)],
        "USOIL":  [(7, 21)],
        # BTC: 24/7 but enforce a minimum liquidity window
        "BTC":    [(0, 24)],  # no restriction by default
    }

    @staticmethod
    def is_preferred_session(asset_name: str) -> bool:
        """
        Returns True if the current UTC hour is within the preferred
        trading session for this asset.

        Called from check_market_hours() when session_filter_enabled is True
        in config.  Assets not in PREFERRED_SESSIONS are allowed through.
        """
        windows = MarketHours.PREFERRED_SESSIONS.get(asset_name.upper())
        if not windows:
            return True  # No restriction defined — allow

        now = MarketHours.get_gmt_time()
        utc_hour = now.hour

        for (open_h, close_h) in windows:
            if open_h < close_h:
                if open_h <= utc_hour < close_h:
                    return True
            else:
                # Overnight window (e.g. 22 – 06)
                if utc_hour >= open_h or utc_hour < close_h:
                    return True

        return False


# Convenience functions
def should_trade_btc() -> bool:
    """Check if BTC trading is allowed (Weekend Gate applied)"""
    return MarketHours.should_trade("btc")


def should_trade_gold() -> bool:
    """Check if GOLD trading is allowed (Forex hours)"""
    return MarketHours.should_trade("forex")


def should_trade_stocks() -> bool:
    """Check if stock trading is allowed"""
    return MarketHours.should_trade("stocks")
