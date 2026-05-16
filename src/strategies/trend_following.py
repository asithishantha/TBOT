# src/strategies/trend_following.py
"""
MULTI-TIMEFRAME Trend Following Strategy with 4H Context
Key improvements:
- 4H timeframe context for trend filtering
- Fixed lookforward for training label consistency (T1.2B)
- Volatility Squeeze Filter (T1.2C)
- Calibrated min_conditions threshold (T1.2A)
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """
     trend following with multi-timeframe analysis
    Uses 4H context to filter and validate 1H signals
    """

    def __init__(self, config: dict):
        super().__init__(config, "TrendFollowing")

        # Moving average periods
        self.fast_ma = config.get("fast_ma", 20)
        self.slow_ma = config.get("slow_ma", 50)

        # MACD parameters
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)

        # ADX parameters
        self.adx_period = config.get("adx_period", 14)
        self.adx_threshold = config.get("adx_threshold", 20)
        self.require_adx = config.get("require_adx", False)

        # 4H context parameters
        self.use_4h_context = config.get("use_4h_context", True)
        self.require_4h_alignment = config.get("require_4h_alignment", True)
        self.h4_trend_weight = config.get(
            "h4_trend_weight", 1.5
        )  # Bonus points for 4H alignment
        self.h4_counter_penalty = config.get(
            "h4_counter_penalty", 2.0
        )  # Penalty for counter-trend

        # Return thresholds
        self.min_return_threshold = config.get("min_return_threshold", 0.001)

        # Score threshold
        # ✅ T1.2A: Standardized to 2.5 to eliminate single-indicator noise
        self.min_score_threshold = config.get("min_conditions", 2.5)

        logger.info(f"[{self.name}] Initialized with:")
        logger.info(f"  Fast MA: {self.fast_ma}, Slow MA: {self.slow_ma}")
        logger.info(
            f"  ADX Threshold: {self.adx_threshold} (Required: {self.require_adx})"
        )
        logger.info(f"  Min Return: {self.min_return_threshold:.3%}")
        logger.info(f"  Min Score: {self.min_score_threshold}")

    def get_warmup_period(self) -> int:
        periods = [
            self.slow_ma,
            self.macd_slow + self.macd_signal,
            self.adx_period,
            100 # For bb_width_norm rolling quantile
        ]
        return max(periods)

    def generate_labels(self, df: pd.DataFrame, df_4h: pd.DataFrame = None) -> pd.Series:
        """Satisfies BaseStrategy requirements"""
        return self.generate_signal(mode='train', df=df, df_4h=df_4h)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend following features"""
        df = df.copy()

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # Moving Averages
        df["sma_fast"] = ta.SMA(close, timeperiod=self.fast_ma)
        df["sma_slow"] = ta.SMA(close, timeperiod=self.slow_ma)
        df["ema_fast"] = ta.EMA(close, timeperiod=self.fast_ma)
        df["ema_slow"] = ta.EMA(close, timeperiod=self.slow_ma)

        # MA relationships
        df["ma_diff"] = df["sma_fast"] - df["sma_slow"]
        df["ma_diff_pct"] = (df["sma_fast"] - df["sma_slow"]) / df["sma_slow"]

        # MA crossovers
        df["ma_cross"] = 0
        ma_diff_shift = df["ma_diff"].shift(1)
        df.loc[(df["ma_diff"] > 0) & (ma_diff_shift <= 0), "ma_cross"] = 1
        df.loc[(df["ma_diff"] < 0) & (ma_diff_shift >= 0), "ma_cross"] = -1

        # MACD
        macd, macd_signal, macd_hist = ta.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist

        # MACD crossovers
        df["macd_cross"] = 0
        macd_hist_shift = df["macd_hist"].shift(1)
        df.loc[(df["macd_hist"] > 0) & (macd_hist_shift <= 0), "macd_cross"] = 1
        df.loc[(df["macd_hist"] < 0) & (macd_hist_shift >= 0), "macd_cross"] = -1

        # ADX (trend strength)
        df["adx"] = ta.ADX(high, low, close, timeperiod=self.adx_period)
        df["strong_trend"] = (df["adx"] > self.adx_threshold).astype(int)

        # Directional indicators
        df["plus_di"] = ta.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        df["minus_di"] = ta.MINUS_DI(high, low, close, timeperiod=self.adx_period)
        df["di_diff"] = df["plus_di"] - df["minus_di"]

        # Price position
        df["close_above_fast"] = (close > df["sma_fast"]).astype(int)
        df["close_above_slow"] = (close > df["sma_slow"]).astype(int)

        # Momentum
        df["momentum"] = ta.MOM(close, timeperiod=10)
        df["roc"] = ta.ROC(close, timeperiod=10)

        # Trend features
        df["trend_strength"] = np.abs(df["ma_diff_pct"])
        df["price_vs_ma_fast"] = (close - df["sma_fast"]) / df["sma_fast"]
        df["price_vs_ma_slow"] = (close - df["sma_slow"]) / df["sma_slow"]

        # MA slopes
        df["ma_fast_slope"] = df["sma_fast"].diff(5) / df["sma_fast"]
        df["ma_slow_slope"] = df["sma_slow"].diff(10) / df["sma_slow"]

        # ✅ T1.2C: Bollinger Bands for Squeeze Filter
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        df["bb_width_norm"] = (bb_upper - bb_lower) / bb_middle

        return df

    def _align_4h_to_1h(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
        """Align 4H data to 1H timeframe using forward-fill"""
        if df_4h is None or df_4h.empty:
            return None

        if not isinstance(df_1h.index, pd.DatetimeIndex):
            df_1h = df_1h.set_index("timestamp")
        if not isinstance(df_4h.index, pd.DatetimeIndex):
            df_4h = df_4h.set_index("timestamp")

        # ✅ FIX: Handle timezone mismatch between 1H and 4H indices
        if df_1h.index.tz is not None:
            df_1h = df_1h.copy()
            df_1h.index = df_1h.index.tz_localize(None)
        if df_4h.index.tz is not None:
            df_4h = df_4h.copy()
            df_4h.index = df_4h.index.tz_localize(None)

        h4_features = [
            "sma_fast", "sma_slow", "adx", "macd_hist", "plus_di", "minus_di", "close"
        ]

        df_4h_aligned = pd.DataFrame(index=df_1h.index)

        for feature in h4_features:
            if feature in df_4h.columns:
                df_4h_aligned[f"h4_{feature}"] = df_4h[feature].reindex(
                    df_1h.index, method="ffill"
                )

        return df_4h_aligned

    def _calculate_4h_trend_score(self, df_4h_aligned: pd.DataFrame, idx: int) -> tuple:
        """Calculate 4H trend direction and strength"""
        if df_4h_aligned is None or idx >= len(df_4h_aligned):
            return 0.0, 0.0

        bullish_4h = 0.0
        bearish_4h = 0.0

        row = df_4h_aligned.iloc[idx]

        if pd.isna(row.get("h4_sma_fast")) or pd.isna(row.get("h4_sma_slow")):
            return 0.0, 0.0

        # 1. MA Alignment (0-2 points)
        if row["h4_sma_fast"] > row["h4_sma_slow"]:
            ma_sep = (row["h4_sma_fast"] - row["h4_sma_slow"]) / row["h4_sma_slow"]
            bullish_4h += 2.0 if ma_sep > 0.005 else 1.0
        elif row["h4_sma_fast"] < row["h4_sma_slow"]:
            ma_sep = (row["h4_sma_slow"] - row["h4_sma_fast"]) / row["h4_sma_slow"]
            bearish_4h += 2.0 if ma_sep > 0.005 else 1.0

        # 2. MACD (0-1 point)
        if not pd.isna(row.get("h4_macd_hist")):
            if row["h4_macd_hist"] > 0: bullish_4h += 1.0
            elif row["h4_macd_hist"] < 0: bearish_4h += 1.0

        # 3. Directional Indicators (0-1 point)
        if not pd.isna(row.get("h4_plus_di")) and not pd.isna(row.get("h4_minus_di")):
            if row["h4_plus_di"] > row["h4_minus_di"]: bullish_4h += 1.0
            elif row["h4_minus_di"] > row["h4_plus_di"]: bearish_4h += 1.0

        # 4. ADX Bonus (0-0.5 point)
        if not pd.isna(row.get("h4_adx")):
            if row["h4_adx"] > 25:
                if bullish_4h > bearish_4h: bullish_4h += 0.5
                elif bearish_4h > bullish_4h: bearish_4h += 0.5

        return bullish_4h, bearish_4h

    def _generate_training_labels(
        self, df: pd.DataFrame, df_4h: pd.DataFrame = None
    ) -> pd.Series:
        """
        MULTI-TIMEFRAME label generation with 4H context for TRAINING.
        """
        labels = pd.Series(0, index=df.index)
        close = df["close"].values
        sma_fast = df["sma_fast"].values
        sma_slow = df["sma_slow"].values
        adx = df["adx"].values
        macd_hist = df["macd_hist"].values
        plus_di = df["plus_di"].values
        minus_di = df["minus_di"].values
        bb_width_norm = df["bb_width_norm"].values

        # Align 4H data if provided
        df_4h_aligned = None
        if self.use_4h_context and df_4h is not None:
            if "sma_fast" not in df_4h.columns:
                df_4h = self.generate_features(df_4h)
            df_4h_aligned = self._align_4h_to_1h(df, df_4h)

        # ✅ T1.2B: Fixed lookforward period for homogeneous labels
        fixed_lookforward = 8

        for i in range(100, len(df) - fixed_lookforward - 1):
            if pd.isna(adx[i]): continue

            # === CALCULATE 1H TIMEFRAME SCORES ===
            bullish_score = 0.0
            bearish_score = 0.0

            # 1. MA Alignment (0-2 points)
            if sma_fast[i] > sma_slow[i]:
                ma_separation = (sma_fast[i] - sma_slow[i]) / sma_slow[i]
                bullish_score += 2.0 if ma_separation > 0.001 else 1.0
            elif sma_fast[i] < sma_slow[i]:
                ma_separation = (sma_slow[i] - sma_fast[i]) / sma_slow[i]
                bearish_score += 2.0 if ma_separation > 0.001 else 1.0

            # 2. MACD (0-1.5 points)
            if macd_hist[i] > 0:
                bullish_score += 1.5 if abs(macd_hist[i]) > 0 else 1.0
            elif macd_hist[i] < 0:
                bearish_score += 1.5 if abs(macd_hist[i]) > 0 else 1.0

            # 3. Directional Indicators (0-1.5 points)
            if plus_di[i] > minus_di[i]:
                bullish_score += 1.5 if (plus_di[i] - minus_di[i]) > 10 else 1.0
            elif minus_di[i] > plus_di[i]:
                bearish_score += 1.5 if (minus_di[i] - plus_di[i]) > 10 else 1.0

            # 4. Price Position (0-1 point)
            if close[i] > sma_fast[i]: bullish_score += 1.0
            elif close[i] < sma_fast[i]: bearish_score += 1.0

            # 5. ADX Bonus (0-1 point)
            if adx[i] > 25:
                if bullish_score > bearish_score: bullish_score += 1.0
                elif bearish_score > bullish_score: bearish_score += 1.0

            # ✅ T1.2C: Squeeze Filter Logic (Bonus/Penalty)
            recent_widths = bb_width_norm[max(0, i-100):i+1]
            if len(recent_widths) >= 50:
                lower_20 = np.percentile(recent_widths, 20)
                upper_70 = np.percentile(recent_widths, 70) # Top 30%
                
                if bb_width_norm[i] <= lower_20:
                    bullish_score += 1.0; bearish_score += 1.0
                elif bb_width_norm[i] >= upper_70:
                    bullish_score -= 0.5; bearish_score -= 0.5

            # === APPLY 4H CONTEXT ===
            if df_4h_aligned is not None:
                h4_bullish, h4_bearish = self._calculate_4h_trend_score(df_4h_aligned, i)
                if h4_bullish > h4_bearish:
                    bullish_score += self.h4_trend_weight
                    bearish_score -= self.h4_counter_penalty
                elif h4_bearish > h4_bullish:
                    bearish_score += self.h4_trend_weight
                    bullish_score -= self.h4_counter_penalty

            # Calculate future return (Fixed horizon)
            future_closes = close[i + 1 : i + 1 + fixed_lookforward]
            future_return = (np.mean(future_closes) - close[i]) / close[i]

            # Adaptive threshold
            vol = pd.Series(close).pct_change().iloc[max(0, i-20):i+1].std()
            min_return = max(self.min_return_threshold, 0.5 * vol)

            if bullish_score >= self.min_score_threshold and future_return > min_return:
                labels.iloc[i] = 1
            elif bearish_score >= self.min_score_threshold and future_return < -min_return:
                labels.iloc[i] = -1

        return labels

    def _generate_live_signal(self, df: pd.DataFrame, df_4h: pd.DataFrame = None, silent: bool = False) -> tuple[int, float]:
        """Generate live signal without lookahead"""
        if len(df) < self.get_warmup_period():
            return 0, 0.0

        latest = df.iloc[-1]
        bullish_score = 0.0
        bearish_score = 0.0

        # 1. MA Alignment (0-2 points)
        if latest["sma_fast"] > latest["sma_slow"]:
            bullish_score += 2.0 if (latest["sma_fast"] - latest["sma_slow"])/latest["sma_slow"] > 0.001 else 1.0
        elif latest["sma_fast"] < latest["sma_slow"]:
            bearish_score += 2.0 if (latest["sma_slow"] - latest["sma_fast"])/latest["sma_slow"] > 0.001 else 1.0

        # 2. MACD (0-1.5 points)
        if latest["macd_hist"] > 0: bullish_score += 1.5
        elif latest["macd_hist"] < 0: bearish_score += 1.5

        # 3. Directional Indicators (0-1.5 points)
        if latest["plus_di"] > latest["minus_di"]: bullish_score += 1.5
        elif latest["minus_di"] > latest["plus_di"]: bearish_score += 1.5

        # 4. Price Position (0-1 point)
        if latest["close"] > latest["sma_fast"]: bullish_score += 1.0
        elif latest["close"] < latest["sma_fast"]: bearish_score += 1.0

        # 5. ADX Bonus (0-1 point)
        if latest["adx"] > 25:
            if bullish_score > bearish_score: bullish_score += 1.0
            elif bearish_score > bullish_score: bearish_score += 1.0

        # ✅ T1.2C: Live Squeeze Filter
        recent_widths = df["bb_width_norm"].tail(100).values
        if len(recent_widths) >= 50:
            lower_20 = np.percentile(recent_widths, 20)
            upper_70 = np.percentile(recent_widths, 70)
            
            if latest["bb_width_norm"] <= lower_20:
                bullish_score += 1.0; bearish_score += 1.0
                if not silent: logger.info(f"[{self.name}] 🌀 Volatility Squeeze detected (+1.0 bonus)")
            elif latest["bb_width_norm"] >= upper_70:
                bullish_score -= 0.5; bearish_score -= 0.5
                if not silent: logger.info(f"[{self.name}] 🌋 High Volatility Expansion (-0.5 penalty)")

        # 4H Context
        if self.use_4h_context and df_4h is not None:
            if "sma_fast" not in df_4h.columns: df_4h = self.generate_features(df_4h)
            df_4h_aligned = self._align_4h_to_1h(df.tail(1), df_4h)
            if df_4h_aligned is not None and not df_4h_aligned.empty:
                h4_bull, h4_bear = self._calculate_4h_trend_score(df_4h_aligned, -1)
                if h4_bull > h4_bear:
                    bullish_score += self.h4_trend_weight
                    bearish_score -= self.h4_counter_penalty
                elif h4_bear > h4_bull:
                    bearish_score += self.h4_trend_weight
                    bullish_score -= self.h4_counter_penalty

        # FINAL SIGNAL
        normalization_factor = 6.5
        signal = 0
        confidence = 0.0

        if bullish_score >= self.min_score_threshold and bullish_score > bearish_score:
            signal = 1
            confidence = min(bullish_score / normalization_factor, 1.0)
        elif bearish_score >= self.min_score_threshold and bearish_score > bullish_score:
            signal = -1
            confidence = min(bearish_score / normalization_factor, 1.0)

        return signal, confidence

    def generate_signal(self, df: pd.DataFrame, mode: str = 'live', df_4h: pd.DataFrame = None, silent: bool = False):
        df = self.generate_features(df)
        if mode == "train": return self._generate_training_labels(df, df_4h)
        return self._generate_live_signal(df, df_4h, silent=silent)
