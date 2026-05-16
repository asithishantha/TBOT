"""
MULTI-TIMEFRAME EMA Crossover Strategy with Optional 4H Context
Realistic signal generation with proper label validation and no data leakage
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class EMAStrategy(BaseStrategy):
    """
    EMA Crossover Strategy (50/200) with optional 4H trend confirmation
    Uses 4H to filter false crossovers and confirm trend strength
    """

    def __init__(self, config: dict):
        super().__init__(config, "EMA")
        
        # EMA periods
        self.fast_period = config.get("ema_fast", 50)
        self.slow_period = config.get("ema_slow", 200)  # Institutional baseline

        # Signal thresholds (relaxed for GOLD)
        self.min_distance_pct = config.get("min_distance_pct", 0.001)  # 0.001%
        self.min_return_threshold = config.get("min_return_threshold", 0.0025)  # 0.25%
        self.min_score_threshold = config.get("min_conditions", 3)  # Lowered for GOLD

        # Filters
        self.use_price_confirmation = config.get("use_price_confirmation", True)  # Enable for GOLD
        self.use_volume_filter = config.get("use_volume_filter", True)  # Enable for GOLD
        self.volume_multiplier = config.get("volume_multiplier", 1.2)

        # 4H context parameters (reduced penalty for GOLD)
        self.use_4h_context = config.get("use_4h_context", True)
        self.require_4h_alignment = config.get("require_4h_alignment", False)
        self.h4_trend_weight = config.get("h4_trend_weight", 1.5)
        self.h4_counter_penalty = config.get("h4_counter_penalty", 1.5)  # Strict enforcement

        logger.info(f"[{self.name}] Initialized with:")
        logger.info(f"  EMA Fast: {self.fast_period}, Slow: {self.slow_period}")
        logger.info(f"  Min Distance: {self.min_distance_pct}%")
        logger.info(f"  Min Score Threshold: {self.min_score_threshold}")
        logger.info(f"  Min Return: {self.min_return_threshold:.4%}")
        logger.info(f"  4H Context: {self.use_4h_context} (Required: {self.require_4h_alignment})")

    def get_warmup_period(self) -> int:
        """Need enough data for slow EMA + other indicators"""
        return max(self.slow_period, 26 + 9) + 50

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate EMA-based features, ensuring all data is numeric."""
        if len(df) < self.get_warmup_period():
            logger.debug(
                f"[{self.name}] Insufficient data: {len(df)} < {self.get_warmup_period()}"
            )
            empty_df = df.copy()
            for col in [
                "ema_fast",
                "ema_slow",
                "ema_diff",
                "ema_diff_pct",
                "ema_cross",
            ]:
                empty_df[col] = 0
            return empty_df

        df = df.copy()
        
        # Ensure all columns are numeric first
        for col in df.columns:
            if df[col].dtype not in ["float64", "int64"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle missing volume
        if "volume" not in df.columns:
            df["volume"] = 1.0

        # Clean price data
        df["close"] = pd.to_numeric(df["close"], errors="coerce").ffill().fillna(0)
        df["high"] = (
            pd.to_numeric(df["high"], errors="coerce").ffill().fillna(df["close"])
        )
        df["low"] = (
            pd.to_numeric(df["low"], errors="coerce").ffill().fillna(df["close"])
        )
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").ffill().fillna(1.0)

        # Extract values
        close = df["close"].values.astype("float64")
        high = df["high"].values.astype("float64")
        low = df["low"].values.astype("float64")
        volume = df["volume"].values.astype("float64")

        # Check for all NaN
        if np.all(np.isnan(close)) or len(close) == 0:
            logger.warning(
                f"[{self.name}] All close prices are NaN, returning zero features"
            )
            df["ema_fast"] = 0
            df["ema_slow"] = 0
            df["ema_diff"] = 0
            df["ema_diff_pct"] = 0
            df["ema_cross"] = 0
            return df

        # Fill NaN values
        close = np.nan_to_num(close, nan=np.nanmean(close))
        high = np.nan_to_num(high, nan=np.nanmean(high))
        low = np.nan_to_num(low, nan=np.nanmean(low))
        volume = np.nan_to_num(volume, nan=np.nanmean(volume))

        # === CORE EMA INDICATORS ===
        try:
            ema_fast_values = ta.EMA(close, timeperiod=self.fast_period)
            ema_slow_values = ta.EMA(close, timeperiod=self.slow_period)

            df["ema_fast"] = ema_fast_values
            df["ema_slow"] = ema_slow_values

            df["ema_fast"] = df["ema_fast"].ffill().bfill()
            df["ema_slow"] = df["ema_slow"].ffill().bfill()

            if df["ema_fast"].isna().any():
                df["ema_fast"] = df["ema_fast"].fillna(df["close"])
            if df["ema_slow"].isna().any():
                df["ema_slow"] = df["ema_slow"].fillna(df["close"])
        except Exception as e:
            logger.error(f"[{self.name}] Error calculating EMAs: {e}")
            df["ema_fast"] = df["close"]
            df["ema_slow"] = df["close"]

        # EMA relationships
        df["ema_diff"] = df["ema_fast"] - df["ema_slow"]
        df["ema_diff_pct"] = np.where(
            df["ema_slow"] != 0, (df["ema_diff"] / df["ema_slow"]) * 100, 0
        )

        # Price vs EMAs
        df["price_vs_fast"] = np.where(
            df["ema_fast"] != 0, (close - df["ema_fast"]) / df["ema_fast"], 0
        )
        df["price_vs_slow"] = np.where(
            df["ema_slow"] != 0, (close - df["ema_slow"]) / df["ema_slow"], 0
        )
        df["price_above_fast"] = (close > df["ema_fast"]).astype(int)
        df["price_above_slow"] = (close > df["ema_slow"]).astype(int)

        # === CROSSOVER DETECTION ===
        df["ema_cross"] = 0
        ema_diff_shift = df["ema_diff"].shift(1).fillna(0)

        # Golden Cross (Fast crosses above Slow)
        df.loc[(df["ema_diff"] > 0) & (ema_diff_shift <= 0), "ema_cross"] = 1
        # Death Cross (Fast crosses below Slow)
        df.loc[(df["ema_diff"] < 0) & (ema_diff_shift >= 0), "ema_cross"] = -1

        # Bars since last crossover (Vectorized cummax approach)
        cross_indices = np.where(df["ema_cross"] != 0)[0]
        if len(cross_indices) > 0:
            # Create an array of indices where crossovers occurred, otherwise 0
            event_indices = np.zeros(len(df))
            event_indices[cross_indices] = cross_indices
            # Use cumulative maximum to propagate the last crossover index forward
            last_event_index = pd.Series(event_indices).cummax().values
            df["bars_since_cross"] = np.arange(len(df)) - last_event_index
            # Set values before the first crossover to 0 (or a large number if preferred)
            df.loc[np.arange(len(df)) < cross_indices[0], "bars_since_cross"] = 0
        else:
            df["bars_since_cross"] = 0

        # === TREND STRENGTH ===
        df["ema_trend"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
        df["trend_strength"] = np.abs(df["ema_diff_pct"])

        # EMA slopes (rate of change)
        df["ema_fast_slope"] = df["ema_fast"].diff(5)
        df["ema_fast_slope"] = np.where(
            df["ema_fast"] != 0, df["ema_fast_slope"] / df["ema_fast"], 0
        )

        df["ema_slow_slope"] = df["ema_slow"].diff(10)
        df["ema_slow_slope"] = np.where(
            df["ema_slow"] != 0, df["ema_slow_slope"] / df["ema_slow"], 0
        )

        # === ADDITIONAL CONFIRMATION INDICATORS ===
        try:
            volume_ma_values = ta.SMA(volume, timeperiod=20)
            df["volume_ma"] = volume_ma_values
            df["volume_ma"] = df["volume_ma"].ffill().bfill()

            if df["volume_ma"].isna().any():
                df["volume_ma"] = df["volume_ma"].fillna(df["volume"])

            df["volume_ratio"] = np.where(
                df["volume_ma"] != 0, df["volume"] / df["volume_ma"], 1.0
            )
            df["high_volume"] = (df["volume_ratio"] > self.volume_multiplier).astype(
                int
            )
        except Exception as e:
            logger.warning(f"[{self.name}] Error calculating volume indicators: {e}")
            df["volume_ma"] = df["volume"]
            df["volume_ratio"] = 1.0
            df["high_volume"] = 1

        # MACD (complementary momentum)
        try:
            macd, macd_signal, macd_hist = ta.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            df["macd"] = macd
            df["macd_signal"] = macd_signal
            df["macd_hist"] = macd_hist

            df["macd"] = df["macd"].fillna(0)
            df["macd_signal"] = df["macd_signal"].fillna(0)
            df["macd_hist"] = df["macd_hist"].fillna(0)

            df["macd_aligned"] = np.where(
                ((df["ema_trend"] == 1) & (macd_hist > 0))
                | ((df["ema_trend"] == -1) & (macd_hist < 0)),
                1,
                0,
            )
        except Exception as e:
            logger.warning(f"[{self.name}] Error calculating MACD: {e}")
            df["macd"] = 0
            df["macd_signal"] = 0
            df["macd_hist"] = 0
            df["macd_aligned"] = 0

        # RSI (for overbought/oversold context)
        try:
            df["rsi"] = ta.RSI(close, timeperiod=14)
            df["rsi"] = df["rsi"].fillna(50)
        except Exception as e:
            logger.warning(f"[{self.name}] Error calculating RSI: {e}")
            df["rsi"] = 50

        # ADX (trend strength)
        try:
            df["adx"] = ta.ADX(high, low, close, timeperiod=14)
            df["adx"] = df["adx"].fillna(20)
            df["strong_trend"] = (df["adx"] > 25).astype(int)
        except Exception as e:
            logger.warning(f"[{self.name}] Error calculating ADX: {e}")
            df["adx"] = 20
            df["strong_trend"] = 0

        # ATR (for bounce detection)
        try:
            df["atr"] = ta.ATR(high, low, close, timeperiod=14)
            df["atr"] = df["atr"].fillna(df["close"] * 0.01)
        except Exception as e:
            logger.warning(f"[{self.name}] Error calculating ATR: {e}")
            df["atr"] = df["close"] * 0.01

        # Final NaN cleanup
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            logger.warning(f"[{self.name}] Still has NaN in columns: {nan_columns}")
            df[nan_columns] = df[nan_columns].fillna(0)

        return df

    def _align_4h_to_1h(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
        """Align 4H data to 1H timeframe using forward-fill"""
        if df_4h is None or df_4h.empty:
            return None
        
        # Ensure both have datetime index
        if not isinstance(df_1h.index, pd.DatetimeIndex):
            df_1h = df_1h.set_index('timestamp')
        if not isinstance(df_4h.index, pd.DatetimeIndex):
            df_4h = df_4h.set_index('timestamp')

        # ✅ FIX: Handle timezone mismatch between 1H and 4H indices
        if df_1h.index.tz is not None:
            df_1h = df_1h.copy()
            df_1h.index = df_1h.index.tz_localize(None)
        if df_4h.index.tz is not None:
            df_4h = df_4h.copy()
            df_4h.index = df_4h.index.tz_localize(None)

        # Select 4H features to align
        h4_features = ['ema_fast', 'ema_slow', 'ema_diff_pct', 'ema_trend',
                       'adx', 'macd_hist', 'trend_strength']

        df_4h_aligned = pd.DataFrame(index=df_1h.index)        
        for feature in h4_features:
            if feature in df_4h.columns:
                df_4h_aligned[f'h4_{feature}'] = df_4h[feature].reindex(
                    df_1h.index, 
                    method='ffill'
                )
        
        return df_4h_aligned

    def _calculate_4h_trend_context(self, df_4h_aligned: pd.DataFrame, idx: int) -> dict:
        """
        Calculate 4H trend context for crossover confirmation
        Returns: {'trend_direction': int, 'trend_strength': float, 'alignment_score': float}
        """
        if df_4h_aligned is None or idx >= len(df_4h_aligned):
            return {'trend_direction': 0, 'trend_strength': 0.0, 'alignment_score': 0.0}
        
        row = df_4h_aligned.iloc[idx]
        
        # Check for valid data
        if pd.isna(row.get('h4_ema_trend')) or pd.isna(row.get('h4_trend_strength')):
            return {'trend_direction': 0, 'trend_strength': 0.0, 'alignment_score': 0.0}
        
        context = {
            'trend_direction': int(row['h4_ema_trend']),  # 1 or -1
            'trend_strength': float(row.get('h4_trend_strength', 0.0)),
            'alignment_score': 0.0
        }
        
        # Calculate alignment score (0-3 points)
        ema_diff = row.get('h4_ema_diff_pct', 0)
        adx = row.get('h4_adx', 0)
        macd_hist = row.get('h4_macd_hist', 0)
        
        # Strong EMA separation
        if abs(ema_diff) > 1.0:  # >1% separation
            context['alignment_score'] += 1.5
        elif abs(ema_diff) > 0.5:
            context['alignment_score'] += 1.0
        
        # Strong trend (ADX)
        if adx > 25:
            context['alignment_score'] += 1.0
        elif adx > 20:
            context['alignment_score'] += 0.5
        
        # MACD confirmation
        if (context['trend_direction'] == 1 and macd_hist > 0) or \
           (context['trend_direction'] == -1 and macd_hist < 0):
            context['alignment_score'] += 0.5
        
        return context

    def generate_signal(self, df: pd.DataFrame, df_4h: pd.DataFrame = None) -> tuple:
        """
        Generate real-time signal based on current EMA conditions.
        Now detects uptrends/downtrends even without crossovers.
        ✅ ENHANCED: Includes 4H trend alignment validation to prevent Train/Serve skew.
        """
        if len(df) < self.get_warmup_period():
            return 0, 0.0

        try:
            features_df = self.generate_features(
                df.tail(max(self.fast_period, self.slow_period) + 50)
            )
            if features_df.empty or len(features_df) == 0:
                return 0, 0.0

            latest = features_df.iloc[-1]

            # Check for NaN in critical features
            critical_features = [
                "ema_fast", "ema_slow", "ema_diff", "ema_cross",
                "macd_hist", "rsi", "price_above_fast", "ema_fast_slope"
            ]
            for feat in critical_features:
                if pd.isna(latest[feat]) or np.isnan(latest[feat]):
                    logger.debug(f"[{self.name}] NaN detected in {feat}, returning HOLD")
                    return 0, 0.0

            ema_cross = latest["ema_cross"]
            trend_strength = latest["trend_strength"]
            price_above_fast = latest["price_above_fast"]
            ema_fast_slope = latest["ema_fast_slope"]
            macd_aligned = latest["macd_aligned"]
            rsi = latest["rsi"]
            high_volume = latest["high_volume"]

            # ── Pull extra latest values needed for scoring ───────────────
            macd_hist_val  = latest["macd_hist"]
            ema_diff_pct   = latest["ema_diff_pct"]
            ema_fast_val   = latest["ema_fast"]
            ema_slow_val   = latest["ema_slow"]
            adx            = latest["adx"]
            close          = latest["close"]
            atr            = latest["atr"]

            # ================================================================
            # FULL SCORING — mirrors generate_labels() so live signals match
            # training labels.  Crossovers score highest; trend-continuation
            # fires between crossovers so EMA contributes every cycle.
            # ================================================================
            bullish_score = 0
            bearish_score = 0

            # --- Bullish conditions -----------------------------------------
            if ema_cross == 1:                                    # Golden Cross
                bullish_score += 3
            if ema_fast_val > ema_slow_val and ema_diff_pct > self.min_distance_pct:
                bullish_score += 2                                # EMA separation
            if self.use_price_confirmation and close > ema_fast_val and close > ema_slow_val:
                bullish_score += 1                                # Price above both EMAs
            if rsi > 50:
                bullish_score += 1                                # RSI above midpoint = sustained bullish pressure
            if 40 < rsi < 70:
                bullish_score += 1                                # RSI not overbought
            if self.use_volume_filter and high_volume == 1:
                bullish_score += 1                                # Volume surge
            if ema_fast_slope > 0:
                bullish_score += 1                                # EMA fast slope rising
            # Trend continuation (no crossover required)
            if (close > ema_fast_val and ema_fast_slope > 0
                    and rsi < 70):
                bullish_score += 2

            # --- Bearish conditions -----------------------------------------
            if ema_cross == -1:                                   # Death Cross
                bearish_score += 3
            if ema_fast_val < ema_slow_val and ema_diff_pct < -self.min_distance_pct:
                bearish_score += 2                                # EMA separation
            if self.use_price_confirmation and close < ema_fast_val and close < ema_slow_val:
                bearish_score += 1                                # Price below both EMAs
            if rsi < 50:
                bearish_score += 1                                # RSI below midpoint = sustained bearish pressure
            if 30 < rsi < 60:
                bearish_score += 1                                # RSI not oversold
            if self.use_volume_filter and high_volume == 1:
                bearish_score += 1                                # Volume surge
            if ema_fast_slope < 0:
                bearish_score += 1                                # EMA fast slope falling
            # Trend continuation (no crossover required)
            if (close < ema_fast_val and ema_fast_slope < 0
                    and rsi > 30):
                bearish_score += 2

            # --- Determine preliminary signal --------------------------------
            signal = 0
            if bullish_score >= self.min_score_threshold and bullish_score > bearish_score:
                signal = 1
            elif bearish_score >= self.min_score_threshold and bearish_score > bullish_score:
                signal = -1

            logger.debug(
                f"[{self.name}] Score → bullish={bullish_score} bearish={bearish_score} "
                f"cross={ema_cross} signal={signal}"
            )

            # ── Confidence (score-driven, capped at 1.0) ─────────────────
            active_score  = bullish_score if signal == 1 else bearish_score
            # Normalise: max achievable raw score is ~13 with everything firing
            normalization_factor = 6.5
            confidence = min(1.0, trend_strength / normalization_factor)
            confidence += 0.05 * min(active_score, 6)     # up to +0.30 from score
            if macd_aligned == 1:
                confidence += 0.10
            if ema_cross != 0:                             # extra boost for actual crossover
                confidence += 0.10
            if high_volume == 1:
                confidence += 0.05

            # ================================================================
            # 4H CONTEXT (alignment boost / counter-trend penalty)
            # ================================================================
            h4_macro_trend = 'NEUTRAL'
            if self.use_4h_context and df_4h is not None:
                try:
                    df_4h_feat = self.generate_features(df_4h) \
                        if 'ema_fast' not in df_4h.columns else df_4h
                    df_4h_aligned = self._align_4h_to_1h(df.tail(1), df_4h_feat)
                    if df_4h_aligned is not None:
                        h4_context = self._calculate_4h_trend_context(df_4h_aligned, -1)
                        if h4_context['trend_direction'] == 1:
                            h4_macro_trend = 'BULLISH'
                        elif h4_context['trend_direction'] == -1:
                            h4_macro_trend = 'BEARISH'

                        if signal != 0 and h4_context['trend_direction'] != 0:
                            if h4_context['trend_direction'] == signal:
                                boost = self.h4_trend_weight * h4_context['alignment_score'] / 30.0
                                confidence += boost
                                logger.debug(f"[{self.name}] 4H Alignment Boost: +{boost:.2f}")
                            else:
                                penalty = self.h4_counter_penalty * 0.5
                                confidence -= penalty
                                logger.info(f"[{self.name}] 4H Counter-Trend Penalty: -{penalty:.2f}")
                                if self.require_4h_alignment and h4_context['alignment_score'] > 2.0:
                                    logger.info(f"[{self.name}] ❌ Blocked by strict 4H alignment.")
                                    return 0, 0.0
                                if confidence < self.min_confidence:
                                    logger.info(
                                        f"[{self.name}] ❌ Confidence {confidence:.2f} below "
                                        f"threshold after 4H penalty."
                                    )
                                    return 0, 0.0
                except Exception as e:
                    logger.warning(f"[{self.name}] 4H context validation failed: {e}")

            # ================================================================
            # EMA-200 BOUNCE MODE (fallback when no trend score met)
            # ================================================================
            if signal == 0:
                if (h4_macro_trend == 'BULLISH'
                        and abs(close - ema_slow_val) <= 0.5 * atr
                        and close > ema_slow_val):
                    signal = 1
                    confidence = 0.75
                    logger.info(f"[{self.name}] 🚀 EMA-200 Bounce (Bullish)")
                elif (h4_macro_trend == 'BEARISH'
                        and abs(close - ema_slow_val) <= 0.5 * atr
                        and close < ema_slow_val):
                    signal = -1
                    confidence = 0.75
                    logger.info(f"[{self.name}] 🔻 EMA-200 Bounce (Bearish)")

            if signal == 0:
                return 0, 0.0

            confidence = max(0.0, min(1.0, confidence))
            return signal, confidence

        except Exception as e:
            logger.error(f"[{self.name}] Error in generate_signal: {e}")
            return 0, 0.0


    def generate_labels(self, df: pd.DataFrame, df_4h: pd.DataFrame = None) -> pd.Series:
        """
        MULTI-TIMEFRAME label generation with optional 4H context
        
        4H Context Usage:
        - Confirms trend direction before crossovers
        - Filters weak/false crossovers in choppy markets
        - Boosts high-conviction crossovers aligned with 4H
        
        Parameters:
        -----------
        df : pd.DataFrame
            Primary timeframe data (1H) with features
        df_4h : pd.DataFrame, optional
            Higher timeframe data (4H) for trend confirmation
        
        Returns:
        --------
        pd.Series
            Labels: -1 (SELL), 0 (HOLD), 1 (BUY)
        """
        df = df.copy()
        logger.info(f"[{self.name}] Starting label generation with {len(df)} rows")

        required_cols = [
            "close",
            "ema_fast",
            "ema_slow",
            "ema_diff_pct",
            "ema_cross",
            "rsi",
            "ema_fast_slope",
            "high_volume",
        ]
        df = df.dropna(subset=required_cols)

        logger.info(
            f"[{self.name}] After filtering NaN in required columns: {len(df)} rows"
        )

        if len(df) == 0:
            logger.error(f"[{self.name}] No valid rows after filtering NaN!")
            return pd.Series(0, index=pd.Index([]))

        # Align 4H data if provided
        df_4h_aligned = None
        if self.use_4h_context and df_4h is not None:
            if 'ema_fast' not in df_4h.columns:
                df_4h = self.generate_features(df_4h)
            df_4h_aligned = self._align_4h_to_1h(df, df_4h)
            logger.info(f"[{self.name}] 4H context aligned successfully")

        # Extract values
        close = df["close"].values
        ema_fast = df["ema_fast"].values
        ema_slow = df["ema_slow"].values
        ema_diff_pct = df["ema_diff_pct"].values
        ema_cross = df["ema_cross"].values.astype(int)
        rsi = df["rsi"].values
        ema_fast_slope = df["ema_fast_slope"].values
        high_volume = df["high_volume"].values

        labels = pd.Series(0, index=df.index)
        lookforward = 5

        filtered_by_4h = 0
        boosted_by_4h = 0

        logger.info(
            f"[{self.name}] Analyzing {len(df) - lookforward} bars for signals..."
        )

        for i in range(len(df) - lookforward):
            if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(close[i]):
                continue

            # Calculate future return for VALIDATION ONLY
            future_closes = close[i + 1 : i + 1 + lookforward]
            if len(future_closes) == 0:
                continue

            future_return = (np.mean(future_closes) - close[i]) / close[i]

            # === BULLISH CONDITIONS (NO FUTURE DATA IN SCORING) ===
            bullish_score = 0

            # Strong signal: Actual crossover
            if ema_cross[i] == 1:
                bullish_score += 3

            # EMA separation (trend strength)
            if ema_fast[i] > ema_slow[i] and ema_diff_pct[i] > self.min_distance_pct:
                bullish_score += 2

            # Price confirmation
            if close[i] > ema_fast[i] and close[i] > ema_slow[i]:
                bullish_score += 1

            # RSI above midpoint — sustained bullish pressure
            if not np.isnan(rsi[i]) and rsi[i] > 50:
                bullish_score += 1

            # RSI not overbought
            if not np.isnan(rsi[i]) and 40 < rsi[i] < 70:
                bullish_score += 1

            # Volume confirmation
            if high_volume[i] == 1:
                bullish_score += 1

            # EMA fast slope rising — momentum behind the move
            if not np.isnan(ema_fast_slope[i]) and ema_fast_slope[i] > 0:
                bullish_score += 1

            # Trend continuation — matches live signal logic exactly
            if (
                (close[i] > ema_fast[i]) and       # Price above fast EMA
                (not np.isnan(ema_fast_slope[i]) and ema_fast_slope[i] > 0) and  # EMA rising
                (not np.isnan(rsi[i]) and rsi[i] < 70)                           # Not overbought
            ):
                bullish_score += 2

            # === BEARISH CONDITIONS (NO FUTURE DATA IN SCORING) ===
            bearish_score = 0

            # Strong signal: Actual crossover
            if ema_cross[i] == -1:
                bearish_score += 3

            # EMA separation (trend strength)
            if ema_fast[i] < ema_slow[i] and ema_diff_pct[i] < -self.min_distance_pct:
                bearish_score += 2

            # Price confirmation
            if close[i] < ema_fast[i] and close[i] < ema_slow[i]:
                bearish_score += 1

            # RSI below midpoint — sustained bearish pressure
            if not np.isnan(rsi[i]) and rsi[i] < 50:
                bearish_score += 1

            # RSI not oversold
            if not np.isnan(rsi[i]) and 30 < rsi[i] < 60:
                bearish_score += 1

            # Volume confirmation
            if high_volume[i] == 1:
                bearish_score += 1

            # EMA fast slope falling — momentum behind the move
            if not np.isnan(ema_fast_slope[i]) and ema_fast_slope[i] < 0:
                bearish_score += 1

            # Trend continuation — matches live signal logic exactly
            if (
                (close[i] < ema_fast[i]) and       # Price below fast EMA
                (not np.isnan(ema_fast_slope[i]) and ema_fast_slope[i] < 0) and  # EMA falling
                (not np.isnan(rsi[i]) and rsi[i] > 30)                           # Not oversold
            ):
                bearish_score += 2

            # === APPLY 4H CONTEXT ===
            if df_4h_aligned is not None:
                h4_context = self._calculate_4h_trend_context(df_4h_aligned, i)

                # BUY signal with 4H context (reduced penalty)
                if bullish_score > 0:
                    if h4_context['trend_direction'] == 1:  # 4H also bullish
                        bullish_score += self.h4_trend_weight * h4_context['alignment_score'] / 3.0
                        boosted_by_4h += 1
                    elif h4_context['trend_direction'] == -1:  # 4H bearish (counter-trend)
                        bullish_score -= self.h4_counter_penalty * 0.5  # Reduced penalty by 50%
                        if self.require_4h_alignment and h4_context['alignment_score'] > 2.5:  # Only skip if 4H is STRONGLY bearish
                            filtered_by_4h += 1
                            continue  # Skip signal entirely

                # SELL signal with 4H context (reduced penalty)
                if bearish_score > 0:
                    if h4_context['trend_direction'] == -1:  # 4H also bearish
                        bearish_score += self.h4_trend_weight * h4_context['alignment_score'] / 3.0
                        boosted_by_4h += 1
                    elif h4_context['trend_direction'] == 1:  # 4H bullish (counter-trend)
                        bearish_score -= self.h4_counter_penalty * 0.5  # Reduced penalty by 50%
                        if self.require_4h_alignment and h4_context['alignment_score'] > 2.5:  # Only skip if 4H is STRONGLY bullish
                            filtered_by_4h += 1
                            continue  # Skip signal entirely
# Skip strong counter-trend signals

            # === ASSIGN LABELS (Future return used ONLY for validation) ===
            if (
                bullish_score >= self.min_score_threshold
                and bullish_score > bearish_score
                and future_return > self.min_return_threshold
            ):
                labels.iloc[i] = 1

            elif (
                bearish_score >= self.min_score_threshold
                and bearish_score > bullish_score
                and future_return < -self.min_return_threshold
            ):
                labels.iloc[i] = -1
            else:
                labels.iloc[i] = 0

        # Remove labels from last N bars
        labels.iloc[-lookforward:] = 0

        signals_generated = (labels != 0).sum()
        logger.info(
            f"[{self.name}] Generated {signals_generated} total signals from {len(df)} bars"
        )

        # Log 4H impact
        if df_4h_aligned is not None:
            logger.info(f"[{self.name}] 4H Context Impact:")
            logger.info(f"  Filtered: {filtered_by_4h} signals")
            logger.info(f"  Boosted: {boosted_by_4h} signals")

        # Log distribution
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        total = len(labels)
        logger.info(f"[{self.name}] Label distribution:")
        logger.info(
            f"  SELL: {dist.get(-1, 0):>5} ({dist.get(-1, 0)/total*100 if total > 0 else 0:>5.2f}%)"
        )
        logger.info(
            f"  HOLD: {dist.get(0, 0):>5} ({dist.get(0, 0)/total*100 if total > 0 else 0:>5.2f}%)"
        )
        logger.info(
            f"  BUY:  {dist.get(1, 0):>5} ({dist.get(1, 0)/total*100 if total > 0 else 0:>5.2f}%)"
        )

        buy_pct = dist.get(1, 0) / total * 100 if total > 0 else 0
        sell_pct = dist.get(-1, 0) / total * 100 if total > 0 else 0

        if total > 0 and (buy_pct < 5 or sell_pct < 5):
            logger.warning(
                f"  ⚠ Low signal rate detected (BUY: {buy_pct:.1f}%, SELL: {sell_pct:.1f}%)"
            )
            logger.warning(
                f"  Consider lowering min_score_threshold or min_return_threshold"
            )
        elif total > 0:
            logger.info(
                f"  ✓ Healthy signal distribution (BUY: {buy_pct:.1f}%, SELL: {sell_pct:.1f}%)"
            )

        return labels