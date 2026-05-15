"""
 Production Training Script - Fixes for 14% accuracy issue
Key improvements:
1. Better class balancing with capped weights
2. Stronger pattern filtering (min 50 samples)
3. Balanced noise generation
4. Longer training with better early stopping
5. Data augmentation improvements
"""

import pickle
import numpy as np
from typing import Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
import pickle
from pathlib import Path
import sys
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# ✅ FIX: Add project root to sys.path (relative to this script)
script_path = Path(__file__).resolve()
project_root = script_path.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if sys.platform == "win32":
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

# Setup logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "dual_timeframe_training.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Import AI components
from src.ai.pattern_miner import PatternMiner
from src.ai.sniper import OHLCSniper
from src.ai.analyst import DynamicAnalyst


def train_dual_timeframe_system(
    # DATA SOURCES (must provide both timeframes)
    assets=["btc", "gold", "ustec", "eurusd", "usoil", "gbpaud", "gbpusd", "usdjpy"],
    data_folder=None,
    # TRAINING PARAMETERS
    samples_per_pattern=2000,
    min_samples_per_class=50,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    use_class_weights=True,
    max_class_weight=10.0,
    # OUTPUT
    model_name="sniper_dual_timeframe",
    save_plots=True,
):
    """
    Complete training pipeline for dual timeframe system
    """
    if data_folder is None:
        # ✅ Look in data/raw where downloader saves files
        data_folder = project_root / "data" / "raw"
    else:
        data_folder = Path(data_folder)

    logger.info("=" * 80)
    logger.info("DUAL TIMEFRAME TRAINING SYSTEM")
    logger.info("Analyst: 4H candles | Sniper: 15min candles")
    logger.info("=" * 80)

    models_dir = project_root / "models" / "ai"
    models_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Validate Data Existence
    # =========================================================================

    logger.info(f"\nSTEP 1: Validating data files in {data_folder}...")

    # ✅ Asset to Filename Mapping (match download_multi_tf_data.py)
    asset_map = {
        "btc": "BTCUSDT",
        "gold": "XAUUSDm",
        "ustec": "USTECm",
        "eurjpy": "EURJPYm",
        "eurusd": "EURUSDm"
    }

    missing_15m = []
    missing_4h = []
    valid_assets = []

    for asset in assets:
        symbol = asset_map.get(asset.lower(), asset.upper())
        
        # Check 15min (required for Sniper)
        path_15m = data_folder / f"{symbol}_15m.csv"
        # Check 4H (required for Analyst)
        path_4h = data_folder / f"{symbol}_4h.csv"

        if path_15m.exists() and path_4h.exists():
            df_15 = pd.read_csv(path_15m)
            df_4 = pd.read_csv(path_4h)
            logger.info(f"  [OK] {asset.upper()}: {len(df_15)} (15m), {len(df_4)} (4h) candles")
            valid_assets.append(asset)
        else:
            if not path_15m.exists():
                missing_15m.append(f"{asset} (15m)")
                logger.error(f"  [MISSING] {asset} 15min: {path_15m}")
            if not path_4h.exists():
                missing_4h.append(f"{asset} (4H)")
                logger.error(f"  [MISSING] {asset} 4H: {path_4h}")

    if not valid_assets:
        logger.error("\nCRITICAL: No valid data found for any requested assets!")
        logger.info("Please run scripts/data_tools/download_multi_tf_data.py first.")
        raise FileNotFoundError("No training data found.")

    # Update asset list to only those we actually have data for
    assets = valid_assets

    # =========================================================================
    # STEP 2: Mine Patterns from 15min Data (SNIPER)
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Mining patterns from 15min candles (SNIPER)")
    logger.info("=" * 80)

    miner = PatternMiner(sequence_length=15)

    # Collect all 15min training files
    all_15m_files = []
    for asset in assets:
        symbol = asset_map.get(asset.lower(), asset.upper())
        path = data_folder / f"{symbol}_15m.csv"
        all_15m_files.append(str(path))

    # Load and mine
    df_15m_combined = miner.load_multiple_sources(
        all_15m_files, expected_timeframe="15min"
    )

    X, y, pattern_map = miner.mine_from_dataframe(
        df_15m_combined,
        samples_per_pattern=samples_per_pattern,
        use_augmentation=True,
        augmentation_strength="medium",
        min_pattern_quality=100,
    )

    # Log distribution
    class_counts = Counter(y)
    logger.info(f"\n✓ Mined {len(X)} pattern samples from 15min data")
    logger.info("Initial distribution:")
    for name, pid in pattern_map.items():
        logger.info(f"  {name} ({pid}): {class_counts.get(pid, 0)}")

    # =========================================================================
    # STEP 3: Filter Weak Patterns
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 3: Filtering patterns < {min_samples_per_class} samples")
    logger.info("=" * 80)

    classes_to_keep = [
        pid for pid, count in class_counts.items() if count >= min_samples_per_class
    ]

    patterns_removed = []
    new_pattern_map = {}
    next_id = 1

    for name, pid in pattern_map.items():
        if pid in classes_to_keep:
            new_pattern_map[name] = next_id
            next_id += 1
        else:
            patterns_removed.append(name)
            logger.info(f"  ✗ {name}: only {class_counts[pid]} samples")

    if patterns_removed:
        old_to_new = {
            old_id: new_pattern_map[name]
            for name, old_id in pattern_map.items()
            if name in new_pattern_map
        }

        mask = np.isin(y, list(old_to_new.keys()))
        X = X[mask]
        y_old = y[mask]
        y = np.array([old_to_new[old_id] for old_id in y_old])
        pattern_map = new_pattern_map

        logger.info(f"✓ Kept {len(pattern_map)}, removed {len(patterns_removed)}")

    # =========================================================================
    # STEP 4: Generate Noise from 15min Data
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Generating noise class from 15min data")
    logger.info("=" * 80)

    pattern_counts = [class_counts[pid] for pid in new_pattern_map.values()]
    noise_target = int(np.median(pattern_counts))

    logger.info(f"  Target: {noise_target} samples (median)")

    noise_X = miner.generate_noise_samples(df_15m_combined, num_samples=noise_target)
    noise_y = np.zeros(len(noise_X), dtype=int)

    # Add noise to pattern map
    pattern_map_with_noise = {"Noise": 0}
    pattern_map_with_noise.update(pattern_map)

    logger.info(f"✓ Generated {len(noise_X)} noise samples from 15min data")

    # =========================================================================
    # STEP 5: Combine Dataset
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Combining final dataset")
    logger.info("=" * 80)

    X_combined = np.vstack([X, noise_X])
    y_combined = np.concatenate([y, noise_y])

    # Shuffle
    shuffle_idx = np.random.permutation(len(X_combined))
    X_combined = X_combined[shuffle_idx]
    y_combined = y_combined[shuffle_idx]

    final_counts = Counter(y_combined)
    logger.info(f"✓ Total: {len(X_combined)} samples")
    logger.info("Final distribution:")
    for name, pid in sorted(pattern_map_with_noise.items(), key=lambda x: x[1]):
        logger.info(f"  {name} ({pid}): {final_counts.get(pid, 0)}")

    # =========================================================================
    # STEP 6: Train/Val Split
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Train/validation split")
    logger.info("=" * 80)

    X_train, X_val, y_train, y_val = train_test_split(
        X_combined,
        y_combined,
        test_size=validation_split,
        random_state=42,
        stratify=y_combined,
    )

    logger.info(f"✓ Training: {len(X_train)} (15min samples)")
    logger.info(f"✓ Validation: {len(X_val)} (15min samples)")

    # =========================================================================
    # STEP 7: Train Sniper Model
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Training Sniper on 15min patterns")
    logger.info("=" * 80)

    num_classes = len(pattern_map_with_noise)
    sniper = OHLCSniper(input_shape=(15, 4), num_classes=num_classes, dropout_rate=0.3)

    logger.info(f"  Timeframe: 15min")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Classes: {num_classes}")

    history = sniper.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        use_class_weights=use_class_weights,
        max_class_weight=max_class_weight,
        early_stopping_patience=30,
        verbose=1,
    )

    # =========================================================================
    # STEP 8: Evaluation
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Model evaluation")
    logger.info("=" * 80)

    val_loss, val_acc, val_top3 = sniper.model.evaluate(X_val, y_val, verbose=0)
    y_pred = np.argmax(sniper.model.predict(X_val, verbose=0), axis=1)

    logger.info(f"✓ Validation Accuracy: {val_acc:.2%}")
    logger.info(f"✓ Validation Top-3: {val_top3:.2%}")
    logger.info(f"✓ Validation Loss: {val_loss:.4f}")

    # Classification report
    reverse_map = {v: k for k, v in pattern_map_with_noise.items()}
    unique_classes = np.unique(np.concatenate([y_val, y_pred]))
    target_names = [reverse_map.get(i, f"Class_{i}") for i in unique_classes]

    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 80)

    print(
        classification_report(
            y_val,
            y_pred,
            labels=unique_classes,
            target_names=target_names,
            zero_division=0,
        )
    )

    # =========================================================================
    # STEP 9: Save Everything
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("STEP 9: Saving model and configuration")
    logger.info("=" * 80)

    # Save model
    model_path = models_dir / f"{model_name}.weights.h5"
    sniper.save_model(str(model_path))

    # Save pattern mapping
    mapping_path = models_dir / f"{model_name}_mapping.pkl"
    with open(mapping_path, "wb") as f:
        pickle.dump(pattern_map_with_noise, f)

    # Save configuration
    config = {
        "model_name": model_name,
        "model_version": "V3_DUAL_TIMEFRAME",
        "analyst_timeframe": "4H",
        "sniper_timeframe": "15min",
        "assets": assets,
        "num_samples": len(X_combined),
        "samples_per_pattern": samples_per_pattern,
        "min_samples_per_class": min_samples_per_class,
        "sequence_length": 15,
        "num_classes": num_classes,
        "patterns": list(pattern_map_with_noise.keys()),
        "removed_patterns": patterns_removed,
        "val_accuracy": float(val_acc),
        "val_top3_accuracy": float(val_top3),
        "val_loss": float(val_loss),
        "epochs_trained": len(history.history["loss"]),
        "training_date": pd.Timestamp.now().isoformat(),
    }

    config_path = models_dir / f"{model_name}_config.pkl"
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    logger.info(f"✓ Model: {model_path}")
    logger.info(f"✓ Mapping: {mapping_path}")
    logger.info(f"✓ Config: {config_path}")

    # Save training plots
    if save_plots:

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Accuracy
        axes[0, 0].plot(history.history["accuracy"], label="Train")
        axes[0, 0].plot(history.history["val_accuracy"], label="Validation")
        axes[0, 0].set_title("Accuracy (15min patterns)")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(history.history["loss"], label="Train")
        axes[0, 1].plot(history.history["val_loss"], label="Validation")
        axes[0, 1].set_title("Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Top-3 Accuracy
        if "top3_acc" in history.history:
            axes[1, 0].plot(history.history["top3_acc"], label="Train Top-3")
            axes[1, 0].plot(history.history["val_top3_acc"], label="Val Top-3")
            axes[1, 0].set_title("Top-3 Accuracy")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Accuracy")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Info text
        axes[1, 1].text(
            0.1, 0.9, f"Dual Timeframe System", fontsize=14, fontweight="bold"
        )
        axes[1, 1].text(0.1, 0.8, f"Analyst: 4H candles", fontsize=11)
        axes[1, 1].text(0.1, 0.7, f"Sniper: 15min candles", fontsize=11)
        axes[1, 1].text(0.1, 0.6, f"Val Accuracy: {val_acc:.2%}", fontsize=11)
        axes[1, 1].text(0.1, 0.5, f"Val Top-3: {val_top3:.2%}", fontsize=11)
        axes[1, 1].text(0.1, 0.4, f"Patterns: {num_classes}", fontsize=11)
        axes[1, 1].text(0.1, 0.3, f"Samples: {len(X_combined)}", fontsize=11)
        axes[1, 1].axis("off")

        plt.tight_layout()
        plot_path = models_dir / f"{model_name}_history.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"✓ Plots: {plot_path}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("🎯 DUAL TIMEFRAME TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"✅ Analyst: Uses 4H candles for S/R detection")
    logger.info(f"✅ Sniper: Trained on 15min candles for patterns")
    logger.info(f"✅ Validation Accuracy: {val_acc:.2%}")
    logger.info(f"✅ Validation Top-3: {val_top3:.2%}")
    logger.info(f"✅ Model ready: {model_path}")
    logger.info(f"✅ Active patterns: {num_classes} (including noise)")

    if patterns_removed:
        logger.info(f"⚠️  Excluded: {', '.join(patterns_removed)}")

    logger.info("\nUsage in Production:")
    logger.info("  1. Analyst loads 4H data: analyst.analyze(df_4h)")
    logger.info("  2. Sniper loads 15min data: sniper.predict(last_15_candles_15m)")
    logger.info("  3. Combine signals for trade decisions")
    logger.info("=" * 80)

    return sniper, pattern_map_with_noise, history, config


# ==============================================================================
# 5. PRODUCTION BOT WITH DUAL TIMEFRAMES
# ==============================================================================


class DualTimeframeTradingBot:
    """
    Production trading bot with proper timeframe separation
    """

    def __init__(
        self, threshold_percent: float = 0.005, confidence_threshold: float = 0.85
    ):
        self.threshold_percent = threshold_percent
        self.confidence_threshold = confidence_threshold

        # Initialize components
        self.analyst = DynamicAnalyst(atr_multiplier=2.0)

        # DON'T initialize Sniper yet - wait for load_model
        self.sniper = None
        self.pattern_map = None
        self.reverse_map = None

        # Cache for 4H analysis
        self.cached_levels = None
        self.last_4h_update = None

        logger.info(
            "[BOT] Initialized\n"
            "  Analyst: 4H candles\n"
            "  Sniper: 15min candles (pending model load)"
        )

    def load_model(self, model_path: str, mapping_path: str):
        """Load trained model and pattern mapping"""

        # Load pattern mapping first to get the correct number of classes
        with open(mapping_path, "rb") as f:
            self.pattern_map = pickle.load(f)

        self.reverse_map = {v: k for k, v in self.pattern_map.items()}
        num_classes = len(self.pattern_map)

        logger.info(f"[BOT] Pattern mapping loaded: {num_classes} classes")
        logger.info(f"[BOT] Patterns: {list(self.pattern_map.keys())}")

        # NOW initialize Sniper with correct number of classes
        self.sniper = OHLCSniper(
            input_shape=(15, 4), num_classes=num_classes, dropout_rate=0.3
        )

        # Load weights
        self.sniper.load_model(model_path)

        logger.info(f"[BOT] ✅ Model loaded successfully")
        logger.info(f"[BOT]   Classes: {num_classes}")
        logger.info(f"[BOT]   Patterns: {', '.join(self.pattern_map.keys())}")

    def _extract_pivots_4h(self, df_4h: pd.DataFrame, window: int = 5) -> np.ndarray:
        """Extract pivots from 4H data"""
        highs = df_4h["high"].values
        lows = df_4h["low"].values

        pivots = []

        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i - window : i + window + 1]):
                pivots.append(highs[i])

        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i - window : i + window + 1]):
                pivots.append(lows[i])

        return np.array(pivots)

    def analyze_market(
        self,
        df_4h: pd.DataFrame,
        df_15m: pd.DataFrame,
        force_update_analyst: bool = False,
    ) -> Optional[Dict]:
        """
        Complete market analysis with dual timeframes

        Args:
            df_4h: Recent 4H candles (for Analyst)
            df_15m: Recent 15min candles (for Sniper)
            force_update_analyst: Force S/R recalculation

        Returns:
            Trading signal dict or None
        """

        if self.sniper is None:
            raise RuntimeError("Model not loaded! Call load_model() first.")

        # =====================================================================
        # STEP 1: ANALYST (4H Strategic Analysis)
        # =====================================================================

        if force_update_analyst or self.cached_levels is None:
            logger.info("[BOT] 🔄 Updating Analyst (4H analysis)...")

            pivot_points = self._extract_pivots_4h(df_4h)

            if len(pivot_points) >= 3:
                self.cached_levels = self.analyst.get_support_resistance_levels(
                    pivot_points,
                    df_4h["high"].values,
                    df_4h["low"].values,
                    df_4h["close"].values,
                    n_levels=5,
                )

                logger.info(
                    f"[BOT] ✓ Analyst: {len(self.cached_levels)} levels from 4H data"
                )
            else:
                logger.warning("[BOT] Insufficient 4H pivots")
                return None

        if not self.cached_levels:
            logger.info("[BOT] No S/R levels available")
            return None

        # Check current price against 4H levels
        current_price = float(df_15m.iloc[-1]["close"])

        near_level = None
        for level in self.cached_levels:
            if self.analyst.is_near_level(current_price, level, self.threshold_percent):
                near_level = level
                logger.info(
                    f"[BOT] 📍 Price ${current_price:.2f} near 4H level ${level:.2f}"
                )
                break

        if not near_level:
            logger.debug("[BOT] Price away from 4H levels")
            return None

        # =====================================================================
        # STEP 2: SNIPER (15min Tactical Analysis)
        # =====================================================================

        logger.info("[BOT] 🎯 Activating Sniper (15min analysis)...")

        # Get last 15 candles from 15min timeframe
        if len(df_15m) < 15:
            logger.warning("[BOT] Need at least 15 candles of 15min data")
            return None

        last_15_candles = df_15m[["open", "high", "low", "close"]].tail(15).values

        pattern_id, confidence, extra = self.sniper.predict(last_15_candles)

        if pattern_id == 0:
            logger.info("[BOT] Sniper: No pattern (noise)")
            return None

        if confidence < self.confidence_threshold:
            pattern_name = self.reverse_map.get(pattern_id, "Unknown")
            logger.info(
                f"[BOT] Sniper: Low confidence ({confidence:.1%}) for {pattern_name}"
            )
            return None

        pattern_name = self.reverse_map.get(pattern_id, "Unknown")

        # =====================================================================
        # STEP 3: COMBINE SIGNALS
        # =====================================================================

        classified = self.analyst.classify_levels([near_level], current_price)

        bullish_patterns = [
            "Engulfing",
            "Morning Star",
            "Hammer",
            "Dragonfly Doji",
            "Inverted Hammer",
            "Three White Soldiers",
            "Piercing",
        ]

        bearish_patterns = [
            "Evening Star",
            "Shooting Star",
            "Hanging Man",
            "Gravestone Doji",
            "Three Black Crows",
            "Dark Cloud",
        ]

        signal = {
            "action": None,
            "pattern": pattern_name,
            "confidence": confidence,
            "level_4h": near_level,
            "price_15m": current_price,
            "timeframe_analyst": "4H",
            "timeframe_sniper": "15min",
            "top3_patterns": [
                self.reverse_map.get(pid, f"ID_{pid}") for pid in extra["top3_ids"]
            ],
            "top3_confidences": extra["top3_confidences"],
        }

        if pattern_name in bullish_patterns and classified["support"]:
            signal["action"] = "BUY"
            logger.info(
                f"[BOT] 🟢 BUY SIGNAL\n"
                f"  Pattern: {pattern_name} ({confidence:.1%})\n"
                f"  4H Support: ${near_level:.2f}\n"
                f"  15min Entry: ${current_price:.2f}"
            )

        elif pattern_name in bearish_patterns and classified["resistance"]:
            signal["action"] = "SELL"
            logger.info(
                f"[BOT] 🔴 SELL SIGNAL\n"
                f"  Pattern: {pattern_name} ({confidence:.1%})\n"
                f"  4H Resistance: ${near_level:.2f}\n"
                f"  15min Entry: ${current_price:.2f}"
            )

        return signal


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("dual_timeframe_training.log"),
            logging.StreamHandler(),
        ],
    )

    # =========================================================================
    # TRAINING PHASE
    # =========================================================================

    print("\n" + "=" * 80)
    print("TRAINING DUAL TIMEFRAME SYSTEM")
    print("=" * 80)

    # Train the system
    sniper, pattern_map, history, config = train_dual_timeframe_system(
        assets=["btc", "gold", "ustec", "eurusd", "usoil", "gbpaud", "gbpusd", "usdjpy"],
        data_folder=None, # ✅ Defaults to project_root / "data" / "raw"
        samples_per_pattern=2000,
        min_samples_per_class=50,
        epochs=300,
        batch_size=64,
        validation_split=0.2,
        use_class_weights=True,
        max_class_weight=10.0,
        model_name="sniper_dual_timeframe_v1",
        save_plots=True,
    )

    # =========================================================================
    # PRODUCTION USAGE EXAMPLE
    # =========================================================================

    print("\n" + "=" * 80)
    print("PRODUCTION USAGE EXAMPLE")
    print("=" * 80)

    # Initialize bot
    bot = DualTimeframeTradingBot(threshold_percent=0.005, confidence_threshold=0.85)

    # Load trained model
    bot.load_model(
        model_path=str(project_root / "models" / "ai" / "sniper_dual_timeframe_v1.weights.h5"),
        mapping_path=str(project_root / "models" / "ai" / "sniper_dual_timeframe_v1_mapping.pkl"),
    )

    # Example: Load test data
    # df_4h = pd.read_csv('data/test_data_btc_4h.csv')
    # df_15m = pd.read_csv('data/test_data_btc_15m.csv')

    # Analyze market
    # signal = bot.analyze_market(df_4h.tail(500), df_15m.tail(500))

    # if signal and signal['action']:
    #     print(f"\n{'='*60}")
    #     print(f"TRADE SIGNAL: {signal['action']}")
    #     print(f"{'='*60}")
    #     print(f"Pattern (15min):    {signal['pattern']} ({signal['confidence']:.1%})")
    #     print(f"S/R Level (4H):     ${signal['level_4h']:.2f}")
    #     print(f"Entry Price (15min): ${signal['price_15m']:.2f}")
    #     print(f"Top-3 Patterns:     {', '.join(signal['top3_patterns'])}")
    #     print(f"{'='*60}")

    print("\n✅ System ready for production!")
    print("Next steps:")
    print("  1. Prepare your data:")
    print("     - data/train_data_btc_15m.csv")
    print("     - data/train_data_btc_4h.csv")
    print("  2. Run training script")
    print("  3. Use bot.analyze_market(df_4h, df_15m) in production")
    print("=" * 80)
