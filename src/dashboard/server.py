"""
Flask Dashboard Server for Trading Bot
======================================
Serves the real-time dashboard and provides API endpoints
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import json
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client, Client
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))  # src/dashboard
src_dir = os.path.dirname(current_dir)  # src
project_root = os.path.dirname(src_dir)  # TBOT root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis.storyteller import TradeStoryteller
from src.analysis.gemini_exporter import GeminiExporter
from src.database.database_manager import TradingDatabaseManager

from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
app = Flask(__name__)
CORS(app)  # Enable CORS for real-time updates


# Load from environment or config
SUPABASE_URL = os.getenv("SUPABASE_URL", "YOUR_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "YOUR_SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# Initialize DB Manager (For Analysis Tools)
db_manager = TradingDatabaseManager(SUPABASE_URL, SUPABASE_KEY)

# ============================================================================
# ROUTES
# ============================================================================


@app.route("/")
def index():
    """Serve the main dashboard"""
    try:
        # Read the HTML dashboard file (absolute path, CWD-independent)
        _dashboard_path = os.path.join(current_dir, "templates", "dashboard.html")
        with open(_dashboard_path, "r", encoding="utf-8") as f:
            dashboard_html = f.read()

        # Inject Supabase credentials (client-safe anon key)
        dashboard_html = dashboard_html.replace(
            "'YOUR_SUPABASE_URL'", f"'{SUPABASE_URL}'"
        )
        dashboard_html = dashboard_html.replace(
            "'YOUR_SUPABASE_ANON_KEY'",
            f"'{os.getenv('SUPABASE_ANON_KEY', SUPABASE_KEY)}'",
        )

        return render_template_string(dashboard_html)

    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return jsonify({"error": "Failed to load dashboard"}), 500


@app.route("/debrief")
def debrief_page():
    """Serve the Daily/Weekly Debrief Page"""
    try:
        # Try multiple possible locations for the template
        possible_paths = [
            "src/dashboard/templates/daily_recap.html",
            "templates/daily_recap.html",
            os.path.join(current_dir, "templates", "daily_recap.html"),
            os.path.join(
                project_root, "src", "dashboard", "templates", "daily_recap.html"
            ),
        ]

        template_path = None
        for path in possible_paths:
            if os.path.exists(path):
                template_path = path
                logger.info(f"Found template at: {path}")
                break

        if not template_path:
            error_msg = f"daily_recap.html not found. Searched in:\n"
            error_msg += "\n".join(f"  - {p}" for p in possible_paths)
            logger.error(error_msg)
            return error_msg, 404

        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        return render_template_string(content)

    except Exception as e:
        logger.error(f"Error serving debrief: {e}")
        return f"Error loading debrief page: {str(e)}", 500


# ============================================================================
# API - ANALYSIS & FEEDBACK LOOP
# ============================================================================


@app.route("/api/debrief_data")
def get_debrief_data():
    """Get structured data for the visual debrief"""
    try:
        date_ref = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        mode = request.args.get("mode", "daily")

        storyteller = TradeStoryteller(db_manager)
        data = storyteller.generate_report(mode, date_ref)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Debrief data error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/export_gemini")
def export_gemini_report():
    """Generate text prompt for Gemini analysis"""
    try:
        date_ref = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        mode = request.args.get("mode", "daily")

        exporter = GeminiExporter(db_manager)
        report_text = exporter.generate_report(mode, date_ref)

        return jsonify({"status": "success", "report": report_text})
    except Exception as e:
        logger.error(f"Gemini export error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history/<asset>")
def get_history(asset):
    """
    Serves candle data for the chart from local CSVs.
    Used by the Lightweight Charts in /debrief
    """
    start = request.args.get("start")
    end = request.args.get("end")

    # Map friendly names to CSV filenames
    filename_map = {
        "BTC":    "BTCUSDT_1h.csv",
        "GOLD":   "XAUUSDm_1h.csv",
        "XAU":    "XAUUSDm_1h.csv",
        "USTEC":  "USTECm_1h.csv",
        "EURJPY": "EURJPYm_1h.csv",
        "EURUSD": "EURUSDm_1h.csv",
        "USOIL":  "USOILm_1h.csv",
        "GBPAUD": "GBPAUDm_1h.csv",
        "GBPUSD": "GBPUSDm_1h.csv",
        "USDJPY": "USDJPYm_1h.csv",
    }

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_file = filename_map.get(asset.upper())
    if not csv_file:
        return jsonify([])

    # Check multiple locations
    paths = [
        os.path.join(BASE_DIR, "data", "raw", csv_file),
    ]

    print("[DEBUG] CWD:", os.getcwd())
    print("[DEBUG] Checking paths:")
    csv_path = None
    for p in paths:
        print(" -", p)
        if os.path.exists(p):
            csv_path = p
            break

    if not csv_path:
        print(f"[ERROR] CSV file not found for {asset}: {csv_file}")
        return jsonify([])

    try:
        df = pd.read_csv(csv_path)

        # Handle different column names for timestamp
        timestamp_col = None
        for col in ["timestamp", "time", "date", "datetime"]:
            if col in df.columns:
                timestamp_col = col
                break

        if not timestamp_col:
            print(f"[ERROR] No timestamp column in {csv_path}")
            print(f"Available columns: {df.columns.tolist()}")
            return jsonify([])

        # Rename to standard 'timestamp' for consistency
        if timestamp_col != "timestamp":
            df.rename(columns={timestamp_col: "timestamp"}, inplace=True)

        # Convert to datetime with UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Parse start/end parameters
        if start:
            try:
                start_dt = pd.to_datetime(start, utc=True)
                df = df[df["timestamp"] >= start_dt]
            except Exception as e:
                print(f"[WARNING] Could not parse start date '{start}': {e}")

        if end:
            try:
                end_dt = pd.to_datetime(end, utc=True)
                df = df[df["timestamp"] <= end_dt]
            except Exception as e:
                print(f"[WARNING] Could not parse end date '{end}': {e}")

        print(f"[INFO] Filtered {len(df)} candles for {asset} from {start} to {end}")

        if len(df) == 0:
            print(
                f"[WARNING] No data in date range. CSV range: {pd.to_datetime(pd.read_csv(csv_path)[timestamp_col]).min()} to {pd.to_datetime(pd.read_csv(csv_path)[timestamp_col]).max()}"
            )

        # Format for Lightweight Charts (Unix timestamp in SECONDS)
        chart_data = []
        for _, row in df.iterrows():
            chart_data.append(
                {
                    "time": int(row["timestamp"].timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )

        return jsonify(chart_data)

    except Exception as e:
        print(f"[ERROR] Failed to load history for {asset}: {e}")
        import traceback

        traceback.print_exc()
        return jsonify([])


@app.route("/api/stats")
def get_stats():
    """Get current portfolio statistics"""
    try:
        # Get latest portfolio snapshot
        snapshot = (
            supabase.table("portfolio_snapshots")
            .select("*")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )

        # Get performance stats
        trades = supabase.table("trades").select("*").eq("status", "closed").execute()
        
        # Calculate strategy stats
        strategy_stats = {}
        if trades.data:
            for t in trades.data:
                strat = t.get("strategy", "UNKNOWN") or "UNKNOWN"
                if strat not in strategy_stats:
                    strategy_stats[strat] = {"wins": 0, "losses": 0, "pnl": 0.0}
                
                pnl = t.get("pnl", 0)
                if pnl > 0:
                    strategy_stats[strat]["wins"] += 1
                else:
                    strategy_stats[strat]["losses"] += 1
                strategy_stats[strat]["pnl"] += pnl

        stats = {
            "portfolio": snapshot.data[0] if snapshot.data else None,
            "trade_count": len(trades.data) if trades.data else 0,
            "win_rate": calculate_win_rate(trades.data) if trades.data else 0,
            "strategy_performance": strategy_stats,
            "timestamp": datetime.now().isoformat(),
        }

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/trades/open")
def get_open_trades():
    """Get all open trades"""
    try:
        result = (
            supabase.table("trades")
            .select("*")
            .eq("status", "open")
            .order("entry_time", desc=True)
            .execute()
        )

        return jsonify({"trades": result.data})

    except Exception as e:
        logger.error(f"Error getting open trades: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/trades/closed")
def get_closed_trades():
    """Get recent closed trades"""
    try:
        limit = request.args.get("limit", 20, type=int)

        result = (
            supabase.table("trades")
            .select("*")
            .eq("status", "closed")
            .order("exit_time", desc=True)
            .limit(limit)
            .execute()
        )

        return jsonify({"trades": result.data})

    except Exception as e:
        logger.error(f"Error getting closed trades: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/regime/<asset>")
def get_regime(asset):
    """
    Fetches the latest multi-timeframe regime analysis for a given asset and date.
    If no data exists for the selected date, returns the most recent data.
    """
    try:
        date = request.args.get("date")

        # Query regime data for the selected asset
        query = (
            supabase.table("mtf_regime_analysis").select("*").eq("asset", asset.upper())
        )

        if date:
            # Convert the date to a range for the day (start and end of the day)
            start_of_day = f"{date}T00:00:00+00:00"
            end_of_day = f"{date}T23:59:59+00:00"

            # Filter by timestamp range
            query = query.gte("timestamp", start_of_day).lte("timestamp", end_of_day)

        # Order by timestamp (descending) and fetch the latest record
        result = query.order("timestamp", desc=True).limit(1).execute()

        if not result.data:
            # If no data for the selected date, fetch the most recent data
            result = (
                supabase.table("mtf_regime_analysis")
                .select("*")
                .eq("asset", asset.upper())
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )

            if not result.data:
                return jsonify({"error": f"No regime data found for {asset}"}), 404

        regime_data = result.data[0]

        # Format the response
        response = {
            "asset": regime_data["asset"],
            "timestamp": regime_data["timestamp"],
            "consensus": {
                "regime": regime_data["consensus_regime"],
                "confidence": regime_data["consensus_confidence"],
                "timeframe_agreement": regime_data["timeframe_agreement"],
                "trend_coherence": regime_data["trend_coherence"],
            },
            "risk": {
                "level": regime_data["risk_level"],
                "volatility": regime_data["volatility_regime"],
            },
            "trading_implications": {
                "recommended_mode": regime_data["recommended_mode"],
                "allow_counter_trend": regime_data["allow_counter_trend"],
                "suggested_max_positions": regime_data["suggested_max_positions"],
            },
            "timeframes": {
                "1h": {
                    "regime": regime_data["h1_regime"],
                    "confidence": regime_data["h1_confidence"],
                    "trend_strength": regime_data["h1_trend_strength"],
                    "trend_direction": regime_data["h1_trend_direction"],
                    "adx": regime_data["h1_adx"],
                    "rsi": regime_data["h1_rsi"],
                    "ema_diff_pct": regime_data["h1_ema_diff_pct"],
                },
                "4h": {
                    "regime": regime_data["h4_regime"],
                    "confidence": regime_data["h4_confidence"],
                    "trend_strength": regime_data["h4_trend_strength"],
                    "trend_direction": regime_data["h4_trend_direction"],
                    "adx": regime_data["h4_adx"],
                    "rsi": regime_data["h4_rsi"],
                    "ema_diff_pct": regime_data["h4_ema_diff_pct"],
                },
                "1d": {
                    "regime": regime_data["d1_regime"],
                    "confidence": regime_data["d1_confidence"],
                    "trend_strength": regime_data["d1_trend_strength"],
                    "trend_direction": regime_data["d1_trend_direction"],
                    "adx": regime_data["d1_adx"],
                    "rsi": regime_data["d1_rsi"],
                    "ema_diff_pct": regime_data["d1_ema_diff_pct"],
                },
            },
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error fetching regime data for {asset}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals")
def get_signals():
    """Get recent trading signals"""
    try:
        limit = request.args.get("limit", 20, type=int)
        asset = request.args.get("asset", None)

        query = (
            supabase.table("signals")
            .select("*")
            .order("timestamp", desc=True)
            .limit(limit)
        )
        if asset:
            query = query.eq("asset", asset)

        result = query.execute()

        return jsonify({"signals": result.data})

    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals/history")
def get_signals_history():
    """Get paginated signal history with optional date-range and asset filter"""
    try:
        days      = request.args.get("days",      30,  type=int)
        page      = request.args.get("page",      1,   type=int)
        page_size = request.args.get("page_size", 50,  type=int)
        asset     = request.args.get("asset",     None)
        is_export = request.args.get("export",    "false").lower() == "true"

        page_size = min(page_size, 10000 if is_export else 200)   # relaxed cap for full export
        page      = max(page, 1)
        offset    = (page - 1) * page_size

        start_time = (datetime.now() - timedelta(days=days)).isoformat()

        query = (
            supabase.table("signals")
            .select("*", count="exact")
            .gte("timestamp", start_time)
            .order("timestamp", desc=True)
            .range(offset, offset + page_size - 1)
        )

        if asset:
            query = query.eq("asset", asset)

        result = query.execute()

        total       = result.count or 0
        total_pages = max(1, (total + page_size - 1) // page_size)

        return jsonify({
            "signals":     result.data,
            "total":       total,
            "page":        page,
            "page_size":   page_size,
            "total_pages": total_pages,
            "days":        days,
        })

    except Exception as e:
        logger.error(f"Error getting signal history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/portfolio/history")
def get_portfolio_history():
    """Get portfolio value history"""
    try:
        hours = request.args.get("hours", 24, type=int)
        start_time = datetime.now() - timedelta(hours=hours)

        result = (
            supabase.table("portfolio_snapshots")
            .select("timestamp, total_value, unrealized_pnl")
            .gte("timestamp", start_time.isoformat())
            .order("timestamp", desc=False)
            .execute()
        )

        return jsonify({"history": result.data})

    except Exception as e:
        logger.error(f"Error getting portfolio history: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/performance")
def get_performance():
    """Get performance metrics"""
    try:
        days = request.args.get("days", 7, type=int)
        start_date = datetime.now() - timedelta(days=days)

        # Get trades in date range
        trades = (
            supabase.table("trades")
            .select("*")
            .eq("status", "closed")
            .gte("exit_time", start_date.isoformat())
            .execute()
        )

        if not trades.data:
            return jsonify(
                {
                    "total_trades": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "profit_factor": 0,
                }
            )

        # Calculate metrics
        total_trades = len(trades.data)
        winning_trades = [t for t in trades.data if t["pnl"] > 0]
        losing_trades = [t for t in trades.data if t["pnl"] < 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)

        total_pnl = sum(t["pnl"] for t in trades.data)
        avg_win = (
            sum(t["pnl"] for t in winning_trades) / win_count if win_count > 0 else 0
        )
        avg_loss = (
            sum(t["pnl"] for t in losing_trades) / loss_count if loss_count > 0 else 0
        )

        profit_factor = (
            abs(avg_win * win_count / (avg_loss * loss_count))
            if loss_count > 0 and avg_loss != 0
            else 0
        )

        return jsonify(
            {
                "total_trades": total_trades,
                "winning_trades": win_count,
                "losing_trades": loss_count,
                "win_rate": (win_count / total_trades * 100) if total_trades > 0 else 0,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "best_trade": (
                    max(trades.data, key=lambda x: x["pnl"])["pnl"]
                    if trades.data
                    else 0
                ),
                "worst_trade": (
                    min(trades.data, key=lambda x: x["pnl"])["pnl"]
                    if trades.data
                    else 0
                ),
            }
        )

    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/system/events")
def get_system_events():
    """Get recent system events"""
    try:
        limit = request.args.get("limit", 50, type=int)
        severity = request.args.get("severity")

        query = (
            supabase.table("system_events")
            .select("*")
            .order("timestamp", desc=True)
            .limit(limit)
        )

        if severity:
            query = query.eq("severity", severity)

        result = query.execute()

        return jsonify({"events": result.data})

    except Exception as e:
        logger.error(f"Error getting system events: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/templates/<path:filename>")
def serve_template(filename):
    """Serve documentation template files"""
    try:
        file_path = os.path.join("templates", filename)

        # Security check - ensure file is in templates directory
        if not os.path.abspath(file_path).startswith(os.path.abspath("templates")):
            return jsonify({"error": "Invalid file path"}), 403

        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return content, 200, {"Content-Type": "text/html; charset=utf-8"}

    except Exception as e:
        logger.error(f"Error serving template {filename}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs")
def get_logs():
    """Read the last N lines of the log file"""
    try:
        log_type = request.args.get("type", "bot")
        
        if log_type == "audit":
            log_file = os.path.join(project_root, "logs", "trade_audit.log")
        else:
            log_file = os.path.join(project_root, "logs", "trading_bot.log")
            
        if not os.path.exists(log_file):
            return jsonify({"logs": f"Log file {os.path.basename(log_file)} not found."})

        # Read last lines
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            # Efficiently read last lines
            lines = f.readlines()
            last_lines = lines[-2000:]
            return jsonify({"logs": "".join(last_lines)})

    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/audit")
def get_audit_events():
    """Parse trade_audit.log and return structured event objects (newest first)"""
    try:
        log_file = os.path.join(project_root, "logs", "trade_audit.log")
        if not os.path.exists(log_file):
            return jsonify({"events": [], "total": 0})

        limit = min(int(request.args.get("limit", 500)), 2000)

        events = []
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                # Format 1: raw JSON line in trade_audit.log
                if line.startswith("{"):
                    ev = json.loads(line)
                    events.append(ev)
                # Format 2: trading_bot.log with [TRADE_EVENT] prefix
                elif "[TRADE_EVENT]" in line:
                    ev = json.loads(line.split("[TRADE_EVENT]", 1)[1].strip())
                    events.append(ev)
            except Exception:
                continue

        # Newest first, capped at limit
        events = events[-limit:]
        events.reverse()
        return jsonify({"events": events, "total": len(events)})
    except Exception as e:
        logger.error(f"Error reading audit: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/shadow/state")
def get_shadow_state():
    """
    Returns the full shadow trading snapshot written by ShadowTradingEngine.dump_state().
    The bot writes this file every candle to logs/shadow_state.json.
    """
    try:
        state_path = os.path.join(project_root, "logs", "shadow_state.json")
        if not os.path.exists(state_path):
            return jsonify({
                "open_positions": [],
                "closed_results": [],
                "gate_scorecard": {},
                "strategy_scorecard": {},
                "summary": {"open_count": 0, "closed_count": 0},
                "last_updated": None,
                "available": False,
            })
        import json as _json
        with open(state_path, "r", encoding="utf-8") as f:
            state = _json.load(f)
        state["available"] = True
        return jsonify(state)
    except Exception as e:
        logger.error(f"Shadow state error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/config/overview")
def get_config_overview():
    """
    Returns a sanitised view of config.json for display in the dashboard.
    Strips API keys and passwords.
    """
    try:
        import json as _json
        cfg_path = os.path.join(project_root, "config", "config.json")
        if not os.path.exists(cfg_path):
            return jsonify({"error": "config.json not found"}), 404

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = _json.load(f)

        # Sanitise — remove credentials
        _REDACT = {
            "api_key", "api_secret", "secret", "password", "token",
            "supabase_url", "supabase_key", "telegram_token",
        }
        def _scrub(obj):
            if isinstance(obj, dict):
                return {
                    k: "***" if k.lower() in _REDACT else _scrub(v)
                    for k, v in obj.items()
                }
            if isinstance(obj, list):
                return [_scrub(x) for x in obj]
            return obj

        safe = _scrub(cfg)

        # Extract the most useful slices for the dashboard
        assets_cfg = {}
        for asset in safe.keys():
            if isinstance(safe[asset], dict) and "exchange" in safe[asset]:
                assets_cfg[asset] = safe[asset]

        # Economic calendar: config.json may not have this key — read the
        # dedicated file instead and synthesize the display object.
        eco_cfg = safe.get("economic_calendar", {})
        if not eco_cfg.get("events"):
            cal_path = os.path.join(project_root, "config", "economic_calendar.json")
            if os.path.exists(cal_path):
                try:
                    with open(cal_path, "r", encoding="utf-8") as _cf:
                        cal_data = _json.load(_cf)
                    from datetime import datetime as _dt, timezone as _tz
                    _now = _dt.now(_tz.utc)
                    # Include only future events (or within the last 24h), sorted soonest first
                    from datetime import timedelta as _td
                    upcoming = [
                        e for e in cal_data.get("events", [])
                        if _dt.fromisoformat(e["datetime"].replace("Z", "+00:00")) >= _now - _td(hours=24)
                    ]
                    upcoming.sort(key=lambda e: e["datetime"])
                    eco_cfg = {
                        "enabled": True,
                        "source": "config/economic_calendar.json",
                        "total_events": len(cal_data.get("events", [])),
                        "upcoming_count": len(upcoming),
                        "block_hours_before": 2,
                        "events": upcoming[:15],   # show next 15
                    }
                except Exception as _e:
                    eco_cfg = {"enabled": False, "error": str(_e), "events": []}
            else:
                eco_cfg = {"enabled": False, "events": [], "note": "economic_calendar.json not found"}

        result = {
            "trading_mode": safe.get("trading_mode", safe.get("mode", "unknown")),
            "assets": assets_cfg,
            "risk": safe.get("risk", {}),
            "strategies": safe.get("strategies", {}),
            "aggregator": safe.get("aggregator", safe.get("signal_aggregator", {})),
            "economic_calendar": eco_cfg,
            "session_blocks": safe.get("session_blocks", {}),
            "circuit_breaker": safe.get("circuit_breaker", {}),
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Config overview error: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# CONTROL CENTER — editable config + restart
# ============================================================================

# The exact config paths the Control Center is allowed to read & write.
# format: (dot_path, label, type, extra)
#   type: "bool" | "float" | "int" | "pct"  (pct = float stored as 0–1, shown as %)
#   extra: dict with optional min/max/step
_EDITABLE_FIELDS = [
    # ── Global: Aggregator ───────────────────────────────────────────────────
    # Paths match config.json key "aggregator_settings.*"
    ("aggregator_settings.use_macro_governor",     "Governor (Macro Filter)",        "bool",  {}),
    ("aggregator_settings.use_gatekeeper",         "Gatekeeper (Smart Routing)",     "bool",  {}),
    ("aggregator_settings.trend_aligned_threshold","Trend Score Threshold",          "float", {"min": 1.0, "max": 6.0, "step": 0.25}),
    ("aggregator_settings.counter_trend_threshold","Counter-Trend Score Threshold",  "float", {"min": 1.0, "max": 6.0, "step": 0.25}),
    # ── Global: Circuit Breaker ──────────────────────────────────────────────
    # max_daily_loss_pct lives under risk_management; max_drawdown under portfolio
    ("risk_management.max_daily_loss_pct",  "Max Daily Loss",  "pct", {"min": 0.005, "max": 0.15, "step": 0.005}),
    ("portfolio.max_drawdown",              "Max Drawdown",    "pct", {"min": 0.05,  "max": 0.40, "step": 0.01}),
    # ── Global: Portfolio ────────────────────────────────────────────────────
    ("portfolio.target_risk_per_trade",   "Risk Per Trade",            "pct",   {"min": 0.005, "max": 0.05, "step": 0.005}),
    ("portfolio.max_portfolio_exposure",  "Max Portfolio Exposure (x)","float", {"min": 1.0,   "max": 10.0, "step": 0.5}),
    # ── Global: VTM Execution ─────────────────────────────────────────────────
    ("trading.enabled",                   "Global Trading Enabled",          "bool", {}),
    ("trading.place_vtm_sl_on_exchange",  "Place VTM Stop Loss on Exchange", "bool", {}),
    ("trading.place_vtm_tp_on_exchange",  "Place VTM Take Profit on Exchange", "bool", {}),
]

_ASSETS = ["BTC", "GOLD", "EURUSD", "EURJPY", "USTEC", "USOIL", "GBPAUD", "GBPUSD", "USDJPY"]
_ASSET_FIELDS = [
    # (sub_path,                            label,                           type,    extra)
    ("enabled",                             "Enabled",                       "bool",  {}),
    ("allow_shorts",                        "Allow Shorts",                  "bool",  {}),
    ("leverage",                            "Leverage",                      "int",   {"min": 1, "max": 200, "step": 1}),
    ("weight",                              "Asset Weight",                  "float", {"min": 0.0, "max": 2.0, "step": 0.1}),
    ("strategies.mean_reversion.enabled",   "Mean Reversion: Enabled",       "bool",  {}),
    ("strategies.mean_reversion.weight",    "Mean Reversion: Weight",        "float", {"min": 0.0, "max": 2.0, "step": 0.1}),
    ("strategies.trend_following.enabled",  "Trend Following: Enabled",      "bool",  {}),
    ("strategies.trend_following.weight",   "Trend Following: Weight",       "float", {"min": 0.0, "max": 2.0, "step": 0.1}),
    ("strategies.exponential_moving_averages.enabled", "EMA Crossover: Enabled", "bool", {}),
    ("strategies.exponential_moving_averages.weight",  "EMA Crossover: Weight",  "float", {"min": 0.0, "max": 2.0, "step": 0.1}),
    ("risk.atr_multiplier",                 "ATR Multiplier (SL distance)",  "float", {"min": 0.5, "max": 5.0, "step": 0.1}),
    ("risk.use_ema_structure",              "EMA Structure (MA Shield/TP)",  "bool",  {}),
    ("risk.use_structure_targets",          "Pivot Structure Targets",       "bool",  {}),
    ("risk.early_scale_enabled",            "Early Scale Exit",              "bool",  {}),
    ("risk.early_scale_threshold",          "Early Scale Threshold",         "pct",   {"min": 0.005, "max": 0.10, "step": 0.005}),
    ("risk.breakeven_after_bars",           "Break-Even After (bars)",       "int",   {"min": 1, "max": 48, "step": 1}),
    ("risk.runner_trail_atr_multiplier",    "Runner Trail ATR Multiplier",   "float", {"min": 0.5, "max": 5.0, "step": 0.1}),
    ("risk.time_stop_trend",                "Time Stop — Trend (bars)",      "int",   {"min": 12, "max": 240, "step": 4}),
    ("risk.time_stop_reversion",            "Time Stop — Reversion (bars)",  "int",   {"min": 4,  "max": 72,  "step": 2}),
]

def _cfg_get(cfg, dot_path):
    """Walk a dot-separated path into a nested dict, return value or None."""
    keys = dot_path.split(".")
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def _cfg_set(cfg, dot_path, value):
    """Walk a dot-separated path and set the leaf value (creates dicts as needed)."""
    keys = dot_path.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


@app.route("/api/config/editable")
def get_editable_config():
    """Returns the subset of config values the Control Center can edit."""
    try:
        import json as _json
        cfg_path = os.path.join(project_root, "config", "config.json")
        if not os.path.exists(cfg_path):
            return jsonify({"error": "config.json not found"}), 404
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = _json.load(f)

        result = {"global": {}, "assets": {}}

        # Global fields
        for dot_path, label, ftype, extra in _EDITABLE_FIELDS:
            section = dot_path.split(".")[0]  # "aggregator", "circuit_breaker", "portfolio"
            val = _cfg_get(cfg, dot_path)
            result["global"].setdefault(section, {})[dot_path] = {
                "label": label, "type": ftype, "value": val, **extra
            }

        # Per-asset fields
        assets_root = cfg.get("assets", cfg)  # handle both nested and flat layouts
        for asset in _ASSETS:
            asset_cfg = assets_root.get(asset, {})
            result["assets"][asset] = {}
            for sub_path, label, ftype, extra in _ASSET_FIELDS:
                val = _cfg_get(asset_cfg, sub_path)
                full_path = f"assets.{asset}.{sub_path}"
                result["assets"][asset][full_path] = {
                    "label": label, "type": ftype, "value": val, **extra
                }

        return jsonify(result)
    except Exception as e:
        logger.error(f"Editable config error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/config/save", methods=["POST"])
def save_config():
    """
    Receives {updates: {dot_path: new_value, ...}}, validates each path is
    in the allowed set, deep-merges into config.json (with backup), then
    writes the updated file.
    """
    try:
        import json as _json, shutil, copy

        data = request.get_json(force=True, silent=True) or {}
        updates = data.get("updates", {})
        if not updates:
            return jsonify({"error": "No updates provided"}), 400

        # Build whitelist of allowed dot paths
        allowed = set()
        for dot_path, *_ in _EDITABLE_FIELDS:
            allowed.add(dot_path)
        for asset in _ASSETS:
            for sub_path, *_ in _ASSET_FIELDS:
                allowed.add(f"assets.{asset}.{sub_path}")

        rejected = [p for p in updates if p not in allowed]
        if rejected:
            return jsonify({"error": f"Forbidden paths: {rejected}"}), 403

        cfg_path = os.path.join(project_root, "config", "config.json")
        if not os.path.exists(cfg_path):
            return jsonify({"error": "config.json not found"}), 404

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = _json.load(f)

        # Backup before writing
        backup_path = cfg_path.replace(".json", ".backup.json")
        shutil.copy2(cfg_path, backup_path)

        # Coerce types and apply updates
        # Build type lookup
        type_map = {}
        for dot_path, _, ftype, _ in _EDITABLE_FIELDS:
            type_map[dot_path] = ftype
        for asset in _ASSETS:
            for sub_path, _, ftype, _ in _ASSET_FIELDS:
                type_map[f"assets.{asset}.{sub_path}"] = ftype

        assets_root_key = "assets" if "assets" in cfg else None

        for dot_path, raw_val in updates.items():
            ftype = type_map.get(dot_path, "float")
            try:
                if ftype == "bool":
                    val = bool(raw_val)
                elif ftype == "int":
                    val = int(raw_val)
                elif ftype in ("float", "pct"):
                    val = float(raw_val)
                else:
                    val = raw_val
            except (TypeError, ValueError):
                return jsonify({"error": f"Invalid value for {dot_path}: {raw_val}"}), 400

            # For asset paths, handle both cfg["assets"]["BTC"]... and cfg["BTC"]...
            if dot_path.startswith("assets.") and assets_root_key:
                _cfg_set(cfg, dot_path, val)
            elif dot_path.startswith("assets.") and not assets_root_key:
                # Flat layout: strip "assets." prefix
                _cfg_set(cfg, dot_path[len("assets."):], val)
            else:
                _cfg_set(cfg, dot_path, val)

        with open(cfg_path, "w", encoding="utf-8") as f:
            _json.dump(cfg, f, indent=2)

        logger.info(f"[CONFIG] Saved {len(updates)} changes via Control Center")
        return jsonify({"status": "saved", "changes": len(updates), "backup": backup_path})

    except Exception as e:
        logger.error(f"Config save error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/bot/restart", methods=["POST"])
def request_bot_restart():
    """Writes logs/restart.flag — main.py picks it up on next loop iteration."""
    try:
        flag_path = os.path.join(project_root, "logs", "restart.flag")
        os.makedirs(os.path.dirname(flag_path), exist_ok=True)
        with open(flag_path, "w") as f:
            f.write(datetime.utcnow().isoformat())
        logger.info("[CONTROL] Bot restart flag written")
        return jsonify({"status": "restart_requested", "flag": flag_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bot/pid")
def bot_pid():
    """Returns bot PID and whether the process is alive."""
    try:
        import psutil
        pid_path = os.path.join(project_root, "logs", "bot.pid")
        if not os.path.exists(pid_path):
            return jsonify({"running": False, "pid": None})
        with open(pid_path) as f:
            pid = int(f.read().strip())
        running = psutil.pid_exists(pid)
        return jsonify({"running": running, "pid": pid})
    except ImportError:
        # psutil not available — just report the PID file exists
        pid_path = os.path.join(project_root, "logs", "bot.pid")
        if os.path.exists(pid_path):
            with open(pid_path) as f:
                return jsonify({"running": "unknown", "pid": int(f.read().strip())})
        return jsonify({"running": False, "pid": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        result = supabase.table("trades").select("id").limit(1).execute()

        # Get last activity
        snapshot = (
            supabase.table("portfolio_snapshots")
            .select("timestamp")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )

        last_activity = snapshot.data[0]["timestamp"] if snapshot.data else None

        # Determine if bot is active (activity in last 10 minutes)
        is_active = False
        if last_activity:
            last_time = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
            is_active = (
                datetime.now(last_time.tzinfo) - last_time
            ).total_seconds() < 600

        return jsonify(
            {
                "status": "healthy",
                "database": "connected",
                "bot_active": is_active,
                "last_activity": last_activity,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def calculate_win_rate(trades):
    """Calculate win rate from trades"""
    if not trades:
        return 0
    winning = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return winning / len(trades) * 100


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Check configuration
    if SUPABASE_URL == "YOUR_SUPABASE_URL":
        logger.error("❌ Please set SUPABASE_URL in environment or code")
        exit(1)

    logger.info("=" * 80)
    logger.info("🚀 TOM's Trading Bot Dashboard Server")
    logger.info("=" * 80)
    logger.info(f"Dashboard: http://localhost:5000")
    logger.info(f"API Docs:  http://localhost:5000/api/health")
    logger.info(f"Debrief:   http://localhost:5000/debrief")
    logger.info("=" * 80)

    # Run server
    app.run(
        host="0.0.0.0",  # Allow external connections
        port=5000,
        debug=True,  # Set to False in production
        threaded=True,
    )
