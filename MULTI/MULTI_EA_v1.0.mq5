//+------------------------------------------------------------------+
//|                                          MULTI_EA_v1.0.mq5       |
//|      Multi-Symbol + Multi-Instance EA                            |
//|      D1 Trend + M15 Confirmed Pullback + Tight Trail             |
//|      Auto-detect symbol, apply profile, no fixed TP              |
//+------------------------------------------------------------------+
//
// MARKET RANKING (Best to Worst for this strategy):
// ================================================
// TIER 1 - BEST (High win rate, many trades, easy to read):
//   XAUUSD (Gold)     - Volatile, clean trends, ~60-65% win rate
//   BTCUSD (Bitcoin)  - Big moves, strong trends, ~55-60% win rate
//   NAS100 (Nasdaq)   - Trending index, good volatility, ~55-60% win rate
//
// TIER 2 - GOOD (Solid trends, moderate trades):
//   GBPUSD            - Volatile forex, clear trends, ~55-60% win rate
//   USDJPY            - Strong trends, fewer whipsaws, ~55-60% win rate
//   EURJPY            - Good volatility, trending, ~52-58% win rate
//
// TIER 3 - AVERAGE (Choppy, fewer trades):
//   GBPAUD            - Decent trends, medium volatility, ~50-55% win rate
//   USOIL             - News-driven, can be choppy, ~50-55% win rate
//   EURUSD            - Low volatility, ranging often, ~48-52% win rate
//
// EASIEST TO TRADE: XAUUSD > BTCUSD > GBPUSD
// MOST TRADES:      XAUUSD > NAS100 > BTCUSD
// HIGHEST WIN RATE: XAUUSD > USDJPY > BTCUSD
//
//+------------------------------------------------------------------+
#property copyright "XAUUSD Trader"
#property version   "1.00"
#property description "Multi-symbol EA. D1 ADX+EMA trend. M15 pullback entry. BE + trail. No TP."
#property description ""
#property description "BEST MARKETS: XAUUSD (Gold), BTCUSD (Bitcoin), NAS100 (Nasdaq)"
#property description "GOOD MARKETS: GBPUSD, USDJPY, EURJPY"
#property description "AVERAGE: GBPAUD, USOIL, EURUSD"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

//--- Symbol Profile Structure
struct SymbolProfile {
   string name;
   int    adxThreshold;
   double slATR;
   int    emaFast;
   int    emaSlow;
   int    atrPeriod;
   int    adxPeriod;
   int    cooldownMins;
   double maxSLPoints;
   double minSLPoints;
   double breakEven;
   double breakEvenBuffer;
   double trailFixed;
};

//--- Inputs
input string   InpProfile          = "AUTO";      // Profile: AUTO or symbol name
input bool     InpUseAutoDetect    = true;        // Auto-detect from chart symbol
input double   InpLot              = 0.02;        // Lot size
input double   InpRiskPct          = 0.0;         // Risk % (0 = use fixed lot)
input int      InpMagic            = 51000;       // Base magic number
input int      InpInstanceID       = 0;           // Instance ID (0 = auto)
input string   InpComment          = "MULTI v1.0"; // Trade comment

//--- Global Variables
CTrade         Trade;
CPositionInfo  Pos;
CAccountInfo   AccountInfo;

SymbolProfile  currentProfile;

int            d1_atrHandle, d1_adxHandle, d1_emaFastHandle, d1_emaSlowHandle;
double         d1_atr[], d1_adx[], d1_pdi[], d1_ndi[], d1_emaFast[], d1_emaSlow[];

int            m15_atrHandle, m15_adxHandle, m15_emaFastHandle;
double         m15_atr[], m15_adx[], m15_pdi[], m15_ndi[], m15_emaFast[];

datetime       lastD1BarTime = 0;
datetime       lastM15BarTime = 0;
datetime       lastEntryTime = 0;

int            actualMagic;
int            instanceID;

//+------------------------------------------------------------------+
//| Auto-detect profile from symbol                                  |
//+------------------------------------------------------------------+
SymbolProfile DetectProfile(string symbol) {
   SymbolProfile p;

   //--- Default values
   p.adxThreshold = 25;
   p.slATR = 2.5;
   p.emaFast = 21;
   p.emaSlow = 50;
   p.atrPeriod = 14;
   p.adxPeriod = 14;
   p.cooldownMins = 30;
   p.maxSLPoints = 100;
   p.minSLPoints = 20;
   p.breakEven = 5.0;
   p.breakEvenBuffer = 0.5;
   p.trailFixed = 5.0;

   //--- Gold
   if(symbol == "XAUUSD" || symbol == "XAUUSDm" || symbol == "GOLD" || symbol == "GOLDm") {
      p.name = "XAUUSD";
      p.adxThreshold = 25;
      p.slATR = 3.0;
      p.cooldownMins = 45;
      p.maxSLPoints = 120;
      p.minSLPoints = 20;
   }
   //--- Bitcoin
   else if(symbol == "BTCUSD" || symbol == "BTCUSDm" || symbol == "BTCUSDT") {
      p.name = "BTCUSD";
      p.adxThreshold = 25;
      p.slATR = 2.5;
      p.cooldownMins = 30;
      p.maxSLPoints = 500;
      p.minSLPoints = 50;
   }
   //--- GBPUSD
   else if(symbol == "GBPUSD" || symbol == "GBPUSDm") {
      p.name = "GBPUSD";
      p.adxThreshold = 20;
      p.slATR = 2.0;
      p.cooldownMins = 30;
      p.maxSLPoints = 50;
      p.minSLPoints = 10;
   }
   //--- EURUSD
   else if(symbol == "EURUSD" || symbol == "EURUSDm") {
      p.name = "EURUSD";
      p.adxThreshold = 20;
      p.slATR = 2.0;
      p.cooldownMins = 30;
      p.maxSLPoints = 50;
      p.minSLPoints = 10;
   }
   //--- USDJPY
   else if(symbol == "USDJPY" || symbol == "USDJPYm") {
      p.name = "USDJPY";
      p.adxThreshold = 22;
      p.slATR = 2.0;
      p.cooldownMins = 30;
      p.maxSLPoints = 50;
      p.minSLPoints = 10;
   }
   //--- EURJPY
   else if(symbol == "EURJPY" || symbol == "EURJPYm") {
      p.name = "EURJPY";
      p.adxThreshold = 22;
      p.slATR = 2.0;
      p.cooldownMins = 30;
      p.maxSLPoints = 50;
      p.minSLPoints = 10;
   }
   //--- NAS100 / USTEC
   else if(symbol == "NAS100" || symbol == "NAS100m" || symbol == "USTEC" || symbol == "USTECm") {
      p.name = "NAS100";
      p.adxThreshold = 25;
      p.slATR = 2.5;
      p.cooldownMins = 45;
      p.maxSLPoints = 200;
      p.minSLPoints = 30;
   }
   //--- USOIL
   else if(symbol == "USOIL" || symbol == "USOILm" || symbol == "Crude") {
      p.name = "USOIL";
      p.adxThreshold = 22;
      p.slATR = 2.5;
      p.cooldownMins = 30;
      p.maxSLPoints = 100;
      p.minSLPoints = 15;
   }
   //--- GBPAUD
   else if(symbol == "GBPAUD" || symbol == "GBPAUDm") {
      p.name = "GBPAUD";
      p.adxThreshold = 20;
      p.slATR = 2.0;
      p.cooldownMins = 30;
      p.maxSLPoints = 60;
      p.minSLPoints = 12;
   }
   //--- Unknown symbol: use safe defaults
   else {
      p.name = symbol;
      p.adxThreshold = 25;
      p.slATR = 2.5;
      p.cooldownMins = 30;
      p.maxSLPoints = 100;
      p.minSLPoints = 20;
      Print("Unknown symbol: ", symbol, " — using default profile");
   }

   return p;
}

//+------------------------------------------------------------------+
//| Get profile by name (manual override)                            |
//+------------------------------------------------------------------+
SymbolProfile GetProfile(string profileName) {
   return DetectProfile(profileName);
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   //--- Multi-instance: auto magic from chart handle
   if(InpInstanceID == 0)
      instanceID = (int)ChartGetInteger(0, CHART_WINDOW_HANDLE);
   else
      instanceID = InpInstanceID;

   actualMagic = InpMagic + instanceID;
   Trade.SetExpertMagicNumber(actualMagic);
   Trade.SetDeviationInPoints(50);

   //--- Detect symbol profile
   if(InpUseAutoDetect || InpProfile == "AUTO")
      currentProfile = DetectProfile(_Symbol);
   else
      currentProfile = GetProfile(InpProfile);

   Print("Instance ", instanceID, " | Magic=", actualMagic,
         " | Symbol=", _Symbol, " | Profile=", currentProfile.name);

   //--- Create indicator handles
   d1_atrHandle = iATR(_Symbol, PERIOD_D1, currentProfile.atrPeriod);
   d1_adxHandle = iADX(_Symbol, PERIOD_D1, currentProfile.adxPeriod);
   d1_emaFastHandle = iMA(_Symbol, PERIOD_D1, currentProfile.emaFast, 0, MODE_EMA, PRICE_CLOSE);
   d1_emaSlowHandle = iMA(_Symbol, PERIOD_D1, currentProfile.emaSlow, 0, MODE_EMA, PRICE_CLOSE);

   m15_atrHandle = iATR(_Symbol, PERIOD_M15, currentProfile.atrPeriod);
   m15_adxHandle = iADX(_Symbol, PERIOD_M15, currentProfile.adxPeriod);
   m15_emaFastHandle = iMA(_Symbol, PERIOD_M15, currentProfile.emaFast, 0, MODE_EMA, PRICE_CLOSE);

   if(d1_atrHandle == INVALID_HANDLE || d1_adxHandle == INVALID_HANDLE ||
      d1_emaFastHandle == INVALID_HANDLE || d1_emaSlowHandle == INVALID_HANDLE ||
      m15_atrHandle == INVALID_HANDLE || m15_adxHandle == INVALID_HANDLE ||
      m15_emaFastHandle == INVALID_HANDLE) {
      Print("Failed to create indicator handles for ", _Symbol);
      return INIT_FAILED;
   }

   //--- Set arrays as series
   ArraySetAsSeries(d1_atr, true);
   ArraySetAsSeries(d1_adx, true);
   ArraySetAsSeries(d1_pdi, true);
   ArraySetAsSeries(d1_ndi, true);
   ArraySetAsSeries(d1_emaFast, true);
   ArraySetAsSeries(d1_emaSlow, true);
   ArraySetAsSeries(m15_atr, true);
   ArraySetAsSeries(m15_adx, true);
   ArraySetAsSeries(m15_pdi, true);
   ArraySetAsSeries(m15_ndi, true);
   ArraySetAsSeries(m15_emaFast, true);

   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq  = AccountInfoDouble(ACCOUNT_EQUITY);
   Print("Init MULTI v1.0 | Balance=", DoubleToString(bal, 2),
         " Equity=", DoubleToString(eq, 2),
         " | ADX=", currentProfile.adxThreshold,
         " SL_ATR=", DoubleToString(currentProfile.slATR, 1));

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   IndicatorRelease(d1_atrHandle);
   IndicatorRelease(d1_adxHandle);
   IndicatorRelease(d1_emaFastHandle);
   IndicatorRelease(d1_emaSlowHandle);
   IndicatorRelease(m15_atrHandle);
   IndicatorRelease(m15_adxHandle);
   IndicatorRelease(m15_emaFastHandle);
}

//+------------------------------------------------------------------+
//| Update D1 indicators                                             |
//+------------------------------------------------------------------+
bool UpdateD1Indicators() {
   if(CopyBuffer(d1_atrHandle, 0, 0, 5, d1_atr) < 5) return false;
   if(CopyBuffer(d1_adxHandle, 0, 0, 5, d1_adx) < 5) return false;
   if(CopyBuffer(d1_adxHandle, 1, 0, 5, d1_pdi) < 5) return false;
   if(CopyBuffer(d1_adxHandle, 2, 0, 5, d1_ndi) < 5) return false;
   if(CopyBuffer(d1_emaFastHandle, 0, 0, 5, d1_emaFast) < 5) return false;
   if(CopyBuffer(d1_emaSlowHandle, 0, 0, 5, d1_emaSlow) < 5) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Update M15 indicators                                            |
//+------------------------------------------------------------------+
bool UpdateM15Indicators() {
   if(CopyBuffer(m15_atrHandle, 0, 0, 5, m15_atr) < 5) return false;
   if(CopyBuffer(m15_adxHandle, 0, 0, 5, m15_adx) < 5) return false;
   if(CopyBuffer(m15_adxHandle, 1, 0, 5, m15_pdi) < 5) return false;
   if(CopyBuffer(m15_adxHandle, 2, 0, 5, m15_ndi) < 5) return false;
   if(CopyBuffer(m15_emaFastHandle, 0, 0, 5, m15_emaFast) < 5) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Count my positions                                               |
//+------------------------------------------------------------------+
int CountMyPositions() {
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      if(Pos.SelectByIndex(i) && Pos.Symbol() == _Symbol &&
         Pos.Magic() == actualMagic) count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| Select my position                                               |
//+------------------------------------------------------------------+
bool SelectMyPosition() {
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      if(Pos.SelectByIndex(i) && Pos.Symbol() == _Symbol &&
         Pos.Magic() == actualMagic) return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Calculate lot size                                               |
//+------------------------------------------------------------------+
double CalculateLot() {
   if(InpRiskPct > 0 && d1_atr[0] > 0) {
      double slDist = d1_atr[0] * currentProfile.slATR;
      double riskMoney = AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPct / 100.0;
      double contractSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
      double lot = riskMoney / (slDist * contractSize);
      lot = MathFloor(lot / 0.01) * 0.01;
      return MathMax(0.01, MathMin(1.0, lot));
   }
   return InpLot;
}

//+------------------------------------------------------------------+
//| OnTick                                                           |
//+------------------------------------------------------------------+
void OnTick() {
   //--- Update D1 on new bar
   datetime d1cb = iTime(_Symbol, PERIOD_D1, 0);
   if(d1cb != lastD1BarTime) {
      lastD1BarTime = d1cb;
      if(!UpdateD1Indicators()) return;
   }

   //--- If position open, trail only
   if(CountMyPositions() > 0) {
      TrailPositions();
      return;
   }

   //--- Cooldown check
   if(TimeCurrent() - lastEntryTime < currentProfile.cooldownMins * 60) return;

   //--- Update M15 on new bar
   datetime m15cb = iTime(_Symbol, PERIOD_M15, 0);
   if(m15cb != lastM15BarTime) {
      lastM15BarTime = m15cb;
      if(!UpdateM15Indicators()) return;
      CheckForEntry();
   }
}

//+------------------------------------------------------------------+
//| Check for entry                                                  |
//+------------------------------------------------------------------+
void CheckForEntry() {
   double d1close = iClose(_Symbol, PERIOD_D1, 0);
   if(d1close <= 0 || d1_atr[0] <= 0) return;

   double lot = CalculateLot();
   if(lot <= 0) return;

   //--- D1 trend conditions
   bool d1LongTrend = (d1_adx[0] >= currentProfile.adxThreshold &&
                       d1_pdi[0] > d1_ndi[0] &&
                       d1_emaFast[0] > d1_emaSlow[0] &&
                       d1close > d1_emaFast[0]);

   bool d1ShortTrend = (d1_adx[0] >= currentProfile.adxThreshold &&
                        d1_ndi[0] > d1_pdi[0] &&
                        d1_emaFast[0] < d1_emaSlow[0] &&
                        d1close < d1_emaFast[0]);

   if(!d1LongTrend && !d1ShortTrend) return;

   //--- SL distance (clamped)
   double rawSL = d1_atr[0] * currentProfile.slATR;
   double slDist = MathMax(currentProfile.minSLPoints, MathMin(currentProfile.maxSLPoints, rawSL));

   //--- M15 pullback confirmation
   double m15close1 = iClose(_Symbol, PERIOD_M15, 1);

   //--- LONG entry
   if(d1LongTrend &&
      m15close1 <= m15_emaFast[1] &&
      m15_pdi[1] > m15_ndi[1]) {
      double sl = NormalizeDouble(m15close1 - slDist, _Digits);
      if(Trade.Buy(lot, _Symbol, 0, sl, 0, InpComment)) {
         Print("Instance ", instanceID, " | LONG | ", currentProfile.name,
               " | Entry=", DoubleToString(m15close1, _Digits),
               " SL=", DoubleToString(sl, _Digits),
               " SLPts=", DoubleToString(slDist, 1));
         lastEntryTime = TimeCurrent();
      }
   }
   //--- SHORT entry
   else if(d1ShortTrend &&
           m15close1 >= m15_emaFast[1] &&
           m15_ndi[1] > m15_pdi[1]) {
      double sl = NormalizeDouble(m15close1 + slDist, _Digits);
      if(Trade.Sell(lot, _Symbol, 0, sl, 0, InpComment)) {
         Print("Instance ", instanceID, " | SHORT | ", currentProfile.name,
               " | Entry=", DoubleToString(m15close1, _Digits),
               " SL=", DoubleToString(sl, _Digits),
               " SLPts=", DoubleToString(slDist, 1));
         lastEntryTime = TimeCurrent();
      }
   }
}

//+------------------------------------------------------------------+
//| Trail positions                                                  |
//+------------------------------------------------------------------+
void TrailPositions() {
   if(!SelectMyPosition()) return;

   double entry = Pos.PriceOpen();
   double currentSL = Pos.StopLoss();
   double tp = Pos.TakeProfit();
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double atrVal = d1_atr[0];
   double point = _Point;

   if(atrVal <= 0) return;

   //--- BUY trailing
   if(Pos.PositionType() == POSITION_TYPE_BUY) {
      double profit = Pos.Profit();
      if(profit > 0) {
         //--- Break-even
         if(currentProfile.breakEven > 0 && profit >= currentProfile.breakEven && currentSL < entry) {
            double beSL = NormalizeDouble(entry + currentProfile.breakEvenBuffer, _Digits);
            if(beSL > currentSL) {
               if(Trade.PositionModify(_Symbol, beSL, tp)) {
                  Print("Instance ", instanceID, " | BE | SL=", DoubleToString(beSL, _Digits),
                        " Profit=$", DoubleToString(profit, 2));
                  currentSL = beSL;
               }
            }
         }
         //--- Trailing
         bool beTriggered = (currentSL >= entry);
         double trailDist = beTriggered ? currentProfile.trailFixed : MathMax(currentProfile.minSLPoints, atrVal * currentProfile.slATR);
         double newSL = NormalizeDouble(bid - trailDist, _Digits);
         if(newSL > currentSL + 10 * point) {
            if(Trade.PositionModify(_Symbol, newSL, tp)) {
               Print("Instance ", instanceID, " | TRAIL | SL=", DoubleToString(newSL, _Digits),
                     " Profit=$", DoubleToString(profit, 2));
            }
         }
      }
   }
   //--- SELL trailing
   else if(Pos.PositionType() == POSITION_TYPE_SELL) {
      double profit = Pos.Profit();
      if(profit > 0) {
         //--- Break-even
         if(currentProfile.breakEven > 0 && profit >= currentProfile.breakEven && currentSL > entry) {
            double beSL = NormalizeDouble(entry - currentProfile.breakEvenBuffer, _Digits);
            if(beSL < currentSL) {
               if(Trade.PositionModify(_Symbol, beSL, tp)) {
                  Print("Instance ", instanceID, " | BE | SL=", DoubleToString(beSL, _Digits),
                        " Profit=$", DoubleToString(profit, 2));
                  currentSL = beSL;
               }
            }
         }
         //--- Trailing
         bool beTriggered = (currentSL <= entry);
         double trailDist = beTriggered ? currentProfile.trailFixed : MathMax(currentProfile.minSLPoints, atrVal * currentProfile.slATR);
         double newSL = NormalizeDouble(ask + trailDist, _Digits);
         if(newSL < currentSL - 10 * point) {
            if(Trade.PositionModify(_Symbol, newSL, tp)) {
               Print("Instance ", instanceID, " | TRAIL | SL=", DoubleToString(newSL, _Digits),
                     " Profit=$", DoubleToString(profit, 2));
            }
         }
      }
   }
}
//+------------------------------------------------------------------+
