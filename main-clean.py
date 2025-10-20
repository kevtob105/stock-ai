# =============================================================================
# AI STOCK TRADING SYSTEM - MINIMAL WORKING VERSION
# =============================================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime
from collections import deque
import asyncio

# CRITICAL: Import yfinance FIRST, CLEAN, NO manipulation
import yfinance as yf
import pandas as pd
import numpy as np

print(f"✅ yfinance imported: version {yf.__version__}")

# =============================================================================
# FastAPI Setup
# =============================================================================

app = FastAPI(title="AI Stock Trading API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Data Storage
# =============================================================================

class DataStore:
    def __init__(self):
        self.stocks = {}
        self.signals = deque(maxlen=50)
    
    def update_stock(self, symbol, data):
        self.stocks[symbol] = data
    
    def get_all_stocks(self):
        return list(self.stocks.values())
    
    def add_signal(self, signal):
        self.signals.appendleft(signal)
    
    def get_signals(self):
        return list(self.signals)

store = DataStore()

# =============================================================================
# Stock Symbols
# =============================================================================

IDX_SYMBOLS = [
    'BBCA.JK',  # Bank Central Asia
    'BBRI.JK',  # Bank Rakyat Indonesia
    'BMRI.JK',  # Bank Mandiri
    'BBNI.JK',  # Bank Negara Indonesia
    'TLKM.JK',  # Telkom Indonesia
    'GOTO.JK',  # GoTo
    'ASII.JK',  # Astra
    'UNVR.JK',  # Unilever
]

# =============================================================================
# CRITICAL: MINIMAL DATA FETCHING - NO SESSION MANIPULATION
# =============================================================================

def fetch_stock_data(symbol):
    """
    Minimal fetch function - EXACTLY like Python console test
    NO session, NO headers, NO timeout - NOTHING!
    """
    print(f"   🔍 {symbol.replace('.JK', '')}...")
    
    try:
        # EXACT copy of working Python console command
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='1mo')
        
        if not df.empty and len(df) >= 5:
            close = df['Close'].iloc[-1]
            print(f"   ✅ {len(df)} days, Rp {close:,.0f}")
            return df
        else:
            print(f"   ⚠️  Empty data")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:50]}")
    
    return None

# =============================================================================
# Technical Analysis (Simplified)
# =============================================================================

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    if down == 0:
        return 100
    rs = up / down
    return 100 - (100 / (1 + rs))

def generate_signal(symbol, df):
    """Generate trading signal"""
    if len(df) < 20:
        return None
    
    prices = df['Close'].values
    current_price = prices[-1]
    prev_price = prices[-2]
    change = current_price - prev_price
    change_pct = (change / prev_price * 100)
    
    rsi = calculate_rsi(prices)
    
    # Simple signal logic
    signal = "HOLD"
    confidence = 50
    reasons = []
    
    if rsi < 30:
        signal = "BUY"
        confidence = 65
        reasons.append(f"RSI Oversold ({rsi:.1f})")
    elif rsi > 70:
        signal = "SELL"
        confidence = 65
        reasons.append(f"RSI Overbought ({rsi:.1f})")
    
    return {
        'symbol': symbol,
        'price': float(current_price),
        'change': float(change),
        'change_percent': float(change_pct),
        'volume': int(df['Volume'].iloc[-1]),
        'rsi': float(rsi),
        'signal': signal,
        'confidence': float(confidence),
        'reasons': reasons,
        'timestamp': datetime.now().isoformat(),
        'currency': 'IDR'
    }

# =============================================================================
# Market Scanner
# =============================================================================

async def scan_market():
    """Scan market and generate signals"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔍 Scanning market...\n")
    
    success = 0
    
    for symbol in IDX_SYMBOLS:
        try:
            df = fetch_stock_data(symbol)
            
            if df is not None and len(df) >= 20:
                symbol_clean = symbol.replace('.JK', '')
                result = generate_signal(symbol_clean, df)
                
                if result:
                    store.update_stock(symbol_clean, result)
                    success += 1
                    
                    if result['signal'] in ['BUY', 'SELL'] and result['confidence'] >= 60:
                        store.add_signal(result)
                        print(f"   🎯 {result['signal']}: {symbol_clean}\n")
            
            # Delay between requests
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"   ❌ {symbol}: {e}\n")
    
    print(f"   ✅ Complete: {success}/{len(IDX_SYMBOLS)} stocks\n")

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "AI Stock Trading API - Indonesia",
        "stocks": len(store.get_all_stocks()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stocks")
def get_stocks():
    stocks = store.get_all_stocks()
    if not stocks:
        return []
    return stocks

@app.get("/api/signals")
def get_signals():
    return store.get_signals()

@app.get("/api/stats")
def get_stats():
    signals = store.get_signals()
    return {
        "total_stocks": len(store.get_all_stocks()),
        "total_signals": len(signals),
        "buy_signals": len([s for s in signals if s['signal'] == 'BUY']),
        "sell_signals": len([s for s in signals if s['signal'] == 'SELL']),
    }

@app.post("/api/scan")
@app.get("/api/scan")
async def trigger_scan():
    await scan_market()
    return {"message": "Scan completed", "stocks": len(store.get_all_stocks())}

# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("🚀 AI STOCK TRADING SYSTEM - INDONESIA")
    print("=" * 60)
    print(f"🇮🇩 Monitoring {len(IDX_SYMBOLS)} IDX stocks")
    print("=" * 60)
    
    # Initial scan
    await scan_market()
    
    print("✅ System ready!")
    print("=" * 60)

# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)