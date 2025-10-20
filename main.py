# =============================================================================
# AI STOCK TRADING SYSTEM - COMPLETE STARTER CODE
# FREE Stack - Ready to Deploy
# =============================================================================

"""
BACKEND - main.py
FastAPI backend dengan AI signal generation
Deploy ke Railway/Render (FREE tier)
"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
import feedparser
import json
import asyncio
from collections import deque
import os
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries


# Load environment variables
load_dotenv()

# Alpha Vantage configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
USE_ALPHA_VANTAGE = bool(ALPHA_VANTAGE_API_KEY)
# ADD THIS after all imports (around line 20)

def get_stock_with_fallback(symbol):
    """Get stock data with fallback to alternative method"""
    import time
    import requests
    
    # Method 1: Try yfinance with proper session
    try:
        ticker = yf.Ticker(symbol)
        
        # Create session if not exists
        if not hasattr(ticker, 'session') or ticker.session is None:
            ticker.session = requests.Session()
        
        ticker.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        df = ticker.history(period='1mo', interval='1d', timeout=10)
        
        if not df.empty and len(df) >= 20:
            return df
    except Exception as e:
        print(f"   Method 1 failed for {symbol}: {str(e)[:50]}")
    
    # Method 2: Try with yf.download
    try:
        time.sleep(1)
        df = yf.download(
            symbol, 
            period='1mo', 
            interval='1d', 
            progress=False, 
            timeout=10,
            ignore_tz=True
        )
        
        if not df.empty and len(df) >= 20:
            return df
    except Exception as e:
        print(f"   Method 2 failed for {symbol}: {str(e)[:50]}")
    
    # Method 3: Generate realistic mock data for development
    print(f"   ⚠️  Using mock data for {symbol}")
    import numpy as np
    
    # Generate 60 days of data for better analysis
    dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='D')
    
    # Base price depends on symbol
    if 'BBCA' in symbol or 'AAPL' in symbol:
        base_price = 10000
    elif 'BBRI' in symbol or 'MSFT' in symbol:
        base_price = 5000
    elif 'GOOGL' in symbol:
        base_price = 150
    else:
        base_price = np.random.randint(3000, 15000)
    
    # Generate realistic price movements with trend
    trend = np.linspace(0, np.random.randn() * 500, 60)
    noise = np.random.randn(60) * (base_price * 0.02)  # 2% volatility
    close_prices = base_price + trend + noise
    
    mock_data = pd.DataFrame({
        'Open': close_prices + np.random.randn(60) * (base_price * 0.01),
        'High': close_prices + np.abs(np.random.randn(60) * (base_price * 0.015)),
        'Low': close_prices - np.abs(np.random.randn(60) * (base_price * 0.015)),
        'Close': close_prices,
        'Volume': np.random.randint(10000000, 100000000, 60)
    }, index=dates)
    
    return mock_data

def get_stock_alpha_vantage(symbol):
    """Fetch stock data from Alpha Vantage API"""
    if not ALPHA_VANTAGE_API_KEY:
        return None
    
    try:
        print(f"   📡 Fetching {symbol} from Alpha Vantage...")
        
        # Initialize Alpha Vantage client
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        
        # Convert symbol format for Alpha Vantage
        # IDX: BBCA.JK → needs mapping to international ticker
        # US stocks: AAPL → direct use
        av_symbol = symbol.replace('.JK', '')  # For now, use without .JK
        
        # Get daily data
        data, meta_data = ts.get_daily(symbol=av_symbol, outputsize='compact')
        
        if data is not None and not data.empty:
            # Rename columns to match our format
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            print(f"   ✅ Got {len(data)} days of data from Alpha Vantage")
            return data
        
    except Exception as e:
        print(f"   ❌ Alpha Vantage error for {symbol}: {str(e)[:100]}")
    
    return None


def get_stock_intelligent(symbol):
    """
    Intelligent stock data fetcher with multiple fallbacks:
    1. Alpha Vantage (if API key available)
    2. Yahoo Finance (primary free source)
    3. Mock data (for development)
    """
    import time
    
    # Priority 1: Alpha Vantage (most reliable, but limited requests)
    if USE_ALPHA_VANTAGE:
        df = get_stock_alpha_vantage(symbol)
        if df is not None and not df.empty:
            return df
        print(f"   ⚠️  Alpha Vantage failed, trying Yahoo Finance...")
    
    # Priority 2: Yahoo Finance (free, unlimited, but sometimes blocked)
    try:
        ticker = yf.Ticker(symbol)
        
        # Create session with headers
        if not hasattr(ticker, 'session') or ticker.session is None:
            import requests
            ticker.session = requests.Session()
        
        ticker.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        
        df = ticker.history(period='2mo', interval='1d', timeout=10)
        
        if not df.empty and len(df) >= 20:
            print(f"   ✅ Got {len(df)} days from Yahoo Finance")
            return df
    
    except Exception as e:
        print(f"   ❌ Yahoo Finance error: {str(e)[:50]}")
    
    # Priority 3: Mock data (fallback for development)
    print(f"   ⚠️  Using mock data for {symbol}")
    import numpy as np
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='D')
    
    # Realistic base prices
    price_map = {
        'BBCA': 10000, 'BBRI': 5000, 'BMRI': 6500, 'BBNI': 5500,
        'TLKM': 4000, 'ASII': 5500, 'UNVR': 4500, 'GOTO': 100,
        'AAPL': 180, 'MSFT': 420, 'GOOGL': 140, 'TSLA': 250, 'NVDA': 500
    }
    
    symbol_clean = symbol.replace('.JK', '').replace('.US', '')
    base_price = price_map.get(symbol_clean, np.random.randint(3000, 15000))
    
    # Generate realistic price movements
    trend = np.linspace(0, np.random.randn() * 500, 60)
    noise = np.random.randn(60) * (base_price * 0.02)
    close_prices = base_price + trend + noise
    
    mock_data = pd.DataFrame({
        'Open': close_prices + np.random.randn(60) * (base_price * 0.01),
        'High': close_prices + np.abs(np.random.randn(60) * (base_price * 0.015)),
        'Low': close_prices - np.abs(np.random.randn(60) * (base_price * 0.015)),
        'Close': close_prices,
        'Volume': np.random.randint(10000000, 100000000, 60)
    }, index=dates)
    
    return mock_data


# Initialize FastAPI
app = FastAPI(title="AI Stock Trading API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stock-ai-five.vercel.app",
                   "http://localhost:3000",
                   "*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA MODELS
# =============================================================================

class StockData(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    rsi: float
    macd: float
    sentiment: str
    signal: str
    confidence: float
    reasons: List[str]
    timestamp: str

class Signal(BaseModel):
    id: int
    symbol: str
    signal: str
    confidence: float
    price: float
    reasons: List[str]
    timestamp: str

# =============================================================================
# IN-MEMORY STORAGE (Replace with database in production)
# =============================================================================

class DataStore:
    def __init__(self):
        self.stocks = {}
        self.signals = deque(maxlen=50)  # Keep last 50 signals
        self.news_cache = []
        self.signal_id_counter = 0
    
    def add_signal(self, signal_data):
        self.signal_id_counter += 1
        signal_data['id'] = self.signal_id_counter
        self.signals.appendleft(signal_data)
        return signal_data
    
    def get_signals(self, limit=20):
        return list(self.signals)[:limit]
    
    def update_stock(self, symbol, data):
        self.stocks[symbol] = data
    
    def get_all_stocks(self):
        return list(self.stocks.values())

store = DataStore()

# =============================================================================
# INDONESIAN STOCK SYMBOLS (IDX)
# =============================================================================

IDX_SYMBOLS = [
    'AAPL',  # Bank Central Asia
    'MSFT',  # Bank Rakyat Indonesia
    'GOOGL',  # Bank Mandiri
    'TSLA',  # Bank Negara Indonesia
    'NVDA',  # Telkom Indonesia
    'BBRI.JK',
    'BBCA.JK',
    'ANTM.jk',
]

# =============================================================================
# TECHNICAL ANALYSIS FUNCTIONS
# =============================================================================

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    if down == 0:
        return 100
    rs = up / down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.iloc[-1], signal_line.iloc[-1]

def calculate_moving_averages(prices):
    """Calculate MA20 and MA50"""
    df = pd.Series(prices)
    ma20 = df.rolling(window=20).mean().iloc[-1] if len(prices) >= 20 else df.mean()
    ma50 = df.rolling(window=50).mean().iloc[-1] if len(prices) >= 50 else df.mean()
    return ma20, ma50

# =============================================================================
# SENTIMENT ANALYSIS (Simple version - upgrade with ML later)
# =============================================================================

def analyze_sentiment_simple(text):
    """Simple keyword-based sentiment analysis"""
    positive_words = ['naik', 'untung', 'positif', 'bullish', 'tumbuh', 'ekspansi', 
                      'profit', 'meningkat', 'optimis', 'kuat', 'bagus']
    negative_words = ['turun', 'rugi', 'negatif', 'bearish', 'anjlok', 'lemah',
                      'merosot', 'penurunan', 'resesi', 'krisis', 'buruk']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "Bullish"
    elif neg_count > pos_count:
        return "Bearish"
    else:
        return "Neutral"

def get_news_sentiment():
    """Fetch news from RSS feeds and analyze sentiment"""
    feeds = [
        'https://www.cnbcindonesia.com/market/rss',
        'https://ekonomi.bisnis.com/index.xml',
    ]
    
    all_articles = []
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:3]:  # Top 3 from each feed
                sentiment = analyze_sentiment_simple(entry.title + ' ' + entry.get('summary', ''))
                all_articles.append({
                    'title': entry.title,
                    'link': entry.link,
                    'sentiment': sentiment,
                    'published': entry.get('published', '')
                })
        except Exception as e:
            print(f"Error fetching feed {feed_url}: {e}")
    
    store.news_cache = all_articles
    
    # Calculate overall sentiment
    sentiments = [a['sentiment'] for a in all_articles]
    if not sentiments:
        return "Neutral"
    
    bullish = sentiments.count("Bullish")
    bearish = sentiments.count("Bearish")
    
    if bullish > bearish:
        return "Bullish"
    elif bearish > bullish:
        return "Bearish"
    else:
        return "Neutral"

# =============================================================================
# AI SIGNAL GENERATION ENGINE
# =============================================================================

def generate_trading_signal(symbol_clean, df, overall_sentiment):
    """
    AI Decision Engine - Generate BUY/SELL/HOLD signal
    Based on multiple factors with weighted scoring
    """
    if len(df) < 20:
        return None
    
    prices = df['Close'].values
    volumes = df['Volume'].values
    
    # Calculate indicators
    current_price = prices[-1]
    prev_price = prices[-2] if len(prices) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price * 100) if prev_price != 0 else 0
    
    rsi = calculate_rsi(prices)
    macd_value, signal_value = calculate_macd(prices)
    ma20, ma50 = calculate_moving_averages(prices)
    
    avg_volume = np.mean(volumes[-20:])
    current_volume = volumes[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    # Scoring system
    score = 0
    reasons = []
    
    # 1. RSI Analysis (Weight: 25 points)
    if rsi < 30:
        score += 25
        reasons.append(f"RSI Oversold ({rsi:.1f})")
    elif rsi > 70:
        score -= 25
        reasons.append(f"RSI Overbought ({rsi:.1f})")
    elif 30 <= rsi <= 45:
        score += 10
        reasons.append(f"RSI Moderate Buy ({rsi:.1f})")
    elif 55 <= rsi <= 70:
        score -= 10
        reasons.append(f"RSI Moderate Sell ({rsi:.1f})")
    
    # 2. MACD Analysis (Weight: 20 points)
    macd_diff = macd_value - signal_value
    if macd_diff > 0:
        score += 20
        reasons.append("MACD Bullish Cross")
    else:
        score -= 20
        reasons.append("MACD Bearish Cross")
    
    # 3. Moving Average Analysis (Weight: 20 points)
    if current_price > ma20 and ma20 > ma50:
        score += 20
        reasons.append("Golden Cross (MA20 > MA50)")
    elif current_price < ma20 and ma20 < ma50:
        score -= 20
        reasons.append("Death Cross (MA20 < MA50)")
    
    # 4. Sentiment Analysis (Weight: 15 points)
    if overall_sentiment == "Bullish":
        score += 15
        reasons.append("Positive Market Sentiment")
    elif overall_sentiment == "Bearish":
        score -= 15
        reasons.append("Negative Market Sentiment")
    
    # 5. Volume Analysis (Weight: 10 points)
    if volume_ratio > 1.5:
        score += 10
        reasons.append(f"High Volume ({volume_ratio:.1f}x avg)")
    elif volume_ratio < 0.5:
        score -= 5
        reasons.append("Low Volume Activity")
    
    # 6. Price Momentum (Weight: 10 points)
    if price_change_pct > 3:
        score += 10
        reasons.append(f"Strong Uptrend (+{price_change_pct:.1f}%)")
    elif price_change_pct < -3:
        score -= 10
        reasons.append(f"Strong Downtrend ({price_change_pct:.1f}%)")
    
    # Generate final signal
    confidence = min(abs(score), 100)
    
    if score >= 40:
        signal = "BUY"
    elif score <= -40:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    return {
        'symbol': symbol_clean,
        'price': float(current_price),
        'change': float(price_change),
        'change_percent': float(price_change_pct),
        'volume': int(current_volume),
        'rsi': float(rsi),
        'macd': float(macd_diff),
        'sentiment': overall_sentiment,
        'signal': signal,
        'confidence': float(confidence),
        'reasons': reasons,
        'timestamp': datetime.now().isoformat()
    }

# =============================================================================
# MARKET SCANNER - Background Job
# =============================================================================

async def scan_market():
    """Scan market and generate signals for all stocks"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Scanning market...")
    
    # Get overall market sentiment from news
    overall_sentiment = get_news_sentiment()
    print(f"   Overall Sentiment: {overall_sentiment}")
    
    successful_scans = 0
    
    for symbol in IDX_SYMBOLS:
        try:
            # Add delay to avoid rate limiting
            import time
            time.sleep(2)  # 2 seconds between requests
            
            # Fetch data with multiple methods
            df = get_stock_intelligent(symbol)
            
            if df is None or df.empty:
                print(f"   ⚠️  Skipping {symbol} - No data available")
                continue
            
            # Generate signal
            symbol_clean = symbol.replace('.JK', '')
            result = generate_trading_signal(symbol_clean, df, overall_sentiment)
            
            if result:
                # Update store
                store.update_stock(symbol_clean, result)
                successful_scans += 1
                
                # Add to signals if strong signal
                if result['signal'] in ['BUY', 'SELL'] and result['confidence'] >= 60:
                    store.add_signal(result)
                    print(f"   🎯 {result['signal']} signal: {symbol_clean} (Confidence: {result['confidence']:.0f}%)")
        
        except Exception as e:
            print(f"   ❌ Error processing {symbol}: {str(e)[:100]}")
    
    print(f"   ✅ Scan complete - {successful_scans}/{len(IDX_SYMBOLS)} stocks updated\n")

# =============================================================================
# SCHEDULER SETUP
# =============================================================================

scheduler = BackgroundScheduler()

@app.on_event("startup")
async def startup_event():
    """Run on app startup"""
    print("=" * 60)
    print("🚀 AI STOCK TRADING SYSTEM - STARTING")
    print("=" * 60)
    # ADD THIS:
    if USE_ALPHA_VANTAGE:
        print(f"✅ Alpha Vantage API: ENABLED")
    else:
        print(f"⚠️  Alpha Vantage API: DISABLED (no API key)")
    
    print("=" * 60)
    
    # Initial scan
    await scan_market()
    
    # Schedule periodic scans (every 5 minutes)
    scheduler.add_job(
        lambda: asyncio.create_task(scan_market()),
        'interval',
        minutes=5,
        id='market_scan'
    )
    
    # Schedule news update (every 30 minutes)
    scheduler.add_job(
        get_news_sentiment,
        'interval',
        minutes=30,
        id='news_update'
    )
    
    scheduler.start()
    print("✅ Scheduler started - Scanning every 5 minutes")
    print("=" * 60)

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    scheduler.shutdown()
    print("👋 Shutting down...")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "AI Stock Trading API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "stocks_monitored": len(store.get_all_stocks()),
        "total_signals": len(store.get_signals())
    }

@app.get("/api/stocks", response_model=List[StockData])
def get_stocks():
    """Get all monitored stocks with latest data"""
    stocks = store.get_all_stocks()
    if not stocks:
        raise HTTPException(status_code=404, detail="No stock data available")
    return stocks

@app.get("/api/stocks/{symbol}")
def get_stock(symbol: str):
    """Get specific stock data"""
    stock = store.stocks.get(symbol.upper())
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    return stock

@app.get("/api/signals", response_model=List[Signal])
def get_signals(limit: int = 20):
    """Get latest trading signals"""
    return store.get_signals(limit)

@app.get("/api/news")
def get_news():
    """Get latest news with sentiment"""
    return {
        "articles": store.news_cache,
        "total": len(store.news_cache),
        "last_update": datetime.now().isoformat()
    }

@app.post("/api/scan")
@app.get("/api/scan")
async def trigger_scan():
    """Manually trigger market scan"""
    await scan_market()
    return {"message": "Scan completed", "stocks": len(store.get_all_stocks())}

@app.get("/api/stats")
def get_stats():
    """Get system statistics"""
    signals = store.get_signals()
    buy_signals = [s for s in signals if s['signal'] == 'BUY']
    sell_signals = [s for s in signals if s['signal'] == 'SELL']
    
    return {
        "total_stocks": len(store.get_all_stocks()),
        "total_signals": len(signals),
        "buy_signals": len(buy_signals),
        "sell_signals": len(sell_signals),
        "news_articles": len(store.news_cache),
        "last_scan": signals[0]['timestamp'] if signals else None,
        "uptime": "Running"
    }

# =============================================================================
# WEBSOCKET for Real-time Updates
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates"""
    await websocket.accept()
    print("🔌 WebSocket client connected")
    
    try:
        while True:
            # Send updates every 10 seconds
            data = {
                "type": "update",
                "stocks": store.get_all_stocks(),
                "signals": store.get_signals(5),
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_json(data)
            await asyncio.sleep(10)
    
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print("🔌 WebSocket client disconnected")

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )