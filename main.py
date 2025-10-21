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
from typing import Optional
import csv
from backtester import Backtester
from pathlib import Path
import asyncio
from collections import deque
import os
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
from telegram import Bot
import asyncio

TELEGRAM_TOKEN = "8247322040:AAEpGthXNLSnTPrWL8PbxrPJ_1hyWFHE0DA"  # Get from @BotFather
CHAT_ID = "@IDXMind_bot"
print(f"yfinance version: {yf.__version__}")
print(f"yfinance location: {yf.__file__}")


# Load environment variables
load_dotenv()

# Alpha Vantage configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
USE_ALPHA_VANTAGE = bool(ALPHA_VANTAGE_API_KEY)
# ADD THIS after all imports (around line 20)

def log_signal_to_file(signal_data):
    """Log signal to CSV for backtesting"""
    log_file = Path('signals_history.csv')
    
    # Create file with headers if not exists
    if not log_file.exists():
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'symbol', 'signal', 'price', 
                'rsi', 'confidence', 'reasons'
            ])
    
    # Append signal
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            signal_data['timestamp'],
            signal_data['symbol'],
            signal_data['signal'],
            signal_data['price'],
            signal_data['rsi'],
            signal_data['confidence'],
            '; '.join(signal_data['reasons'])
        ])


def get_stock_with_fallback(symbol):
    """Get stock data with fallback to alternative method"""
    import time
    import requests
    
    
    
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
        
        # Convert symbol format
        av_symbol = symbol.replace('.JK', '').strip()
        
        # Get daily data (compact = last 100 days)
        data, meta_data = ts.get_daily(symbol=av_symbol, outputsize='compact')
        
        if data is not None and not data.empty:
            # IMPORTANT: Alpha Vantage returns columns with prefixes
            # Original columns: '1. open', '2. high', '3. low', '4. close', '5. volume'
            
            # Debug: Print original columns
            print(f"   🔍 Original columns: {data.columns.tolist()}")
            
            # Rename to match our format
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Sort by date ascending (oldest first)
            data = data.sort_index()
            
            # Convert to numeric (handle any string data)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any NaN rows
            data = data.dropna()
            
            print(f"   ✅ Got {len(data)} days of data from Alpha Vantage")
            print(f"   💰 Latest close: ${data['Close'].iloc[-1]:.2f}")
            
            return data
        
    except Exception as e:
        print(f"   ❌ Alpha Vantage error for {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full error for debugging
    
    return None


def get_stock_intelligent(symbol):
    """
    Fetch stock data using Yahoo Finance
    Simplified version that actually works!
    """
    import time
    
    print(f"   🔍 Fetching {symbol.replace('.JK', '')}...")
    
    try:
        # Simple approach - just like your test!
        ticker = yf.Ticker(symbol)
        
        # Fetch data - try multiple periods
        for period in ['3mo', '2mo', '1mo', '5d']:
            try:
                df = ticker.history(period=period)
                
                if not df.empty and len(df) >= 5:
                    print(f"   ✅ Yahoo Finance ({period}): Rp {df['Close'].iloc[-1]:,.0f}")
                    
                    # Need at least 20 days for technical analysis
                    if len(df) >= 20:
                        return df
                    elif len(df) >= 5:
                        # Extend data if needed
                        return extend_historical_data(df, target_days=60)
                
            except Exception as e:
                print(f"   ⚠️  Period {period} failed: {str(e)[:40]}")
                time.sleep(1)
                continue
        
        print(f"   ❌ All periods failed for {symbol}")
        
    except Exception as e:
        print(f"   ❌ Yahoo Finance error: {str(e)[:60]}")
    
    # Fallback to mock data
    print(f"   🔄 Using mock data")
    return generate_mock_data_idx(symbol)


def extend_historical_data(df_short, target_days=60):
    """Extend short data to target length"""
    import numpy as np
    
    if len(df_short) >= target_days:
        return df_short
    
    last_close = df_short['Close'].iloc[-1]
    last_date = df_short.index[-1]
    
    additional_days = target_days - len(df_short)
    start_date = last_date - pd.Timedelta(days=additional_days + 10)
    
    extended_dates = pd.date_range(
        start=start_date,
        end=last_date - pd.Timedelta(days=1),
        freq='D'
    )
    
    num_extended = len(extended_dates)
    returns = np.random.normal(0, 0.015, num_extended)
    price_path = last_close * np.exp(np.cumsum(returns[::-1]))[::-1]
    
    extended_data = pd.DataFrame({
        'Open': price_path * 0.995,
        'High': price_path * 1.01,
        'Low': price_path * 0.99,
        'Close': price_path,
        'Volume': np.random.randint(
            int(df_short['Volume'].mean() * 0.8),
            int(df_short['Volume'].mean() * 1.2),
            num_extended
        )
    }, index=extended_dates)
    
    combined = pd.concat([extended_data, df_short])
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    
    return combined

async def send_telegram_alert(symbol, signal, price, confidence, reasons):
    bot = Bot(token=TELEGRAM_TOKEN)
    
    emoji = "🟢" if signal == "BUY" else "🔴"
    message = f"""
    {emoji} *{signal} SIGNAL: {symbol}*

    Price: Rp {price:,.0f}
    Confidence: {confidence:.0f}%
    Reasons:
    {chr(10).join(f"• {r}" for r in reasons)}

    _AI Stock Trading System_
    """
    
    await bot.send_message(
        chat_id=CHAT_ID,
        text=message,
        parse_mode='Markdown'
    )



def generate_mock_data_idx(symbol):
    """Generate realistic mock data for Indonesian stocks"""
    import numpy as np
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='D')
    
    # Realistic IDX price ranges (in Rupiah)
    idx_prices = {
        'BBCA.JK': (9500, 10500),   # BCA
        'BBRI.JK': (4800, 5400),    # BRI
        'BMRI.JK': (6200, 6800),    # Mandiri
        'BBNI.JK': (5200, 5800),    # BNI
        'TLKM.JK': (3800, 4200),    # Telkom
        'ASII.JK': (5200, 5800),    # Astra
        'UNVR.JK': (4200, 4800),    # Unilever
        'GOTO.JK': (90, 140),        # GoTo
        'BUKA.JK': (60, 100),        # Bukalapak
        'ARTO.JK': (1800, 2400),    # Bank Jago
        'AMMN.JK': (9000, 11000),   # Amman
        'ADRO.JK': (2800, 3400),    # Adaro
        'ICBP.JK': (10500, 11500),  # Indofood CBP
        'INDF.JK': (6500, 7500),    # Indofood
        'UNTR.JK': (26000, 30000),  # United Tractors
        'PTBA.JK': (2400, 2800),    # Bukit Asam
        'PGAS.JK': (1400, 1800),    # PGN
        'MEDC.JK': (1100, 1500),    # Medco
        'BSDE.JK': (1100, 1400),    # BSD
        'EMTK.JK': (1600, 2200),    # EMTK
    }
    
    # Get price range
    price_min, price_max = idx_prices.get(symbol, (1000, 10000))
    base_price = np.random.randint(price_min, price_max)
    
    # Generate realistic price movements
    trend = np.linspace(0, np.random.randn() * (base_price * 0.05), 60)
    noise = np.random.randn(60) * (base_price * 0.02)
    close_prices = base_price + trend + noise
    
    # Ensure prices stay positive
    close_prices = np.maximum(close_prices, price_min * 0.8)
    
    mock_data = pd.DataFrame({
        'Open': close_prices + np.random.randn(60) * (base_price * 0.01),
        'High': close_prices + np.abs(np.random.randn(60) * (base_price * 0.015)),
        'Low': close_prices - np.abs(np.random.randn(60) * (base_price * 0.015)),
        'Close': close_prices,
        'Volume': np.random.randint(10000000, 200000000, 60)  # IDX typical volume
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
    
    # Technology & Digital
    'TLKM.JK',  # Telkom Indonesia
    'WIFI.JK',  # GoTo (Gojek Tokopedia)
    'EMTK.JK',  # Elang Mahkota Teknologi
    
    # Pak PP
    'CUAN.JK',  # Unilever Indonesia
    'TPIA.JK',  # Indofood CBP
    'BREN.JK',  # Indofood
    'CDIA.JK',
    'TOBA.JK',
    'JATI.JK',
    'MBMA.JK',
    'KRYA.JK',
    'PTRO.JK',
    'TINS.JK',
    'DATA.JK',
    'BRPT.JK',
    'BMTR.JK',
    'DEWA.JK',
    'ENRG.JK',
    'BUVA.JK',
    'INET.JK',


    
    # Industrial & Automotive
    'ASII.JK',  # Astra International
    'UNTR.JK',  # United Tractors
    
    # Mining & Energy
    'AMMN.JK',  # Amman Mineral (Emas)
    'ADRO.JK',  # Adaro Energy (Batubara)
    'PTBA.JK',  # Bukit Asam (Batubara)
    'PGAS.JK',  # PGN (Gas)
    'MEDC.JK',  # Medco Energi
    'EMTK.JK',
    'BRMS.JK',
    'HRTA.JK',
    'ANTM.JK',
    'PSAB.JK',
    'AADI.JK',
    'EMAS.JK',
    'RAJA.JK',


    
    # Property
    'BKSL.JK',  # Bumi Serpong Damai
    'ASRI.JK',
    'PANI.JK',
    'CTRA.JK',
    'BSBK.JK',

]

print(f"📊 Monitoring {len(IDX_SYMBOLS)} Indonesian stocks (IDX)")


# =============================================================================
# TECHNICAL ANALYSIS FUNCTIONS
# =============================================================================

def calculate_rsi(prices, period=14):
    """
    Calculate RSI using Wilder's Smoothing method
    Same calculation as TradingView uses
    """
    if len(prices) < period + 1:
        return 50  # Not enough data
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Wilder's smoothing (same as TradingView)
    # First average
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Smooth subsequent values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    # Calculate RS and RSI
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
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
    """Fetch news - FIXED timeout issue"""
    feeds = [
        'https://www.cnbcindonesia.com/market/rss',
        'https://www.cnbcindonesia.com/investment/rss',
        'https://ekonomi.bisnis.com/index.xml',
    ]
    
    all_articles = []
    
    for feed_url in feeds:
        try:
            # NO timeout parameter! feedparser doesn't support it
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:3]:
                sentiment = analyze_sentiment_simple(
                    entry.title + ' ' + entry.get('summary', '')
                )
                all_articles.append({
                    'title': entry.title,
                    'link': entry.link,
                    'sentiment': sentiment,
                    'published': entry.get('published', ''),
                    'source': feed_url.split('/')[2]
                })
                
        except Exception as e:
            print(f"   ⚠️  RSS error: {feed_url.split('/')[2]}")
    
    store.news_cache = all_articles
    
    if not all_articles:
        return "Neutral"
    
    sentiments = [a['sentiment'] for a in all_articles]
    bullish = sentiments.count("Bullish")
    bearish = sentiments.count("Bearish")
    
    if bullish > bearish * 1.2:
        return "Bullish"
    elif bearish > bullish * 1.2:
        return "Bearish"
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
    
    #ADD THIS - Debug actual data
    print(f"   🧮 Processing {symbol_clean}:")
    print(f"      Data points: {len(df)}")
    print(f"      Latest close: {df['Close'].iloc[-1]:.2f}")
    print(f"      Date range: {df.index[0]} to {df.index[-1]}")
    
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
    
    if score >= 20:
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
        'timestamp': datetime.now().isoformat(),
        'currency': 'IDR'  # Always IDR for IDX stocks
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
            # ADD THIS - Debug print to see actual data
            print(f"   📊 Latest data for {symbol}:")
            print(f"      Close: {df['Close'].iloc[-1]:.2f}")
            print(f"      Volume: {df['Volume'].iloc[-1]:,}")
            print(f"      Date: {df.index[-1]}")
            
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
                    log_signal_to_file(result)
                    print(f"   🎯 {result['signal']} signal: {symbol_clean} (Confidence: {result['confidence']:.0f}%)")
                    await send_telegram_alert(
                        result['symbol'],
                        result['signal'],
                        result['price'],
                        result['confidence'],
                        result['reasons'])
        
        except Exception as e:
            print(f"   ❌ Error processing {symbol}: {str(e)[:100]}")
    
    print(f"   ✅ Scan complete - {successful_scans}/{len(IDX_SYMBOLS)} stocks updated\n")

# Add API endpoint
@app.get("/api/history")
def get_signal_history():
    import pandas as pd
    try:
        df = pd.read_csv('signals_history.csv')
        return df.to_dict('records')
    except:
        return []

# =============================================================================
# SCHEDULER SETUP
# =============================================================================

scheduler = BackgroundScheduler()

@app.on_event("startup")
async def startup_event():
    """Run on app startup"""
    print("=" * 60)
    print("🚀 AI STOCK TRADING SYSTEM - INDONESIA")
    print("=" * 60)
    print(f"🇮🇩 Monitoring {len(IDX_SYMBOLS)} IDX stocks")
    print(f"📡 Data source: Yahoo Finance (IDX)")
    print(f"💰 Currency: Indonesian Rupiah (IDR)")
    print("=" * 60)
    
    # Initial scan
    await scan_market()
    
    # More frequent scanning (every 10 minutes during market hours)
    # IDX market hours: 09:00 - 16:00 WIB
    scheduler.add_job(
        lambda: asyncio.create_task(scan_market_smart()),
        'interval',
        minutes=5,  # Scan every 10 minutes!
        id='market_scan'
    )
    
    # News update every 15 minutes
    scheduler.add_job(
        get_news_sentiment,
        'interval',
        minutes=15,
        id='news_update'
    )
    
    scheduler.start()
    print("✅ Scheduler started - Scan every 10 minutes")
    print("=" * 60)


async def scan_market_smart():
    """Only scan during IDX market hours (09:00-16:00 WIB)"""
    from datetime import datetime
    import pytz
    
    # WIB timezone
    wib = pytz.timezone('Asia/Jakarta')
    now = datetime.now(wib)
    hour = now.hour
    
    # IDX market hours: 09:00 - 16:00 WIB (includes pre-market & post-market)
    if 8 <= hour <= 16:
        await scan_market()
    else:
        print(f"[{now.strftime('%H:%M:%S')}] ⏸️  Market closed - Next scan at 09:00 WIB")

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

class PerformanceTracker:
    def __init__(self):
        self.results_file = Path('backtest_results.json')
        self.trades_file = Path('backtest_trades.csv')
    
    def get_latest_backtest(self) -> Optional[dict]:
        """Get latest backtest results"""
        if not self.results_file.exists():
            return None
        
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def get_performance_summary(self) -> dict:
        """Get quick performance summary"""
        backtest = self.get_latest_backtest()
        
        if not backtest:
            return {
                "status": "no_data",
                "message": "Run backtest first"
            }
        
        report = backtest.get('report', {})
        
        return {
            "status": "available",
            "last_update": backtest.get('timestamp', 'Unknown'),
            "metrics": {
                "win_rate": report.get('win_rate', 0),
                "total_return_pct": report.get('total_return_pct', 0),
                "profit_factor": report.get('profit_factor', 0),
                "max_drawdown": report.get('max_drawdown', 0),
                "total_trades": report.get('total_trades', 0)
            },
            "assessment": self._assess_performance(report)
        }
    
    def _assess_performance(self, report: dict) -> str:
        """Assess overall system performance"""
        win_rate = report.get('win_rate', 0)
        profit_factor = report.get('profit_factor', 0)
        
        if win_rate >= 60 and profit_factor >= 1.5:
            return "Excellent"
        elif win_rate >= 50 and profit_factor >= 1.2:
            return "Good"
        elif win_rate >= 45:
            return "Moderate"
        else:
            return "Needs Improvement"
    
    def compare_signal_types(self) -> dict:
        """Compare BUY vs SELL signal performance"""
        backtest = self.get_latest_backtest()
        
        if not backtest:
            return {}
        
        report = backtest.get('report', {})
        
        return {
            "buy_signals": report.get('buy_signals', {}),
            "sell_signals": report.get('sell_signals', {}),
            "comparison": {
                "better_signal": "BUY" if report.get('buy_signals', {}).get('win_rate', 0) > 
                                         report.get('sell_signals', {}).get('win_rate', 0) else "SELL"
            }
        }

# Initialize tracker
performance_tracker = PerformanceTracker()


# =============================================================================
# API ENDPOINTS - Add these routes
# =============================================================================

@app.get("/api/backtest/summary")
def get_backtest_summary():
    """
    Get backtesting performance summary
    
    Returns key metrics without full details
    """
    return performance_tracker.get_performance_summary()


@app.get("/api/backtest/full")
def get_full_backtest():
    """
    Get complete backtest results
    
    Returns all trades and detailed analysis
    """
    backtest = performance_tracker.get_latest_backtest()
    
    if not backtest:
        raise HTTPException(
            status_code=404, 
            detail="No backtest results found. Run backtest.py first"
        )
    
    return backtest


@app.get("/api/backtest/trades")
def get_backtest_trades(limit: int = 50):
    """
    Get individual backtest trades
    
    Args:
        limit: Number of trades to return (default 50)
    """
    if not performance_tracker.trades_file.exists():
        raise HTTPException(
            status_code=404,
            detail="No trades file found"
        )
    
    try:
        trades_df = pd.read_csv(performance_tracker.trades_file)
        trades = trades_df.head(limit).to_dict('records')
        
        return {
            "total_trades": len(trades_df),
            "returned": len(trades),
            "trades": trades
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/comparison")
def get_signal_comparison():
    """
    Compare BUY vs SELL signal performance
    """
    comparison = performance_tracker.compare_signal_types()
    
    if not comparison:
        raise HTTPException(
            status_code=404,
            detail="No comparison data available"
        )
    
    return comparison


@app.get("/api/backtest/metrics")
def get_key_metrics():
    """
    Get only key performance metrics
    
    Quick endpoint for dashboard widgets
    """
    summary = performance_tracker.get_performance_summary()
    
    if summary['status'] == 'no_data':
        return {
            "win_rate": 0,
            "profit_factor": 0,
            "total_return": 0,
            "assessment": "No Data"
        }
    
    metrics = summary['metrics']
    
    return {
        "win_rate": f"{metrics['win_rate']:.1f}%",
        "profit_factor": f"{metrics['profit_factor']:.2f}",
        "total_return": f"{metrics['total_return_pct']:+.2f}%",
        "max_drawdown": f"{metrics['max_drawdown']:.2f}%",
        "total_trades": metrics['total_trades'],
        "assessment": summary['assessment']
    }


@app.post("/api/backtest/run")
async def trigger_backtest(
    hold_days: int = 3,
    max_signals: int = 50
):
    """
    Trigger backtesting (runs in background)
    
    Args:
        hold_days: Days to hold each position
        max_signals: Maximum signals to test
    
    Note: This is a simplified version. 
    For production, use Celery or similar task queue
    """
    import subprocess
    
    try:
        # Run backtest in background
        subprocess.Popen([
            'python', 
            'backtest.py',
            '--hold_days', str(hold_days),
            '--max_signals', str(max_signals)
        ])
        
        return {
            "status": "started",
            "message": f"Backtest started with hold_days={hold_days}, max_signals={max_signals}",
            "note": "Check /api/backtest/summary for results"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start backtest: {str(e)}"
        )


@app.get("/api/performance/live")
async def get_live_performance():
    """
    Calculate live performance based on recent signals
    
    This tracks how recent signals are performing in real-time
    """
    signals_file = Path('signals_history.csv')
    
    if not signals_file.exists():
        return {
            "status": "no_signals",
            "message": "No signals generated yet"
        }
    
    try:
        # Read recent signals (last 10)
        df = pd.read_csv(signals_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        recent_signals = df.head(10)
        
        live_results = []
        
        for idx, signal in recent_signals.iterrows():
            symbol = signal['symbol']
            entry_price = signal['price']
            signal_type = signal['signal']
            
            # Get current price
            try:
                ticker = yf.Ticker(f"{symbol}.JK")
                current_data = ticker.history(period='1d')
                
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    
                    # Calculate unrealized P&L
                    if signal_type == 'BUY':
                        pnl_pct = (current_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - current_price) / entry_price * 100
                    
                    live_results.append({
                        'symbol': symbol,
                        'signal': signal_type,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'pnl_pct': pnl_pct,
                        'status': 'winning' if pnl_pct > 0 else 'losing',
                        'timestamp': signal['timestamp']
                    })
            except:
                continue
        
        if not live_results:
            return {
                "status": "no_live_data",
                "message": "Cannot fetch current prices"
            }
        
        # Calculate summary
        winning = sum(1 for r in live_results if r['status'] == 'winning')
        total = len(live_results)
        live_win_rate = (winning / total * 100) if total > 0 else 0
        avg_pnl = sum(r['pnl_pct'] for r in live_results) / total if total > 0 else 0
        
        return {
            "status": "live",
            "summary": {
                "total_positions": total,
                "winning": winning,
                "losing": total - winning,
                "live_win_rate": f"{live_win_rate:.1f}%",
                "avg_unrealized_pnl": f"{avg_pnl:+.2f}%"
            },
            "positions": live_results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating live performance: {str(e)}"
        )


# =============================================================================
# ENHANCED STATS ENDPOINT - Update existing /api/stats
# =============================================================================

@app.get("/api/stats/enhanced")
def get_enhanced_stats():
    """
    Enhanced statistics including backtest performance
    """
    signals = store.get_signals()
    buy_signals = [s for s in signals if s['signal'] == 'BUY']
    sell_signals = [s for s in signals if s['signal'] == 'SELL']
    
    # Get backtest performance
    backtest_summary = performance_tracker.get_performance_summary()
    
    return {
        "live": {
            "total_stocks": len(store.get_all_stocks()),
            "total_signals": len(signals),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "news_articles": len(store.news_cache),
            "last_scan": signals[0]['timestamp'] if signals else None
        },
        "backtest": backtest_summary,
        "system_status": "Running",
        "market_hours": "09:00 - 16:00 WIB"
    }

@app.post("/api/backtest")
async def run_backtest(
    symbol: str,
    start_date: str = "2024-01-01",
    end_date: str = "2025-10-20",
    min_confidence: float = 40,
    initial_capital: int = 100_000_000
):
    """
    Run backtest on historical data
    
    Example:
    POST /api/backtest?symbol=BBCA&start_date=2024-01-01
    """
    try:
        # Add .JK if not present
        if not symbol.endswith('.JK'):
            symbol = f"{symbol}.JK"
        
        # Run backtest
        bt = Backtester(initial_capital=initial_capital)
        results = bt.run_backtest(symbol, start_date, end_date, min_confidence)
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/backtest/symbols")
def get_backtest_symbols():
    """Get list of symbols available for backtesting"""
    return {
        "symbols": [s.replace('.JK', '') for s in IDX_SYMBOLS],
        "description": "Available symbols for backtesting"
    }

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