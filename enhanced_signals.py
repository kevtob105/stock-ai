"""
ENHANCED SIGNAL GENERATION ENGINE
With Additional Technical Indicators for Better Accuracy

New Indicators:
1. Stochastic Oscillator
2. ADX (Average Directional Index)
3. OBV (On-Balance Volume)
4. ATR (Average True Range)
5. Support/Resistance Levels
"""

import pandas as pd
import numpy as np


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator
    
    Args:
        high: High prices array
        low: Low prices array
        close: Close prices array
        k_period: %K period (default 14)
        d_period: %D period (default 3)
    
    Returns:
        k: %K line
        d: %D line (signal line)
    """
    if len(close) < k_period:
        return 50, 50
    
    # Get last k_period values
    recent_high = pd.Series(high[-k_period:]).rolling(k_period).max().iloc[-1]
    recent_low = pd.Series(low[-k_period:]).rolling(k_period).min().iloc[-1]
    
    # Calculate %K
    if recent_high - recent_low == 0:
        k = 50
    else:
        k = ((close[-1] - recent_low) / (recent_high - recent_low)) * 100
    
    # Calculate %D (SMA of %K)
    # For simplicity, we'll use current K as D
    # In production, you'd track K history
    d = k  # Simplified
    
    return k, d


def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index)
    
    Args:
        high: High prices array
        low: Low prices array
        close: Close prices array
        period: ADX period (default 14)
    
    Returns:
        adx: ADX value
    """
    if len(close) < period + 1:
        return 25  # Neutral
    
    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    })
    
    # Calculate True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate Directional Movement
    df['dm_plus'] = np.where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
        np.maximum(df['high'] - df['high'].shift(1), 0),
        0
    )
    df['dm_minus'] = np.where(
        (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
        np.maximum(df['low'].shift(1) - df['low'], 0),
        0
    )
    
    # Smooth with Wilder's method
    atr = df['tr'].rolling(period).mean().iloc[-1]
    
    if atr == 0:
        return 25
    
    di_plus = (df['dm_plus'].rolling(period).mean().iloc[-1] / atr) * 100
    di_minus = (df['dm_minus'].rolling(period).mean().iloc[-1] / atr) * 100
    
    # Calculate DX
    if di_plus + di_minus == 0:
        return 25
    
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    
    # ADX is smoothed DX (simplified here)
    adx = dx
    
    return adx


def calculate_obv(close, volume):
    """
    Calculate OBV (On-Balance Volume)
    
    Args:
        close: Close prices array
        volume: Volume array
    
    Returns:
        obv: Current OBV value
        obv_trend: 'rising', 'falling', or 'neutral'
    """
    if len(close) < 20:
        return 0, 'neutral'
    
    df = pd.DataFrame({
        'close': close,
        'volume': volume
    })
    
    # Calculate OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    
    # Determine trend
    obv_ma = pd.Series(obv[-20:]).rolling(10).mean()
    
    if obv_ma.iloc[-1] > obv_ma.iloc[-5]:
        trend = 'rising'
    elif obv_ma.iloc[-1] < obv_ma.iloc[-5]:
        trend = 'falling'
    else:
        trend = 'neutral'
    
    return obv[-1], trend


def calculate_atr(high, low, close, period=14):
    """
    Calculate ATR (Average True Range)
    
    Args:
        high: High prices array
        low: Low prices array
        close: Close prices array
        period: ATR period (default 14)
    
    Returns:
        atr: Current ATR value
    """
    if len(close) < period + 1:
        return 0
    
    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    })
    
    # Calculate True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # ATR is smoothed TR
    atr = df['tr'].rolling(period).mean().iloc[-1]
    
    return atr


def find_support_resistance(close, window=20):
    """
    Find support and resistance levels
    
    Args:
        close: Close prices array
        window: Lookback window (default 20)
    
    Returns:
        support: Support level
        resistance: Resistance level
    """
    if len(close) < window:
        return close[-1] * 0.95, close[-1] * 1.05
    
    recent_prices = close[-window:]
    
    # Simple method: use recent high/low
    resistance = np.max(recent_prices)
    support = np.min(recent_prices)
    
    return support, resistance


def calculate_price_position(current_price, support, resistance):
    """
    Calculate where price is relative to support/resistance
    
    Returns:
        position: 0-100 (0=support, 100=resistance)
        status: 'near_support', 'near_resistance', or 'middle'
    """
    if resistance == support:
        return 50, 'middle'
    
    position = ((current_price - support) / (resistance - support)) * 100
    
    if position < 10:
        status = 'near_support'
    elif position > 90:
        status = 'near_resistance'
    else:
        status = 'middle'
    
    return position, status


# =============================================================================
# ENHANCED SIGNAL GENERATION
# =============================================================================

def generate_enhanced_signal(df, overall_sentiment="Neutral"):
    """
    Generate trading signal with enhanced indicators
    
    Args:
        df: DataFrame with OHLCV data
        overall_sentiment: Market sentiment
    
    Returns:
        dict with signal, confidence, and reasons
    """
    if len(df) < 50:
        return {
            'signal': 'HOLD',
            'confidence': 0,
            'reasons': ['Insufficient data']
        }
    
    # Extract data
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    volume = df['Volume'].values
    
    current_price = close[-1]
    
    # =================================================================
    # CALCULATE ALL INDICATORS
    # =================================================================
    
    # Original indicators
    from main import calculate_rsi, calculate_macd, calculate_moving_averages
    
    rsi = calculate_rsi(close)
    macd_value, signal_value = calculate_macd(close)
    macd_diff = macd_value - signal_value
    ma20, ma50 = calculate_moving_averages(close)
    
    # New indicators
    stoch_k, stoch_d = calculate_stochastic(high, low, close)
    adx = calculate_adx(high, low, close)
    obv_value, obv_trend = calculate_obv(close, volume)
    atr = calculate_atr(high, low, close)
    support, resistance = find_support_resistance(close)
    price_pos, price_status = calculate_price_position(current_price, support, resistance)
    
    # Volume analysis
    avg_volume = np.mean(volume[-20:])
    volume_ratio = volume[-1] / avg_volume if avg_volume > 0 else 1
    
    # Price change
    price_change_pct = ((current_price - close[-2]) / close[-2] * 100) if len(close) > 1 else 0
    
    # =================================================================
    # SCORING SYSTEM (Enhanced)
    # =================================================================
    
    score = 0
    reasons = []
    confidence_multiplier = 1.0
    
    # 1. RSI Analysis (20 points) - ORIGINAL
    if rsi < 30:
        score += 20
        reasons.append(f"RSI Oversold ({rsi:.1f})")
    elif rsi > 70:
        score -= 20
        reasons.append(f"RSI Overbought ({rsi:.1f})")
    elif 30 <= rsi <= 45:
        score += 10
        reasons.append(f"RSI Moderate Buy ({rsi:.1f})")
    elif 55 <= rsi <= 70:
        score -= 10
        reasons.append(f"RSI Moderate Sell ({rsi:.1f})")
    
    # 2. MACD Analysis (15 points) - ORIGINAL
    if macd_diff > 0:
        score += 15
        reasons.append("MACD Bullish")
    else:
        score -= 15
        reasons.append("MACD Bearish")
    
    # 3. Moving Average (15 points) - ORIGINAL
    if current_price > ma20 and ma20 > ma50:
        score += 15
        reasons.append("Golden Cross")
    elif current_price < ma20 and ma20 < ma50:
        score -= 15
        reasons.append("Death Cross")
    
    # 4. Stochastic Oscillator (20 points) - NEW ⭐
    if stoch_k < 20:
        score += 20
        reasons.append(f"Stochastic Oversold ({stoch_k:.1f})")
    elif stoch_k > 80:
        score -= 20
        reasons.append(f"Stochastic Overbought ({stoch_k:.1f})")
    elif stoch_k < 40:
        score += 10
        reasons.append(f"Stochastic Moderate ({stoch_k:.1f})")
    
    # 5. ADX Trend Strength (Confidence Multiplier) - NEW ⭐
    if adx > 40:
        confidence_multiplier = 1.3
        reasons.append(f"Very Strong Trend (ADX:{adx:.0f})")
    elif adx > 25:
        confidence_multiplier = 1.1
        reasons.append(f"Strong Trend (ADX:{adx:.0f})")
    elif adx < 20:
        confidence_multiplier = 0.7
        reasons.append(f"Weak Trend (ADX:{adx:.0f})")
    
    # 6. OBV Analysis (15 points) - NEW ⭐
    price_trend = 'rising' if current_price > close[-10] else 'falling'
    
    if price_trend == obv_trend and obv_trend == 'rising':
        score += 15
        reasons.append("OBV Confirms Uptrend")
    elif price_trend == obv_trend and obv_trend == 'falling':
        score -= 15
        reasons.append("OBV Confirms Downtrend")
    elif price_trend != obv_trend:
        score -= 10
        reasons.append("OBV Divergence (Warning)")
    
    # 7. Support/Resistance (15 points) - NEW ⭐
    if price_status == 'near_support':
        score += 15
        reasons.append(f"Near Support (Rp {support:,.0f})")
    elif price_status == 'near_resistance':
        score -= 15
        reasons.append(f"Near Resistance (Rp {resistance:,.0f})")
    
    # 8. Volume Analysis (10 points) - ORIGINAL
    if volume_ratio > 1.5:
        score += 10
        reasons.append(f"High Volume ({volume_ratio:.1f}x)")
    elif volume_ratio < 0.5:
        score -= 5
        reasons.append("Low Volume")
    
    # 9. ATR Volatility Check - NEW ⭐
    atr_pct = (atr / current_price) * 100
    if atr_pct > 5:
        confidence_multiplier *= 0.9
        reasons.append(f"High Volatility (ATR:{atr_pct:.1f}%)")
    
    # 10. Market Sentiment (10 points) - ORIGINAL
    if overall_sentiment == "Bullish":
        score += 10
        reasons.append("Bullish Market Sentiment")
    elif overall_sentiment == "Bearish":
        score -= 10
        reasons.append("Bearish Market Sentiment")
    
    # 11. Price Momentum (5 points) - ORIGINAL
    if price_change_pct > 3:
        score += 5
        reasons.append(f"Strong Momentum (+{price_change_pct:.1f}%)")
    elif price_change_pct < -3:
        score -= 5
        reasons.append(f"Weak Momentum ({price_change_pct:.1f}%)")
    
    # =================================================================
    # FINAL SIGNAL DETERMINATION
    # =================================================================
    
    # Apply confidence multiplier (from ADX and ATR)
    final_score = score * confidence_multiplier
    confidence = min(abs(final_score), 100)
    
    # Signal thresholds (adjusted for new scoring)
    if final_score >= 60:
        signal = "BUY"
    elif final_score <= -60:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    # =================================================================
    # ADDITIONAL FILTERS (Safety checks)
    # =================================================================
    
    # Filter 1: Don't trade in very weak trends
    if adx < 15 and signal != "HOLD":
        signal = "HOLD"
        reasons.append("⚠️ Trend too weak (ADX < 15)")
        confidence *= 0.5
    
    # Filter 2: Don't trade against strong S/R
    if signal == "BUY" and price_status == "near_resistance":
        confidence *= 0.7
        reasons.append("⚠️ Near resistance (Reduced confidence)")
    
    if signal == "SELL" and price_status == "near_support":
        confidence *= 0.7
        reasons.append("⚠️ Near support (Reduced confidence)")
    
    # =================================================================
    # RETURN RESULT
    # =================================================================
    
    return {
        'signal': signal,
        'confidence': float(confidence),
        'reasons': reasons,
        'raw_score': float(final_score),
        'indicators': {
            'rsi': float(rsi),
            'macd': float(macd_diff),
            'stochastic': float(stoch_k),
            'adx': float(adx),
            'obv_trend': obv_trend,
            'atr_pct': float(atr_pct),
            'support': float(support),
            'resistance': float(resistance),
            'price_position': float(price_pos)
        }
    }


# =============================================================================
# COMPARISON FUNCTION
# =============================================================================

def compare_signals(df, sentiment):
    """
    Compare original vs enhanced signal
    """
    from main import generate_trading_signal
    
    # Original signal
    original = generate_trading_signal(df['Close'].values, df, sentiment)
    
    # Enhanced signal
    enhanced = generate_enhanced_signal(df, sentiment)
    
    print("\n" + "="*60)
    print("SIGNAL COMPARISON")
    print("="*60)
    
    print(f"\n📊 ORIGINAL SIGNAL:")
    print(f"   Signal: {original['signal']}")
    print(f"   Confidence: {original['confidence']:.1f}%")
    print(f"   Reasons: {', '.join(original['reasons'][:3])}")
    
    print(f"\n✨ ENHANCED SIGNAL:")
    print(f"   Signal: {enhanced['signal']}")
    print(f"   Confidence: {enhanced['confidence']:.1f}%")
    print(f"   Raw Score: {enhanced['raw_score']:.1f}")
    print(f"   Reasons: {', '.join(enhanced['reasons'][:3])}")
    
    print(f"\n🔬 ADDITIONAL INDICATORS:")
    ind = enhanced['indicators']
    print(f"   Stochastic: {ind['stochastic']:.1f}")
    print(f"   ADX: {ind['adx']:.1f}")
    print(f"   OBV Trend: {ind['obv_trend']}")
    print(f"   ATR: {ind['atr_pct']:.2f}%")
    print(f"   Support: Rp {ind['support']:,.0f}")
    print(f"   Resistance: Rp {ind['resistance']:,.0f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("Enhanced Signal Generation Module")
    print("Ready to be integrated into main.py")