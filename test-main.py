"""
Minimal test to debug yfinance in main.py context
"""

import yfinance as yf

print("Testing yfinance in main.py environment...")

symbol = 'BBCA.JK'
print(f"\nFetching {symbol}...")

try:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='1mo')
    
    if not df.empty:
        print(f"✅ SUCCESS!")
        print(f"   Days: {len(df)}")
        print(f"   Close: Rp {df['Close'].iloc[-1]:,.0f}")
        print(f"\nFirst few rows:")
        print(df.head())
    else:
        print("❌ FAILED - Empty dataframe")
        
except Exception as e:
    print(f"❌ EXCEPTION: {e}")
    import traceback
    traceback.print_exc()