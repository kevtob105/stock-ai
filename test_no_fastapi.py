"""
Test yfinance WITHOUT FastAPI - isolate the issue
"""

import yfinance as yf
import time

print("=" * 60)
print("Testing yfinance WITHOUT FastAPI...")
print("=" * 60)

symbols = ['BBCA.JK', 'BBRI.JK', 'BMRI.JK']

for symbol in symbols:
    print(f"\n{symbol}:")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='1mo')
        
        if not df.empty:
            print(f"  ✅ SUCCESS - {len(df)} days")
            print(f"  Close: Rp {df['Close'].iloc[-1]:,.0f}")
        else:
            print(f"  ❌ EMPTY")
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
    
    time.sleep(2)

print("\n" + "=" * 60)