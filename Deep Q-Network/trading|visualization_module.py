# black_scholes_yfinance.py
import yfinance as yf
import numpy as np
from math import log, sqrt, exp, erf
from datetime import datetime, date

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def black_scholes_call_put(S, K, t, r, sigma):
    """
    S: spot price
    K: strike
    t: time to maturity in years (float). If t == 0 returns intrinsic values.
    r: annual risk-free rate (continuous compounding)
    sigma: annual volatility (std dev)
    returns: (call_price, put_price)
    """
    if t <= 0:
        call = max(S - K, 0.0)
        put  = max(K - S, 0.0)
        return call, put

    sqrt_t = sqrt(t)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    call = S * norm_cdf(d1) - K * exp(-r * t) * norm_cdf(d2)
    put  = K * exp(-r * t) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return call, put

def fetch_spot_and_vol(ticker, hist_days=252):
    """
    Fetches historical daily closes for `hist_days` trading days (or available).
    Returns (spot_price, annualized_volatility).
    Volatility is computed as std(log returns) * sqrt(252).
    """
    tk = yf.Ticker(ticker)
    # get up to hist_days+1 closes to compute returns
    period = f"{max(30, hist_days)}d"
    hist = tk.history(period=period, interval="1d")  # DataFrame with 'Close'
    if hist.empty:
        raise RuntimeError(f"No historical data returned for {ticker}.")
    closes = hist['Close'].dropna()
    # spot: most recent close (or regularMarketPrice fallback)
    try:
        spot = float(closes.iloc[-1])
    except Exception:
        info = tk.info if hasattr(tk, "info") else {}
        spot = float(info.get("regularMarketPrice", closes.mean()))
    # compute log returns
    logrets = np.log(closes / closes.shift(1)).dropna()
    if logrets.empty:
        raise RuntimeError("Not enough price history to estimate volatility.")
    # use last `hist_days` returns if available
    if len(logrets) >= hist_days:
        sample = logrets.iloc[-hist_days:]
    else:
        sample = logrets
    sigma = float(sample.std(ddof=1) * sqrt(252))  # annualize
    
    return spot, sigma

def parse_time_to_maturity(maturity_date=None, years=None):
    """
    Provide either maturity_date as 'YYYY-MM-DD' or years as float.
    Returns time-to-maturity in years (float).
    """
    if years is not None:
        return float(years)
    if maturity_date is None:
        raise ValueError("Provide maturity_date or years.")
    mat = datetime.strptime(maturity_date, "%Y-%m-%d").date()
    today = date.today()
    delta_days = (mat - today).days
    return max(delta_days / 365.0, 0.0)

# -------- Example usage --------
if __name__ == "__main__":
    ticker = "AAPL"              # example ticker
    strike = 180.0               # example strike
    maturity = "2026-01-17"      # or set years=0.25
    risk_free = 0.035            # 3.5% annual continuous rate (adjust as needed)

    # fetch spot and vol
    S, implied_sigma = fetch_spot_and_vol(ticker, hist_days=252)
    t = parse_time_to_maturity(maturity_date=maturity)
    call, put = black_scholes_call_put(S, strike, t, risk_free, implied_sigma)

    print(f"Ticker: {ticker}")
    print(f"Spot: {S:.4f}")
    print(f"Estimated annual vol (sigma): {implied_sigma:.4%}")
    print(f"Time to maturity (years): {t:.6f}")
    print(f"Risk-free rate (r): {risk_free:.4%}")
    print(f"Call price: {call:.4f}")
    print(f"Put price:  {put:.4f}")
