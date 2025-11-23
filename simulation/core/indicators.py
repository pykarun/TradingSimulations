"""Technical indicator calculations."""
import pandas as pd


def calculate_ema(data, period):
    """Calculate EMA for given period.
    
    Args:
        data: DataFrame with 'Close' column
        period: EMA period
        
    Returns:
        DataFrame with added 'EMA' column
    """
    df = data.copy()
    df['EMA'] = df['Close'].ewm(span=period, adjust=False).mean()
    return df


def calculate_double_ema(data, fast_period, slow_period):
    """Calculate two EMAs for crossover strategy.
    
    Args:
        data: DataFrame with 'Close' column
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        
    Returns:
        DataFrame with added 'EMA_Fast' and 'EMA_Slow' columns
    """
    df = data.copy()
    df['EMA_Fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    return df


def calculate_rsi(data, period=14):
    """Calculate RSI.
    
    Args:
        data: DataFrame with 'Close' column
        period: RSI period (default 14)
        
    Returns:
        DataFrame with added 'RSI' column
    """
    df = data.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_bollinger_bands(data, period=20, std_dev=2.0):
    """Calculate Bollinger Bands.
    
    Args:
        data: DataFrame with 'Close' column
        period: Moving average period
        std_dev: Number of standard deviations for bands
        
    Returns:
        DataFrame with added BB columns
    """
    df = data.copy()
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    df['BB_Std'] = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * df['BB_Std'])
    # Calculate position within bands (0 = lower band, 1 = upper band)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    return df


def calculate_atr(data, period=14):
    """Calculate Average True Range (ATR).
    
    ATR measures market volatility.
    
    Args:
        data: DataFrame with 'High', 'Low', 'Close' columns
        period: ATR period
        
    Returns:
        DataFrame with added 'ATR' column
    """
    df = data.copy()
    
    # Calculate True Range
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Calculate ATR as moving average of True Range
    df['ATR'] = df['TR'].rolling(window=period).mean()
    
    # Clean up temporary columns
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    
    return df


def calculate_msl_msh(data, msl_period=20, msh_period=20, msl_lookback=5, msh_lookback=5):
    """Calculate MSL (Multi-timeframe Stop Low) and MSH (Multi-timeframe Stop High).
    
    MSL: Lowest low over lookback period, smoothed with moving average
    MSH: Highest high over lookback period, smoothed with moving average
    
    These provide dynamic support/resistance levels for stop loss.
    
    Args:
        data: DataFrame with 'High' and 'Low' columns
        msl_period: Smoothing period for MSL
        msh_period: Smoothing period for MSH
        msl_lookback: Lookback period for lowest low
        msh_lookback: Lookback period for highest high
        
    Returns:
        DataFrame with added MSL/MSH columns
    """
    df = data.copy()
    
    # Calculate rolling lowest low and highest high
    df['Lowest_Low'] = df['Low'].rolling(window=msl_lookback).min()
    df['Highest_High'] = df['High'].rolling(window=msh_lookback).max()
    
    # Smooth with moving average to create MSL and MSH
    df['MSL'] = df['Lowest_Low'].rolling(window=msl_period).mean()
    df['MSH'] = df['Highest_High'].rolling(window=msh_period).mean()
    
    # Calculate distance from current price to MSL/MSH (as percentage)
    df['MSL_Distance'] = ((df['Close'] - df['MSL']) / df['Close']) * 100
    df['MSH_Distance'] = ((df['MSH'] - df['Close']) / df['Close']) * 100
    
    return df


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data: DataFrame with 'Close' column
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
        
    Returns:
        DataFrame with added 'MACD', 'MACD_Signal', 'MACD_Hist' columns
    """
    df = data.copy()
    
    # Calculate Fast and Slow EMAs
    ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    df['MACD'] = ema_fast - ema_slow
    
    # Calculate Signal line
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    
    # Calculate MACD Histogram
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df


def calculate_adx(data, period=14):
    """Calculate ADX (Average Directional Index).
    
    Args:
        data: DataFrame with 'High', 'Low', 'Close' columns
        period: ADX period (default 14)
        
    Returns:
        DataFrame with added 'ADX', '+DI', '-DI' columns
    """
    df = data.copy()
    
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    df['+DM'] = (df['High'] - df['High'].shift(1)).where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 0)
    df['-DM'] = (df['Low'].shift(1) - df['Low']).where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 0)
    
    # Smoothed values
    atr = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (df['+DM'].ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (df['-DM'].ewm(alpha=1/period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    dx = dx.fillna(0)  # Handle potential division by zero
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    df['+DI'] = plus_di
    df['-DI'] = minus_di
    df['ADX'] = adx
    
    # Clean up temporary columns
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR', '+DM', '-DM'], axis=1, inplace=True, errors='ignore')
    
    return df


def calculate_supertrend(data, period=10, multiplier=3.0):
    """Calculate Supertrend indicator (pure pandas implementation).

    This implementation follows the common TradingView Supertrend logic:
    - ATR is calculated using `calculate_atr` (simple rolling mean of True Range)
    - Basic Upper/Lower bands derived from HL2 +/- multiplier * ATR
    - Final bands are adjusted to avoid whipsaws
    - Supertrend flips between the final bands and a direction flag is provided

    Args:
        data: DataFrame with 'High', 'Low', 'Close' columns and a DatetimeIndex
        period: ATR period (default 10)
        multiplier: ATR multiplier (default 3.0)

    Returns:
        DataFrame with added columns: 'ATR', 'ST_upperband', 'ST_lowerband', 'Supertrend', 'ST_dir'
            - 'ST_dir' = 1 for up trend, -1 for down trend
    """
    df = data.copy()

    # Ensure required columns exist
    for col in ['High', 'Low', 'Close']:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column to calculate Supertrend")

    # HL2
    hl2 = (df['High'] + df['Low']) / 2.0

    # ATR (use rolling mean of True Range to stay dependency-free)
    high_low = df['High'] - df['Low']
    high_prevclose = (df['High'] - df['Close'].shift(1)).abs()
    low_prevclose = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_prevclose, low_prevclose], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()

    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    supertrend = pd.Series(index=df.index, dtype='float64')
    dir_series = pd.Series(index=df.index, dtype='int8')

    # Initialize first row
    last_final_ub = basic_ub.iloc[0]
    last_final_lb = basic_lb.iloc[0]
    last_supertrend = basic_ub.iloc[0]
    last_dir = -1  # start as down by convention

    final_ub.iloc[0] = last_final_ub
    final_lb.iloc[0] = last_final_lb
    supertrend.iloc[0] = last_supertrend
    dir_series.iloc[0] = last_dir

    for i in range(1, len(df)):
        curr_basic_ub = basic_ub.iat[i]
        curr_basic_lb = basic_lb.iat[i]
        prev_close = df['Close'].iat[i-1]

        # Final upper band
        if (curr_basic_ub < last_final_ub) or (prev_close > last_final_ub):
            curr_final_ub = curr_basic_ub
        else:
            curr_final_ub = last_final_ub

        # Final lower band
        if (curr_basic_lb > last_final_lb) or (prev_close < last_final_lb):
            curr_final_lb = curr_basic_lb
        else:
            curr_final_lb = last_final_lb

        final_ub.iat[i] = curr_final_ub
        final_lb.iat[i] = curr_final_lb

        curr_close = df['Close'].iat[i]

        # Determine Supertrend value
        if last_supertrend == last_final_ub:
            if curr_close <= curr_final_ub:
                curr_supertrend = curr_final_ub
            else:
                curr_supertrend = curr_final_lb
        else:
            if curr_close >= curr_final_lb:
                curr_supertrend = curr_final_lb
            else:
                curr_supertrend = curr_final_ub

        # Direction: up (1) if price above supertrend, else down (-1)
        curr_dir = 1 if curr_close > curr_supertrend else -1

        supertrend.iat[i] = curr_supertrend
        dir_series.iat[i] = curr_dir

        # Shift last values for next iteration
        last_final_ub = curr_final_ub
        last_final_lb = curr_final_lb
        last_supertrend = curr_supertrend
        last_dir = curr_dir

    df['ATR'] = atr
    df['ST_upperband'] = final_ub
    df['ST_lowerband'] = final_lb
    df['Supertrend'] = supertrend
    df['ST_dir'] = dir_series

    return df
