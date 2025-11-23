"""Trading strategy execution."""
import pandas as pd
from .indicators import (
    calculate_ema, calculate_double_ema, calculate_rsi,
    calculate_bollinger_bands, calculate_atr, calculate_msl_msh,
    calculate_macd, calculate_adx, calculate_supertrend
)


def run_tqqq_only_strategy(
    qqq_data, tqqq_data, start_date, end_date, initial_capital, ema_period,
    rsi_threshold=0, use_rsi=False, rsi_oversold=30, rsi_overbought=70,
    stop_loss_pct=0, use_stop_loss=False, use_double_ema=False, ema_fast=9,
    ema_slow=21, use_bb=False, bb_period=20, bb_std_dev=2.0,
    bb_buy_threshold=0.2, bb_sell_threshold=0.8, use_atr=False, atr_period=14,
    atr_multiplier=2.0, use_msl_msh=False, msl_period=20, msh_period=20,
    msl_lookback=5, msh_lookback=5, use_ema=True, use_macd=False,
    macd_fast=12, macd_slow=26, macd_signal_period=9, use_adx=False,
    adx_period=14, adx_threshold=25,
    use_supertrend=False, st_period=10, st_multiplier=3.0
):
    """Smart Leverage Strategy - TQQQ with EMA, RSI, Bollinger Bands & Stop-Loss.
    
    Args:
        qqq_data: QQQ price data
        tqqq_data: TQQQ price data
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        ema_period: EMA period for single EMA strategy
        rsi_threshold: RSI momentum threshold
        use_rsi: Enable RSI filter
        rsi_oversold: RSI oversold level
        rsi_overbought: RSI overbought level
        stop_loss_pct: Stop-loss percentage
        use_stop_loss: Enable stop-loss
        use_double_ema: Use double EMA crossover
        ema_fast: Fast EMA period
        ema_slow: Slow EMA period
        use_bb: Enable Bollinger Bands filter
        bb_period: Bollinger Bands period
        bb_std_dev: Bollinger Bands standard deviation
        bb_buy_threshold: BB buy threshold
        bb_sell_threshold: BB sell threshold
        use_atr: Enable ATR stop-loss
        atr_period: ATR period
        atr_multiplier: ATR multiplier
        use_msl_msh: Enable MSL/MSH stop-loss
        msl_period: MSL smoothing period
        msh_period: MSH smoothing period
        msl_lookback: MSL lookback period
        msh_lookback: MSH lookback period
        use_ema: Enable EMA strategy
        
    Returns:
        Dictionary with backtest results
    """
    
    SLIPPAGE_BUY_PCT = 0.0010   # 0.10% slippage when buying
    SLIPPAGE_SELL_PCT = 0.0010  # 0.10% slippage when selling
    COMMISSION_PCT = 0.0000     # 0% commission (Alpaca is free)
    
    # Only calculate EMA if enabled
    if use_ema:
        if use_double_ema:
            qqq_data = calculate_double_ema(qqq_data, ema_fast, ema_slow)
        else:
            qqq_data = calculate_ema(qqq_data, ema_period)
    
    qqq_data = calculate_rsi(qqq_data, period=14)
    if use_bb:
        qqq_data = calculate_bollinger_bands(qqq_data, bb_period, bb_std_dev)
    if use_atr:
        qqq_data = calculate_atr(qqq_data, atr_period)
        tqqq_data = calculate_atr(tqqq_data, atr_period)
    if use_msl_msh:
        qqq_data = calculate_msl_msh(qqq_data, msl_period, msh_period, msl_lookback, msh_lookback)
        tqqq_data = calculate_msl_msh(tqqq_data, msl_period, msh_period, msl_lookback, msh_lookback)
    
    if use_macd:
        qqq_data = calculate_macd(qqq_data, macd_fast, macd_slow, macd_signal_period)
        
    if use_adx:
        qqq_data = calculate_adx(qqq_data, adx_period)

    if use_supertrend:
        # Calculate Supertrend on QQQ and TQQQ data
        try:
            qqq_data = calculate_supertrend(qqq_data, period=st_period, multiplier=st_multiplier)
            tqqq_data = calculate_supertrend(tqqq_data, period=st_period, multiplier=st_multiplier)
        except Exception:
            # If Supertrend calculation fails, ensure the rest still runs
            pass
    
    sim_data = qqq_data[start_date:end_date].copy()
    
    capital = initial_capital
    position = None
    shares = 0
    portfolio_value = []
    trade_log = []
    entry_value = 0
    peak_value = 0
    
    for i in range(len(sim_data)):
        date = sim_data.index[i]
        qqq_close = sim_data.iloc[i]['Close']
        
        # Get EMA values if EMA is enabled
        if use_ema:
            if use_double_ema:
                ema_fast_val = sim_data.iloc[i]['EMA_Fast']
                ema_slow_val = sim_data.iloc[i]['EMA_Slow']
                qqq_ema = ema_slow_val
            else:
                qqq_ema = sim_data.iloc[i]['EMA']
        else:
            ema_fast_val = None
            ema_slow_val = None
            qqq_ema = None
        
        rsi = sim_data.iloc[i]['RSI']
        
        if date not in tqqq_data.index:
            continue
        tqqq_close = tqqq_data.loc[date]['Close']
        
        if position == 'TQQQ':
            current_value = shares * tqqq_close
        else:
            current_value = capital
        
        if position == 'TQQQ':
            peak_value = max(peak_value, current_value)
        
        stop_loss_triggered = False
        stop_loss_reason = ''
        
        # Traditional percentage-based stop loss
        if use_stop_loss and position == 'TQQQ' and peak_value > 0:
            drawdown_from_peak = ((current_value - peak_value) / peak_value) * 100
            if drawdown_from_peak <= -stop_loss_pct:
                stop_loss_triggered = True
                stop_loss_reason = f'Stop-Loss ({drawdown_from_peak:.2f}% from peak)'
        
        # ATR-based stop loss
        if use_atr and position == 'TQQQ' and date in tqqq_data.index:
            tqqq_atr = tqqq_data.loc[date].get('ATR', None)
            if pd.notna(tqqq_atr) and shares > 0:
                atr_stop_price = tqqq_close - (atr_multiplier * tqqq_atr)
                if tqqq_close <= atr_stop_price:
                    stop_loss_triggered = True
                    stop_loss_reason = f'ATR Stop-Loss (Price: ${tqqq_close:.2f} <= Stop: ${atr_stop_price:.2f})'
        
        # MSL/MSH-based stop loss
        if use_msl_msh and position == 'TQQQ' and date in tqqq_data.index:
            tqqq_msl = tqqq_data.loc[date].get('MSL', None)
            if pd.notna(tqqq_msl) and tqqq_close < tqqq_msl:
                stop_loss_triggered = True
                stop_loss_reason = f'MSL Stop-Loss (Price: ${tqqq_close:.2f} < MSL: ${tqqq_msl:.2f})'
        
        # Generate base signal
        if use_ema:
            if use_double_ema:
                base_signal = 'BUY' if ema_fast_val > ema_slow_val else 'SELL'
            else:
                base_signal = 'BUY' if qqq_close > qqq_ema else 'SELL'
        else:
            base_signal = 'BUY'
        
        if use_rsi:
            if pd.notna(rsi):
                rsi_buy_signal = rsi < rsi_oversold or (rsi > rsi_threshold and base_signal == 'BUY')
                rsi_sell_signal = rsi > rsi_overbought
                
                if rsi_sell_signal:
                    signal = 'SELL'
                elif base_signal == 'BUY' and rsi_buy_signal:
                    signal = 'BUY'
                else:
                    signal = 'SELL'
            else:
                signal = base_signal
        else:
            signal = base_signal
        
        # Apply filters that can turn a BUY signal into a SELL
        if signal == 'BUY':
            # Bollinger Bands buy filter
            if use_bb:
                bb_position = sim_data.iloc[i]['BB_Position']
                if pd.notna(bb_position) and bb_position > bb_buy_threshold:
                    signal = 'SELL'
            
            # MACD buy filter
            if use_macd and signal == 'BUY':
                macd_hist = sim_data.iloc[i].get('MACD_Hist', None)
                if pd.notna(macd_hist) and macd_hist <= 0:
                    signal = 'SELL'

            # ADX buy filter
            if use_adx and signal == 'BUY':
                adx = sim_data.iloc[i].get('ADX', None)
                plus_di = sim_data.iloc[i].get('+DI', None)
                minus_di = sim_data.iloc[i].get('-DI', None)
                if pd.notna(adx) and pd.notna(plus_di) and pd.notna(minus_di):
                    if adx < adx_threshold or plus_di < minus_di:
                        signal = 'SELL'

            # Supertrend buy filter: require Supertrend direction up for buys
            if use_supertrend and signal == 'BUY':
                st_dir = sim_data.iloc[i].get('ST_dir', None)
                if pd.isna(st_dir) or st_dir != 1:
                    signal = 'SELL'

        # Standalone SELL conditions that can override anything
        if use_bb:
            bb_position = sim_data.iloc[i]['BB_Position']
            if pd.notna(bb_position) and position == 'TQQQ' and bb_position >= bb_sell_threshold:
                signal = 'SELL'

        
        if stop_loss_triggered:
            signal = 'SELL'
        
        action = 'HOLD'
        if signal == 'BUY' and position != 'TQQQ':
            execution_price = tqqq_close * (1 + SLIPPAGE_BUY_PCT)
            shares = capital / execution_price
            position = 'TQQQ'
            entry_value = current_value
            peak_value = current_value
            action = f'BUY TQQQ @ ${execution_price:.2f} (slippage: {SLIPPAGE_BUY_PCT*100:.2f}%)'
            capital = 0
        elif signal == 'SELL' and position == 'TQQQ':
            execution_price = tqqq_close * (1 - SLIPPAGE_SELL_PCT)
            capital = shares * execution_price * (1 - COMMISSION_PCT)
            shares = 0
            position = None
            entry_value = 0
            peak_value = 0
            if stop_loss_triggered:
                action = f'SELL (STOP-LOSS) @ ${execution_price:.2f} (slippage: {SLIPPAGE_SELL_PCT*100:.2f}%)'
            else:
                action = f'SELL to CASH @ ${execution_price:.2f} (slippage: {SLIPPAGE_SELL_PCT*100:.2f}%)'
        else:
            if position == 'TQQQ':
                action = 'HOLD TQQQ'
            else:
                action = 'HOLD CASH'
        
        if position == 'TQQQ':
            current_value = shares * tqqq_close
        else:
            current_value = capital
        
        if use_ema:
            if use_double_ema:
                signal_text = f'Fast({ema_fast_val:.2f}) > Slow({ema_slow_val:.2f})' if ema_fast_val > ema_slow_val else f'Fast({ema_fast_val:.2f}) < Slow({ema_slow_val:.2f})'
                ema_display = f'${ema_fast_val:.2f} / ${ema_slow_val:.2f}'
            else:
                signal_text = 'Above EMA' if qqq_close > qqq_ema else 'Below EMA'
                ema_display = f'${qqq_ema:.2f}'
        else:
            signal_text = 'EMA Disabled'
            ema_display = 'N/A'
        
        trade_log.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Action': action,
            'QQQ_Price': f'${qqq_close:.2f}',
            'QQQ_EMA': ema_display,
            'RSI': f'{rsi:.1f}' if pd.notna(rsi) else 'N/A',
            'MACD_Hist': f'{sim_data.iloc[i].get("MACD_Hist", 0):.2f}' if use_macd else 'N/A',
            'ADX': f'{sim_data.iloc[i].get("ADX", 0):.1f}' if use_adx else 'N/A',
            'Signal': signal_text,
            'TQQQ_Price': f'${tqqq_close:.2f}',
            'Shares': f'{shares:.2f}',
            'Portfolio_Value': f'${current_value:,.2f}',
            'Position': position if position else 'Cash',
            'Stop_Loss': stop_loss_reason if stop_loss_triggered else ''
        })
        
        portfolio_value.append({'Date': date, 'Value': current_value})
    
    portfolio_df = pd.DataFrame(portfolio_value).set_index('Date')
    final_value = portfolio_df['Value'].iloc[-1]
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    
    portfolio_df['Peak'] = portfolio_df['Value'].cummax()
    portfolio_df['Drawdown'] = (portfolio_df['Value'] - portfolio_df['Peak']) / portfolio_df['Peak'] * 100
    max_drawdown = portfolio_df['Drawdown'].min()
    
    num_trades = len([log for log in trade_log if 'BUY TQQQ' in log['Action']])
    num_stop_loss_exits = len([log for log in trade_log if 'STOP-LOSS' in log['Action']])
    
    round_trip_cost_pct = SLIPPAGE_BUY_PCT + SLIPPAGE_SELL_PCT + COMMISSION_PCT
    estimated_total_costs_pct = num_trades * round_trip_cost_pct * 100
    
    return {
        'portfolio_df': portfolio_df,
        'trade_log': trade_log,
        'final_value': final_value,
        'total_return_pct': total_return,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'num_stop_loss_exits': num_stop_loss_exits,
        'slippage_buy_pct': SLIPPAGE_BUY_PCT * 100,
        'slippage_sell_pct': SLIPPAGE_SELL_PCT * 100,
        'commission_pct': COMMISSION_PCT * 100,
        'estimated_total_costs_pct': estimated_total_costs_pct
    }
