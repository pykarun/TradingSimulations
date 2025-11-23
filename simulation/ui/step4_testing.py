"""Step 4: Testing - Run strategy tests and simulations.

This module provides three testing modes:
1. Daily Signal - Get today's buy/sell recommendation
2. Custom Simulation - Backtest on historical data
3. Monte Carlo Simulation - Probability analysis

Due to the size of this module, it's split into logical sections.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from curl_cffi import requests
import plotly.graph_objects as go

from core import get_data, run_tqqq_only_strategy, calculate_double_ema, calculate_ema, calculate_rsi, calculate_bollinger_bands, calculate_macd, calculate_adx
from utils import show_applied_params_banner, create_performance_chart

try:
    rerun = st.rerun
except AttributeError:
    rerun = st.experimental_rerun


def render_step4():
    """Render Step 4: Testing interface."""
    
    st.markdown("## 📊 Step 4: Testing")
    st.markdown("---")
    
    # Show applied parameters banner
    show_applied_params_banner()
    
    st.markdown("---")
    
    # Auto-load strategy
    if st.session_state.get('best_params') is not None:
        if not st.session_state.get('testing_params_loaded', False):
            _load_testing_params()
            st.session_state.testing_params_loaded = True
        
    else:
        st.warning("⚠️ No strategy selected. Please find best signals in Step 2 first.")
        st.stop()
    
    # Clear navigation flag
    if st.session_state.get('navigate_to_step2', False):
        st.session_state.navigate_to_step2 = False
    
    # Load parameters
    params = _get_loaded_params()

    
    # Create tabs for different testing modes
    test_tab1, test_tab2, test_tab3 = st.tabs([
        "📊 Daily Signal",
        "📈 Custom Simulation",
        "🎲 Monte Carlo Simulation"
    ])
    
    # Initialize tab tracking
    if 'last_analysis_tab' not in st.session_state:
        st.session_state.last_analysis_tab = None
    if 'clear_results' not in st.session_state:
        st.session_state.clear_results = False
    
    # Render tabs and get button states
    run_button_daily, run_button_custom, run_button_monte, test_params = _render_testing_tabs(
        test_tab1, test_tab2, test_tab3, params
    )
    
    # Determine which test to run
    if run_button_daily:
        _run_daily_signal(params)
    elif run_button_custom:
        _run_custom_simulation(params, test_params)
    elif run_button_monte:
        _run_monte_carlo(params, test_params)
    
    # Navigation buttons (always show at bottom)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("⬅️ Previous Step", use_container_width=True, key="step4_prev"):
            st.session_state.wizard_step = 3
            rerun()
    
    with col2:
        pass  # Empty middle column for spacing
    
    with col3:
        if st.button("Next Step ➡️", type="primary", use_container_width=True, key="step4_next"):
            st.session_state.wizard_step = 5
            rerun()
    
    # Add bottom padding for mobile visibility
    st.markdown("<br><br>", unsafe_allow_html=True)


def _load_testing_params():
    """Load best_params into testing session state."""
    best_params = st.session_state.best_params
    
    st.session_state.use_ema = best_params.get('use_ema', True)
    st.session_state.use_double_ema = best_params.get('use_double_ema', False)
    st.session_state.ema_period = best_params.get('ema_period', 50)
    st.session_state.ema_fast = best_params.get('ema_fast', 9)
    st.session_state.ema_slow = best_params.get('ema_slow', 21)
    st.session_state.use_rsi = best_params.get('use_rsi', False)
    st.session_state.rsi_threshold = best_params.get('rsi_threshold', 50)
    st.session_state.rsi_oversold = best_params.get('rsi_oversold', 30)
    st.session_state.rsi_overbought = best_params.get('rsi_overbought', 70)
    st.session_state.use_stop_loss = best_params.get('use_stop_loss', False)
    st.session_state.stop_loss_pct = best_params.get('stop_loss_pct', 10)
    st.session_state.use_bb = best_params.get('use_bb', False)
    st.session_state.bb_period = best_params.get('bb_period', 20)
    st.session_state.bb_std_dev = best_params.get('bb_std_dev', 2.0)
    st.session_state.bb_buy_threshold = best_params.get('bb_buy_threshold', 0.2)
    st.session_state.bb_sell_threshold = best_params.get('bb_sell_threshold', 0.8)
    st.session_state.use_atr = best_params.get('use_atr', False)
    st.session_state.atr_period = best_params.get('atr_period', 14)
    st.session_state.atr_multiplier = best_params.get('atr_multiplier', 2.0)
    st.session_state.use_msl_msh = best_params.get('use_msl_msh', False)
    st.session_state.msl_period = best_params.get('msl_period', 20)
    st.session_state.msh_period = best_params.get('msh_period', 20)
    st.session_state.msl_lookback = best_params.get('msl_lookback', 5)
    st.session_state.msh_lookback = best_params.get('msh_lookback', 5)
    st.session_state.use_macd = best_params.get('use_macd', False)
    st.session_state.macd_fast = best_params.get('macd_fast', 12)
    st.session_state.macd_slow = best_params.get('macd_slow', 26)
    st.session_state.macd_signal_period = best_params.get('macd_signal_period', 9)
    st.session_state.use_adx = best_params.get('use_adx', False)
    st.session_state.adx_period = best_params.get('adx_period', 14)
    st.session_state.adx_threshold = best_params.get('adx_threshold', 25)


def _get_loaded_params():
    """Get loaded parameters from session state."""
    return {
        'use_ema': st.session_state.use_ema,
        'use_double_ema': st.session_state.use_double_ema,
        'ema_period': st.session_state.ema_period,
        'ema_fast': st.session_state.ema_fast,
        'ema_slow': st.session_state.ema_slow,
        'use_rsi': st.session_state.use_rsi,
        'rsi_threshold': st.session_state.rsi_threshold,
        'rsi_oversold': st.session_state.rsi_oversold,
        'rsi_overbought': st.session_state.rsi_overbought,
        'use_stop_loss': st.session_state.use_stop_loss,
        'stop_loss_pct': st.session_state.stop_loss_pct,
        'use_bb': st.session_state.use_bb,
        'bb_period': st.session_state.bb_period,
        'bb_std_dev': st.session_state.bb_std_dev,
        'bb_buy_threshold': st.session_state.bb_buy_threshold,
        'bb_sell_threshold': st.session_state.bb_sell_threshold,
        'use_atr': st.session_state.use_atr,
        'atr_period': st.session_state.atr_period,
        'atr_multiplier': st.session_state.atr_multiplier,
        'use_msl_msh': st.session_state.use_msl_msh,
        'msl_period': st.session_state.msl_period,
        'msh_period': st.session_state.msh_period,
        'msl_lookback': st.session_state.msl_lookback,
        'msh_lookback': st.session_state.msh_lookback,
        'use_macd': st.session_state.use_macd,
        'macd_fast': st.session_state.macd_fast,
        'macd_slow': st.session_state.macd_slow,
        'macd_signal_period': st.session_state.macd_signal_period,
        'use_adx': st.session_state.use_adx,
        'adx_period': st.session_state.adx_period,
        'adx_threshold': st.session_state.adx_threshold
    }


def _display_current_configuration(params):
    """Display current strategy configuration."""
    st.markdown("### 🎯 Current Strategy Configuration")
    
    # Debug info
    if st.session_state.get('user_applied_params', False):
        applied_rank = st.session_state.get('applied_rank', 'Unknown')
        st.caption(f"🔵 Using user-selected parameters (Rank #{applied_rank})")
    
    # Build summary
    strategy_summary = []
    
    if not params['use_ema']:
        strategy_summary.append("**EMA:** Disabled")
    elif params['use_double_ema']:
        strategy_summary.append(f"**EMA:** Double Crossover ({params['ema_fast']}/{params['ema_slow']})")
    else:
        strategy_summary.append(f"**EMA:** Single ({params['ema_period']})")
    
    if params['use_rsi']:
        strategy_summary.append(f"**RSI:** Enabled (>{params['rsi_threshold']})")
    else:
        strategy_summary.append("**RSI:** Disabled")
    
    if params['use_stop_loss']:
        strategy_summary.append(f"**Stop-Loss:** {params['stop_loss_pct']}%")
    else:
        strategy_summary.append("**Stop-Loss:** Disabled")
    
    if params['use_bb']:
        strategy_summary.append(f"**BB:** Enabled ({params['bb_period']},{params['bb_std_dev']})")
    else:
        strategy_summary.append("**BB:** Disabled")
    
    if params['use_atr']:
        strategy_summary.append(f"**ATR:** Enabled ({params['atr_period']},{params['atr_multiplier']}x)")
    else:
        strategy_summary.append("**ATR:** Disabled")
    
    if params['use_msl_msh']:
        strategy_summary.append(f"**MSL/MSH:** Enabled ({params['msl_period']},{params['msl_lookback']})")
    else:
        strategy_summary.append("**MSL/MSH:** Disabled")

    if params.get('use_macd', False):
        strategy_summary.append(f"**MACD:** Enabled ({params['macd_fast']},{params['macd_slow']},{params['macd_signal_period']})")
    else:
        strategy_summary.append("**MACD:** Disabled")

    if params.get('use_adx', False):
        strategy_summary.append(f"**ADX:** Enabled ({params['adx_period']},{params['adx_threshold']})")
    else:
        strategy_summary.append("**ADX:** Disabled")
    
    st.info(" | ".join(strategy_summary))


def _render_testing_tabs(test_tab1, test_tab2, test_tab3, params):
    """Render testing tabs and return button states."""
    
    run_button_daily = False
    run_button_custom = False
    run_button_monte = False
    test_params = {}
    
    # Daily Signal Tab
    with test_tab1:
        st.markdown("### 📊 Daily Signal")
        st.info("✅ Get today's buy/sell signal based on your strategy. Check at 3:55 PM ET before market close.")
        st.markdown("---")
        run_button_daily = st.button("🔔 Execute - Get Today's Signal", type="primary", use_container_width=True, key="daily_signal_btn")
    
    # Custom Simulation Tab
    with test_tab2:
        st.markdown("### 📈 Custom Simulation")
        st.info("Backtest your strategy on historical data with detailed performance analysis.")
        
        st.markdown("**Parameters:**")
        # Quick presets for common historical periods
        presets = [
            "Custom (manual)",
            "COVID Crash (Feb-Apr 2020)",
            "Whole of 2020",
            "Trade War (2018-2019)",
            "Ukraine War (2022- )",
            "Post-2011 (Long Term)"
        ]
        preset = st.selectbox("Quick Period Presets", options=presets, index=0, help="Choose a preset period or select Custom to set dates manually")

        # Determine default start/end based on preset
        today = datetime.date.today()
        if preset == "COVID Crash (Feb-Apr 2020)":
            default_start = datetime.date(2020, 2, 20)
            default_end = datetime.date(2020, 4, 30)
        elif preset == "Whole of 2020":
            default_start = datetime.date(2020, 1, 1)
            default_end = datetime.date(2020, 12, 31)
        elif preset == "Trade War (2018-2019)":
            default_start = datetime.date(2018, 1, 1)
            default_end = datetime.date(2019, 12, 31)
        elif preset == "Ukraine War (2022- )":
            default_start = datetime.date(2022, 2, 24)
            default_end = today
        elif preset == "Post-2011 (Long Term)":
            default_start = datetime.date(2011, 1, 1)
            default_end = today
        else:
            default_start = datetime.date(2020, 1, 1)
            default_end = today

        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("Start Date", value=default_start, min_value=datetime.date(2010, 3, 31), max_value=datetime.date.today())
        with col2:
            end_date = st.date_input("End Date", value=default_end, min_value=datetime.date(2010, 3, 31), max_value=datetime.date.today())
        with col3:
            initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
        
        st.markdown("---")
        run_button_custom = st.button("🚀 Execute - Run Custom Simulation", type="primary", use_container_width=True, key="custom_sim_btn")
        
        test_params['custom'] = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital
        }
    
    # Monte Carlo Tab
    with test_tab3:
        st.markdown("### 🎲 Monte Carlo Simulation")
        st.info("Run probabilistic simulations to see the range of possible future outcomes.")
        
        st.markdown("**Historical Data Period:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date_mc = st.date_input("Start", value=datetime.date(2020, 1, 1), min_value=datetime.date(2010, 3, 31), max_value=datetime.date.today(), key="mc_start")
        with col2:
            end_date_mc = st.date_input("End", value=datetime.date.today(), min_value=datetime.date(2010, 3, 31), max_value=datetime.date.today(), key="mc_end")
        with col3:
            initial_capital_mc = st.number_input("Capital ($)", min_value=1000, max_value=10000000, value=10000, step=1000, key="mc_capital")
        
        st.markdown("**Simulation Settings:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            num_simulations = st.number_input("Simulations", min_value=100, max_value=10000, value=1000, step=100)
        with col2:
            simulation_days = st.number_input("Days", min_value=30, max_value=1825, value=252, step=30)
            st.caption(f"= {simulation_days/252:.1f} year(s)")
        with col3:
            confidence_level = st.slider("Confidence %", min_value=80, max_value=99, value=95, step=1)
        
        st.markdown("---")
        run_button_monte = st.button("🎲 Execute - Run Monte Carlo Simulation", type="primary", use_container_width=True, key="monte_carlo_btn")
        
        test_params['monte'] = {
            'start_date': start_date_mc,
            'end_date': end_date_mc,
            'initial_capital': initial_capital_mc,
            'num_simulations': num_simulations,
            'simulation_days': simulation_days,
            'confidence_level': confidence_level
        }
    
    return run_button_daily, run_button_custom, run_button_monte, test_params



def _run_daily_signal(params):
    """Run daily signal analysis."""
    st.markdown("---")
    st.markdown("## 📊 Results")
    st.markdown("---")
    
    with st.spinner("Fetching latest market data..."):
        try:
            session = requests.Session(impersonate="chrome110", verify=False)
            buffer_days = max(365, params['ema_period'] + 100)
            fetch_start = (datetime.date.today() - datetime.timedelta(days=buffer_days)).strftime('%Y-%m-%d')
            
            historical_data = yf.download(
                ["QQQ", "TQQQ"],
                start=fetch_start,
                end=datetime.date.today(),
                session=session,
                auto_adjust=False,
                group_by='ticker',
                progress=False
            )
            
            qqq_ticker = yf.Ticker("QQQ", session=session)
            tqqq_ticker = yf.Ticker("TQQQ", session=session)
            
            qqq_current_info = qqq_ticker.info
            tqqq_current_info = tqqq_ticker.info
            
            current_qqq_price = qqq_current_info.get('currentPrice') or qqq_current_info.get('regularMarketPrice')
            current_tqqq_price = tqqq_current_info.get('currentPrice') or tqqq_current_info.get('regularMarketPrice')
            
            if current_qqq_price is None or current_tqqq_price is None:
                st.error("❌ Unable to fetch current prices. Market may be closed.")
                st.stop()
            
            qqq_data = historical_data["QQQ"].copy()
            
            today = pd.Timestamp.now().normalize()
            if today not in qqq_data.index:
                new_row_qqq = pd.DataFrame({
                    'Open': [current_qqq_price],
                    'High': [current_qqq_price],
                    'Low': [current_qqq_price],
                    'Close': [current_qqq_price],
                    'Volume': [0],
                    'Adj Close': [current_qqq_price]
                }, index=[today])
                qqq_data = pd.concat([qqq_data, new_row_qqq])
            else:
                qqq_data.loc[today, 'Close'] = current_qqq_price
            
            if params['use_double_ema']:
                qqq_data = calculate_double_ema(qqq_data, params['ema_fast'], params['ema_slow'])
            else:
                qqq_data = calculate_ema(qqq_data, params['ema_period'])
            
            qqq_data = calculate_rsi(qqq_data, period=14)
            
            if params['use_bb']:
                qqq_data = calculate_bollinger_bands(qqq_data, params['bb_period'], params['bb_std_dev'])

            if params.get('use_macd', False):
                qqq_data = calculate_macd(qqq_data, params['macd_fast'], params['macd_slow'], params['macd_signal_period'])

            if params.get('use_adx', False):
                qqq_data = calculate_adx(qqq_data, params['adx_period'])
            
            latest_date = qqq_data.index[-1]
            latest_qqq_close = qqq_data.iloc[-1]['Close']
            
            if params['use_double_ema']:
                latest_ema_fast = qqq_data.iloc[-1]['EMA_Fast']
                latest_ema_slow = qqq_data.iloc[-1]['EMA_Slow']
                latest_ema = latest_ema_slow
            else:
                latest_ema = qqq_data.iloc[-1]['EMA']
                latest_ema_fast = None
                latest_ema_slow = None
            
            latest_rsi = qqq_data.iloc[-1]['RSI']
            latest_tqqq_close = current_tqqq_price
            
            # Determine signal
            if params['use_double_ema']:
                base_signal = 'BUY' if latest_ema_fast > latest_ema_slow else 'SELL'
            else:
                base_signal = 'BUY' if latest_qqq_close > latest_ema else 'SELL'
            
            if params['use_rsi']:
                rsi_ok = pd.notna(latest_rsi) and latest_rsi > params['rsi_threshold']
                final_signal = 'BUY' if base_signal == 'BUY' and rsi_ok else 'SELL'
            else:
                final_signal = base_signal
            
            # Apply filters that can turn a BUY signal into a SELL
            if final_signal == 'BUY':
                # BB buy filter
                if params['use_bb']:
                    latest_bb_position = qqq_data.iloc[-1]['BB_Position']
                    if pd.notna(latest_bb_position) and latest_bb_position > params['bb_buy_threshold']:
                        final_signal = 'SELL'
                
                # MACD buy filter
                if params.get('use_macd', False) and final_signal == 'BUY':
                    macd_hist = qqq_data.iloc[-1].get('MACD_Hist', None)
                    if pd.notna(macd_hist) and macd_hist <= 0:
                        final_signal = 'SELL'

                # ADX buy filter
                if params.get('use_adx', False) and final_signal == 'BUY':
                    adx = qqq_data.iloc[-1].get('ADX', None)
                    plus_di = qqq_data.iloc[-1].get('+DI', None)
                    minus_di = qqq_data.iloc[-1].get('-DI', None)
                    if pd.notna(adx) and pd.notna(plus_di) and pd.notna(minus_di):
                        if adx < params['adx_threshold'] or plus_di < minus_di:
                            final_signal = 'SELL'
            
            # Standalone SELL conditions
            if params['use_bb']:
                latest_bb_position = qqq_data.iloc[-1]['BB_Position']
                if pd.notna(latest_bb_position) and latest_bb_position >= params['bb_sell_threshold']:
                    final_signal = 'SELL'
            
            st.markdown(f"### 📅 Signal for {latest_date.strftime('%Y-%m-%d')}")
            
            if final_signal == 'BUY':
                st.success("### 🟢 BUY TQQQ (or HOLD if already in)")
            else:
                st.error("### 🔴 SELL TQQQ / STAY CASH")
            
            market_state = qqq_current_info.get('marketState', 'UNKNOWN')
            if market_state == 'REGULAR':
                st.success("🟢 Market: OPEN")
            elif market_state == 'CLOSED':
                st.error("🔴 Market: CLOSED")
            else:
                st.warning(f"🟡 Market: {market_state}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("QQQ Price", f"${latest_qqq_close:.2f}")
                st.metric("TQQQ Price", f"${latest_tqqq_close:.2f}")
            
            with col2:
                if params['use_double_ema']:
                    st.metric(f"Fast EMA ({params['ema_fast']}d)", f"${latest_ema_fast:.2f}")
                    st.metric(f"Slow EMA ({params['ema_slow']}d)", f"${latest_ema_slow:.2f}")
                else:
                    st.metric(f"{params['ema_period']}-day EMA", f"${latest_ema:.2f}")
            
            with col3:
                if params['use_rsi']:
                    st.metric("RSI (14-day)", f"{latest_rsi:.1f}")
                else:
                    st.info("RSI: Disabled")
                
                if params['use_bb']:
                    latest_bb_position = qqq_data.iloc[-1]['BB_Position']
                    if pd.notna(latest_bb_position):
                        st.metric("BB Position", f"{latest_bb_position:.2f}", help="0.0=lower band, 0.5=middle, 1.0=upper band")
                    else:
                        st.info("BB: Calculating...")
            
            if latest_date.date() != datetime.date.today():
                st.warning("⚠️ Using stale data. Try again during market hours.")
            
            # Save results to session state for AI summary
            st.session_state.daily_signal_results = {
                'params': params,
                'date': latest_date,
                'qqq_price': latest_qqq_close,
                'tqqq_price': latest_tqqq_close,
                'ema': latest_ema,
                'rsi': latest_rsi,
                'signal': final_signal,
                'market_state': market_state,
                'ema_fast': latest_ema_fast,
                'ema_slow': latest_ema_slow,
                'bb_position': latest_bb_position if params['use_bb'] else None
            }
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    st.info("💡 **Tip:** Go to Step 5 to generate a comprehensive AI summary report of this signal")
    st.markdown("---")
    
    st.caption("⚠️ **Disclaimer:** Educational purposes only. Past performance does not guarantee future results. Trading leveraged ETFs involves significant risk.")



def _run_custom_simulation(params, test_params):
    """Run custom simulation backtest."""
    
    custom_params = test_params['custom']
    start_date = custom_params['start_date']
    end_date = custom_params['end_date']
    initial_capital = custom_params['initial_capital']
    
    st.markdown("---")
    st.markdown("## 📊 Results")
    st.markdown("---")
    
    with st.spinner("Running custom simulation..."):
        tickers = ["QQQ", "TQQQ"]
        raw_data = get_data(tickers, start_date, end_date, buffer_days=max(365, params['ema_period'] + 100))
        
        qqq = raw_data["QQQ"].copy()
        tqqq = raw_data["TQQQ"].copy()
        
        result = run_tqqq_only_strategy(
            qqq.copy(), tqqq.copy(), start_date, end_date, initial_capital,
            params['ema_period'], params['rsi_threshold'], params['use_rsi'],
            params['rsi_oversold'], params['rsi_overbought'],
            params['stop_loss_pct'], params['use_stop_loss'],
            params['use_double_ema'], params['ema_fast'], params['ema_slow'],
            params['use_bb'], params['bb_period'], params['bb_std_dev'],
            params['bb_buy_threshold'], params['bb_sell_threshold'],
            params['use_atr'], params['atr_period'], params['atr_multiplier'],
            params['use_msl_msh'], params['msl_period'], params['msh_period'],
            params['msl_lookback'], params['msh_lookback'],
            params['use_ema'],
            params.get('use_macd', False), params.get('macd_fast', 12),
            params.get('macd_slow', 26), params.get('macd_signal_period', 9),
            params.get('use_adx', False), params.get('adx_period', 14),
            params.get('adx_threshold', 25), params.get('use_supertrend', False), params.get('st_period', 10), params.get('st_multiplier', 3.0)
        )
        
        # Calculate QQQ benchmark
        qqq_start = qqq.loc[start_date:end_date].iloc[0]['Close']
        qqq_end = qqq.loc[start_date:end_date].iloc[-1]['Close']
        qqq_bh_value = (qqq_end / qqq_start) * initial_capital
        qqq_bh_return = ((qqq_bh_value - initial_capital) / initial_capital) * 100
    
    st.success("✅ Custom simulation complete!")
    
    # Performance Summary
    with st.expander("📊 Performance Summary", expanded=True):
        days = (end_date - start_date).days
        years = days / 365.25
        cagr_strategy = ((result['final_value'] / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        cagr_qqq = ((qqq_bh_value / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        st.info(f"💡 **Realistic Execution Modeling:** This backtest simulates intraday execution (3:50 PM analyze → 3:55 PM execute). Slippage ({result['slippage_buy_pct']:.2f}% buy, {result['slippage_sell_pct']:.2f}% sell) covers bid-ask spread, market impact, and 5-minute price movement.")
        
        st.markdown("**Strategy Performance:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Final Value", f"${result['final_value']:,.2f}", f"{result['total_return_pct']:.2f}%")
        with col2:
            st.metric("CAGR", f"{cagr_strategy:.2f}%")
        with col3:
            st.metric("Max Drawdown", f"{result['max_drawdown']:.2f}%")
        with col4:
            st.metric("Total Trades", result['num_trades'])
        with col5:
            st.metric("Execution Costs", f"{result['estimated_total_costs_pct']:.2f}%")
        
        st.markdown("**QQQ Buy & Hold Comparison:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("QQQ Final Value", f"${qqq_bh_value:,.2f}", f"{qqq_bh_return:.2f}%")
        with col2:
            st.metric("QQQ CAGR", f"{cagr_qqq:.2f}%")
        with col3:
            outperformance = result['total_return_pct'] - qqq_bh_return
            st.metric("Outperformance", f"{outperformance:+.2f}%", "✅ Winning" if outperformance > 0 else "❌ Losing")
        with col4:
            value_diff = result['final_value'] - qqq_bh_value
            st.metric("Value Difference", f"${value_diff:+,.2f}")
    
    # Performance Chart
    with st.expander("📉 Performance Chart", expanded=True):
        fig = create_performance_chart(result, qqq, tqqq, start_date, initial_capital, params['ema_period'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade Log
    with st.expander("📋 Trade Log", expanded=False):
        trade_df = pd.DataFrame(result['trade_log'])
        show_all = st.checkbox("Show all days", value=False)
        
        if show_all:
            display_df = trade_df
        else:
            display_df = trade_df[trade_df['Action'].str.contains('BUY TQQQ|SELL', case=False, na=False)]
        
        st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)
        
        st.download_button(
            "📥 Download Trade Log CSV",
            trade_df.to_csv(index=False),
            "trade_log.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Save results to session state for AI summary
    st.session_state.custom_sim_results = {
        'params': params,
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'result': result,
        'qqq_bh_value': qqq_bh_value,
        'qqq_bh_return': qqq_bh_return
    }
    
    st.markdown("---")
    st.info("💡 **Tip:** Go to Step 5 to generate a comprehensive AI summary report of this backtest")
    st.markdown("---")
    
    st.caption("⚠️ **Disclaimer:** Educational purposes only. Past performance does not guarantee future results. Trading leveraged ETFs involves significant risk.")



def _run_monte_carlo(params, test_params):
    """Run Monte Carlo simulation."""
    
    monte_params = test_params['monte']
    start_date = monte_params['start_date']
    end_date = monte_params['end_date']
    initial_capital = monte_params['initial_capital']
    num_simulations = monte_params['num_simulations']
    simulation_days = monte_params['simulation_days']
    confidence_level = monte_params['confidence_level']
    
    st.markdown("---")
    st.markdown("## 📊 Results")
    st.markdown("---")
    
    with st.spinner(f"Running {num_simulations:,} simulations..."):
        tickers = ["QQQ", "TQQQ"]
        raw_data = get_data(tickers, start_date, end_date, buffer_days=max(365, params['ema_period'] + 100))
        
        qqq = raw_data["QQQ"].copy()
        tqqq = raw_data["TQQQ"].copy()
        
        # Run strategy to get historical returns
        result = run_tqqq_only_strategy(
            qqq.copy(), tqqq.copy(), start_date, end_date,
            initial_capital, params['ema_period'], params['rsi_threshold'],
            params['use_rsi'], params['rsi_oversold'], params['rsi_overbought'],
            params['stop_loss_pct'], params['use_stop_loss'],
            params['use_double_ema'], params['ema_fast'], params['ema_slow'],
            params['use_bb'], params['bb_period'], params['bb_std_dev'],
            params['bb_buy_threshold'], params['bb_sell_threshold'],
            params['use_atr'], params['atr_period'], params['atr_multiplier'],
            params['use_msl_msh'], params['msl_period'], params['msh_period'],
            params['msl_lookback'], params['msh_lookback'],
            params['use_ema'],
            params.get('use_macd', False), params.get('macd_fast', 12),
            params.get('macd_slow', 26), params.get('macd_signal_period', 9),
            params.get('use_adx', False), params.get('adx_period', 14),
            params.get('adx_threshold', 25), params.get('use_supertrend', False), params.get('st_period', 10), params.get('st_multiplier', 3.0)
        )
        
        portfolio_df = result['portfolio_df'].copy()
        portfolio_df['Daily_Return'] = portfolio_df['Value'].pct_change()
        daily_returns = portfolio_df['Daily_Return'].dropna().values
        
        if len(daily_returns) < 30:
            st.error("Not enough historical data. Please select a longer period.")
            st.stop()
        
        # Calculate QQQ returns
        qqq_benchmark = qqq.loc[start_date:end_date]['Close'].copy()
        qqq_returns = qqq_benchmark.pct_change().dropna().values
        
        # Run Monte Carlo for strategy
        np.random.seed(42)
        simulations = np.zeros((num_simulations, simulation_days + 1))
        simulations[:, 0] = initial_capital
        
        for sim in range(num_simulations):
            for day in range(1, simulation_days + 1):
                random_return = np.random.choice(daily_returns)
                simulations[sim, day] = simulations[sim, day - 1] * (1 + random_return)
        
        # Run Monte Carlo for QQQ
        np.random.seed(42)
        qqq_simulations = np.zeros((num_simulations, simulation_days + 1))
        qqq_simulations[:, 0] = initial_capital
        
        for sim in range(num_simulations):
            for day in range(1, simulation_days + 1):
                random_return = np.random.choice(qqq_returns)
                qqq_simulations[sim, day] = qqq_simulations[sim, day - 1] * (1 + random_return)
        
        # Calculate statistics
        final_values = simulations[:, -1]
        mean_final_value = np.mean(final_values)
        median_final_value = np.median(final_values)
        
        lower_percentile = (100 - confidence_level) / 2
        upper_percentile = 100 - lower_percentile
        
        ci_lower = np.percentile(final_values, lower_percentile)
        ci_upper = np.percentile(final_values, upper_percentile)
        
        prob_profit = (final_values > initial_capital).sum() / num_simulations * 100
        
        qqq_final_values = qqq_simulations[:, -1]
        qqq_mean_final_value = np.mean(qqq_final_values)
        qqq_median_final_value = np.median(qqq_final_values)
        
        outperformance = final_values - qqq_final_values
        prob_outperform = (outperformance > 0).sum() / num_simulations * 100
        mean_outperformance = np.mean(outperformance)
        median_outperformance = np.median(outperformance)
    
    st.success(f"✅ Completed {num_simulations:,} simulations!")
    
    # Strategy Results
    st.subheader("📈 Strategy Monte Carlo Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Final Value", f"${mean_final_value:,.2f}")
        mean_return = (mean_final_value - initial_capital) / initial_capital * 100
        st.metric("Mean Return", f"{mean_return:.2f}%")
    
    with col2:
        st.metric("Median Final Value", f"${median_final_value:,.2f}")
        median_return = (median_final_value - initial_capital) / initial_capital * 100
        st.metric("Median Return", f"{median_return:.2f}%")
    
    with col3:
        st.metric(f"{confidence_level}% CI Lower", f"${ci_lower:,.2f}")
        ci_lower_return = (ci_lower - initial_capital) / initial_capital * 100
        st.metric("Lower Return", f"{ci_lower_return:.2f}%")
    
    with col4:
        st.metric(f"{confidence_level}% CI Upper", f"${ci_upper:,.2f}")
        ci_upper_return = (ci_upper - initial_capital) / initial_capital * 100
        st.metric("Upper Return", f"{ci_upper_return:.2f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probability of Profit", f"{prob_profit:.1f}%")
    with col2:
        std_dev = np.std(final_values)
        st.metric("Std Deviation", f"${std_dev:,.2f}")
    
    # Distribution Chart
    st.markdown("### 📊 Distribution of Final Portfolio Values")
    
    fig = go.Figure()
    
    # Strategy distribution
    fig.add_trace(go.Histogram(
        x=final_values,
        name='Strategy',
        nbinsx=50,
        opacity=0.7,
        marker_color='blue'
    ))
    
    # QQQ distribution
    fig.add_trace(go.Histogram(
        x=qqq_final_values,
        name='QQQ',
        nbinsx=50,
        opacity=0.7,
        marker_color='green'
    ))
    
    # Add vertical lines for mean values
    fig.add_vline(x=mean_final_value, line_dash="dash", line_color="blue", 
                  annotation_text=f"Strategy Mean: ${mean_final_value:,.0f}")
    fig.add_vline(x=qqq_mean_final_value, line_dash="dash", line_color="green",
                  annotation_text=f"QQQ Mean: ${qqq_mean_final_value:,.0f}")
    fig.add_vline(x=initial_capital, line_dash="dot", line_color="red",
                  annotation_text=f"Initial: ${initial_capital:,.0f}")
    
    fig.update_layout(
        title=f"Distribution of Final Values ({num_simulations:,} simulations over {simulation_days} days)",
        xaxis_title="Final Portfolio Value ($)",
        yaxis_title="Frequency",
        barmode='overlay',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample Simulation Paths
    st.markdown("### 📈 Sample Simulation Paths")
    
    num_paths_to_show = min(100, num_simulations)
    
    fig = go.Figure()
    
    # Plot sample strategy paths
    for i in range(num_paths_to_show):
        fig.add_trace(go.Scatter(
            x=list(range(simulation_days + 1)),
            y=simulations[i],
            mode='lines',
            line=dict(color='blue', width=0.5),
            opacity=0.3,
            showlegend=False,
            hovertemplate='Day %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))
    
    # Plot mean path
    mean_path = np.mean(simulations, axis=0)
    fig.add_trace(go.Scatter(
        x=list(range(simulation_days + 1)),
        y=mean_path,
        mode='lines',
        name='Mean Path',
        line=dict(color='darkblue', width=3),
        hovertemplate='Day %{x}<br>Mean Value: $%{y:,.2f}<extra></extra>'
    ))
    
    # Plot QQQ mean path
    qqq_mean_path = np.mean(qqq_simulations, axis=0)
    fig.add_trace(go.Scatter(
        x=list(range(simulation_days + 1)),
        y=qqq_mean_path,
        mode='lines',
        name='QQQ Mean Path',
        line=dict(color='green', width=3, dash='dash'),
        hovertemplate='Day %{x}<br>QQQ Mean: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add initial capital line
    fig.add_hline(y=initial_capital, line_dash="dot", line_color="red",
                  annotation_text=f"Initial Capital: ${initial_capital:,.0f}")
    
    fig.update_layout(
        title=f"Sample Simulation Paths (showing {num_paths_to_show} of {num_simulations:,})",
        xaxis_title="Days",
        yaxis_title="Portfolio Value ($)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison Summary
    st.markdown("---")
    st.subheader("📊 Strategy vs QQQ Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Probability of Outperforming QQQ", f"{prob_outperform:.1f}%")
    with col2:
        st.metric("Mean Outperformance", f"${mean_outperformance:+,.2f}")
    with col3:
        st.metric("Median Outperformance", f"${median_outperformance:+,.2f}")
    with col4:
        mean_return_diff = ((mean_final_value - initial_capital) / initial_capital * 100) - ((qqq_mean_final_value - initial_capital) / initial_capital * 100)
        st.metric("Mean Return Difference", f"{mean_return_diff:+.2f}%")
    
    if prob_outperform > 50:
        st.success(f"✅ Your strategy has a {prob_outperform:.1f}% chance of beating QQQ Buy & Hold!")
    else:
        st.warning(f"⚠️ Your strategy has only a {prob_outperform:.1f}% chance of beating QQQ Buy & Hold.")
    
    # Risk Metrics
    st.markdown("---")
    st.subheader("⚠️ Risk Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_loss = np.min(final_values) - initial_capital
        max_loss_pct = (max_loss / initial_capital) * 100
        st.metric("Worst Case Scenario", f"${max_loss:+,.2f}", f"{max_loss_pct:+.2f}%")
    
    with col2:
        max_gain = np.max(final_values) - initial_capital
        max_gain_pct = (max_gain / initial_capital) * 100
        st.metric("Best Case Scenario", f"${max_gain:+,.2f}", f"{max_gain_pct:+.2f}%")
    
    with col3:
        value_at_risk = initial_capital - ci_lower
        var_pct = (value_at_risk / initial_capital) * 100
        st.metric(f"Value at Risk ({confidence_level}% CI)", f"${value_at_risk:,.2f}", f"{var_pct:.2f}%")
    
    # Save results to session state for AI summary
    st.session_state.monte_carlo_results = {
        'params': params,
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'num_simulations': num_simulations,
        'simulation_days': simulation_days,
        'confidence_level': confidence_level,
        'mean_final_value': mean_final_value,
        'median_final_value': median_final_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'prob_profit': prob_profit,
        'qqq_mean_final_value': qqq_mean_final_value,
        'prob_outperform': prob_outperform,
        'mean_outperformance': mean_outperformance,
        'final_values': final_values
    }
    
    st.markdown("---")
    st.info("💡 **Tip:** Go to Step 5 to generate a comprehensive AI summary report of this simulation")
    st.markdown("---")
    
    st.caption("⚠️ **Disclaimer:** Educational purposes only. Past performance does not guarantee future results. Trading leveraged ETFs involves significant risk.")
