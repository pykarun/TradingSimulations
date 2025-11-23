"""Step 2: Grid Search - Find optimal strategy parameters.

This module contains the grid search interface where users can:
- Select time periods to test
- Configure EMA, RSI, Stop-Loss, BB, ATR, and MSL/MSH parameters
- Run comprehensive parameter optimization
- View and apply top-performing strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from collections import defaultdict
import plotly.graph_objects as go
import itertools

from core import get_data, run_tqqq_only_strategy

try:
    rerun = st.rerun
except AttributeError:
    rerun = st.experimental_rerun


def render_step2():
    """Render Step 2: Grid Search interface."""
    
    # Initialize session state for grid search
    if 'grid_search_results' not in st.session_state:
        st.session_state.grid_search_results = None
    if 'best_params' not in st.session_state:
        st.session_state.best_params = None
    if 'show_grid_search' not in st.session_state:
        st.session_state.show_grid_search = True

    st.markdown("### 🎯 Find the Best Trading Signals")
    st.info("""
    **How it works:** Enable the features you want to test below. Our **Grid Search Algorithm** will:
    - Test thousands of parameter combinations automatically
    - Evaluate each combination across your selected time periods
    - Rank strategies by performance vs QQQ benchmark
    - Find the optimal settings for your risk tolerance
    
    Simply enable the features you want, and let the algorithm do the heavy lifting!
    """)
    
    # Time Period Selection
    _render_time_period_selection()
    
    # Get selected periods
    selected_periods = st.session_state.get('selected_periods', [])
    test_multiple_periods = True
    
    # Feature Configuration
    st.markdown("---")
    st.markdown("### ⚙️ Configure Strategy Features")
    st.caption("Enable and configure the trading signals you want to test. The sections will stack on mobile devices.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        enable_ema, ema_config = _render_ema_section()
    with col2:
        enable_rsi, rsi_config = _render_rsi_section()
    with col3:
        enable_sl, sl_config = _render_stop_loss_section()

    col1, col2, col3 = st.columns(3)
    with col1:
        enable_bb, bb_config = _render_bollinger_bands_section()
    with col2:
        enable_atr, atr_config = _render_atr_section()
    with col3:
        enable_msl, msl_config = _render_msl_section()

    col1, col2, col3 = st.columns(3)
    with col1:
        enable_macd, macd_config = _render_macd_section()
    with col2:
        enable_adx, adx_config = _render_adx_section()
    # Supertrend section
    with st.container(border=True):
        enable_supertrend, st_config = _render_supertrend_section()
    
    # Initial Capital
    st.markdown("---")
    st.markdown("### 💰 Initial Capital")
    grid_capital = st.number_input(
        "Initial Capital for Testing",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Starting portfolio value for backtesting"
    )
    
    # Run Grid Search Button
    st.markdown("---")
    run_grid_search = st.button("🚀 Run Grid Search", type="primary", use_container_width=True)
    
    if run_grid_search:
        _execute_grid_search(
            selected_periods, test_multiple_periods, grid_capital,
            enable_ema, ema_config, enable_rsi, rsi_config,
            enable_sl, sl_config, enable_bb, bb_config,
            enable_atr, atr_config, enable_msl, msl_config,
            enable_macd, macd_config, enable_adx, adx_config
            , enable_supertrend, st_config
        )
    
    # Display Results
    if st.session_state.grid_search_results is not None:
        _display_grid_search_results(selected_periods, test_multiple_periods)


def _render_time_period_selection():
    """Render time period selection UI."""
    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("**📅 Testing duration**")
    with col2:
        st.markdown("**Time Period Selection** - Choose which historical periods to test")
    
    st.caption("Testing across multiple time periods ensures robust strategy performance")
    
    time_periods = {
        "last 3 Months": 90,
        "last 6 Months": 180,
        "last 9 Months": 270,
        "last 1 Year": 365,
        "last 2 Years": 730,
        "last 3 Years": 1095,
        "last 4 Years": 1460,
        "last 5 Years": 1825
    }
    
    selected_periods = st.multiselect(
        "Select Time Periods to Test",
        options=list(time_periods.keys()),
        default=["last 3 Months", "last 6 Months", "last 9 Months", "last 1 Year"],
        help="Testing across multiple periods finds strategies that work in different market conditions"
    )
    
    # Store in session state
    st.session_state.selected_periods = selected_periods
    st.session_state.time_periods = time_periods
    
    if selected_periods:
        period_summary = ", ".join(selected_periods)
    else:
        st.warning("⚠️ Please select at least one time period")


def _render_ema_section():
    """Render EMA configuration section."""
    with st.container(border=True):
        st.markdown("**📈 EMA Strategy** (Primary)")
        enable_ema = st.checkbox("Enable EMA", value=True, key="enable_ema", help="Include EMA strategy in grid search")

        if not enable_ema:
            st.caption("⚠️ EMA is disabled.")
            return False, {}

        ema_strategy_options = st.multiselect(
            "EMA Strategies",
            options=["Single EMA", "Double EMA Crossover"],
            default=["Single EMA", "Double EMA Crossover"],
            help="Single EMA: Price vs EMA | Double EMA: Fast/Slow crossover"
        )

        if "Single EMA" in ema_strategy_options:
            ema_range = st.multiselect(
                "Single EMA Periods",
                options=[10, 20, 21, 30, 40, 50, 60, 80, 100],
                default=[21, 30, 50],
                help="Common: 21, 50, 80"
            )
        else:
            ema_range = []

        if "Double EMA Crossover" in ema_strategy_options:
            fast_ema_range = st.multiselect(
                "Fast EMA",
                options=[5, 8, 9, 10, 12, 15, 20, 21],
                default=[9, 12, 21],
                help="Faster response"
            )
            slow_ema_range = st.multiselect(
                "Slow EMA",
                options=[15, 20, 21, 25, 30, 40, 50],
                default=[21, 30, 50],
                help="Smoother trend"
            )
        else:
            fast_ema_range = []
            slow_ema_range = []

        return enable_ema, {
            'strategy_options': ema_strategy_options,
            'ema_range': ema_range,
            'fast_ema_range': fast_ema_range,
            'slow_ema_range': slow_ema_range
        }


def _render_rsi_section():
    """Render RSI configuration section."""
    with st.container(border=True):
        st.markdown("**🎯 RSI Filter** (Optional)")
        enable_rsi = st.checkbox("Enable RSI", value=True, key="enable_rsi", help="Include RSI filter in grid search")

        if not enable_rsi:
            st.caption("⚠️ RSI is disabled.")
            return False, {}

        rsi_range = st.multiselect(
            "Momentum (0=off)",
            options=[0, 40, 45, 50, 55, 60],
            default=[0, 50],
            help="Buy when RSI > threshold"
        )

        rsi_oversold_range = st.multiselect(
            "Oversold (Buy)",
            options=[20, 25, 30, 35, 40],
            default=[30],
            help="Buy signal level"
        )

        rsi_overbought_range = st.multiselect(
            "Overbought (Sell)",
            options=[60, 65, 70, 75, 80],
            default=[70],
            help="Sell signal level"
        )

        return enable_rsi, {
            'rsi_range': rsi_range,
            'rsi_oversold_range': rsi_oversold_range,
            'rsi_overbought_range': rsi_overbought_range
        }


def _render_stop_loss_section():
    """Render Stop-Loss configuration section."""
    with st.container(border=True):
        st.markdown("**🛡️ Stop-Loss** (Risk Mgmt)")
        enable_sl = st.checkbox("Enable Stop-Loss", value=True, key="enable_sl", help="Include stop-loss in grid search")

        if not enable_sl:
            st.caption("⚠️ Stop-Loss is disabled.")
            return False, {}

        stop_loss_range = st.multiselect(
            "Stop-Loss % (0=disabled)",
            options=[0, 5, 8, 10, 12, 15, 20],
            default=[0, 10, 15],
            help="Exit if portfolio drops X% from peak"
        )

        return enable_sl, {'stop_loss_range': stop_loss_range}


def _render_bollinger_bands_section():
    """Render Bollinger Bands configuration section."""
    with st.container(border=True):
        st.markdown("**📊 Bollinger Bands** (Optional)")
        enable_bb = st.checkbox("Enable BBands", value=False, key="enable_bb", help="Include Bollinger Bands in grid search")

        if not enable_bb:
            st.caption("⚠️ BBands are disabled.")
            return False, {}

        st.caption("Tests with/without BB filter")

        bb_period_range = st.multiselect(
            "BB Period",
            options=[10, 15, 20, 25, 30],
            default=[20],
            help="Moving average period (20 is standard)"
        )
        bb_std_dev_range = st.multiselect(
            "BB Std Dev",
            options=[1.5, 2.0, 2.5],
            default=[2.0],
            help="Band width (2.0 is standard)"
        )
        bb_buy_threshold_range = st.multiselect(
            "Buy Threshold",
            options=[0.0, 0.1, 0.2, 0.3],
            default=[0.2],
            help="Buy when price in lower X%"
        )
        bb_sell_threshold_range = st.multiselect(
            "Sell Threshold",
            options=[0.7, 0.8, 0.9, 1.0],
            default=[0.8],
            help="Sell when price in upper X%"
        )

        return enable_bb, {
            'bb_period_range': bb_period_range,
            'bb_std_dev_range': bb_std_dev_range,
            'bb_buy_threshold_range': bb_buy_threshold_range,
            'bb_sell_threshold_range': bb_sell_threshold_range
        }


def _render_atr_section():
    """Render ATR configuration section."""
    with st.container(border=True):
        st.markdown("**🎢 ATR Stop-Loss** (Optional)")
        enable_atr = st.checkbox("Enable ATR", value=False, key="enable_atr", help="Include ATR stop-loss in grid search")

        if not enable_atr:
            st.caption("⚠️ ATR Stop-Loss is disabled.")
            return False, {}

        st.caption("Tests with/without ATR stop")

        atr_period_range = st.multiselect(
            "ATR Period",
            options=[7, 10, 14, 20, 30],
            default=[14],
            help="Period for ATR (14 is standard)"
        )
        atr_multiplier_range = st.multiselect(
            "ATR Multiplier",
            options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            default=[2.0, 2.5],
            help="1.5-2.0 Day Trading, 2.0-3.0 Swing, 3.0-4.0 Position"
        )

        return enable_atr, {
            'atr_period_range': atr_period_range,
            'atr_multiplier_range': atr_multiplier_range
        }


def _render_msl_section():
    """Render MSL/MSH configuration section."""
    with st.container(border=True):
        st.markdown("**📉 MSL/MSH Stop-Loss** (Optional)")
        enable_msl = st.checkbox("Enable MSL/MSH", value=False, key="enable_msl", help="Include MSL/MSH stop-loss in grid search")

        if not enable_msl:
            st.caption("⚠️ MSL/MSH is disabled.")
            return False, {}

        st.caption("Tests with/without MSL/MSH stop")

        msl_period_range = st.multiselect(
            "MSL Smoothing",
            options=[10, 15, 20, 25, 30],
            default=[20],
            help="Smoothing period for moving stop-loss"
        )
        msl_lookback_range = st.multiselect(
            "MSL Lookback",
            options=[3, 5, 7, 10, 15],
            default=[5],
            help="Lookback for stop-loss levels"
        )

        return enable_msl, {
            'msl_period_range': msl_period_range,
            'msl_lookback_range': msl_lookback_range
        }


def _render_macd_section():
    """Render MACD configuration section."""
    with st.container(border=True):
        st.markdown("**📈 MACD Filter** (Optional)")
        enable_macd = st.checkbox("Enable MACD", value=False, key="enable_macd", help="Include MACD filter in grid search")

        if not enable_macd:
            st.caption("⚠️ MACD is disabled.")
            return False, {}

        st.caption("Tests with/without MACD filter. Buy if MACD > Signal.")

        macd_fast_range = st.multiselect(
            "MACD Fast",
            options=[9, 10, 12, 15],
            default=[12],
            help="Fast EMA (12 is standard)"
        )
        macd_slow_range = st.multiselect(
            "MACD Slow",
            options=[20, 21, 25, 26],
            default=[26],
            help="Slow EMA (26 is standard)"
        )
        macd_signal_range = st.multiselect(
            "MACD Signal",
            options=[7, 8, 9, 10],
            default=[9],
            help="Signal line EMA (9 is standard)"
        )

        return enable_macd, {
            'macd_fast_range': macd_fast_range,
            'macd_slow_range': macd_slow_range,
            'macd_signal_range': macd_signal_range,
        }

def _render_adx_section():
    """Render ADX configuration section."""
    with st.container(border=True):
        st.markdown("**💪 ADX Filter** (Optional)")
        enable_adx = st.checkbox("Enable ADX", value=False, key="enable_adx", help="Include ADX filter in grid search")

        if not enable_adx:
            st.caption("⚠️ ADX is disabled.")
            return False, {}

        st.caption("Tests with/without ADX filter. Buy if ADX > threshold & +DI > -DI.")

        adx_period_range = st.multiselect(
            "ADX Period",
            options=[10, 12, 14, 20],
            default=[14],
            help="ADX period (14 is standard)"
        )
        adx_threshold_range = st.multiselect(
            "ADX Threshold",
            options=[20, 25, 30],
            default=[25],
            help="Min ADX for trend (25 is standard)"
        )

        return enable_adx, {
            'adx_period_range': adx_period_range,
            'adx_threshold_range': adx_threshold_range,
        }


def _render_supertrend_section():
    """Render Supertrend configuration section."""
    with st.container(border=True):
        st.markdown("**⚡ Supertrend Filter** (Optional)")
        enable_supertrend = st.checkbox("Enable Supertrend", value=False, key="enable_supertrend", help="Include Supertrend filter in grid search")

        if not enable_supertrend:
            st.caption("⚠️ Supertrend is disabled.")
            return False, {}

        st.caption("Use Supertrend as a directional buy filter (requires price above Supertrend to buy)")

        st_period_range = st.multiselect(
            "Supertrend ATR Period",
            options=[7, 10, 14, 21],
            default=[10],
            help="ATR period used by Supertrend"
        )

        st_multiplier_range = st.multiselect(
            "Supertrend Multiplier",
            options=[1.5, 2.0, 2.5, 3.0],
            default=[3.0],
            help="ATR multiplier used by Supertrend"
        )

        return enable_supertrend, {
            'st_period_range': st_period_range,
            'st_multiplier_range': st_multiplier_range
        }


def _execute_grid_search(
    selected_periods, test_multiple_periods, grid_capital,
    enable_ema, ema_config, enable_rsi, rsi_config,
    enable_sl, sl_config, enable_bb, bb_config,
    enable_atr, atr_config, enable_msl, msl_config,
    enable_macd, macd_config, enable_adx, adx_config
    , enable_supertrend=None, st_config=None
):
    """Execute the grid search with given parameters."""
    
    # Validate inputs
    if test_multiple_periods and not selected_periods:
        st.error("Please select at least one time period to test")
        st.stop()
    
    if not enable_ema or not ema_config.get('strategy_options'):
        st.error("Please select at least one EMA strategy to test")
        st.stop()
    
    # Generate parameter combinations
    param_combinations = _generate_param_combinations(
        enable_ema, ema_config, enable_rsi, rsi_config,
        enable_sl, sl_config, enable_bb, bb_config,
        enable_atr, atr_config, enable_msl, msl_config,
        enable_macd, macd_config, enable_adx, adx_config
        , enable_supertrend, st_config
    )
    
    # Determine periods to test
    time_periods = st.session_state.time_periods
    periods_to_test = [(period, time_periods[period]) for period in selected_periods]
    
    total_combinations = len(param_combinations) * len(periods_to_test)
    
    st.info(f"Testing {len(param_combinations)} parameter combinations across {len(periods_to_test)} time period(s) = {total_combinations} total tests...")
    
    # Download data
    with st.spinner("Downloading historical data..."):
        tickers = ["QQQ", "TQQQ"]
        max_ema = max([p['ema_period'] for p in param_combinations])
        max_days = max([days for _, days in periods_to_test])
        
        grid_end_date = datetime.date.today()
        grid_start_date = grid_end_date - datetime.timedelta(days=max_days)
        
        raw_data = get_data(tickers, grid_start_date, grid_end_date, buffer_days=max(365, max_ema + 100))
        
        qqq = raw_data["QQQ"].copy()
        tqqq = raw_data["TQQQ"].copy()
    
    # Run grid search
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    test_counter = 0
    
    for period_name, days_back in periods_to_test:
        period_end_date = datetime.date.today()
        period_start_date = period_end_date - datetime.timedelta(days=days_back)
        
        # Calculate QQQ benchmark
        qqq_period = qqq.loc[period_start_date:period_end_date]
        if len(qqq_period) == 0:
            st.warning(f"No data available for period: {period_name}")
            continue
        
        qqq_start = qqq_period.iloc[0]['Close']
        qqq_end = qqq_period.iloc[-1]['Close']
        qqq_bh_value = (qqq_end / qqq_start) * grid_capital
        qqq_bh_return = ((qqq_bh_value - grid_capital) / grid_capital) * 100
        
        for params in param_combinations:
            test_counter += 1
            status_text.text(f"Testing {test_counter}/{total_combinations}: {period_name} - Combination {(test_counter-1) % len(param_combinations) + 1}/{len(param_combinations)}...")
            
            try:
                result = run_tqqq_only_strategy(
                    qqq.copy(), tqqq.copy(),
                    period_start_date, period_end_date,
                    grid_capital,
                    params['ema_period'],
                    params['rsi_threshold'],
                    params['use_rsi'],
                    params['rsi_oversold'],
                    params['rsi_overbought'],
                    params['stop_loss_pct'],
                    params['use_stop_loss'],
                    params['use_double_ema'],
                    params['ema_fast'],
                    params['ema_slow'],
                    params['use_bb'],
                    params['bb_period'],
                    params['bb_std_dev'],
                    params['bb_buy_threshold'],
                    params['bb_sell_threshold'],
                    params['use_atr'],
                    params['atr_period'],
                    params['atr_multiplier'],
                    params['use_msl_msh'],
                    params['msl_period'],
                    params['msh_period'],
                    params['msl_lookback'],
                    params['msh_lookback'],
                    params.get('use_ema', True),
                    params['use_macd'],
                    params['macd_fast'],
                    params['macd_slow'],
                    params['macd_signal_period'],
                    params['use_adx'],
                    params['adx_period'],
                    params['adx_threshold']
                    , params.get('use_supertrend', False), params.get('st_period', 10), params.get('st_multiplier', 3.0)
                )
                
                # Calculate metrics
                days = (period_end_date - period_start_date).days
                years = days / 365.25
                cagr = ((result['final_value'] / grid_capital) ** (1/years) - 1) * 100 if years > 0 else 0
                
                # Calculate Sharpe Ratio
                portfolio_df = result['portfolio_df'].copy()
                portfolio_df['Daily_Return'] = portfolio_df['Value'].pct_change()
                daily_returns = portfolio_df['Daily_Return'].dropna()
                
                if len(daily_returns) > 0 and daily_returns.std() > 0:
                    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
                else:
                    sharpe = 0
                
                outperformance = result['total_return_pct'] - qqq_bh_return
                
                # Build parameter string
                param_str = _build_param_string(params, period_name if test_multiple_periods else None)
                
                results.append({
                    'Period': period_name,
                    'Parameters': param_str,
                    'Final Value': result['final_value'],
                    'Total Return %': result['total_return_pct'],
                    'CAGR %': cagr,
                    'Max Drawdown %': result['max_drawdown'],
                    'Sharpe Ratio': sharpe,
                    'Trades': result['num_trades'],
                    'vs QQQ %': outperformance,
                    'QQQ Return %': qqq_bh_return,
                    'params_dict': params
                })
                
            except Exception as e:
                st.warning(f"Error testing {period_name} - combination {test_counter}: {str(e)}")
            
            progress_bar.progress(test_counter / total_combinations)
    
    progress_bar.empty()
    status_text.empty()
    
    if len(results) == 0:
        st.error("No valid results found. Please adjust your parameters.")
        st.stop()
    
    # Store results
    st.session_state.grid_search_results = results
    
    # Clear any previously applied params
    for key in ['best_params', 'user_applied_params', 'applied_rank', 'auto_applied_rank']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success(f"✅ Grid search complete! Tested {len(results)} combinations.")
    rerun()


def _generate_param_combinations(
    enable_ema, ema_config, enable_rsi, rsi_config,
    enable_sl, sl_config, enable_bb, bb_config,
    enable_atr, atr_config, enable_msl, msl_config,
    enable_macd, macd_config, enable_adx, adx_config
    , enable_supertrend=None, st_config=None
):
    """Generate all parameter combinations for grid search using itertools."""
    
    param_combinations = []

    # EMA base parameters
    ema_strategy_params = []
    if enable_ema:
        if "Single EMA" in ema_config.get('strategy_options', []):
            for ema in ema_config['ema_range']:
                ema_strategy_params.append({'use_double_ema': False, 'ema_period': ema, 'ema_fast': 9, 'ema_slow': 21})
        
        if "Double EMA Crossover" in ema_config.get('strategy_options', []):
            for fast in ema_config['fast_ema_range']:
                for slow in ema_config['slow_ema_range']:
                    if fast < slow:
                        ema_strategy_params.append({'use_double_ema': True, 'ema_period': slow, 'ema_fast': fast, 'ema_slow': slow})
    else:
        # If EMA is disabled, create a single placeholder
        ema_strategy_params.append({'use_double_ema': False, 'ema_period': 50, 'ema_fast': 9, 'ema_slow': 21})

    # Prepare iterables for all other parameters
    rsi_params = list(itertools.product(
        rsi_config.get('rsi_range', [0]) if enable_rsi else [0],
        rsi_config.get('rsi_oversold_range', [30]) if enable_rsi else [30],
        rsi_config.get('rsi_overbought_range', [70]) if enable_rsi else [70]
    ))
    
    sl_params = sl_config.get('stop_loss_range', [0]) if enable_sl else [0]
    
    bb_params = list(itertools.product(
        ["Enabled", "Disabled"] if enable_bb else ["Disabled"],
        bb_config.get('bb_period_range', [20]) if enable_bb else [20],
        bb_config.get('bb_std_dev_range', [2.0]) if enable_bb else [2.0],
        bb_config.get('bb_buy_threshold_range', [0.2]) if enable_bb else [0.2],
        bb_config.get('bb_sell_threshold_range', [0.8]) if enable_bb else [0.8]
    ))

    atr_params = list(itertools.product(
        ["Enabled", "Disabled"] if enable_atr else ["Disabled"],
        atr_config.get('atr_period_range', [14]) if enable_atr else [14],
        atr_config.get('atr_multiplier_range', [2.0]) if enable_atr else [2.0]
    ))

    msl_params = list(itertools.product(
        ["Enabled", "Disabled"] if enable_msl else ["Disabled"],
        msl_config.get('msl_period_range', [20]) if enable_msl else [20],
        msl_config.get('msl_lookback_range', [5]) if enable_msl else [5]
    ))
    
    macd_params = list(itertools.product(
        ["Enabled", "Disabled"] if enable_macd else ["Disabled"],
        macd_config.get('macd_fast_range', [12]) if enable_macd else [12],
        macd_config.get('macd_slow_range', [26]) if enable_macd else [26],
        macd_config.get('macd_signal_range', [9]) if enable_macd else [9]
    ))

    adx_params = list(itertools.product(
        ["Enabled", "Disabled"] if enable_adx else ["Disabled"],
        adx_config.get('adx_period_range', [14]) if enable_adx else [14],
        adx_config.get('adx_threshold_range', [25]) if enable_adx else [25]
    ))

    # Supertrend parameters
    st_params = list(itertools.product(
        ["Enabled", "Disabled"] if enable_supertrend else ["Disabled"],
        st_config.get('st_period_range', [10]) if enable_supertrend and st_config else [10],
        st_config.get('st_multiplier_range', [3.0]) if enable_supertrend and st_config else [3.0]
    ))

    # Combine all parameter sets
    all_combinations = itertools.product(
        ema_strategy_params, rsi_params, sl_params, bb_params,
        atr_params, msl_params, macd_params, adx_params
        , st_params
    )

    for combo in all_combinations:
        ema_p, rsi_p, sl_p, bb_p, atr_p, msl_p, macd_p, adx_p, st_p = combo

        use_bb = bb_p[0] == "Enabled"
        use_atr = atr_p[0] == "Enabled"
        use_msl = msl_p[0] == "Enabled"
        use_macd = macd_p[0] == "Enabled"
        use_adx = adx_p[0] == "Enabled"
        use_st = st_p[0] == "Enabled"

        param_dict = _create_param_dict(
            use_ema=enable_ema,
            use_double_ema=ema_p['use_double_ema'],
            ema_period=ema_p['ema_period'],
            ema_fast=ema_p['ema_fast'],
            ema_slow=ema_p['ema_slow'],
            rsi=rsi_p[0],
            rsi_oversold=rsi_p[1],
            rsi_overbought=rsi_p[2],
            sl=sl_p,
            use_bb=use_bb,
            bb_period=bb_p[1],
            bb_std=bb_p[2],
            bb_buy=bb_p[3],
            bb_sell=bb_p[4],
            use_atr=use_atr,
            atr_period=atr_p[1],
            atr_mult=atr_p[2],
            use_msl_msh=use_msl,
            msl_period=msl_p[1],
            msl_lookback=msl_p[2],
            use_macd=use_macd,
            macd_fast=macd_p[1],
            macd_slow=macd_p[2],
            macd_signal=macd_p[3],
            use_adx=use_adx,
            adx_period=adx_p[1],
            adx_thresh=adx_p[2]
            , use_supertrend=use_st, st_period=st_p[1], st_multiplier=st_p[2]
        )
        param_combinations.append(param_dict)
        
    return param_combinations



def _create_param_dict(
    use_ema, use_double_ema, ema_period, ema_fast, ema_slow,
    rsi, rsi_oversold, rsi_overbought, sl,
    use_bb, bb_period, bb_std, bb_buy, bb_sell,
    use_atr, atr_period, atr_mult,
    use_msl_msh, msl_period, msl_lookback,
    use_macd, macd_fast, macd_slow, macd_signal,
    use_adx, adx_period, adx_thresh,
    use_supertrend=False, st_period=10, st_multiplier=3.0
):
    """Create a parameter dictionary."""
    return {
        'use_ema': use_ema,
        'use_double_ema': use_double_ema,
        'ema_period': ema_period,
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'rsi_threshold': rsi,
        'use_rsi': rsi > 0,
        'rsi_oversold': rsi_oversold,
        'rsi_overbought': rsi_overbought,
        'stop_loss_pct': sl,
        'use_stop_loss': sl > 0,
        'use_bb': use_bb,
        'bb_period': bb_period,
        'bb_std_dev': bb_std,
        'bb_buy_threshold': bb_buy,
        'bb_sell_threshold': bb_sell,
        'use_atr': use_atr,
        'atr_period': atr_period,
        'atr_multiplier': atr_mult,
        'use_msl_msh': use_msl_msh,
        'msl_period': msl_period,
        'msh_period': msl_period,
        'msl_lookback': msl_lookback,
        'msh_lookback': msl_lookback,
        'use_macd': use_macd,
        'macd_fast': macd_fast,
        'macd_slow': macd_slow,
        'macd_signal_period': macd_signal,
        'use_adx': use_adx,
        'adx_period': adx_period,
        'adx_threshold': adx_thresh,
        'use_supertrend': use_supertrend,
        'st_period': st_period,
        'st_multiplier': st_multiplier
    }


def _build_param_string(params, period_name=None):
    """Build a human-readable parameter string."""
    if params['use_double_ema']:
        param_str = f"EMA({params['ema_fast']}/{params['ema_slow']})"
    else:
        param_str = f"EMA({params['ema_period']})"
    
    if params['use_rsi']:
        param_str += f" | RSI>{params['rsi_threshold']}"
    
    if params['use_stop_loss']:
        param_str += f" | SL:{params['stop_loss_pct']}%"
    
    if params['use_bb']:
        param_str += f" | BB({params['bb_period']},{params['bb_std_dev']},{params['bb_buy_threshold']}/{params['bb_sell_threshold']})"
    
    if params['use_atr']:
        param_str += f" | ATR({params['atr_period']},{params['atr_multiplier']}x)"
    
    if params['use_msl_msh']:
        param_str += f" | MSL({params['msl_period']},{params['msl_lookback']})"
        
    if params.get('use_macd', False):
        param_str += f" | MACD({params['macd_fast']},{params['macd_slow']},{params['macd_signal_period']})"
        
    if params.get('use_adx', False):
        param_str += f" | ADX({params['adx_period']},{params['adx_threshold']})"
    if params.get('use_supertrend', False):
        param_str += f" | ST({params['st_period']},{params['st_multiplier']})"
    
    if period_name:
        param_str = f"[{period_name}] {param_str}"
    
    return param_str


def _display_grid_search_results(selected_periods, test_multiple_periods):
    """Display grid search results with apply buttons."""
    
    st.markdown("---")
    st.markdown("### 📊 Grid Search Results")
    
    results = st.session_state.grid_search_results
    sorted_results = sorted(results, key=lambda x: x['vs QQQ %'], reverse=True)
    
    st.markdown(f"**All {len(sorted_results)} Combinations Tested** (Top 10 highlighted in green)")
    st.caption("💡 Use the 'Apply' buttons to quickly test any configuration")
    
    show_all_results = st.checkbox("Show all tested combinations", value=False)
    
    if show_all_results:
        _display_all_results(sorted_results)
    else:
        _display_top_10_results(sorted_results)
    
    # Show aggregated results if multiple periods
    if test_multiple_periods and len(selected_periods) > 1:
        _display_robust_strategies(results, selected_periods)
    
    # Clear results button
    if st.button("🔄 Clear Results", use_container_width=True):
        for key in ['best_params', 'grid_search_results', 'user_applied_params', 'applied_rank', 'auto_applied_rank']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Cleared!")
        rerun()

    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ Previous Step", use_container_width=True, key="step2_prev"):
            st.session_state.wizard_step = 1
            rerun()

    with col2:
        pass  # Empty middle column for spacing

    with col3:
        if st.button("Next Step ➡️", type="primary", use_container_width=True, key="step2_next"):
            if st.session_state.get('best_params') is not None:
                st.session_state.wizard_step = 3
                rerun()
            else:
                st.warning("⚠️ Please find best signals and apply a strategy first!")

    # Add bottom padding for mobile visibility
    st.markdown("<br><br>", unsafe_allow_html=True)


def _display_top_10_results(sorted_results):
    """Display top 10 results with inline apply buttons."""
    
    st.markdown("**Top 10 Parameter Combinations:**")
    st.caption("💡 Click the 'Apply' button in each row to select that strategy")
    
    # Display each result as a row with inline apply button
    for i, r in enumerate(sorted_results[:10]):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        result_emoji = '✅' if r['vs QQQ %'] > 0 else '❌'
        params = r['params_dict']
        
        # Create expandable row
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Main metrics in a compact format
                st.markdown(f"""
                **{rank_emoji} {r['Parameters']}**  
                💰 Final: ${r['Final Value']:,.2f} | 📈 Return: {r['Total Return %']:.2f}% | 📊 CAGR: {r['CAGR %']:.2f}% | 📉 Max DD: {r['Max Drawdown %']:.2f}% | ⚡ Sharpe: {r['Sharpe Ratio']:.2f} | 🔄 Trades: {r['Trades']} | 🆚 vs QQQ: {r['vs QQQ %']:+.2f}% {result_emoji}
                """)
            
            with col2:
                button_key = f"apply_top10_rank_{i+1}_{id(params)}"
                if st.button("Apply", key=button_key, use_container_width=True, type="primary" if i == 0 else "secondary"):
                    # Build confirmation message
                    if params['use_double_ema']:
                        strategy_desc = f"Double EMA ({params['ema_fast']}/{params['ema_slow']})"
                    else:
                        strategy_desc = f"Single EMA ({params['ema_period']})"
                    
                    if params['use_rsi'] and params['rsi_threshold'] > 0:
                        strategy_desc += f" + RSI>{params['rsi_threshold']}"
                    
                    if params['use_stop_loss']:
                        strategy_desc += f" + Stop-Loss {params['stop_loss_pct']}%"
                    
                    if params['use_bb']:
                        strategy_desc += f" + Bollinger Bands"
                    
                    if params['use_atr']:
                        strategy_desc += f" + ATR Stop-Loss"
                    
                    if params['use_msl_msh']:
                        strategy_desc += f" + MSL/MSH"

                    if params.get('use_macd', False):
                        strategy_desc += f" + MACD"
                    
                    if params.get('use_adx', False):
                        strategy_desc += f" + ADX"
                    
                    # Apply parameters
                    st.session_state.best_params = params.copy()
                    st.session_state.user_applied_params = True
                    st.session_state.applied_rank = i + 1
                    st.session_state.navigate_to_step2 = True
                    st.session_state.manual_config_loaded = False
                    st.session_state.testing_params_loaded = False
                    
                    # Show confirmation and redirect
                    st.success(f"✅ **Applied Rank #{i + 1} Strategy:** {strategy_desc}")
                    st.info("🔄 Redirecting to Step 3: Verify & Customize...")
                    st.session_state.wizard_step = 3
                    rerun()
            
            st.markdown("---")


def _display_all_results(sorted_results):
    """Display all results in a table."""
    
    all_display_results = []
    for i, r in enumerate(sorted_results):
        row_data = {
            'Rank': i + 1,
            'Parameters': r['Parameters'],
            'Final Value': f"${r['Final Value']:,.2f}",
            'Total Return': f"{r['Total Return %']:.2f}%",
            'CAGR': f"{r['CAGR %']:.2f}%",
            'Max DD': f"{r['Max Drawdown %']:.2f}%",
            'Sharpe': f"{r['Sharpe Ratio']:.2f}",
            'Trades': r['Trades'],
            'vs QQQ': f"{r['vs QQQ %']:+.2f}%",
            'QQQ Return': f"{r['QQQ Return %']:.2f}%",
            'Result': '✅ Win' if r['vs QQQ %'] > 0 else '❌ Loss'
        }
        all_display_results.append(row_data)
    
    all_results_df = pd.DataFrame(all_display_results)
    
    def highlight_top_10(row):
        if row['Rank'] <= 10:
            return ['background-color: #d4edda'] * len(row)
        else:
            return ['background-color: #f8f9fa'] * len(row)
    
    styled_df = all_results_df.style.apply(highlight_top_10, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=600, hide_index=True)
    
    st.download_button(
        "📥 Download All Results CSV",
        all_results_df.to_csv(index=False),
        "grid_search_all_results.csv",
        "text/csv",
        use_container_width=True
    )
    
    # Summary statistics
    st.markdown("---")
    st.markdown("**Summary Statistics:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        win_count = sum(1 for r in sorted_results if r['vs QQQ %'] > 0)
        win_rate = (win_count / len(sorted_results) * 100) if len(sorted_results) > 0 else 0
        st.metric("Win Rate vs QQQ", f"{win_rate:.1f}%", f"{win_count}/{len(sorted_results)}")
    
    with col2:
        avg_return = sum(r['Total Return %'] for r in sorted_results) / len(sorted_results)
        st.metric("Avg Total Return", f"{avg_return:.2f}%")
    
    with col3:
        avg_outperformance = sum(r['vs QQQ %'] for r in sorted_results) / len(sorted_results)
        st.metric("Avg vs QQQ", f"{avg_outperformance:+.2f}%")
    
    with col4:
        avg_sharpe = sum(r['Sharpe Ratio'] for r in sorted_results) / len(sorted_results)
        st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")


def _display_robust_strategies(results, selected_periods):
    """Display most robust strategies across all periods."""
    
    st.markdown("---")
    st.markdown("### 🏆 Best Parameters Across All Periods")
    
    # Group by parameter combination
    param_performance = defaultdict(list)
    
    for r in results:
        if r['params_dict']['use_double_ema']:
            key = f"EMA({r['params_dict']['ema_fast']}/{r['params_dict']['ema_slow']})"
        else:
            key = f"EMA({r['params_dict']['ema_period']})"
        
        if r['params_dict']['use_rsi']:
            key += f" | RSI>{r['params_dict']['rsi_threshold']}"
        
        if r['params_dict']['use_stop_loss']:
            key += f" | SL:{r['params_dict']['stop_loss_pct']}%"
        
        if r['params_dict']['use_bb']:
            key += f" | BB({r['params_dict']['bb_period']},{r['params_dict']['bb_std_dev']})"
        
        param_performance[key].append({
            'period': r['Period'],
            'vs_qqq': r['vs QQQ %'],
            'cagr': r['CAGR %'],
            'sharpe': r['Sharpe Ratio'],
            'params_dict': r['params_dict']
        })
    
    # Calculate averages
    aggregated_results = []
    for param_key, performances in param_performance.items():
        avg_vs_qqq = sum(p['vs_qqq'] for p in performances) / len(performances)
        avg_cagr = sum(p['cagr'] for p in performances) / len(performances)
        avg_sharpe = sum(p['sharpe'] for p in performances) / len(performances)
        win_rate = sum(1 for p in performances if p['vs_qqq'] > 0) / len(performances) * 100
        
        aggregated_results.append({
            'Parameters': param_key,
            'Avg vs QQQ %': avg_vs_qqq,
            'Avg CAGR %': avg_cagr,
            'Avg Sharpe': avg_sharpe,
            'Win Rate': win_rate,
            'Periods Tested': len(performances),
            'params_dict': performances[0]['params_dict']
        })
    
    aggregated_results.sort(key=lambda x: x['Avg vs QQQ %'], reverse=True)
    
    st.markdown("**Top 5 Most Robust Parameters (Best Average Performance):**")
    
    # Display table
    robust_display = []
    for i, r in enumerate(aggregated_results[:5]):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        
        robust_display.append({
            'Rank': rank_emoji,
            'Parameters': r['Parameters'],
            'Avg vs QQQ': f"{r['Avg vs QQQ %']:+.2f}%",
            'Avg CAGR': f"{r['Avg CAGR %']:.2f}%",
            'Avg Sharpe': f"{r['Avg Sharpe']:.2f}",
            'Win Rate': f"{r['Win Rate']:.0f}%",
            'Periods': r['Periods Tested']
        })
    
    st.caption("💡 Click the 'Apply' button in each row to select that strategy")
    
    # Display each result as a row with inline apply button
    num_to_show = min(5, len(aggregated_results))
    for i in range(num_to_show):
        r = aggregated_results[i]
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        params = r['params_dict']
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                **{rank_emoji} {r['Parameters']}**  
                📊 Avg vs QQQ: {r['Avg vs QQQ %']:+.2f}% | 📈 Avg CAGR: {r['Avg CAGR %']:.2f}% | ⚡ Avg Sharpe: {r['Avg Sharpe']:.2f} | 🎯 Win Rate: {r['Win Rate']:.0f}% | 📅 Periods: {r['Periods Tested']}
                """)
            
            with col2:
                button_key = f"apply_robust_rank_{i+1}_{id(params)}"
                if st.button("Apply", key=button_key, use_container_width=True, type="primary" if i == 0 else "secondary"):
                    # Build confirmation message
                    if params['use_double_ema']:
                        strategy_desc = f"Double EMA ({params['ema_fast']}/{params['ema_slow']})"
                    else:
                        strategy_desc = f"Single EMA ({params['ema_period']})"
                    
                    if params['use_rsi'] and params['rsi_threshold'] > 0:
                        strategy_desc += f" + RSI>{params['rsi_threshold']}"
                    
                    if params['use_stop_loss']:
                        strategy_desc += f" + Stop-Loss {params['stop_loss_pct']}%"
                    
                    if params['use_bb']:
                        strategy_desc += f" + Bollinger Bands"
                    
                    if params['use_atr']:
                        strategy_desc += f" + ATR Stop-Loss"
                    
                    if params['use_msl_msh']:
                        strategy_desc += f" + MSL/MSH"

                    if params.get('use_macd', False):
                        strategy_desc += f" + MACD"
                    
                    if params.get('use_adx', False):
                        strategy_desc += f" + ADX"
                    
                    # Apply parameters
                    st.session_state.best_params = params.copy()
                    st.session_state.user_applied_params = True
                    st.session_state.applied_rank = i + 1
                    st.session_state.navigate_to_step2 = True
                    st.session_state.manual_config_loaded = False
                    st.session_state.testing_params_loaded = False
                    
                    # Show confirmation and redirect
                    st.success(f"✅ **Applied Rank #{i + 1} Strategy:** {strategy_desc}")
                    st.info("🔄 Redirecting to Step 3: Verify & Customize...")
                    st.session_state.wizard_step = 3
                    rerun()
            
            st.markdown("---")
    
    st.info(f"""
    **Most Robust Strategy (Best across {len(selected_periods)} periods):**
    - **Parameters:** {aggregated_results[0]['Parameters']}
    - **Average Outperformance vs QQQ:** {aggregated_results[0]['Avg vs QQQ %']:+.2f}%
    - **Average CAGR:** {aggregated_results[0]['Avg CAGR %']:.2f}%
    - **Average Sharpe Ratio:** {aggregated_results[0]['Avg Sharpe']:.2f}
    - **Win Rate:** {aggregated_results[0]['Win Rate']:.0f}% ({int(aggregated_results[0]['Win Rate']/100 * len(selected_periods))}/{len(selected_periods)} periods)
    
    This strategy consistently performs well across different market conditions!
    """)
