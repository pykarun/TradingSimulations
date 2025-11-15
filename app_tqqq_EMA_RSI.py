import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from curl_cffi import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Handle different Streamlit versions
try:
    rerun = st.rerun
except AttributeError:
    rerun = st.experimental_rerun

# --- Page Configuration (Mobile Optimized) ---
st.set_page_config(
    page_title="Smart Leverage Strategy",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Custom CSS for mobile optimization
st.markdown("""
<style>
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Fix header spacing - prevent overlap with deploy button */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 1rem;
    }
    
    /* Additional top margin for main content */
    .main .block-container {
        margin-top: 2rem;
    }
    
    /* Ensure title is visible */
    h1:first-of-type {
        margin-top: 1rem;
        padding-top: 1rem;
    }
    
    /* Responsive text */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.5rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
        
        /* More top padding on mobile */
        .block-container {
            padding-top: 4rem;
        }
    }
    
    /* Better button spacing on mobile */
    .stButton button {
        width: 100%;
    }
    
    /* Fix for Streamlit's top toolbar */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Smart Leverage Strategy")
st.caption("TQQQ Trading Strategy with EMA, RSI & Stop-Loss Protection")

# --- Core Functions ---

def calculate_ema(data, period):
    """Calculate EMA for given period"""
    df = data.copy()
    df['EMA'] = df['Close'].ewm(span=period, adjust=False).mean()
    return df

def calculate_double_ema(data, fast_period, slow_period):
    """Calculate two EMAs for crossover strategy"""
    df = data.copy()
    df['EMA_Fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    return df

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    df = data.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

@st.cache_data(ttl=3600)
def get_data(tickers, start_date, end_date, buffer_days=365):
    """Download data using curl_cffi session"""
    session = requests.Session(impersonate="chrome110", verify=False)
    fetch_start_date = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
    data = yf.download(tickers, start=fetch_start_date, end=end_date, session=session, 
                      auto_adjust=False, group_by='ticker', progress=False)
    return data

def run_tqqq_only_strategy(qqq_data, tqqq_data, start_date, end_date, initial_capital, ema_period, rsi_threshold=0, use_rsi=False, stop_loss_pct=0, use_stop_loss=False, use_double_ema=False, ema_fast=9, ema_slow=21):
    """Smart Leverage Strategy - TQQQ with EMA, RSI & Stop-Loss"""
    if use_double_ema:
        qqq_data = calculate_double_ema(qqq_data, ema_fast, ema_slow)
    else:
        qqq_data = calculate_ema(qqq_data, ema_period)
    qqq_data = calculate_rsi(qqq_data, period=14)
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
        
        if use_double_ema:
            ema_fast_val = sim_data.iloc[i]['EMA_Fast']
            ema_slow_val = sim_data.iloc[i]['EMA_Slow']
            qqq_ema = ema_slow_val
        else:
            qqq_ema = sim_data.iloc[i]['EMA']
        
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
        
        if use_stop_loss and position == 'TQQQ' and peak_value > 0:
            drawdown_from_peak = ((current_value - peak_value) / peak_value) * 100
            if drawdown_from_peak <= -stop_loss_pct:
                stop_loss_triggered = True
                stop_loss_reason = f'Stop-Loss ({drawdown_from_peak:.2f}% from peak)'
        
        if use_double_ema:
            base_signal = 'BUY' if ema_fast_val > ema_slow_val else 'SELL'
        else:
            base_signal = 'BUY' if qqq_close > qqq_ema else 'SELL'
        
        if use_rsi:
            rsi_ok = pd.notna(rsi) and rsi > rsi_threshold
            signal = 'BUY' if base_signal == 'BUY' and rsi_ok else 'SELL'
        else:
            signal = base_signal
        
        if stop_loss_triggered:
            signal = 'SELL'
        
        action = 'HOLD'
        if signal == 'BUY' and position != 'TQQQ':
            shares = capital / tqqq_close
            position = 'TQQQ'
            entry_value = current_value
            peak_value = current_value
            action = 'BUY TQQQ'
        elif signal == 'SELL' and position == 'TQQQ':
            capital = shares * tqqq_close
            shares = 0
            position = None
            entry_value = 0
            peak_value = 0
            if stop_loss_triggered:
                action = f'SELL (STOP-LOSS)'
            else:
                action = 'SELL to CASH'
        else:
            if position == 'TQQQ':
                action = 'HOLD TQQQ'
            else:
                action = 'HOLD CASH'
        
        if position == 'TQQQ':
            current_value = shares * tqqq_close
        else:
            current_value = capital
        
        if use_double_ema:
            signal_text = f'Fast({ema_fast_val:.2f}) > Slow({ema_slow_val:.2f})' if ema_fast_val > ema_slow_val else f'Fast({ema_fast_val:.2f}) < Slow({ema_slow_val:.2f})'
            ema_display = f'${ema_fast_val:.2f} / ${ema_slow_val:.2f}'
        else:
            signal_text = 'Above EMA' if qqq_close > qqq_ema else 'Below EMA'
            ema_display = f'${qqq_ema:.2f}'
        
        trade_log.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Action': action,
            'QQQ_Price': f'${qqq_close:.2f}',
            'QQQ_EMA': ema_display,
            'RSI': f'{rsi:.1f}' if pd.notna(rsi) else 'N/A',
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
    
    num_trades = len([log for log in trade_log if log['Action'] in ['BUY TQQQ', 'SELL to CASH', 'SELL (STOP-LOSS)']])
    num_stop_loss_exits = len([log for log in trade_log if 'STOP-LOSS' in log['Action']])
    
    return {
        'portfolio_df': portfolio_df,
        'trade_log': trade_log,
        'final_value': final_value,
        'total_return_pct': total_return,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'num_stop_loss_exits': num_stop_loss_exits
    }

def create_performance_chart(result, qqq_data, tqqq_data, start_date, initial_capital, ema_period):
    """Create performance chart with QQQ benchmark"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            'Portfolio Value vs QQQ Benchmark',
            f'QQQ Price vs {ema_period}-day EMA',
            'Drawdown %'
        )
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=result['portfolio_df'].index,
            y=result['portfolio_df']['Value'],
            name='Smart Leverage Strategy',
            line=dict(color='green', width=3)
        ),
        row=1, col=1
    )
    
    # Trade markers
    trade_log_df = pd.DataFrame(result['trade_log'])
    trade_log_df['Date'] = pd.to_datetime(trade_log_df['Date'])
    trade_log_df = trade_log_df.set_index('Date')
    trade_log_df['Portfolio_Value_Numeric'] = trade_log_df['Portfolio_Value'].str.replace('$', '').str.replace(',', '').astype(float)
    
    buy_trades = trade_log_df[trade_log_df['Action'] == 'BUY TQQQ']
    if len(buy_trades) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_trades.index,
                y=buy_trades['Portfolio_Value_Numeric'],
                mode='markers',
                name='Buy TQQQ',
                marker=dict(symbol='triangle-up', size=12, color='lime', line=dict(color='darkgreen', width=1)),
                hovertemplate='<b>BUY TQQQ</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    sell_trades = trade_log_df[trade_log_df['Action'] == 'SELL to CASH']
    if len(sell_trades) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_trades.index,
                y=sell_trades['Portfolio_Value_Numeric'],
                mode='markers',
                name='Sell to Cash',
                marker=dict(symbol='triangle-down', size=12, color='yellow', line=dict(color='orange', width=1)),
                hovertemplate='<b>SELL to CASH</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # QQQ benchmark
    qqq_benchmark = qqq_data.loc[start_date:]['Close']
    qqq_performance = (qqq_benchmark / qqq_benchmark.iloc[0]) * initial_capital
    fig.add_trace(
        go.Scatter(
            x=qqq_performance.index,
            y=qqq_performance,
            name='QQQ Buy & Hold',
            line=dict(color='gray', dash='dot', width=2)
        ),
        row=1, col=1
    )
    
    # QQQ Price vs EMA
    qqq_with_ema = calculate_ema(qqq_data, ema_period)
    qqq_display = qqq_with_ema.loc[start_date:]
    
    fig.add_trace(
        go.Scatter(
            x=qqq_display.index,
            y=qqq_display['Close'],
            name='QQQ Price',
            line=dict(color='black', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=qqq_display.index,
            y=qqq_display['EMA'],
            name=f'{ema_period}-day EMA',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=2, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=result['portfolio_df'].index,
            y=result['portfolio_df']['Drawdown'],
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text="Smart Leverage Strategy - Performance Analysis",
        showlegend=True,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

# ============================================================================
# MAIN APPLICATION - STEP-BASED WORKFLOW
# ============================================================================

st.markdown("---")

# Initialize session state for analysis type
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = "📊 Daily Signal"

# Initialize session state for showing sections
if 'show_backtest' not in st.session_state:
    st.session_state.show_backtest = True
if 'show_historical' not in st.session_state:
    st.session_state.show_historical = True

# --- NAVIGATION TABS ---
# Initialize current_step in session state if not exists
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0  # Default to Step 1

# Check if we should show navigation banner (keep it visible until user goes to Step 3)
if st.session_state.get('navigate_to_step3', False):
    st.success("✅ **Parameters Applied!** → Click on **'📊 Step 3: Testing'** tab above to run your strategy.")

# Create main step tabs
main_tab1, main_tab2, main_tab3 = st.tabs(["📖 Step 1: How It Works", "🔍 Step 2: Find Best Strategy", "📊 Step 3: Testing"])

# Initialize run_button to False by default
run_button = False

# --- STEP 1: HOW IT WORKS ---
with main_tab1:
    st.markdown("## 📖 Step 1: How the Strategy Works")
    
    st.markdown("""
    ### 🎯 What This Tool Does
    
    This tool helps you trade **TQQQ** (3x leveraged NASDAQ ETF) using technical indicators to maximize gains while managing risk.
    
    ---
    
    ### 📚 Key Concepts (For Beginners)
    
    **What is QQQ?**
    - QQQ is an ETF that tracks the NASDAQ-100 index (top 100 tech companies)
    - Includes companies like Apple, Microsoft, Amazon, Google, Tesla
    - 1x leverage = normal stock market returns
    
    **What is TQQQ?**
    - TQQQ is a 3x leveraged version of QQQ
    - If QQQ goes up 1%, TQQQ goes up ~3%
    - If QQQ goes down 1%, TQQQ goes down ~3%
    - Higher risk, higher reward - NOT for buy-and-hold
    
    **What is EMA (Exponential Moving Average)?**
    - A trend-following indicator that smooths out price movements
    - Gives more weight to recent prices
    - **Single EMA:** Buy when price crosses above EMA (trend is up)
    - **Double EMA:** Buy when fast EMA crosses above slow EMA (stronger confirmation)
    
    **What is RSI (Relative Strength Index)?**
    - Measures momentum on a scale of 0-100
    - RSI > 50 = bullish momentum (good time to buy)
    - RSI < 50 = bearish momentum (avoid buying)
    - Optional filter to avoid buying in weak conditions
    
    **What is Stop-Loss?**
    - Automatic exit when portfolio drops X% from its peak
    - Protects you from large losses during market crashes
    - Example: 10% stop-loss exits if you're down 10% from your highest value
    
    ---
    
    ### 🚀 Quick Start (3 Steps)
    
    1. **Step 2: Find Your Best Strategy**
       - Run Grid Search to test hundreds of parameter combinations
       - System ranks strategies by performance vs QQQ
       - Apply the best parameters automatically
    
    2. **Step 3: Test & Execute**
       - Get today's buy/sell signal (Daily Signal)
       - Backtest on historical data (Custom Simulation)
       - See probability of success (Monte Carlo)
    
    ---
    
    ### ⚠️ Important Warnings
    
    - **TQQQ is 3x leveraged** - Gains and losses are amplified
    - **Not buy-and-hold** - Requires active daily management
    - **Past performance ≠ future results**
    - **Educational purposes only** - NOT financial advice
    - **High risk** - Only use money you can afford to lose
    """)

with main_tab2:
    st.markdown("## 🔍 Step 2: Find Best Strategy")
    
    # Initialize session state for grid search
    if 'grid_search_results' not in st.session_state:
        st.session_state.grid_search_results = None
    if 'best_params' not in st.session_state:
        st.session_state.best_params = None
    if 'show_grid_search' not in st.session_state:
        st.session_state.show_grid_search = True  # Default to grid search mode

    # Configuration Mode Selection using tabs
    config_tab1, config_tab2 = st.tabs(["🔍 Grid Search Optimizer", "📝 Manual Configuration"])
    
    # Determine which tab to show based on session state
    # Note: Streamlit tabs don't support programmatic selection, so we use conditional rendering
    
    # === GRID SEARCH SECTION ===
    with config_tab1:
        st.markdown("### 🔍 Grid Search Optimizer")
        st.info("Find the best parameters by testing multiple combinations on historical data")
        
        # Time Period Selection
        st.markdown("**Select Historical Periods for Optimization:**")
        
        time_periods = {
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "3 Years": 1095,
            "4 Years": 1460,
            "5 Years": 1825
        }
        
        test_multiple_periods = st.checkbox(
            "Test multiple time periods",
            value=False,
            help="Test strategies across different historical periods to find the most robust parameters"
        )
        
        if test_multiple_periods:
            selected_periods = st.multiselect(
                "Historical Data Periods",
                options=list(time_periods.keys()),
                default=["1 Year", "2 Years", "3 Years"],
                help="Select multiple periods to test - results will show best parameters across all periods"
            )
            
            if selected_periods:
                period_summary = ", ".join(selected_periods)
                st.caption(f"Will test on: {period_summary}")
            else:
                st.warning("Please select at least one time period")
        else:
            selected_period = st.selectbox(
                "Historical Data Period",
                options=list(time_periods.keys()),
                index=3,  # Default to 2 years
                help="Select how far back to test the strategy"
            )
            
            days_back = time_periods[selected_period]
            grid_end_date = datetime.date.today()
            grid_start_date = grid_end_date - datetime.timedelta(days=days_back)
            
            st.caption(f"Testing period: {grid_start_date} to {grid_end_date}")
        
        # Parameter Ranges
        st.markdown("**Parameter Ranges to Test:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**EMA Strategy:**")
            ema_strategy_options = st.multiselect(
                "Test EMA Strategies",
                options=["Single EMA", "Double EMA Crossover"],
                default=["Single EMA", "Double EMA Crossover"],
                help="Select which EMA strategies to test"
            )
            
            if "Single EMA" in ema_strategy_options:
                ema_range = st.multiselect(
                    "Single EMA Periods",
                    options=[10, 20, 21, 30, 40, 50, 60, 80, 100],
                    default=[21, 30, 50, 80]
                )
            else:
                ema_range = []
            
            if "Double EMA Crossover" in ema_strategy_options:
                fast_ema_range = st.multiselect(
                    "Fast EMA Periods",
                    options=[5, 8, 9, 10, 12, 15, 20, 21],
                    default=[9, 12, 21]
                )
                slow_ema_range = st.multiselect(
                    "Slow EMA Periods",
                    options=[15, 20, 21, 25, 30, 40, 50],
                    default=[21, 30, 50]
                )
            else:
                fast_ema_range = []
                slow_ema_range = []
        
        with col2:
            st.markdown("**Risk Management:**")
            rsi_range = st.multiselect(
                "RSI Thresholds (0 = disabled)",
                options=[0, 40, 45, 50, 55, 60],
                default=[0, 50]
            )
            stop_loss_range = st.multiselect(
                "Stop-Loss % (0 = disabled)",
                options=[0, 5, 8, 10, 12, 15],
                default=[0, 5, 10, 15]
            )
        
        grid_capital = st.number_input(
            "Initial Capital for Testing",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
        
        # Run Grid Search Button
        st.markdown("---")
        run_grid_search = st.button("🚀 Run Grid Search", type="primary", use_container_width=True)
        
        if run_grid_search:
            # Validate inputs
            if test_multiple_periods and not selected_periods:
                st.error("Please select at least one time period to test")
                st.stop()
            
            if not ema_strategy_options:
                st.error("Please select at least one EMA strategy to test")
                st.stop()
            
            if "Single EMA" in ema_strategy_options and not ema_range:
                st.error("Please select at least one Single EMA period")
                st.stop()
            
            if "Double EMA Crossover" in ema_strategy_options:
                if not fast_ema_range or not slow_ema_range:
                    st.error("Please select at least one Fast and Slow EMA period for Double EMA")
                    st.stop()
            
            if not rsi_range or not stop_loss_range:
                st.error("Please select at least one RSI threshold and Stop-Loss percentage")
                st.stop()
            
            # Generate parameter combinations
            param_combinations = []
            
            # Single EMA combinations
            if "Single EMA" in ema_strategy_options:
                for ema in ema_range:
                    for rsi in rsi_range:
                        for sl in stop_loss_range:
                            param_combinations.append({
                                'use_double_ema': False,
                                'ema_period': ema,
                                'ema_fast': 9,
                                'ema_slow': 21,
                                'rsi_threshold': rsi,
                                'use_rsi': rsi > 0,
                                'stop_loss_pct': sl,
                                'use_stop_loss': sl > 0
                            })
            
            # Double EMA combinations
            if "Double EMA Crossover" in ema_strategy_options:
                for fast in fast_ema_range:
                    for slow in slow_ema_range:
                        if fast >= slow:
                            continue  # Skip invalid combinations
                        for rsi in rsi_range:
                            for sl in stop_loss_range:
                                param_combinations.append({
                                    'use_double_ema': True,
                                    'ema_fast': fast,
                                    'ema_slow': slow,
                                    'ema_period': slow,
                                    'rsi_threshold': rsi,
                                    'use_rsi': rsi > 0,
                                    'stop_loss_pct': sl,
                                    'use_stop_loss': sl > 0
                                })
            
            # Determine which periods to test
            if test_multiple_periods:
                periods_to_test = [(period, time_periods[period]) for period in selected_periods]
            else:
                periods_to_test = [(selected_period, days_back)]
            
            total_combinations = len(param_combinations) * len(periods_to_test)
            
            st.info(f"Testing {len(param_combinations)} parameter combinations across {len(periods_to_test)} time period(s) = {total_combinations} total tests...")
            
            # Download data once (use longest period)
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
                
                # Calculate QQQ benchmark for this period
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
                            params['stop_loss_pct'],
                            params['use_stop_loss'],
                            params['use_double_ema'],
                            params['ema_fast'],
                            params['ema_slow']
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
                        if params['use_double_ema']:
                            param_str = f"EMA({params['ema_fast']}/{params['ema_slow']})"
                        else:
                            param_str = f"EMA({params['ema_period']})"
                        
                        if params['use_rsi']:
                            param_str += f" | RSI>{params['rsi_threshold']}"
                        
                        if params['use_stop_loss']:
                            param_str += f" | SL:{params['stop_loss_pct']}%"
                        
                        # Add period to parameter string if testing multiple periods
                        if test_multiple_periods:
                            param_str = f"[{period_name}] {param_str}"
                        
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
            
            # Store results in session state
            st.session_state.grid_search_results = results
            st.session_state.best_params = results[0]['params_dict']  # Will be updated after sorting
            
            st.success(f"✅ Grid search complete! Tested {len(results)} combinations.")
            rerun()
        
        # Display Results
        if st.session_state.grid_search_results is not None:
            st.markdown("---")
            st.markdown("### 📊 Grid Search Results")
            
            results = st.session_state.grid_search_results
            
            # Sort by vs QQQ % (default and only option)
            sorted_results = sorted(results, key=lambda x: x['vs QQQ %'], reverse=True)
            
            # Update best params
            st.session_state.best_params = sorted_results[0]['params_dict']
            
            # Display all combinations with color coding
            st.markdown(f"**All {len(sorted_results)} Combinations Tested** (Top 10 highlighted in green)")
            
            show_all_results = st.checkbox("Show all tested combinations", value=False)
            
            if show_all_results:
                # Create styled dataframe
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
                
                # Apply styling function
                def highlight_top_10(row):
                    if row['Rank'] <= 10:
                        return ['background-color: #d4edda'] * len(row)  # Light green for top 10
                    else:
                        return ['background-color: #f8f9fa'] * len(row)  # Light gray for others
                
                styled_df = all_results_df.style.apply(highlight_top_10, axis=1)
                
                st.dataframe(styled_df, use_container_width=True, height=600, hide_index=True)
                
                # Download button for all results
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
            else:
                # Show only top 10
                st.markdown("**Top 10 Parameter Combinations:**")
                
                display_results = []
                for i, r in enumerate(sorted_results[:10]):
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
                        'Result': '✅ Win' if r['vs QQQ %'] > 0 else '❌ Loss'
                    }
                    display_results.append(row_data)
                
                results_df = pd.DataFrame(display_results)
                
                # Apply green background to top 10
                def highlight_all_green(row):
                    return ['background-color: #d4edda'] * len(row)
                
                styled_top_10 = results_df.style.apply(highlight_all_green, axis=1)
                
                st.dataframe(styled_top_10, use_container_width=True, hide_index=True)
            
            # Best Parameters Summary
            st.markdown("---")
            
            # If multiple periods tested, show aggregated best parameters
            if test_multiple_periods and len(selected_periods) > 1:
                st.markdown("### 🏆 Best Parameters Across All Periods")
                
                # Group by parameter combination (excluding period)
                from collections import defaultdict
                param_performance = defaultdict(list)
                
                for r in results:
                    # Create key without period
                    if r['params_dict']['use_double_ema']:
                        key = f"EMA({r['params_dict']['ema_fast']}/{r['params_dict']['ema_slow']})"
                    else:
                        key = f"EMA({r['params_dict']['ema_period']})"
                    
                    if r['params_dict']['use_rsi']:
                        key += f" | RSI>{r['params_dict']['rsi_threshold']}"
                    
                    if r['params_dict']['use_stop_loss']:
                        key += f" | SL:{r['params_dict']['stop_loss_pct']}%"
                    
                    param_performance[key].append({
                        'period': r['Period'],
                        'vs_qqq': r['vs QQQ %'],
                        'cagr': r['CAGR %'],
                        'sharpe': r['Sharpe Ratio'],
                        'params_dict': r['params_dict']
                    })
                
                # Calculate average performance across periods
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
                
                # Sort by average vs QQQ
                aggregated_results.sort(key=lambda x: x['Avg vs QQQ %'], reverse=True)
                
                st.markdown("**Top 5 Most Robust Parameters (Best Average Performance):**")
                
                robust_display = []
                for i, r in enumerate(aggregated_results[:5]):
                    robust_display.append({
                        'Rank': i + 1,
                        'Parameters': r['Parameters'],
                        'Avg vs QQQ': f"{r['Avg vs QQQ %']:+.2f}%",
                        'Avg CAGR': f"{r['Avg CAGR %']:.2f}%",
                        'Avg Sharpe': f"{r['Avg Sharpe']:.2f}",
                        'Win Rate': f"{r['Win Rate']:.0f}%",
                        'Periods': r['Periods Tested']
                    })
                
                robust_df = pd.DataFrame(robust_display)
                
                def highlight_robust(row):
                    return ['background-color: #d4edda'] * len(row)
                
                styled_robust = robust_df.style.apply(highlight_robust, axis=1)
                st.dataframe(styled_robust, use_container_width=True, hide_index=True)
                
                st.info(f"""
                **Most Robust Strategy (Best across {len(selected_periods)} periods):**
                - **Parameters:** {aggregated_results[0]['Parameters']}
                - **Average Outperformance vs QQQ:** {aggregated_results[0]['Avg vs QQQ %']:+.2f}%
                - **Average CAGR:** {aggregated_results[0]['Avg CAGR %']:.2f}%
                - **Average Sharpe Ratio:** {aggregated_results[0]['Avg Sharpe']:.2f}
                - **Win Rate:** {aggregated_results[0]['Win Rate']:.0f}% ({int(aggregated_results[0]['Win Rate']/100 * len(selected_periods))}/{len(selected_periods)} periods)
                
                This strategy consistently performs well across different market conditions!
                """)
                
                # Update best params to most robust
                st.session_state.best_params = aggregated_results[0]['params_dict']
                
                st.markdown("---")
            
            st.markdown("### 🏆 Best Single Result (Rank #1)")
            
            best = sorted_results[0]
            best_params = best['params_dict']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Value", f"${best['Final Value']:,.2f}")
            with col2:
                st.metric("Total Return", f"{best['Total Return %']:.2f}%")
            with col3:
                st.metric("vs QQQ", f"{best['vs QQQ %']:+.2f}%", 
                         "✅ Winning" if best['vs QQQ %'] > 0 else "❌ Losing")
            with col4:
                st.metric("Sharpe Ratio", f"{best['Sharpe Ratio']:.2f}")
            
            # Show which parameters will be applied
            if test_multiple_periods and len(selected_periods) > 1 and 'aggregated_results' in locals():
                params_to_apply = aggregated_results[0]['params_dict']
                st.warning("**Note:** The 'Apply Best Parameters' button will use the **Most Robust Parameters** (shown above), not this single best result.")
            else:
                params_to_apply = best_params
            
            st.info(f"""
            **Configuration (Rank #1):**
            - **Strategy:** {'Double EMA Crossover' if best_params['use_double_ema'] else 'Single EMA'}
            - **EMA:** {f"{best_params['ema_fast']}/{best_params['ema_slow']}" if best_params['use_double_ema'] else best_params['ema_period']}
            - **RSI Filter:** {'Enabled (>' + str(best_params['rsi_threshold']) + ')' if best_params['use_rsi'] else 'Disabled'}
            - **Stop-Loss:** {str(best_params['stop_loss_pct']) + '%' if best_params['use_stop_loss'] else 'Disabled'}
            - **CAGR:** {best['CAGR %']:.2f}%
            - **Max Drawdown:** {best['Max Drawdown %']:.2f}%
            - **Total Trades:** {best['Trades']}
            """)
            
            # Show what will be applied
            st.success(f"""
            **Parameters that will be applied when you click 'Apply Best Parameters':**
            - **Strategy:** {'Double EMA Crossover' if params_to_apply['use_double_ema'] else 'Single EMA'}
            - **EMA:** {f"{params_to_apply['ema_fast']}/{params_to_apply['ema_slow']}" if params_to_apply['use_double_ema'] else params_to_apply['ema_period']}
            - **RSI Filter:** {'Enabled (>' + str(params_to_apply['rsi_threshold']) + ')' if params_to_apply['use_rsi'] else 'Disabled'}
            - **Stop-Loss:** {str(params_to_apply['stop_loss_pct']) + '%' if params_to_apply['use_stop_loss'] else 'Disabled'}
            """)
            
            # Comparison Chart
            st.markdown("---")
            st.markdown("### 📈 Top 10 Performance Comparison")
            
            fig_comparison = go.Figure()
            
            top_10 = sorted_results[:10]
            params_labels = [r['Parameters'] for r in top_10]
            strategy_returns = [r['Total Return %'] for r in top_10]
            qqq_returns = [r['QQQ Return %'] for r in top_10]
            
            fig_comparison.add_trace(go.Bar(
                name='Strategy',
                x=params_labels,
                y=strategy_returns,
                marker_color=['green' if r > qqq_returns[i] else 'red' for i, r in enumerate(strategy_returns)],
                text=[f'{r:.1f}%' for r in strategy_returns],
                textposition='outside'
            ))
            
            fig_comparison.add_trace(go.Scatter(
                name='QQQ Buy & Hold',
                x=params_labels,
                y=qqq_returns,
                mode='lines+markers',
                line=dict(color='orange', width=2, dash='dash'),
                marker=dict(size=8)
            ))
            
            fig_comparison.update_layout(
                title='Top 10 Strategies vs QQQ Buy & Hold',
                xaxis_title='Parameters',
                yaxis_title='Total Return %',
                height=500,
                xaxis={'tickangle': -45},
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Apply Best Parameters Button
            st.markdown("---")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("✅ Apply and Run Further Testing", type="primary", use_container_width=True):
                    # Store best params in session state
                    if test_multiple_periods and len(selected_periods) > 1 and 'aggregated_results' in locals():
                        # Use most robust parameters
                        st.session_state.best_params = aggregated_results[0]['params_dict'].copy()
                    else:
                        # Use single best result
                        st.session_state.best_params = sorted_results[0]['params_dict'].copy()
                    
                    # Set flag to show navigation banner
                    st.session_state.navigate_to_step3 = True
                    st.success("✅ **Parameters Applied!** → Now click on **'📊 Step 3: Testing'** tab above to run your strategy.")
                    st.info("💡 Your optimized parameters have been saved and will be used in Step 3, you may now go to Step 3 and test this further.")
            
            with col2:
                if st.button("🔄 Clear", use_container_width=True):
                    st.session_state.best_params = None
                    st.session_state.grid_search_results = None
                    st.success("Cleared!")
                    rerun()
    
    # === MANUAL CONFIGURATION SECTION ===
    with config_tab2:
        st.markdown("### 📝 Manual Configuration")
        
        # Apply best params if available
        if st.session_state.best_params is not None:
            default_double_ema = st.session_state.best_params.get('use_double_ema', False)
            default_ema_period = st.session_state.best_params.get('ema_period', 50)
            default_ema_fast = st.session_state.best_params.get('ema_fast', 9)
            default_ema_slow = st.session_state.best_params.get('ema_slow', 21)
            default_use_rsi = st.session_state.best_params.get('use_rsi', True)
            default_rsi_threshold = st.session_state.best_params.get('rsi_threshold', 50)
            default_use_stop_loss = st.session_state.best_params.get('use_stop_loss', True)
            default_stop_loss_pct = st.session_state.best_params.get('stop_loss_pct', 10)
            
            # Display what was applied
            st.success("✅ Using optimized parameters from grid search:")
            
            param_summary = []
            if default_double_ema:
                param_summary.append(f"**EMA:** Double Crossover ({default_ema_fast}/{default_ema_slow})")
            else:
                param_summary.append(f"**EMA:** Single ({default_ema_period})")
            
            if default_use_rsi:
                param_summary.append(f"**RSI:** Enabled (>{default_rsi_threshold})")
            else:
                param_summary.append(f"**RSI:** Disabled")
            
            if default_use_stop_loss:
                param_summary.append(f"**Stop-Loss:** {default_stop_loss_pct}%")
            else:
                param_summary.append(f"**Stop-Loss:** Disabled")
            
            st.info(" | ".join(param_summary))
            st.caption("You can modify these parameters below if needed.")
        else:
            default_double_ema = False
            default_ema_period = 50
            default_ema_fast = 9
            default_ema_slow = 21
            default_use_rsi = True
            default_rsi_threshold = 50
            default_use_stop_loss = True
            default_stop_loss_pct = 10
            
            st.info("Using default parameters. Run Grid Search to find optimized values.")
    
        # Condition 1: EMA Strategy
        st.markdown("**Condition 1: EMA Strategy**")
        
        use_double_ema = st.checkbox(
            "Use Double EMA Crossover",
            value=default_double_ema,
            help="Use two EMAs (fast/slow crossover) instead of single EMA"
        )

        if use_double_ema:
            col1, col2 = st.columns(2)
            with col1:
                ema_fast = st.number_input("Fast EMA", min_value=5, max_value=100, value=default_ema_fast, step=1)
            with col2:
                ema_slow = st.number_input("Slow EMA", min_value=10, max_value=200, value=default_ema_slow, step=1)
            ema_period = ema_slow
        else:
            ema_period = st.number_input("EMA Period", min_value=5, max_value=200, value=default_ema_period, step=5)
            ema_fast = 9
            ema_slow = 21

        # Condition 2: RSI Filter
        st.markdown("**Condition 2: RSI Filter (Optional)**")
        
        col1, col2 = st.columns(2)
        with col1:
            use_rsi = st.checkbox("Enable RSI Filter", value=default_use_rsi)
        with col2:
            if use_rsi:
                rsi_threshold = st.number_input("RSI Threshold", min_value=30, max_value=70, value=default_rsi_threshold, step=5)
            else:
                rsi_threshold = 0

        # Condition 3: Stop-Loss
        st.markdown("**Condition 3: Stop-Loss (Optional)**")
        
        col1, col2 = st.columns(2)
        with col1:
            use_stop_loss = st.checkbox("Enable Stop-Loss", value=default_use_stop_loss)
        with col2:
            if use_stop_loss:
                stop_loss_pct = st.number_input("Stop-Loss %", min_value=5, max_value=30, value=default_stop_loss_pct, step=1)
            else:
                stop_loss_pct = 0
    
        # === SIGNAL SUMMARY ===
        st.markdown("---")
        st.markdown("### 📋 Your Strategy Signals")
    
        # Build BUY signal
        buy_conditions = []
        if use_double_ema:
            buy_conditions.append(f"**Fast EMA ({ema_fast}d) > Slow EMA ({ema_slow}d)**")
        else:
            buy_conditions.append(f"**QQQ Price > {ema_period}-day EMA**")
        
        if use_rsi:
            buy_conditions.append(f"**RSI > {rsi_threshold}**")
        
        # Build SELL signal
        sell_conditions = []
        if use_double_ema:
            sell_conditions.append(f"**Fast EMA ({ema_fast}d) < Slow EMA ({ema_slow}d)**")
        else:
            sell_conditions.append(f"**QQQ Price < {ema_period}-day EMA**")
        
        if use_rsi:
            sell_conditions.append(f"**RSI ≤ {rsi_threshold}**")
        
        if use_stop_loss:
            sell_conditions.append(f"**Portfolio drops {stop_loss_pct}% from peak**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**🟢 BUY TQQQ Signal**")
            st.markdown("**When ALL conditions are met:**")
            for i, condition in enumerate(buy_conditions, 1):
                st.markdown(f"{i}. {condition}")
            st.caption("→ Enter or hold TQQQ position")
        
        with col2:
            st.error("**🔴 SELL TQQQ Signal**")
            st.markdown("**When ANY condition is met:**")
            for i, condition in enumerate(sell_conditions, 1):
                st.markdown(f"{i}. {condition}")
            st.caption("→ Exit TQQQ and move to cash")
        
        # Apply and Run Further Testing Button
        st.markdown("---")
        if st.button("✅ Apply and Run Further Testing", type="primary", use_container_width=True, key="manual_apply"):
            # Store manual params in session state
            st.session_state.best_params = {
                'use_double_ema': use_double_ema,
                'ema_period': ema_period,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'use_rsi': use_rsi,
                'rsi_threshold': rsi_threshold,
                'use_stop_loss': use_stop_loss,
                'stop_loss_pct': stop_loss_pct
            }
            
            # Set flag to show navigation banner
            st.session_state.navigate_to_step3 = True
            st.success("✅ **Parameters Applied!** → Now click on **'📊 Step 3: Testing'** tab above to run your strategy.")
            st.info("💡 Your manual parameters have been saved and will be used in Step 3.")

with main_tab3:
    st.markdown("## 📊 Step 3: Testing")
    
    # Initialize best_params if not exists
    if 'best_params' not in st.session_state:
        st.session_state.best_params = None
    
    # Check if parameters are set
    if st.session_state.best_params is None:
        st.warning("⚠️ **No strategy parameters set!**")
        st.info("👈 Please go to **Step 2: Find Best Strategy** to run Grid Search and apply the best parameters, or use Manual Configuration to set your own parameters.")
        st.stop()
    
    # Clear navigation flag when user reaches Step 3 (after the stop check)
    if st.session_state.get('navigate_to_step3', False):
        st.session_state.navigate_to_step3 = False

    # Set default values if not defined (when in Step 3)
    if 'use_double_ema' not in locals():
        if st.session_state.best_params is not None:
            use_double_ema = st.session_state.best_params.get('use_double_ema', False)
            ema_period = st.session_state.best_params.get('ema_period', 50)
            ema_fast = st.session_state.best_params.get('ema_fast', 9)
            ema_slow = st.session_state.best_params.get('ema_slow', 21)
            use_rsi = st.session_state.best_params.get('use_rsi', True)
            rsi_threshold = st.session_state.best_params.get('rsi_threshold', 50)
            use_stop_loss = st.session_state.best_params.get('use_stop_loss', True)
            stop_loss_pct = st.session_state.best_params.get('stop_loss_pct', 10)
        else:
            use_double_ema = False
            ema_period = 50
            ema_fast = 9
            ema_slow = 21
            use_rsi = True
            rsi_threshold = 50
            use_stop_loss = True
            stop_loss_pct = 10

    # Display current strategy configuration
    st.markdown("### 🎯 Current Strategy Configuration")
    
    strategy_summary = []
    
    # EMA Strategy
    if use_double_ema:
        strategy_summary.append(f"**EMA:** Double Crossover ({ema_fast}/{ema_slow})")
    else:
        strategy_summary.append(f"**EMA:** Single ({ema_period})")
    
    # RSI Filter
    if use_rsi:
        strategy_summary.append(f"**RSI:** Enabled (>{rsi_threshold})")
    else:
        strategy_summary.append(f"**RSI:** Disabled")
    
    # Stop-Loss
    if use_stop_loss:
        strategy_summary.append(f"**Stop-Loss:** {stop_loss_pct}%")
    else:
        strategy_summary.append(f"**Stop-Loss:** Disabled")
    
    st.info(" | ".join(strategy_summary))
    
    st.markdown("---")
    
    # Create tabs for different analysis types
    test_tab1, test_tab2, test_tab3 = st.tabs(["📊 Daily Signal", "📈 Custom Simulation", "🎲 Monte Carlo Simulation"])
    
    # Initialize variables
    run_button_daily = False
    run_button_custom = False
    run_button_monte = False
    
    # Initialize last_analysis_tab to track tab changes
    if 'last_analysis_tab' not in st.session_state:
        st.session_state.last_analysis_tab = None
    if 'clear_results' not in st.session_state:
        st.session_state.clear_results = False
    
    # === DAILY SIGNAL TAB ===
    with test_tab1:
        st.markdown("### 📊 Daily Signal")
        st.info("✅ Get today's buy/sell signal based on your strategy. Check at 3:55 PM ET before market close.")
        
        st.markdown("---")
        
        run_button_daily = st.button("🔔 Execute - Get Today's Signal", type="primary", use_container_width=True, key="daily_signal_btn")
    
    # === CUSTOM SIMULATION TAB ===
    with test_tab2:
        st.markdown("### 📈 Custom Simulation")
        st.info("Backtest your strategy on historical data with detailed performance analysis.")
        
        st.markdown("**Parameters:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.date(2020, 1, 1), min_value=datetime.date(2010, 3, 31), max_value=datetime.date.today())
        with col2:
            end_date = st.date_input("End Date", value=datetime.date.today(), min_value=datetime.date(2010, 3, 31), max_value=datetime.date.today())
        with col3:
            initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
        
        st.markdown("---")
        
        run_button_custom = st.button("🚀 Execute - Run Custom Simulation", type="primary", use_container_width=True, key="custom_sim_btn")
    
    # === MONTE CARLO TAB ===
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
    
    # Determine which button was clicked and set analysis type
    if run_button_daily:
        run_button = True
        analysis_type = "📊 Daily Signal"
        current_tab = "daily"
    elif run_button_custom:
        run_button = True
        analysis_type = "📈 Custom Simulation"
        current_tab = "custom"
    elif run_button_monte:
        run_button = True
        analysis_type = "🎲 Monte Carlo Simulation"
        current_tab = "monte"
        # Use Monte Carlo specific variables
        start_date = start_date_mc
        end_date = end_date_mc
        initial_capital = initial_capital_mc
    else:
        run_button = False
        analysis_type = "📊 Daily Signal"
        current_tab = None
    
    # Check if tab changed (clear results if switching tabs)
    # Only set clear flag if we're switching tabs AND not executing a new analysis
    if current_tab is not None and st.session_state.last_analysis_tab != current_tab and not run_button:
        st.session_state.clear_results = True
    
    # Update last tab when button is clicked
    if run_button and current_tab is not None:
        st.session_state.last_analysis_tab = current_tab
        st.session_state.clear_results = False  # Clear the flag when running new analysis
    
    # ============================================================================
    # RESULTS SECTION (INSIDE STEP 3 TAB)
    # ============================================================================
    
    # Only show results if button was clicked and not clearing results
    if run_button and not st.session_state.get('clear_results', False):
        st.markdown("---")
        st.markdown("## 📊 Results")
        st.markdown("---")
        
        # --- DAILY SIGNAL ---
        if analysis_type == "📊 Daily Signal":
            with st.spinner("Fetching latest market data..."):
                try:
                    session = requests.Session(impersonate="chrome110", verify=False)
                    buffer_days = max(365, ema_period + 100)
                    fetch_start = (datetime.date.today() - datetime.timedelta(days=buffer_days)).strftime('%Y-%m-%d')
                    
                    historical_data = yf.download(["QQQ", "TQQQ"], start=fetch_start, end=datetime.date.today(), 
                                                 session=session, auto_adjust=False, group_by='ticker', progress=False)
                    
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
                    
                    if use_double_ema:
                        qqq_data = calculate_double_ema(qqq_data, ema_fast, ema_slow)
                    else:
                        qqq_data = calculate_ema(qqq_data, ema_period)
                    qqq_data = calculate_rsi(qqq_data, period=14)
                    
                    latest_date = qqq_data.index[-1]
                    latest_qqq_close = qqq_data.iloc[-1]['Close']
                    
                    if use_double_ema:
                        latest_ema_fast = qqq_data.iloc[-1]['EMA_Fast']
                        latest_ema_slow = qqq_data.iloc[-1]['EMA_Slow']
                        latest_ema = latest_ema_slow
                    else:
                        latest_ema = qqq_data.iloc[-1]['EMA']
                        latest_ema_fast = None
                        latest_ema_slow = None
                    
                    latest_rsi = qqq_data.iloc[-1]['RSI']
                    latest_tqqq_close = current_tqqq_price
                    
                    is_today = latest_date.date() == datetime.date.today()
                    
                    if use_double_ema:
                        base_signal = 'BUY' if latest_ema_fast > latest_ema_slow else 'SELL'
                    else:
                        base_signal = 'BUY' if latest_qqq_close > latest_ema else 'SELL'
                    
                    if use_rsi:
                        rsi_ok = pd.notna(latest_rsi) and latest_rsi > rsi_threshold
                        final_signal = 'BUY' if base_signal == 'BUY' and rsi_ok else 'SELL'
                    else:
                        final_signal = base_signal
                    
                    st.markdown(f"### 📅 Signal for {latest_date.strftime('%Y-%m-%d')}")
                    
                    if final_signal == 'BUY':
                        st.success("### 🟢 BUY TQQQ (or HOLD if already in)")
                    else:
                        st.error("### 🔴 SELL TQQQ / STAY CASH")
                    
                    market_state = qqq_current_info.get('marketState', 'UNKNOWN')
                    if market_state == 'REGULAR':
                        st.success(f"🟢 Market: OPEN")
                    elif market_state == 'CLOSED':
                        st.error(f"🔴 Market: CLOSED")
                    else:
                        st.warning(f"🟡 Market: {market_state}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("QQQ Price", f"${latest_qqq_close:.2f}")
                        st.metric("TQQQ Price", f"${latest_tqqq_close:.2f}")
                    
                    with col2:
                        if use_double_ema:
                            st.metric(f"Fast EMA ({ema_fast}d)", f"${latest_ema_fast:.2f}")
                            st.metric(f"Slow EMA ({ema_slow}d)", f"${latest_ema_slow:.2f}")
                        else:
                            st.metric(f"{ema_period}-day EMA", f"${latest_ema:.2f}")
                    
                    with col3:
                        if use_rsi:
                            st.metric("RSI (14-day)", f"{latest_rsi:.1f}")
                        else:
                            st.info("RSI: Disabled")
                    
                    if not is_today:
                        st.warning("⚠️ Using stale data. Try again during market hours.")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # --- CUSTOM SIMULATION (Combined Backtest & Historical Analysis) ---
        elif analysis_type == "📈 Custom Simulation":
            with st.spinner("Running custom simulation..."):
                tickers = ["QQQ", "TQQQ"]
                raw_data = get_data(tickers, start_date, end_date, buffer_days=max(365, ema_period + 100))
                
                qqq = raw_data["QQQ"].copy()
                tqqq = raw_data["TQQQ"].copy()
                
                result = run_tqqq_only_strategy(qqq.copy(), tqqq.copy(), start_date, end_date, initial_capital, ema_period, rsi_threshold, use_rsi, stop_loss_pct, use_stop_loss, use_double_ema, ema_fast, ema_slow)
                
                # Calculate QQQ benchmark
                qqq_start = qqq.loc[start_date:end_date].iloc[0]['Close']
                qqq_end = qqq.loc[start_date:end_date].iloc[-1]['Close']
                qqq_bh_value = (qqq_end / qqq_start) * initial_capital
                qqq_bh_return = ((qqq_bh_value - initial_capital) / initial_capital) * 100
                
                # Prepare data for historical analysis
                portfolio_df = result['portfolio_df'].copy()
                portfolio_df['Year'] = portfolio_df.index.year
                portfolio_df['Quarter'] = portfolio_df.index.quarter
                portfolio_df['YearQuarter'] = portfolio_df['Year'].astype(str) + '-Q' + portfolio_df['Quarter'].astype(str)
                
                qqq_benchmark = qqq.loc[start_date:end_date]['Close'].copy()
                qqq_benchmark_df = pd.DataFrame({'Close': qqq_benchmark})
                qqq_benchmark_df['Value'] = (qqq_benchmark_df['Close'] / qqq_benchmark_df['Close'].iloc[0]) * initial_capital
                qqq_benchmark_df['Year'] = qqq_benchmark_df.index.year
                qqq_benchmark_df['Quarter'] = qqq_benchmark_df.index.quarter
                qqq_benchmark_df['YearQuarter'] = qqq_benchmark_df['Year'].astype(str) + '-Q' + qqq_benchmark_df['Quarter'].astype(str)
            
            st.success("✅ Custom simulation complete!")
            
            # === SECTION 1: PERFORMANCE SUMMARY ===
            with st.expander("📊 Performance Summary", expanded=True):
                days = (end_date - start_date).days
                years = days / 365.25
                cagr_strategy = ((result['final_value'] / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
                cagr_qqq = ((qqq_bh_value / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
                
                st.markdown("**Strategy Performance:**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Final Value", f"${result['final_value']:,.2f}", f"{result['total_return_pct']:.2f}%")
                with col2:
                    st.metric("CAGR", f"{cagr_strategy:.2f}%")
                with col3:
                    st.metric("Max Drawdown", f"{result['max_drawdown']:.2f}%")
                with col4:
                    st.metric("Total Trades", result['num_trades'])
                
                st.markdown("**QQQ Buy & Hold Comparison:**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("QQQ Final Value", f"${qqq_bh_value:,.2f}", f"{qqq_bh_return:.2f}%")
                with col2:
                    st.metric("QQQ CAGR", f"{cagr_qqq:.2f}%")
                with col3:
                    outperformance = result['total_return_pct'] - qqq_bh_return
                    st.metric("Outperformance", f"{outperformance:+.2f}%", 
                             "✅ Winning" if outperformance > 0 else "❌ Losing")
                with col4:
                    value_diff = result['final_value'] - qqq_bh_value
                    st.metric("Value Difference", f"${value_diff:+,.2f}")
            
            # === SECTION 2: PERFORMANCE CHART ===
            with st.expander("📉 Performance Chart", expanded=True):
                fig = create_performance_chart(result, qqq, tqqq, start_date, initial_capital, ema_period)
                st.plotly_chart(fig, use_container_width=True)
            
            # === SECTION 3: YEARLY PERFORMANCE ===
            with st.expander("📅 Yearly Performance vs QQQ", expanded=True):
                yearly_stats = []
                for year in sorted(portfolio_df['Year'].unique()):
                    year_data = portfolio_df[portfolio_df['Year'] == year]
                    qqq_year_data = qqq_benchmark_df[qqq_benchmark_df['Year'] == year]
                    
                    if len(year_data) == 0:
                        continue
                    
                    start_value = year_data['Value'].iloc[0]
                    end_value = year_data['Value'].iloc[-1]
                    year_return = ((end_value - start_value) / start_value) * 100
                    year_max_dd = year_data['Drawdown'].min()
                    
                    qqq_start = qqq_year_data['Value'].iloc[0] if len(qqq_year_data) > 0 else 0
                    qqq_end = qqq_year_data['Value'].iloc[-1] if len(qqq_year_data) > 0 else 0
                    qqq_year_return = ((qqq_end - qqq_start) / qqq_start) * 100 if qqq_start > 0 else 0
                    
                    outperformance = year_return - qqq_year_return
                    
                    yearly_stats.append({
                        'Year': year,
                        'Strategy Return': f'{year_return:.2f}%',
                        'QQQ Return': f'{qqq_year_return:.2f}%',
                        'Outperformance': f'{outperformance:+.2f}%',
                        'Strategy Value': f'${end_value:,.2f}',
                        'QQQ Value': f'${qqq_end:,.2f}',
                        'Max Drawdown': f'{year_max_dd:.2f}%',
                        'Result': '✅ Win' if outperformance > 0 else '❌ Loss'
                    })
                
                yearly_df = pd.DataFrame(yearly_stats)
                st.dataframe(yearly_df, use_container_width=True, hide_index=True)
                
                yearly_wins = sum(1 for stat in yearly_stats if float(stat['Outperformance'].replace('%', '').replace('+', '')) > 0)
                yearly_total = len(yearly_stats)
                yearly_win_rate = (yearly_wins / yearly_total * 100) if yearly_total > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Years", yearly_total)
                with col2:
                    st.metric("Years Won", f"{yearly_wins}", f"{yearly_win_rate:.1f}% win rate")
                with col3:
                    st.metric("Years Lost", f"{yearly_total - yearly_wins}")
                with col4:
                    avg_outperformance = sum([float(stat['Outperformance'].replace('%', '').replace('+', '')) for stat in yearly_stats]) / yearly_total if yearly_total > 0 else 0
                    st.metric("Avg Outperformance", f"{avg_outperformance:+.2f}%")
                
                fig_yearly = go.Figure()
                yearly_returns = [float(stat['Strategy Return'].replace('%', '')) for stat in yearly_stats]
                qqq_yearly_returns = [float(stat['QQQ Return'].replace('%', '')) for stat in yearly_stats]
                years = [stat['Year'] for stat in yearly_stats]
                
                fig_yearly.add_trace(go.Bar(
                    name='Strategy',
                    x=years,
                    y=yearly_returns,
                    marker_color=['green' if r > 0 else 'red' for r in yearly_returns],
                    text=[f'{r:.1f}%' for r in yearly_returns],
                    textposition='outside'
                ))
                
                fig_yearly.add_trace(go.Bar(
                    name='QQQ Buy & Hold',
                    x=years,
                    y=qqq_yearly_returns,
                    marker_color=['lightgreen' if r > 0 else 'lightcoral' for r in qqq_yearly_returns],
                    text=[f'{r:.1f}%' for r in qqq_yearly_returns],
                    textposition='outside'
                ))
                
                fig_yearly.update_layout(
                    title='Yearly Returns: Strategy vs QQQ (%)',
                    xaxis_title='Year',
                    yaxis_title='Return %',
                    height=400,
                    barmode='group'
                )
                
                st.plotly_chart(fig_yearly, use_container_width=True)
            
            # === SECTION 4: QUARTERLY PERFORMANCE ===
            with st.expander("📊 Quarterly Performance vs QQQ", expanded=False):
                quarterly_stats = []
                for yq in sorted(portfolio_df['YearQuarter'].unique()):
                    quarter_data = portfolio_df[portfolio_df['YearQuarter'] == yq]
                    qqq_quarter_data = qqq_benchmark_df[qqq_benchmark_df['YearQuarter'] == yq]
                    
                    if len(quarter_data) == 0:
                        continue
                    
                    start_value = quarter_data['Value'].iloc[0]
                    end_value = quarter_data['Value'].iloc[-1]
                    quarter_return = ((end_value - start_value) / start_value) * 100
                    quarter_max_dd = quarter_data['Drawdown'].min()
                    
                    qqq_start = qqq_quarter_data['Value'].iloc[0] if len(qqq_quarter_data) > 0 else 0
                    qqq_end = qqq_quarter_data['Value'].iloc[-1] if len(qqq_quarter_data) > 0 else 0
                    qqq_quarter_return = ((qqq_end - qqq_start) / qqq_start) * 100 if qqq_start > 0 else 0
                    
                    outperformance = quarter_return - qqq_quarter_return
                    
                    quarterly_stats.append({
                        'Period': yq,
                        'Strategy Return': f'{quarter_return:.2f}%',
                        'QQQ Return': f'{qqq_quarter_return:.2f}%',
                        'Outperformance': f'{outperformance:+.2f}%',
                        'Strategy Value': f'${end_value:,.2f}',
                        'QQQ Value': f'${qqq_end:,.2f}',
                        'Max Drawdown': f'{quarter_max_dd:.2f}%',
                        'Result': '✅ Win' if outperformance > 0 else '❌ Loss'
                    })
                
                quarterly_df = pd.DataFrame(quarterly_stats)
                st.dataframe(quarterly_df, use_container_width=True, hide_index=True)
                
                quarterly_wins = sum(1 for stat in quarterly_stats if float(stat['Outperformance'].replace('%', '').replace('+', '')) > 0)
                quarterly_total = len(quarterly_stats)
                quarterly_win_rate = (quarterly_wins / quarterly_total * 100) if quarterly_total > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Quarters", quarterly_total)
                with col2:
                    st.metric("Quarters Won", f"{quarterly_wins}", f"{quarterly_win_rate:.1f}% win rate")
                with col3:
                    st.metric("Quarters Lost", f"{quarterly_total - quarterly_wins}")
                with col4:
                    avg_q_outperformance = sum([float(stat['Outperformance'].replace('%', '').replace('+', '')) for stat in quarterly_stats]) / quarterly_total if quarterly_total > 0 else 0
                    st.metric("Avg Outperformance", f"{avg_q_outperformance:+.2f}%")
                
                fig_quarterly = go.Figure()
                quarterly_returns = [float(stat['Strategy Return'].replace('%', '')) for stat in quarterly_stats]
                qqq_quarterly_returns = [float(stat['QQQ Return'].replace('%', '')) for stat in quarterly_stats]
                quarters = [stat['Period'] for stat in quarterly_stats]
                
                fig_quarterly.add_trace(go.Bar(
                    name='Strategy',
                    x=quarters,
                    y=quarterly_returns,
                    marker_color=['green' if r > 0 else 'red' for r in quarterly_returns],
                    text=[f'{r:.1f}%' for r in quarterly_returns],
                    textposition='outside'
                ))
                
                fig_quarterly.add_trace(go.Bar(
                    name='QQQ Buy & Hold',
                    x=quarters,
                    y=qqq_quarterly_returns,
                    marker_color=['lightgreen' if r > 0 else 'lightcoral' for r in qqq_quarterly_returns],
                    text=[f'{r:.1f}%' for r in qqq_quarterly_returns],
                    textposition='outside'
                ))
                
                fig_quarterly.update_layout(
                    title='Quarterly Returns: Strategy vs QQQ (%)',
                    xaxis_title='Quarter',
                    yaxis_title='Return %',
                    height=400,
                    barmode='group',
                    xaxis={'tickangle': -45}
                )
                
                st.plotly_chart(fig_quarterly, use_container_width=True)
            
            # === SECTION 5: TRADE LOG ===
            with st.expander("📋 Trade Log", expanded=False):
                trade_df = pd.DataFrame(result['trade_log'])
                show_all = st.checkbox("Show all days", value=False)
                
                if show_all:
                    display_df = trade_df
                else:
                    display_df = trade_df[trade_df['Action'].isin(['BUY TQQQ', 'SELL to CASH', 'SELL (STOP-LOSS)'])]
                
                st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Trade Log CSV",
                        trade_df.to_csv(index=False),
                        "trade_log.csv",
                        "text/csv",
                        use_container_width=True
                    )
                with col2:
                    st.download_button(
                        "📥 Download Yearly Stats CSV",
                        yearly_df.to_csv(index=False),
                        "yearly_performance.csv",
                        "text/csv",
                        use_container_width=True
                    )
        
        # --- MONTE CARLO SIMULATION ---
        else:  # Monte Carlo
            with st.spinner(f"Running {num_simulations:,} simulations..."):
                tickers = ["QQQ", "TQQQ"]
                raw_data = get_data(tickers, start_date, end_date, buffer_days=max(365, ema_period + 100))
                
                qqq = raw_data["QQQ"].copy()
                tqqq = raw_data["TQQQ"].copy()
                
                # Run strategy to get historical returns
                result = run_tqqq_only_strategy(qqq.copy(), tqqq.copy(), start_date, end_date, 
                                               initial_capital, ema_period, rsi_threshold, use_rsi, stop_loss_pct, use_stop_loss, use_double_ema, ema_fast, ema_slow)
                
                portfolio_df = result['portfolio_df'].copy()
                portfolio_df['Daily_Return'] = portfolio_df['Value'].pct_change()
                daily_returns = portfolio_df['Daily_Return'].dropna().values
                
                if len(daily_returns) < 30:
                    st.error("Not enough historical data. Please select a longer period.")
                    st.stop()
                
                # Calculate QQQ daily returns for baseline simulation
                qqq_benchmark = qqq.loc[start_date:end_date]['Close'].copy()
                qqq_returns = qqq_benchmark.pct_change().dropna().values
                
                # Run Monte Carlo simulations for STRATEGY
                np.random.seed(42)
                simulations = np.zeros((num_simulations, simulation_days + 1))
                simulations[:, 0] = initial_capital
                
                for sim in range(num_simulations):
                    for day in range(1, simulation_days + 1):
                        random_return = np.random.choice(daily_returns)
                        simulations[sim, day] = simulations[sim, day - 1] * (1 + random_return)
                
                # Run Monte Carlo simulations for QQQ BASELINE
                np.random.seed(42)  # Same seed for fair comparison
                qqq_simulations = np.zeros((num_simulations, simulation_days + 1))
                qqq_simulations[:, 0] = initial_capital
                
                for sim in range(num_simulations):
                    for day in range(1, simulation_days + 1):
                        random_return = np.random.choice(qqq_returns)
                        qqq_simulations[sim, day] = qqq_simulations[sim, day - 1] * (1 + random_return)
                
                # Calculate statistics for STRATEGY
                final_values = simulations[:, -1]
                final_returns = ((final_values - initial_capital) / initial_capital) * 100
                
                mean_final_value = np.mean(final_values)
                median_final_value = np.median(final_values)
                
                lower_percentile = (100 - confidence_level) / 2
                upper_percentile = 100 - lower_percentile
                
                ci_lower = np.percentile(final_values, lower_percentile)
                ci_upper = np.percentile(final_values, upper_percentile)
                
                prob_profit = (final_values > initial_capital).sum() / num_simulations * 100
                
                # Calculate statistics for QQQ
                qqq_final_values = qqq_simulations[:, -1]
                qqq_final_returns = ((qqq_final_values - initial_capital) / initial_capital) * 100
                
                qqq_mean_final_value = np.mean(qqq_final_values)
                qqq_median_final_value = np.median(qqq_final_values)
                
                qqq_ci_lower = np.percentile(qqq_final_values, lower_percentile)
                qqq_ci_upper = np.percentile(qqq_final_values, upper_percentile)
                
                qqq_prob_profit = (qqq_final_values > initial_capital).sum() / num_simulations * 100
                
                # Calculate outperformance
                outperformance = final_values - qqq_final_values
                prob_outperform = (outperformance > 0).sum() / num_simulations * 100
                mean_outperformance = np.mean(outperformance)
                median_outperformance = np.median(outperformance)
            
            st.success(f"✅ Completed {num_simulations:,} simulations!")
            
            # Comparison Summary
            st.subheader("📊 Strategy vs QQQ Comparison")
            
            comparison_data = {
                'Metric': [
                    'Mean Final Value',
                    'Median Final Value',
                    'Probability of Profit',
                    f'{confidence_level}% CI Lower',
                    f'{confidence_level}% CI Upper',
                    'Mean Return %',
                    'Median Return %'
                ],
                'Strategy': [
                    f'${mean_final_value:,.2f}',
                    f'${median_final_value:,.2f}',
                    f'{prob_profit:.1f}%',
                    f'${ci_lower:,.2f}',
                    f'${ci_upper:,.2f}',
                    f'{((mean_final_value - initial_capital) / initial_capital * 100):.2f}%',
                    f'{((median_final_value - initial_capital) / initial_capital * 100):.2f}%'
                ],
                'QQQ Buy & Hold': [
                    f'${qqq_mean_final_value:,.2f}',
                    f'${qqq_median_final_value:,.2f}',
                    f'{qqq_prob_profit:.1f}%',
                    f'${qqq_ci_lower:,.2f}',
                    f'${qqq_ci_upper:,.2f}',
                    f'{((qqq_mean_final_value - initial_capital) / initial_capital * 100):.2f}%',
                    f'{((qqq_median_final_value - initial_capital) / initial_capital * 100):.2f}%'
                ],
                'Difference': [
                    f'${mean_final_value - qqq_mean_final_value:+,.2f}',
                    f'${median_final_value - qqq_median_final_value:+,.2f}',
                    f'{prob_profit - qqq_prob_profit:+.1f}%',
                    f'${ci_lower - qqq_ci_lower:+,.2f}',
                    f'${ci_upper - qqq_ci_upper:+,.2f}',
                    f'{((mean_final_value - initial_capital) / initial_capital * 100) - ((qqq_mean_final_value - initial_capital) / initial_capital * 100):+.2f}%',
                    f'{((median_final_value - initial_capital) / initial_capital * 100) - ((qqq_median_final_value - initial_capital) / initial_capital * 100):+.2f}%'
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Outperformance Metrics
            st.markdown("---")
            st.subheader("🎯 Outperformance Analysis")
            
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
            
            st.markdown("---")
            st.subheader("📉 Simulation Paths Comparison")
            
            fig_paths = go.Figure()
            
            # Sample paths for STRATEGY
            sample_size = min(50, num_simulations)
            sample_indices = np.random.choice(num_simulations, sample_size, replace=False)
            
            for idx in sample_indices:
                fig_paths.add_trace(go.Scatter(
                    x=list(range(simulation_days + 1)),
                    y=simulations[idx, :],
                    mode='lines',
                    line=dict(width=0.5, color='lightblue'),
                    showlegend=False,
                    hoverinfo='skip',
                    opacity=0.3
                ))
            
            # Sample paths for QQQ
            for idx in sample_indices:
                fig_paths.add_trace(go.Scatter(
                    x=list(range(simulation_days + 1)),
                    y=qqq_simulations[idx, :],
                    mode='lines',
                    line=dict(width=0.5, color='lightcoral'),
                    showlegend=False,
                    hoverinfo='skip',
                    opacity=0.3
                ))
            
            # Mean paths
            mean_path = np.mean(simulations, axis=0)
            qqq_mean_path = np.mean(qqq_simulations, axis=0)
            
            fig_paths.add_trace(go.Scatter(
                x=list(range(simulation_days + 1)),
                y=mean_path,
                mode='lines',
                name='Strategy Mean',
                line=dict(width=3, color='blue')
            ))
            
            fig_paths.add_trace(go.Scatter(
                x=list(range(simulation_days + 1)),
                y=qqq_mean_path,
                mode='lines',
                name='QQQ Mean',
                line=dict(width=3, color='red')
            ))
            
            # Median paths
            median_path = np.median(simulations, axis=0)
            qqq_median_path = np.median(qqq_simulations, axis=0)
            
            fig_paths.add_trace(go.Scatter(
                x=list(range(simulation_days + 1)),
                y=median_path,
                mode='lines',
                name='Strategy Median',
                line=dict(width=3, color='green')
            ))
            
            fig_paths.add_trace(go.Scatter(
                x=list(range(simulation_days + 1)),
                y=qqq_median_path,
                mode='lines',
                name='QQQ Median',
                line=dict(width=3, color='orange')
            ))
            
            # Initial capital line
            fig_paths.add_trace(go.Scatter(
                x=[0, simulation_days],
                y=[initial_capital, initial_capital],
                mode='lines',
                name='Initial Capital',
                line=dict(width=2, color='black', dash='dot')
            ))
            
            fig_paths.update_layout(
                title=f'Monte Carlo: Strategy vs QQQ ({num_simulations:,} Simulations)',
                xaxis_title='Days',
                yaxis_title='Portfolio Value ($)',
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_paths, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📊 Distribution Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Strategy Final Values**")
                fig_dist_strategy = go.Figure()
                
                fig_dist_strategy.add_trace(go.Histogram(
                    x=final_values,
                    nbinsx=50,
                    marker_color='lightblue',
                    opacity=0.7,
                    name='Strategy'
                ))
                
                fig_dist_strategy.add_vline(x=initial_capital, line_dash="dot", line_color="black", annotation_text="Initial")
                fig_dist_strategy.add_vline(x=mean_final_value, line_dash="dash", line_color="blue", annotation_text="Mean")
                fig_dist_strategy.add_vline(x=median_final_value, line_dash="dash", line_color="green", annotation_text="Median")
                
                fig_dist_strategy.update_layout(
                    title='Strategy Distribution',
                    xaxis_title='Final Value ($)',
                    yaxis_title='Frequency',
                    height=400
                )
                
                st.plotly_chart(fig_dist_strategy, use_container_width=True)
            
            with col2:
                st.markdown("**QQQ Buy & Hold Final Values**")
                fig_dist_qqq = go.Figure()
                
                fig_dist_qqq.add_trace(go.Histogram(
                    x=qqq_final_values,
                    nbinsx=50,
                    marker_color='lightcoral',
                    opacity=0.7,
                    name='QQQ'
                ))
                
                fig_dist_qqq.add_vline(x=initial_capital, line_dash="dot", line_color="black", annotation_text="Initial")
                fig_dist_qqq.add_vline(x=qqq_mean_final_value, line_dash="dash", line_color="red", annotation_text="Mean")
                fig_dist_qqq.add_vline(x=qqq_median_final_value, line_dash="dash", line_color="orange", annotation_text="Median")
                
                fig_dist_qqq.update_layout(
                    title='QQQ Distribution',
                    xaxis_title='Final Value ($)',
                    yaxis_title='Frequency',
                    height=400
                )
                
                st.plotly_chart(fig_dist_qqq, use_container_width=True)
            
            # Outperformance Distribution
            st.markdown("---")
            st.subheader("📈 Outperformance Distribution")
            
            fig_outperf = go.Figure()
            
            fig_outperf.add_trace(go.Histogram(
                x=outperformance,
                nbinsx=50,
                marker_color=['green' if x > 0 else 'red' for x in outperformance],
                opacity=0.7
            ))
            
            fig_outperf.add_vline(x=0, line_dash="solid", line_color="black", annotation_text="Break Even", line_width=2)
            fig_outperf.add_vline(x=mean_outperformance, line_dash="dash", line_color="blue", annotation_text=f"Mean: ${mean_outperformance:+,.0f}")
            fig_outperf.add_vline(x=median_outperformance, line_dash="dash", line_color="green", annotation_text=f"Median: ${median_outperformance:+,.0f}")
            
            fig_outperf.update_layout(
                title='Distribution of Outperformance (Strategy - QQQ)',
                xaxis_title='Outperformance ($)',
                yaxis_title='Frequency',
                height=400
            )
            
            st.plotly_chart(fig_outperf, use_container_width=True)
            
            st.info(f"""
            **Interpretation:**
            - Values > $0: Strategy outperforms QQQ
            - Values < $0: QQQ outperforms Strategy
            - **{prob_outperform:.1f}%** of simulations show strategy outperforming QQQ
            - **{100 - prob_outperform:.1f}%** of simulations show QQQ outperforming strategy
            """)
            
            # Percentile Comparison Table
            st.markdown("---")
            st.subheader("📋 Percentile Comparison")
            
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            percentile_data = []
            
            for p in percentiles:
                strategy_value = np.percentile(final_values, p)
                qqq_value = np.percentile(qqq_final_values, p)
                diff = strategy_value - qqq_value
                
                percentile_data.append({
                    'Percentile': f'{p}th',
                    'Strategy Value': f'${strategy_value:,.2f}',
                    'QQQ Value': f'${qqq_value:,.2f}',
                    'Difference': f'${diff:+,.2f}',
                    'Winner': '✅ Strategy' if diff > 0 else '❌ QQQ'
                })
            
            percentile_df = pd.DataFrame(percentile_data)
            st.dataframe(percentile_df, use_container_width=True, hide_index=True)
            
            # Key Insights
            st.markdown("---")
            st.subheader("💡 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Strategy Performance:**")
                st.info(f"""
                - **Median Outcome:** ${median_final_value:,.2f}
                - **Mean Outcome:** ${mean_final_value:,.2f}
                - **{confidence_level}% Range:** ${ci_lower:,.2f} - ${ci_upper:,.2f}
                - **Profit Probability:** {prob_profit:.1f}%
                """)
            
            with col2:
                st.markdown("**QQQ Baseline:**")
                st.info(f"""
                - **Median Outcome:** ${qqq_median_final_value:,.2f}
                - **Mean Outcome:** ${qqq_mean_final_value:,.2f}
                - **{confidence_level}% Range:** ${qqq_ci_lower:,.2f} - ${qqq_ci_upper:,.2f}
                - **Profit Probability:** {qqq_prob_profit:.1f}%
                """)
            
            # Final Recommendation
            if prob_outperform > 60:
                st.success(f"""
                ✅ **Strong Strategy Performance**
                
                Your strategy has a {prob_outperform:.1f}% probability of outperforming QQQ Buy & Hold.
                Expected outperformance: ${median_outperformance:+,.2f} (median)
                
                This suggests your strategy adds value over simply buying and holding QQQ.
                """)
            elif prob_outperform > 40:
                st.warning(f"""
                ⚠️ **Moderate Strategy Performance**
                
                Your strategy has a {prob_outperform:.1f}% probability of outperforming QQQ Buy & Hold.
                Expected outperformance: ${median_outperformance:+,.2f} (median)
                
                Performance is mixed. Consider if the added complexity is worth the marginal difference.
                """)
            else:
                st.error(f"""
                ❌ **Underperforming Strategy**
                
                Your strategy has only a {prob_outperform:.1f}% probability of outperforming QQQ Buy & Hold.
                Expected underperformance: ${median_outperformance:+,.2f} (median)
                
                You may be better off simply buying and holding QQQ instead.
                """)
        
        # Disclaimer at the end of results
        st.markdown("---")
        st.caption("⚠️ **Disclaimer:** Educational purposes only. Past performance does not guarantee future results. Trading leveraged ETFs involves significant risk.")
