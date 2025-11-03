# /Users/pykarun/Documents/TradingStrategies/app_ma250.py

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from curl_cffi import requests
import plotly.graph_objects as go
import streamlit as st

# --- Core Logic Functions (from your original script) ---

# Use st.cache_data to prevent re-downloading data on every interaction
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_data(tickers, start_date, end_date, ma_period):
    """
    Downloads historical data using yfinance.
    Uses a curl_cffi session to bypass potential SSL/firewall issues.
    """
    session = requests.Session(impersonate="chrome110", verify=False)
    # Fetch data starting earlier to ensure MA is calculated correctly from day 1
    buffer_days = ma_period + 50 # Add extra buffer
    fetch_start_date = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
    data = yf.download(tickers, start=fetch_start_date, end=end_date, session=session, auto_adjust=False, group_by='ticker')
    return data

def prepare_data(raw_data, ma_type, ma_period):
    """Extracts, copies, and computes the required moving average."""
    if raw_data.empty:
        st.error("No data downloaded. Please check tickers and date range.")
        st.stop()

    qqq_main = raw_data["QQQ"].copy()
    tqqq = raw_data["TQQQ"].copy()
    sqqq = raw_data["SQQQ"].copy()

    # Calculate the selected moving average based on user input
    ma_col_name = f'{ma_type}_{ma_period}'
    if ma_type == 'SMA':
        qqq_main[ma_col_name] = qqq_main['Close'].rolling(window=ma_period).mean()
    elif ma_type == 'EMA':
        qqq_main[ma_col_name] = qqq_main['Close'].ewm(span=ma_period, adjust=False).mean()

    return qqq_main, tqqq, sqqq, ma_col_name

def run_simulation(
    qqq_main_df,
    tqqq_df,
    sqqq_df,
    initial_capital, 
    start_sim_date,
    ma_col,
    downtrend_strategy
):
    """
    Runs a historical backtest based on a single moving average.
    - If QQQ Close > MA, position is TQQQ.
    - If QQQ Close < MA, position is SQQQ or Cash based on downtrend_strategy.
    """
    portfolio = []
    actions_log = []
    position = 'Cash' # Start in cash
    capital = initial_capital
    num_trades = 0

    # Find the actual start date for the simulation in the data
    try:
        first_trade_date = qqq_main_df.loc[start_sim_date:].index[0]
        sim_start_index = qqq_main_df.index.get_loc(first_trade_date)
    except (KeyError, IndexError):
        st.error(f"Start date {start_sim_date} is outside the range of downloaded data.")
        return None, [], None, 0

    # --- First Day Logic ---
    # Determine the position on the first day of the simulation.
    # This cannot be a HOLD, it must be a BUY based on the signal from the day before the simulation starts.
    first_day_index = sim_start_index
    day_before_sim_start = qqq_main_df.index[first_day_index - 1]

    # This check handles an edge case where the very first day of the entire dataset is the simulation start day.
    if first_day_index == 0:
        st.error("Cannot start simulation on the very first day of the dataset. Not enough data for initial signal. Please select a later start date.")
        return None, [], None, 0

    # Ensure we have MA data for the day before the simulation starts
    if pd.isna(qqq_main_df.loc[day_before_sim_start, ma_col]):
        st.error(f"Not enough historical data to calculate the Moving Average for the day before the simulation start date. Please select an earlier start date or a shorter MA period.")
        return None, [], None, 0

    is_uptrend_on_first_day = qqq_main_df.loc[day_before_sim_start, 'Close'] > qqq_main_df.loc[day_before_sim_start, ma_col]
    
    if is_uptrend_on_first_day:
        position = 'TQQQ'
    else: # Downtrend
        position = 'SQQQ' if downtrend_strategy == 'Invest in SQQQ' else 'Cash'

    # If the first day's action is to hold cash, we start the loop as normal.
    # Otherwise, we log the initial BUY action.
    if position != 'Cash':
        num_trades = 1
        actions_log.append({
            'Date': qqq_main_df.index[first_day_index].strftime('%Y-%m-%d'),
            'Action': f"BUY {position}",
            'Capital': f"${capital:,.2f}",
            'QQQ_Close': f"{qqq_main_df.loc[qqq_main_df.index[first_day_index], 'Close']:,.2f}",
            ma_col: f"{qqq_main_df.loc[qqq_main_df.index[first_day_index], ma_col]:,.2f}"
        })
    portfolio.append({'Date': qqq_main_df.index[first_day_index], 'Capital': capital})

    # Loop through the data from the simulation start date
    for i in range(sim_start_index + 1, len(qqq_main_df)):
        date = qqq_main_df.index[i]
        prev_date = qqq_main_df.index[i-1]

        # Apply daily return based on the asset held from the previous day
        if position != 'Cash':
            asset_df = tqqq_df if position == 'TQQQ' else sqqq_df
            if date in asset_df.index and prev_date in asset_df.index:
                daily_ret = asset_df.loc[date, 'Close'] / asset_df.loc[prev_date, 'Close'] - 1
                capital *= (1 + daily_ret)

        # Determine today's target position based on yesterday's close vs MA
        is_uptrend = qqq_main_df.loc[prev_date, 'Close'] > qqq_main_df.loc[prev_date, ma_col]

        if is_uptrend:
            target_position = 'TQQQ'
        else: # Downtrend
            target_position = 'SQQQ' if downtrend_strategy == 'Invest in SQQQ' else 'Cash'

        # Log trades if position changes
        action_details = "HOLD"
        if target_position != position:
            num_trades += 1
            action_details = f"SELL {position}, BUY {target_position}" if position != 'Cash' else f"BUY {target_position}"
            position = target_position

        actions_log.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Action': action_details,
            'Capital': f"${capital:,.2f}",
            'QQQ_Close': f"{qqq_main_df.loc[date, 'Close']:,.2f}",
            ma_col: f"{qqq_main_df.loc[date, ma_col]:,.2f}"
        })
        portfolio.append({'Date': date, 'Capital': capital})

    if not portfolio:
        return None, [], None, 0

    return pd.DataFrame(portfolio).set_index('Date'), actions_log, position, num_trades

def calculate_performance_metrics(portfolio_df, qqq_main_df, initial_capital, num_trades):
    """Calculates and returns key performance metrics."""
    if portfolio_df is None or portfolio_df.empty:
        return {}
    
    # Strategy metrics
    final_value = portfolio_df['Capital'].iloc[-1]
    start_date = portfolio_df.index[0]
    end_date = portfolio_df.index[-1]
    days = (end_date - start_date).days
    years = days / 365.25
    cagr = ((final_value / initial_capital) ** (1 / years)) - 1 if years > 0 else 0
    drawdown = (portfolio_df['Capital'] / portfolio_df['Capital'].cummax() - 1).min()
    
    # QQQ Buy & Hold metrics for the same period
    qqq_benchmark = qqq_main_df['Close'].loc[start_date:end_date]
    if not qqq_benchmark.empty and years > 0:
        qqq_cagr = ((qqq_benchmark.iloc[-1] / qqq_benchmark.iloc[0]) ** (1 / years)) - 1
    else:
        qqq_cagr = 0

    return {
        "Final Portfolio Value": final_value,
        "CAGR": cagr,
        "Max Drawdown": drawdown,
        "Total Trades": num_trades,
        "QQQ CAGR": qqq_cagr
    }

def plot_results(portfolio_df, qqq_main_df, initial_capital, actions_log, ma_col):
    """Plots the strategy equity curve and returns the plotly figure."""
    if portfolio_df is None or portfolio_df.empty:
        return None
        
    backtest_start_date = portfolio_df.index[0]
    qqq_benchmark = qqq_main_df['Close'].loc[backtest_start_date:]
    qqq_performance = (qqq_benchmark / qqq_benchmark.iloc[0]) * initial_capital

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.05, row_heights=[0.7, 0.3]
    )

    # Add strategy and benchmark lines
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['Capital'],
        mode='lines',
        name='Strategy Equity',
        line=dict(color='royalblue', width=2),
        hovertemplate="<b>Date</b>: %{x}<br><b>Portfolio Value</b>: $%{y:,.2f}<extra></extra>"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=qqq_performance.index, y=qqq_performance, mode='lines', name='QQQ Buy & Hold', line=dict(color='grey', dash='dash')), row=1, col=1)

    # Add trade markers
    if actions_log:
        trades_df = pd.DataFrame(actions_log)
        trades_df = trades_df[trades_df['Action'] != 'HOLD'].copy()
        trades_df['Date'] = pd.to_datetime(trades_df['Date'])
        trades_df = trades_df.set_index('Date')
        
        buy_tqqq_trades = trades_df[trades_df['Action'].str.contains('BUY TQQQ')]
        sell_tqqq_trades = trades_df[trades_df['Action'].str.contains('SELL TQQQ')]
        sell_sqqq_trades = trades_df[trades_df['Action'].str.contains('SELL SQQQ')]
        buy_sqqq_trades = trades_df[trades_df['Action'].str.contains('BUY SQQQ')]

        if not buy_tqqq_trades.empty:
            buy_capital_values = portfolio_df.loc[buy_tqqq_trades.index, 'Capital']
            fig.add_trace(go.Scatter(x=buy_tqqq_trades.index, y=buy_capital_values,
                                     mode='markers', name='Buy TQQQ', marker=dict(color='green', symbol='triangle-up', size=10)), row=1, col=1)
        if not buy_sqqq_trades.empty:
            buy_capital_values = portfolio_df.loc[buy_sqqq_trades.index, 'Capital']
            fig.add_trace(go.Scatter(x=buy_sqqq_trades.index, y=buy_capital_values,
                                     mode='markers', name='Buy SQQQ', marker=dict(color='red', symbol='triangle-down', size=10)), row=1, col=1)
        
        sell_trades = pd.concat([sell_tqqq_trades, sell_sqqq_trades])
        if not sell_trades.empty:
            sell_capital_values = portfolio_df.loc[sell_trades.index, 'Capital']
            fig.add_trace(go.Scatter(x=sell_trades.index, y=sell_capital_values, mode='markers', name='Sell to Cash', marker=dict(color='orange', symbol='x', size=8)), row=1, col=1)

    # --- Bottom Plot: QQQ Price and EMA ---
    chart_data = qqq_main_df.loc[backtest_start_date:]
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=chart_data['Close'], mode='lines', name='QQQ Close',
        line=dict(color='black', width=1.5)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=chart_data[ma_col], mode='lines', name=ma_col,
        line=dict(color='orange', width=1.5, dash='dot')
    ), row=2, col=1)
    
    fig.update_layout(
        title_text='Strategy Performance vs. Benchmarks',
        xaxis_rangeslider_visible=False,
        legend_title_text='Legend',
        height=700,
        showlegend=True
    )
    # Update y-axis titles
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="QQQ Price ($)", row=2, col=1)

    return fig

# --- Streamlit Web App Interface ---

st.set_page_config(layout="wide")
st.title("TQQQ/SQQQ Moving Average Strategy Backtester")

# Short description + clickable expander for full details
st.markdown("A simple moving-average based TQQQ/SQQQ backtester. Click for details:")
with st.expander("How The Strategy Works"):
    st.markdown("""
    ### How The Strategy Works
    This backtester simulates a simple trend-following strategy based on a single Moving Average (MA) of the QQQ ETF. You can choose between a Simple Moving Average (SMA) or an Exponential Moving Average [...]

    **Decision Logic:**
    The core of the strategy is a daily decision made near the market close. For this simulation, we use the official daily closing price to represent this decision point (e.g., the price at 3:55 PM EST).

    1.  **Signal Check**: At the end of each trading day, the strategy compares QQQ's closing price to its selected Moving Average (SMA or EMA).
    2.  **Uptrend (Price > MA)**: If the price is above the MA, the strategy decides to hold **TQQQ** for the next trading day to capture leveraged upside moves.
    3.  **Downtrend (Price < MA)**: If the price is below the MA, the strategy takes a defensive position for the next day, either by holding **SQQQ** (to profit from a downturn) or by moving to **Cash**.

    The backtest assumes a trade is executed based on the previous day's closing signal and held for the entire following day, with the return calculated from close-to-close.
    """)
st.markdown("---")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Simulation Parameters")
    start_date = st.date_input("Start Date", datetime.date(2025, 1, 1))
    end_date = st.date_input("End Date", datetime.date.today())
    initial_capital = st.number_input("Initial Capital", min_value=1000, value=10000, step=1000)
    
    st.markdown("---")
    st.subheader("Strategy Configuration")
    st.markdown("""
    - **Uptrend**: If QQQ Close > MA, the strategy will hold TQQQ.
    - **Downtrend**: If QQQ Close < MA, the strategy will either hold SQQQ or move to Cash.
    """)
    ma_period = st.number_input("Moving Average Period (days)", min_value=5, max_value=250, value=15, step=5)
    ma_type = st.radio("Moving Average Type", ('SMA', 'EMA'), index=1)
    downtrend_strategy = st.radio(
        "Downtrend Strategy (when Close < MA)",
        ('Invest in SQQQ', 'Hold Cash'),
        index=1
    )

    run_button = st.button("Run Simulation")

# --- Main App Body ---

if run_button:
    tickers = ["QQQ", "TQQQ", "SQQQ"]

    with st.spinner("Downloading historical data..."):
        raw_data = get_data(tickers, start_date, end_date, ma_period)
    
    qqq_main, tqqq, sqqq, ma_col_name = prepare_data(raw_data, ma_type, ma_period)
    with st.spinner(f"Running simulation with {ma_type}({ma_period})..."):
        portfolio_df, actions_log, _, num_trades = run_simulation(
            qqq_main, tqqq, sqqq, initial_capital, start_date, ma_col_name, downtrend_strategy
        )

    if portfolio_df is not None:
        st.header("Backtest Performance")
        metrics = calculate_performance_metrics(portfolio_df, qqq_main, initial_capital, num_trades)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Final Portfolio Value", f"${metrics.get('Final Portfolio Value', 0):,.2f}")
        col2.metric("Strategy CAGR", f"{metrics.get('CAGR', 0):.2%}")
        col3.metric("QQQ CAGR", f"{metrics.get('QQQ CAGR', 0):.2%}")
        col4.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0):.2%}")
        col5.metric("Total Trades", metrics.get('Total Trades', 0))

        st.subheader("Performance Chart")
        fig = plot_results(portfolio_df, qqq_main, initial_capital, actions_log, ma_col_name)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Daily Actions Log"):
            st.dataframe(pd.DataFrame(actions_log))

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to start.")
