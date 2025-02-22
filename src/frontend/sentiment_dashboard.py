"""
🌙 Mavens Sentiment Analysis Dashboard - Built with love by Mavens 🚀
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import subprocess
import time
from datetime import datetime
import asyncio
from threading import Thread
import queue
from src.agents import (
    ApprovalContractIntegrator,
    ADDRESSES,
    ERC20_ABI,
)

# Constants
DATA_FOLDER = Path(project_root) / "src/data/sentiment_game"
TWEETS_HISTORY_FILE = DATA_FOLDER / "tweets_history.csv"
TOKENS_CONFIG_FILE = DATA_FOLDER / "tokens_config.json"
DEFAULT_TOKENS = ["solana", "GAME by Virtuals", "ethereum", "virtuals"]
AGENT_SCRIPT = Path(project_root) / "src/agents/game_sentiment_agent.py"

# Create a queue for communication between the agent process and the dashboard
result_queue = queue.Queue()
stop_analysis = False  # Global flag for stopping analysis

# Initialize contract integrator
contract_integrator = ApprovalContractIntegrator()


def update_status_display(message, token=None, status_type="info"):
    """Update the live status display with token processing information"""
    status_container = st.sidebar.container()

    status_styles = {
        "info": {"color": "#1E88E5", "icon": "ℹ️"},
        "success": {"color": "#4CAF50", "icon": "✅"},
        "warning": {"color": "#FFC107", "icon": "⚠️"},
        "error": {"color": "#F44336", "icon": "❌"},
    }

    style = status_styles.get(status_type, status_styles["info"])

    status_html = f"""
    <div style="padding: 10px; border-radius: 5px; margin: 5px 0; 
                background-color: rgba(0,0,0,0.1); border-left: 4px solid {style['color']};">
        {style['icon']} {message}
        {f'<br><small style="color: #666">Token: {token}</small>' if token else ''}
    </div>
    """

    status_container.markdown(status_html, unsafe_allow_html=True)


def run_sentiment_agent(selected_tokens):
    """Run the sentiment analysis agent in a separate process"""
    global stop_analysis

    # Create containers for status display
    progress_container = st.sidebar.empty()
    status_container = st.sidebar.empty()
    error_container = st.sidebar.empty()

    try:
        # Start the agent process
        cmd = [sys.executable, str(AGENT_SCRIPT), "--tokens", ",".join(selected_tokens)]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )

        # Track progress for each token
        token_progress = {token: "Pending" for token in selected_tokens}

        def update_progress():
            # Create progress display
            progress_md = "### Current Progress:\n"
            for token, status in token_progress.items():
                if status == "Pending":
                    icon = "⏳"
                elif status == "Fetching":
                    icon = "🔄"
                elif status == "Analyzing":
                    icon = "🤖"
                elif status == "Complete":
                    icon = "✅"
                else:
                    icon = "❌"
                progress_md += f"{icon} **{token}**: {status}\n"
            progress_container.markdown(progress_md)

        def should_show_error(error_text):
            """Filter out warning messages we don't want to show"""
            ignored_patterns = [
                "FutureWarning",
                "resume_download is deprecated",
                "huggingface_hub",
                "UserWarning",
                "DeprecationWarning",
            ]
            return not any(pattern in error_text for pattern in ignored_patterns)

        update_progress()  # Initial display

        while True:
            if stop_analysis:
                process.terminate()
                status_container.warning("⚠️ Analysis stopped by user")
                return False

            # Read output line by line
            output = process.stdout.readline()
            error = process.stderr.readline()

            if output == "" and process.poll() is not None:
                break

            if error:
                error_text = error.strip()
                if should_show_error(error_text):
                    error_container.error(f"❌ {error_text}")
                continue

            if output:
                output = output.strip()

                # Update token progress based on output
                if "Collecting tweets for" in output:
                    token = output.split("Collecting tweets for")[-1].strip()
                    token_progress[token] = "Fetching"
                    status_container.info(f"🔄 Fetching tweets for {token}")
                    update_progress()

                elif "Collected" in output and "tweets" in output:
                    # Extract token from the message
                    for token in selected_tokens:
                        if token in output:
                            token_progress[token] = "Analyzing"
                            status_container.info(f"🤖 Analyzing tweets for {token}")
                            update_progress()
                            break

                elif "Analyzed" in output and "tweets for" in output:
                    token = output.split("tweets for")[-1].split(",")[0].strip()
                    token_progress[token] = "Complete"
                    status_container.success(f"✅ Analysis complete for {token}")
                    update_progress()

                elif "Error" in output:
                    for token in selected_tokens:
                        if token in output:
                            token_progress[token] = "Error"
                            error_container.error(
                                f"❌ Error processing {token}: {output}"
                            )
                            update_progress()
                            break

                # Display other important messages
                if "Starting" in output:
                    status_container.info("🚀 Starting sentiment analysis...")
                elif "Completed" in output:
                    status_container.success("✅ Analysis completed successfully!")

        # Check final process status
        if process.poll() != 0:
            error_output = process.stderr.read()
            if error_output and should_show_error(error_output):
                error_container.error(f"❌ Process failed: {error_output}")
            return False

        return True

    except Exception as e:
        error_container.error(f"❌ Failed to run analysis: {str(e)}")
        return False
    finally:
        stop_analysis = False


def load_tokens():
    """Load tracked tokens from config file"""
    if TOKENS_CONFIG_FILE.exists():
        with open(TOKENS_CONFIG_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_TOKENS


def save_tokens(tokens):
    """Save tracked tokens to config file"""
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    with open(TOKENS_CONFIG_FILE, "w") as f:
        json.dump(tokens, f)


def load_tweet_data():
    """Load and process tweet data"""
    if not TWEETS_HISTORY_FILE.exists():
        st.error("No tweet history found! Please run the sentiment agent first.")
        return None

    df = pd.read_csv(TWEETS_HISTORY_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def token_management_section():
    """Token management UI section"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Token Management")

    # Load existing tokens
    tracked_tokens = load_tokens()

    # Add new token
    new_token = st.sidebar.text_input("Add New Token:", key="new_token").strip()
    if st.sidebar.button("Add Token") and new_token:
        if new_token not in tracked_tokens:
            tracked_tokens.append(new_token)
            save_tokens(tracked_tokens)
            st.sidebar.success(f"Added {new_token} to tracked tokens!")
        else:
            st.sidebar.warning("Token already exists!")

    # Remove tokens
    if tracked_tokens:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Remove Tokens")
        token_to_remove = st.sidebar.selectbox(
            "Select token to remove:", tracked_tokens
        )
        if st.sidebar.button("Remove Token"):
            tracked_tokens.remove(token_to_remove)
            save_tokens(tracked_tokens)
            st.sidebar.success(f"Removed {token_to_remove} from tracked tokens!")

    return tracked_tokens


def run_analysis_section(selected_tokens):
    """Analysis control section"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Run Analysis")

    # Create persistent containers for status and errors
    status_container = st.sidebar.empty()
    error_container = st.sidebar.container()

    col1, col2 = st.sidebar.columns(2)

    with col1:
        run_button = st.button(
            "Run Analysis", key="run_analysis", use_container_width=True
        )

    with col2:
        stop_button = st.button(
            "Stop Analysis", key="stop_analysis", use_container_width=True
        )

    if run_button:
        if not selected_tokens:
            error_container.error("⚠️ Please select at least one token to analyze")
            return

        global stop_analysis
        stop_analysis = False

        status_container.info("🚀 Starting analysis...")

        thread = Thread(target=run_sentiment_agent, args=(selected_tokens,))
        thread.start()

        try:
            while thread.is_alive():
                try:
                    msg_type, message = result_queue.get_nowait()

                    # Update status based on message type
                    if msg_type == "error":
                        error_container.error(f"❌ {message}")
                    elif msg_type == "success":
                        status_container.success(f"✅ {message}")
                    else:
                        status_container.info(f"ℹ️ {message}")

                except queue.Empty:
                    time.sleep(0.1)

                if stop_button:
                    stop_analysis = True
                    status_container.warning("⚠️ Analysis stopped by user")
                    break

            # Wait for thread to fully complete
            thread.join()

            if not stop_analysis:  # Only rerun if not stopped
                try:
                    # Add small delay to ensure status is visible
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    error_msg = f"Error refreshing dashboard: {str(e)}"
                    error_container.error(f"❌ {error_msg}")
                    time.sleep(2)
                    st.rerun()

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            error_container.error(f"❌ {error_msg}")

            # Show detailed error information
            with st.sidebar.expander("Error Details", expanded=True):
                st.code(str(e))

            # Log the full error for debugging
            print(f"Error in run_analysis_section: {str(e)}", file=sys.stderr)


def clear_tweet_history():
    """Clear all tweet history data"""
    try:
        if TWEETS_HISTORY_FILE.exists():
            # Create a backup before clearing
            backup_file = (
                TWEETS_HISTORY_FILE.parent
                / f"tweets_history_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            os.rename(TWEETS_HISTORY_FILE, backup_file)
            # Create new empty file with headers
            pd.DataFrame(columns=["timestamp", "token", "tweet", "sentiment"]).to_csv(
                TWEETS_HISTORY_FILE, index=False
            )
            return True, "Tweet history cleared successfully! (Backup created)"
    except Exception as e:
        return False, f"Error clearing tweet history: {str(e)}"
    return False, "No tweet history found"


def show_work_in_progress():
    """Display a work in progress message with animation"""
    st.markdown(
        """
        <style>
        .coming-soon {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(45deg, #1E1E1E, #2E2E2E);
            border-radius: 10px;
            margin: 2rem 0;
        }
        .coming-soon h2 {
            color: #4CAF50;
            margin-bottom: 1rem;
        }
        .progress-bar {
            width: 50%;
            margin: 20px auto;
            height: 4px;
            background: #333;
            border-radius: 2px;
            overflow: hidden;
        }
        .progress-bar-fill {
            height: 100%;
            background: #4CAF50;
            animation: progress 2s ease-in-out infinite;
            transform-origin: left;
        }
        @keyframes progress {
            0% { transform: scaleX(0); }
            50% { transform: scaleX(1); }
            100% { transform: scaleX(0); }
        }
        </style>
        <div class="coming-soon">
            <h2>🚧 Coming Soon! 🚧</h2>
            <p>We're working hard to bring you this exciting new feature.</p>
            <div class="progress-bar">
                <div class="progress-bar-fill"></div>
            </div>
            <p>Stay tuned for updates!</p>
        </div>
    """,
        unsafe_allow_html=True,
    )


def display_metrics_and_charts(df, selected_tokens):
    """Display metrics and charts for selected tokens"""
    if not selected_tokens:
        st.info("Please select tokens to display metrics and charts.")
        return

    # Filter data for selected tokens
    filtered_df = df[df["token"].isin(selected_tokens)].copy()

    if filtered_df.empty:
        st.warning("No data available for selected tokens.")
        return

    # Create metrics section
    st.markdown("### 📊 Analysis Results")
    metrics_cols = st.columns(len(selected_tokens))

    for idx, token in enumerate(selected_tokens):
        token_data = filtered_df[filtered_df["token"] == token]
        if not token_data.empty:
            avg_sentiment = token_data["sentiment"].mean()
            tweet_count = len(token_data)

            with metrics_cols[idx]:
                st.markdown(
                    f"""
                    <div style="padding: 1rem; background: linear-gradient(45deg, #1E1E1E, #2E2E2E); border-radius: 10px; border: 1px solid #4CAF50;">
                        <h3 style="color: #4CAF50; margin: 0; font-size: 1.2rem;">{token}</h3>
                        <p style="color: #888; margin: 0.5rem 0;">Average Sentiment</p>
                        <h4 style="color: #4CAF50; margin: 0; font-size: 1.8rem;">{avg_sentiment:.2f}</h4>
                        <p style="color: #888; margin: 0.5rem 0;">Tweets Analyzed: {tweet_count}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Create sentiment trend chart
    st.markdown("### 📈 Sentiment Visualizations")

    # Sentiment over time
    sentiment_trend = (
        filtered_df.groupby(["token", "timestamp"])["sentiment"].mean().reset_index()
    )
    fig_trend = px.line(
        sentiment_trend,
        x="timestamp",
        y="sentiment",
        color="token",
        title="Sentiment Trends Over Time",
        template="plotly_dark",
    )
    fig_trend.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        legend_title="Tokens",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Sentiment distribution
    fig_dist = px.box(
        filtered_df,
        x="token",
        y="sentiment",
        color="token",
        title="Sentiment Distribution by Token",
        template="plotly_dark",
    )
    fig_dist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Token",
        yaxis_title="Sentiment Score",
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Recent tweets section
    st.markdown("### 📱 Recent Tweets")
    for token in selected_tokens:
        token_tweets = (
            filtered_df[filtered_df["token"] == token]
            .sort_values("timestamp", ascending=False)
            .head(5)
        )
        if not token_tweets.empty:
            st.markdown(f"#### {token}")
            for _, tweet in token_tweets.iterrows():
                st.markdown(
                    f"""
                    <div class="tweet-card">
                        <p style="margin: 0; color: #888;">
                            {tweet['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                            <span style="float: right; color: {'#4CAF50' if tweet['sentiment'] > 0 else '#F44336'}">
                                Sentiment: {tweet['sentiment']:.2f}
                            </span>
                        </p>
                        <p style="margin: 0.5rem 0;">{tweet['tweet']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def display_contract_section():
    st.sidebar.markdown("### 🔗 Contract Status")

    # Contract metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Token1 Balance", get_token1_balance())
        st.metric("Pending Swaps", get_pending_swaps())
    with col2:
        st.metric("Token2 Balance", get_token2_balance())
        st.metric("Total Volume", get_total_volume())

    # Action buttons
    if st.button("Approve Token1"):
        approve_token1()
    if st.button("Approve Token2"):
        approve_token2()


def display_sentiment_actions():
    st.markdown("### 📊 Sentiment Actions")

    # Manual sentiment testing
    sentiment = st.slider("Test Sentiment", -1.0, 1.0, 0.0, 0.1)
    if st.button("Trigger Action"):
        handle_sentiment_action(sentiment)


def sentiment_dashboard():
    """Main sentiment analysis dashboard"""
    st.title("🤖 TokenPulse Agents Dashboard")
    st.subheader("Real-time Crypto Sentiment Analysis")
    st.caption("Autonomous AI agent monitoring and analyzing market sentiment 24/7")

    # Token management section
    tracked_tokens = token_management_section()
    df = load_tweet_data()

    # Sidebar filters and controls
    st.sidebar.header("📊 Data Filters")
    selected_tokens = st.sidebar.multiselect(
        "Select Tokens to Display", options=tracked_tokens, default=tracked_tokens[:3]
    )

    # Run Analysis section
    run_analysis_section(selected_tokens)

    if df is not None and not df.empty:
        # Display metrics and visualizations
        display_metrics_and_charts(df, selected_tokens)

    # Display contract section
    display_contract_section()

    # Display sentiment actions
    display_sentiment_actions()


def risk_agent_dashboard():
    """Risk Agent Dashboard (Coming Soon)"""
    st.title("📊 Risk Analysis Dashboard")
    show_work_in_progress()


def strategy_agent_dashboard():
    """Strategy Agent Dashboard (Coming Soon)"""
    st.title("🎯 Strategy Analysis Dashboard")
    show_work_in_progress()


def main():
    # Page config with custom CSS
    st.set_page_config(
        page_title="🌙 Sentiment Analysis", page_icon="🚀", layout="wide"
    )

    # Custom CSS for better spacing and styling
    st.markdown(
        """
        <style>
        /* Main container styling */
        .main {
            padding: 1rem;
        }
        
        /* Header styling */
        .stTitle {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            padding: 1rem 0 !important;
            background: linear-gradient(90deg, #4CAF50, #2E7D32);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Metrics container styling */
        [data-testid="stMetric"] {
            background: linear-gradient(45deg, #1E1E1E, #2E2E2E);
            padding: 1rem !important;
            border-radius: 10px;
            border: 1px solid #4CAF50;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            color: #888 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #4CAF50 !important;
        }
        
        /* Card styling */
        .card {
            background: linear-gradient(45deg, #1E1E1E, #2E2E2E);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid #333;
            margin: 1rem 0;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #4CAF50;
            color: #4CAF50;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
            padding: 2rem 1rem;
        }
        .sidebar-header {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #4CAF50;
        }
        
        /* Button styling */
        .stButton button {
            width: 100%;
            background: linear-gradient(45deg, #4CAF50, #2E7D32) !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        }
        
        /* Table styling */
        [data-testid="stDataFrame"] {
            background: #1E1E1E;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #333;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #1E1E1E;
            padding: 0.5rem;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #2E2E2E;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            color: #888;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50 !important;
            color: white !important;
        }
        
        /* Plot container styling */
        [data-testid="stPlotlyChart"] {
            background: #1E1E1E;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #333;
        }
        
        /* Recent tweets styling */
        .tweet-card {
            background: #1E1E1E;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #333;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        .tweet-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = {
        "Sentiment Analysis": sentiment_dashboard,
        "Risk Analysis": risk_agent_dashboard,
        "Strategy Analysis": strategy_agent_dashboard,
    }

    selected_page = st.sidebar.radio("Select Dashboard", list(pages.keys()))

    # Display selected page
    pages[selected_page]()


def get_subscription_count():
    return contract_integrator.get_subscription_count()


def get_pending_swaps():
    return contract_integrator.get_pending_swaps()


def get_total_volume():
    return f"{contract_integrator.get_total_volume():.2f} ETH"


def get_token1_balance():
    try:
        token1_contract = contract_integrator.w3.eth.contract(
            address=ADDRESSES["TOKEN1"], abi=ERC20_ABI  # We need to add this ABI
        )
        balance = token1_contract.functions.balanceOf(
            contract_integrator.account.address
        ).call()
        return contract_integrator.w3.from_wei(balance, "ether")
    except Exception as e:
        st.error(f"Error getting Token1 balance: {e}")
        return 0


def get_token2_balance():
    try:
        token2_contract = contract_integrator.w3.eth.contract(
            address=ADDRESSES["TOKEN2"], abi=ERC20_ABI
        )
        balance = token2_contract.functions.balanceOf(
            contract_integrator.account.address
        ).call()
        return contract_integrator.w3.from_wei(balance, "ether")
    except Exception as e:
        st.error(f"Error getting Token2 balance: {e}")
        return 0


def approve_token1():
    try:
        token1_contract = contract_integrator.w3.eth.contract(
            address=ADDRESSES["TOKEN1"], abi=ERC20_ABI
        )
        amount = contract_integrator.w3.to_wei(0.1, "ether")
        tx = token1_contract.functions.approve(
            ADDRESSES["SWAP"], amount
        ).build_transaction(
            {
                "from": contract_integrator.account.address,
                "nonce": contract_integrator.w3.eth.get_transaction_count(
                    contract_integrator.account.address
                ),
            }
        )
        signed_tx = contract_integrator.account.sign_transaction(tx)
        tx_hash = contract_integrator.w3.eth.send_raw_transaction(
            signed_tx.rawTransaction
        )
        st.success(f"Token1 approval submitted: {tx_hash.hex()}")
    except Exception as e:
        st.error(f"Error approving Token1: {e}")


def approve_token2():
    try:
        token2_contract = contract_integrator.w3.eth.contract(
            address=ADDRESSES["TOKEN2"], abi=ERC20_ABI
        )
        amount = contract_integrator.w3.to_wei(0.1, "ether")
        tx = token2_contract.functions.approve(
            ADDRESSES["SWAP"], amount
        ).build_transaction(
            {
                "from": contract_integrator.account.address,
                "nonce": contract_integrator.w3.eth.get_transaction_count(
                    contract_integrator.account.address
                ),
            }
        )
        signed_tx = contract_integrator.account.sign_transaction(tx)
        tx_hash = contract_integrator.w3.eth.send_raw_transaction(
            signed_tx.rawTransaction
        )
        st.success(f"Token2 approval submitted: {tx_hash.hex()}")
    except Exception as e:
        st.error(f"Error approving Token2: {e}")


def handle_sentiment_action(sentiment):
    try:
        if abs(sentiment) > 0.6:
            if sentiment > 0:
                approve_token1()
            else:
                approve_token2()
            st.success(f"Sentiment action triggered: {sentiment}")
    except Exception as e:
        st.error(f"Error handling sentiment: {e}")


if __name__ == "__main__":
    main()
