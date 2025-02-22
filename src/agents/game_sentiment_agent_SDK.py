"""
🌙 TokenPulse Sentiment Agent - Built with love by Mavens 🚀
Wraps existing sentiment analysis with Virtuals Protocol Game SDK
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from termcolor import cprint
import pandas as pd
from datetime import datetime, timedelta
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import httpx
import asyncio
from twikit import Client, TooManyRequests, BadRequest
from random import randint
import time
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.append(project_root)

# Get the absolute path to src/.env
env_path = Path(__file__).parent.parent / ".env"
print(f"Loading .env from: {env_path}")

# First load
load_dotenv(dotenv_path=env_path)

# Debug: Print API key (first 4 chars)
api_key = os.getenv("GAME_API_KEY")
if api_key:
    print(f"API Key loaded: {api_key[:4]}...")
else:
    print("No API key found!")

# Import Game SDK components
try:
    from game_sdk.game.agent import Agent, WorkerConfig
    from game_sdk.game.custom_types import (
        Function,
        Argument,
        FunctionResultStatus,
        FunctionResult,
    )
except ImportError:
    cprint("❌ Error: game_sdk not found. Please install it first:", "red")
    cprint("pip install game-sdk", "yellow")
    sys.exit(1)

# Constants
TOKENS_TO_TRACK = ["solana", "GAME by Virtuals", "ethereum", "virtuals", "bonk"]
SENTIMENT_ANNOUNCE_THRESHOLD = 0.4
DATA_FOLDER = Path(project_root) / "src/data/sentiment_game"
SENTIMENT_HISTORY_FILE = DATA_FOLDER / "sentiment_history.csv"
TWEETS_HISTORY_FILE = DATA_FOLDER / "tweets_history.csv"
LOGS_FILE = DATA_FOLDER / "agent_logs.txt"
PLOTS_FOLDER = DATA_FOLDER / "plots"

# Additional Constants
TWEETS_PER_RUN = 30  # Number of tweets to collect per run
IGNORE_LIST = ["t.co", "discord", "join", "telegram", "discount", "pay"]

# Create all necessary directories
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Add these as global variables at the top of the file, after the imports
_MODEL = None
_TOKENIZER = None
_TWITTER_CLIENT = None

# Patch httpx for Twitter client
original_client = httpx.Client


def patched_client(*args, **kwargs):
    if "headers" not in kwargs:
        kwargs["headers"] = {}

    user_agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    kwargs["headers"].update(
        {
            "User-Agent": user_agents[randint(0, len(user_agents) - 1)],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    kwargs.pop("proxy", None)
    return original_client(*args, **kwargs)


httpx.Client = patched_client


def get_or_create_model_tokenizer():
    """Helper function to manage model and tokenizer instances"""
    global _MODEL, _TOKENIZER

    if _MODEL is None or _TOKENIZER is None:
        print("Debug - Initializing/Reinitializing model and tokenizer")
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(
                "finiteautomata/bertweet-base-sentiment-analysis"
            )
            _MODEL = AutoModelForSequenceClassification.from_pretrained(
                "finiteautomata/bertweet-base-sentiment-analysis"
            )
        except Exception as e:
            print(f"Error initializing model/tokenizer: {str(e)}")
            return None, None

    return _MODEL, _TOKENIZER


def analyze_sentiment(text: str, model, tokenizer) -> float:
    """Analyze sentiment of a single text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        neg, neu, pos = predictions[0].tolist()
        return pos - neg  # Convert to -1 to 1 scale


async def fetch_tweets_for_all_tokens(tokens: list) -> dict:
    """Fetch tweets for all tokens in one cycle with enhanced tracking"""
    all_tweets = {}
    cycle_stats = {
        "start_time": datetime.now().isoformat(),
        "tokens_processed": 0,
        "total_tweets": 0,
        "token_status": {},
    }

    log_activity(
        f"Starting new fetch cycle for {len(tokens)} tokens: {', '.join(tokens)}"
    )

    for token in tokens:
        try:
            token_start_time = datetime.now()
            log_activity(
                f"[{cycle_stats['tokens_processed'] + 1}/{len(tokens)}] Collecting tweets for {token}"
            )

            tweets = await fetch_tweets_for_token(token)
            all_tweets[token] = tweets

            # Update cycle statistics
            cycle_stats["tokens_processed"] += 1
            cycle_stats["total_tweets"] += len(tweets)
            cycle_stats["token_status"][token] = {
                "status": "success" if tweets else "no_tweets",
                "tweets_collected": len(tweets),
                "collection_time": (datetime.now() - token_start_time).total_seconds(),
                "timestamp": datetime.now().isoformat(),
            }

            log_activity(
                f"✅ Token: {token} - Collected {len(tweets)} tweets "
                f"({cycle_stats['tokens_processed']}/{len(tokens)} tokens processed)"
            )

            # Add small delay between tokens to avoid rate limits
            if cycle_stats["tokens_processed"] < len(tokens):
                delay = randint(1, 2)
                log_activity(f"Waiting {delay}s before next token...")
                time.sleep(delay)

        except Exception as e:
            error_msg = str(e)
            log_activity(f"Error collecting tweets for {token}: {error_msg}", "ERROR")
            all_tweets[token] = []
            cycle_stats["token_status"][token] = {
                "status": "error",
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
            }

    # Log cycle summary
    cycle_stats["end_time"] = datetime.now().isoformat()
    cycle_stats["total_time"] = (
        datetime.fromisoformat(cycle_stats["end_time"])
        - datetime.fromisoformat(cycle_stats["start_time"])
    ).total_seconds()

    log_activity(
        f"Fetch cycle completed:\n"
        f"- Total tokens processed: {cycle_stats['tokens_processed']}\n"
        f"- Total tweets collected: {cycle_stats['total_tweets']}\n"
        f"- Total time: {cycle_stats['total_time']:.2f}s"
    )

    return all_tweets, cycle_stats


def analyze_token_sentiment(
    token: str, **kwargs
) -> tuple[FunctionResultStatus, str, dict]:
    """Enhanced sentiment analysis with better cycle management and tracking"""
    try:
        if "worker_state" not in kwargs:
            return (
                FunctionResultStatus.FAILED,
                "worker_state not provided in kwargs",
                {},
            )

        model, tokenizer = get_or_create_model_tokenizer()
        if model is None or tokenizer is None:
            return (
                FunctionResultStatus.FAILED,
                "Failed to initialize model/tokenizer",
                {},
            )

        # Get all tokens from worker state
        tokens = kwargs["worker_state"].get("tokens", [token])

        # Fetch tweets for all tokens in one cycle
        all_tweets, cycle_stats = asyncio.run(fetch_tweets_for_all_tokens(tokens))

        # Process all tokens' sentiments
        all_sentiments = {}
        analysis_stats = {
            "start_time": datetime.now().isoformat(),
            "tokens_analyzed": 0,
            "token_results": {},
        }

        for current_token, tweets in all_tweets.items():
            token_start = datetime.now()

            if not tweets:
                log_activity(f"Using mock tweets for {current_token}", "WARNING")
                tweets = [
                    f"Really excited about the future of {current_token}! 🚀",
                    f"Just bought more {current_token}, feeling bullish! 💪",
                    f"Not sure about {current_token} right now, market looks uncertain 🤔",
                ]

            # Analyze sentiment
            sentiments = [
                analyze_sentiment(tweet, model, tokenizer) for tweet in tweets
            ]
            avg_sentiment = sum(sentiments) / len(sentiments)
            all_sentiments[current_token] = avg_sentiment

            # Save data
            save_tweets_to_csv(current_token, tweets, sentiments)

            # Update analysis statistics
            analysis_stats["tokens_analyzed"] += 1
            analysis_stats["token_results"][current_token] = {
                "sentiment_score": avg_sentiment,
                "tweets_analyzed": len(tweets),
                "analysis_time": (datetime.now() - token_start).total_seconds(),
                "used_mock_data": len(all_tweets[current_token]) == 0,
            }

            # Log analysis results
            log_activity(
                f"[{analysis_stats['tokens_analyzed']}/{len(tokens)}] "
                f"Analyzed {current_token}: {len(tweets)} tweets, "
                f"sentiment: {avg_sentiment:.2f}"
            )

            # Add contract integration
            if abs(avg_sentiment) > SENTIMENT_ANNOUNCE_THRESHOLD:
                self.contract_integrator.handle_sentiment_trigger(
                    current_token, avg_sentiment
                )

        # Update final statistics
        analysis_stats["end_time"] = datetime.now().isoformat()
        analysis_stats["total_time"] = (
            datetime.fromisoformat(analysis_stats["end_time"])
            - datetime.fromisoformat(analysis_stats["start_time"])
        ).total_seconds()

        # Plot updated sentiment trends
        plot_sentiment_trends()
        plot_sentiment_distribution()

        # Return DONE to indicate we should stop after one run
        return (
            FunctionResultStatus.DONE,
            f"Completed sentiment analysis for {len(tokens)} tokens",
            {
                "sentiment_updates": all_sentiments,
                "cycle_stats": cycle_stats,
                "analysis_stats": analysis_stats,
                "should_stop": True,  # Signal to stop after this run
            },
        )

    except Exception as e:
        log_activity(f"Error in sentiment analysis cycle: {str(e)}", "ERROR")
        return FunctionResultStatus.FAILED, str(e), {"should_stop": True}


def get_worker_state_fn(function_result: FunctionResult, current_state: dict) -> dict:
    """Enhanced state management for sentiment workers"""
    # Define state without model/tokenizer
    state = {
        "tokens": TOKENS_TO_TRACK,
        "sentiment_scores": {},
        "last_check": None,
        "threshold": SENTIMENT_ANNOUNCE_THRESHOLD,
        "cycle_count": current_state.get("cycle_count", 0),
        "last_cycle_time": current_state.get("last_cycle_time", None),
        "should_stop": False,
    }

    if current_state is not None:
        # Preserve existing state values
        state.update(
            {k: v for k, v in current_state.items() if k not in ["model", "tokenizer"]}
        )

        # Check if we should stop
        if function_result and hasattr(function_result, "info"):
            if function_result.info.get("should_stop", False):
                state["should_stop"] = True
            if function_result.info.get("sentiment_updates"):
                state["cycle_count"] = state.get("cycle_count", 0) + 1
                state["last_cycle_time"] = datetime.now().isoformat()
                state["sentiment_scores"].update(
                    function_result.info["sentiment_updates"]
                )

    # Initialize model and tokenizer globally but don't store in state
    get_or_create_model_tokenizer()

    return state


def get_agent_state_fn(function_result: FunctionResult, current_state: dict) -> dict:
    """Enhanced state management for the main sentiment agent"""
    init_state = {
        "total_analyses": 0,
        "average_sentiment": 0.0,
        "tokens_tracked": TOKENS_TO_TRACK,
        "history": [],
        "cycles_completed": 0,
        "should_stop": False,
    }

    if current_state is None:
        return init_state

    # Create a new state based on current state
    new_state = current_state.copy()

    # Update aggregate statistics if we have new sentiment data
    if function_result and hasattr(function_result, "info"):
        if function_result.info.get("should_stop", False):
            new_state["should_stop"] = True

        sentiment_updates = function_result.info.get("sentiment_updates", {})
        if sentiment_updates:
            scores = list(sentiment_updates.values())
            new_state["total_analyses"] += 1
            new_state["cycles_completed"] += 1
            new_state["average_sentiment"] = (
                new_state["average_sentiment"] * (new_state["total_analyses"] - 1)
                + sum(scores) / len(scores)
            ) / new_state["total_analyses"]

            # Add cycle timestamp and data
            cycle_entry = {
                "timestamp": datetime.now().isoformat(),
                "cycle": new_state["cycles_completed"],
                "scores": sentiment_updates,
            }
            new_state["history"].append(cycle_entry)

            # Save to CSV
            save_sentiment_history(new_state["history"])

    return new_state


def save_sentiment_history(history):
    """Save sentiment history to CSV file"""
    try:
        df = pd.DataFrame(
            [
                {"timestamp": entry["timestamp"], "token": token, "sentiment": score}
                for entry in history
                for token, score in entry["scores"].items()
            ]
        )
        df.to_csv(SENTIMENT_HISTORY_FILE, index=False)
        cprint(f"✅ Saved sentiment history to {SENTIMENT_HISTORY_FILE}", "green")
    except Exception as e:
        cprint(f"❌ Error saving sentiment history: {e}", "red")


def get_or_create_twitter_client():
    """Helper function to manage Twitter client instance"""
    global _TWITTER_CLIENT

    if _TWITTER_CLIENT is None:
        try:
            if not os.path.exists("cookies.json"):
                cprint(
                    "❌ No cookies.json found! Please run twitter_login.py first", "red"
                )
                return None

            _TWITTER_CLIENT = Client()
            _TWITTER_CLIENT.load_cookies("cookies.json")
            cprint("🚀 Twitter client initialized successfully!", "green")
        except Exception as e:
            cprint(f"❌ Error initializing Twitter client: {str(e)}", "red")
            return None

    return _TWITTER_CLIENT


def save_tweets_to_csv(token: str, tweets: list, sentiments: list):
    """Save tweets and their sentiment scores to CSV"""
    try:
        timestamp = datetime.now().isoformat()
        data = {
            "timestamp": [timestamp] * len(tweets),
            "token": [token] * len(tweets),
            "tweet": tweets,
            "sentiment": sentiments,
        }
        df = pd.DataFrame(data)

        # Append to existing file or create new
        if TWEETS_HISTORY_FILE.exists():
            df.to_csv(TWEETS_HISTORY_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(TWEETS_HISTORY_FILE, index=False)

        cprint(f"✅ Saved {len(tweets)} tweets for {token}", "green")
    except Exception as e:
        cprint(f"❌ Error saving tweets: {e}", "red")


def log_activity(message: str, level: str = "INFO"):
    """Log activities with timestamp"""
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] [{level}] {message}\n"

    with open(LOGS_FILE, "a") as f:
        f.write(log_entry)

    if level == "ERROR":
        cprint(f"❌ {message}", "red")
    elif level == "WARNING":
        cprint(f"⚠️ {message}", "yellow")
    else:
        cprint(f"ℹ️ {message}", "cyan")


async def fetch_tweets_for_token(token: str) -> list:
    """Enhanced tweet fetching with progress tracking"""
    client = get_or_create_twitter_client()
    if not client:
        return []

    collected_tweets = []
    try:
        log_activity(f"Starting tweet collection for {token}")
        tweets = await client.search_tweet(token, product="Latest")

        if tweets:
            for tweet in tweets:
                if len(collected_tweets) >= TWEETS_PER_RUN:
                    break
                if not any(word.lower() in tweet.text.lower() for word in IGNORE_LIST):
                    collected_tweets.append(tweet.text)
                    cprint(
                        f"📝 Collected {len(collected_tweets)}/{TWEETS_PER_RUN} tweets",
                        "cyan",
                        end="\r",
                    )

            try:
                while len(collected_tweets) < TWEETS_PER_RUN:
                    time.sleep(randint(1, 3))
                    more_tweets = await tweets.next()
                    if not more_tweets:
                        break

                    for tweet in more_tweets:
                        if len(collected_tweets) >= TWEETS_PER_RUN:
                            break
                        if not any(
                            word.lower() in tweet.text.lower() for word in IGNORE_LIST
                        ):
                            collected_tweets.append(tweet.text)
                            cprint(
                                f"📝 Collected {len(collected_tweets)}/{TWEETS_PER_RUN} tweets",
                                "cyan",
                                end="\r",
                            )
            except Exception as e:
                log_activity(f"Pagination stopped: {str(e)}", "WARNING")

    except TooManyRequests:
        log_activity("Rate limit hit, will use mock tweets", "WARNING")
    except Exception as e:
        log_activity(f"Error fetching tweets: {str(e)}", "ERROR")

    print()  # New line after progress
    return collected_tweets


def plot_sentiment_trends(days_back: int = 7):
    """Plot sentiment trends for all tokens over time"""
    try:
        if not TWEETS_HISTORY_FILE.exists():
            log_activity("No tweet history found to plot", "WARNING")
            return

        # Read the data
        df = pd.read_csv(TWEETS_HISTORY_FILE)
        if df.empty:
            log_activity("No data found in tweets history file", "WARNING")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter for recent data
        cutoff_date = datetime.now() - timedelta(days=days_back)
        df = df[df["timestamp"] > cutoff_date]

        if df.empty:
            log_activity(f"No data found in the last {days_back} days", "WARNING")
            return

        # Set up the plot style
        plt.style.use("seaborn")
        fig = plt.figure(figsize=(12, 6))

        # Plot sentiment trends for each token
        for token in TOKENS_TO_TRACK:
            token_data = df[df["token"] == token]
            if not token_data.empty:
                # Calculate hourly average sentiment
                hourly_sentiment = token_data.groupby(
                    pd.Grouper(key="timestamp", freq="H")
                )["sentiment"].mean()

                plt.plot(
                    hourly_sentiment.index,
                    hourly_sentiment.values,
                    label=token,
                    marker="o",
                    markersize=4,
                )

        plt.title("Sentiment Trends Over Time")
        plt.xlabel("Time")
        plt.ylabel("Sentiment Score")
        plt.legend()
        plt.grid(True)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plot_file = PLOTS_FOLDER / f"sentiment_trends_{timestamp}.png"

        # Ensure directory exists
        plot_file.parent.mkdir(parents=True, exist_ok=True)

        # Save with error handling
        try:
            plt.savefig(plot_file, bbox_inches="tight", dpi=300)
            log_activity(f"Sentiment trend plot saved to {plot_file}")
        except Exception as save_error:
            log_activity(f"Error saving plot: {str(save_error)}", "ERROR")
        finally:
            plt.close(fig)

        return plot_file
    except Exception as e:
        log_activity(f"Error plotting sentiment trends: {str(e)}", "ERROR")
        if "fig" in locals():
            plt.close(fig)
        return None


def plot_sentiment_distribution():
    """Plot sentiment distribution for each token"""
    try:
        if not TWEETS_HISTORY_FILE.exists():
            log_activity("No tweet history found to plot", "WARNING")
            return

        # Read the data
        df = pd.read_csv(TWEETS_HISTORY_FILE)
        if df.empty:
            log_activity("No data found in tweets history file", "WARNING")
            return

        # Create subplots for each token
        fig, axes = plt.subplots(
            len(TOKENS_TO_TRACK), 1, figsize=(10, 4 * len(TOKENS_TO_TRACK)), sharex=True
        )

        has_data = False
        for idx, token in enumerate(TOKENS_TO_TRACK):
            token_data = df[df["token"] == token]
            if not token_data.empty:
                has_data = True
                ax = axes[idx] if len(TOKENS_TO_TRACK) > 1 else axes
                sns.histplot(
                    data=token_data,
                    x="sentiment",
                    ax=ax,
                    bins=30,
                    kde=True,
                )
                ax.set_title(f"Sentiment Distribution for {token}")
                ax.set_xlabel("Sentiment Score")
                ax.set_ylabel("Count")

        if not has_data:
            log_activity("No data found for any tokens", "WARNING")
            plt.close(fig)
            return None

        plt.tight_layout()

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plot_file = PLOTS_FOLDER / f"sentiment_distribution_{timestamp}.png"

        # Ensure directory exists
        plot_file.parent.mkdir(parents=True, exist_ok=True)

        # Save with error handling
        try:
            plt.savefig(plot_file, bbox_inches="tight", dpi=300)
            log_activity(f"Sentiment distribution plot saved to {plot_file}")
        except Exception as save_error:
            log_activity(f"Error saving plot: {str(save_error)}", "ERROR")
        finally:
            plt.close(fig)

        return plot_file
    except Exception as e:
        log_activity(f"Error plotting sentiment distribution: {str(e)}", "ERROR")
        if "fig" in locals():
            plt.close(fig)
        return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Sentiment Analysis Agent")
    parser.add_argument(
        "--tokens",
        type=str,
        help="Comma-separated list of tokens to analyze (optional, will use TOKENS_TO_TRACK if not provided)",
        default=None,
    )
    return parser.parse_args()


def get_tokens_to_analyze():
    """Get tokens to analyze from command line args or use TOKENS_TO_TRACK"""
    args = parse_args()

    # If tokens provided via command line, use those
    if args.tokens:
        provided_tokens = [token.strip() for token in args.tokens.split(",")]
        print(f"Using tokens from command line: {provided_tokens}")
        return provided_tokens

    # Otherwise use the tokens defined in TOKENS_TO_TRACK
    print(f"Using default TOKENS_TO_TRACK: {TOKENS_TO_TRACK}")
    return TOKENS_TO_TRACK


def main():
    """Main entry point for the game sentiment agent"""
    try:
        # Get tokens to analyze
        tokens_to_analyze = get_tokens_to_analyze()

        cprint("\n🌟 Starting TokenPulse Sentiment Agent...", "green")
        cprint(f"Analyzing tokens: {', '.join(tokens_to_analyze)}", "cyan")

        # Get API key from environment with detailed debugging
        game_api_key = os.getenv("GAME_API_KEY")

        if not game_api_key:
            cprint("❌ Error: GAME_API_KEY not found in environment variables!", "red")
            sys.exit(1)

        # Define sentiment analysis function
        analyze_fn = Function(
            fn_name="analyze_sentiment",
            fn_description="Analyze sentiment for a specific token",
            args=[
                Argument(
                    name="token",
                    type="str",
                    description="Token to analyze sentiment for",
                ),
                Argument(
                    name="worker_state",
                    type="dict",
                    description="Current state of the worker",
                ),
            ],
            executable=analyze_token_sentiment,
        )

        # Create worker with custom tokens
        sentiment_worker = WorkerConfig(
            id="sentiment_worker",
            worker_description="Worker that analyzes crypto token sentiment",
            get_state_fn=get_worker_state_fn,
            action_space=[analyze_fn],
        )

        # Create the game agent
        game_agent = Agent(
            api_key=game_api_key,
            name="GameSentimentTracker",
            agent_goal=f"Track and analyze sentiment for tokens: {', '.join(tokens_to_analyze)}",
            agent_description="A wrapper around the existing sentiment agent using Game SDK",
            get_agent_state_fn=get_agent_state_fn,
            workers=[sentiment_worker],
        )

        # Compile and run
        cprint("\n🔄 Compiling game agent...", "cyan")
        game_agent.compile()

        cprint("\n🚀 Running game agent...", "green")
        game_agent.run()

        # After running the agent, generate visualizations
        cprint("\n📊 Generating sentiment visualizations...", "cyan")
        trend_plot = plot_sentiment_trends(days_back=7)  # Last 7 days
        dist_plot = plot_sentiment_distribution()

        if trend_plot or dist_plot:
            cprint("\n✨ Visualization files generated:", "green")
            if trend_plot:
                cprint(f"  - Trends: {trend_plot}", "cyan")
            if dist_plot:
                cprint(f"  - Distribution: {dist_plot}", "cyan")

    except KeyboardInterrupt:
        cprint("\n👋 Game agent shutting down gracefully...", "yellow")
    except Exception as e:
        cprint(f"\n❌ Fatal error: {str(e)}", "red")
        raise


if __name__ == "__main__":
    main()
