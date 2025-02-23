"""
🌙 TokenPulse Sentiment Agent - Built with love by Mavens 🚀
"""

import os
from pathlib import Path
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from twikit import Client, TooManyRequests
import httpx
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from src.agents.contract_integration import ContractIntegrator
from termcolor import cprint
from random import randint
import time

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Constants
SENTIMENT_THRESHOLD = 0.4
DATA_FOLDER = Path(project_root) / "src/data/sentiment_game"
SENTIMENT_HISTORY_FILE = DATA_FOLDER / "sentiment_history.csv"
TWEETS_HISTORY_FILE = DATA_FOLDER / "tweets_history.csv"
TWEETS_PER_RUN = 30
IGNORE_LIST = ["t.co", "discord", "join", "telegram", "discount", "pay"]

# Create data folder
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

# Patch httpx for Twitter client
original_client = httpx.Client


def patched_client(*args, **kwargs):
    if "headers" not in kwargs:
        kwargs["headers"] = {}

    user_agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
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


class GameSentimentAgent:
    def __init__(self):
        # Initialize Twitter client
        if not os.path.exists("cookies.json"):
            cprint("❌ No cookies.json found! Please run twitter_login.py first", "red")
            return

        self.twitter_client = Client()
        self.twitter_client.load_cookies("cookies.json")

        # Initialize sentiment analyzer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "finiteautomata/bertweet-base-sentiment-analysis"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "finiteautomata/bertweet-base-sentiment-analysis"
        )

        # Initialize contract integrator
        self.contract = ContractIntegrator()

    async def get_tweets(self, token: str) -> list:
        """Fetch tweets for a token"""
        collected_tweets = []
        try:
            tweets = await self.twitter_client.search_tweet(token, product="Latest")

            if tweets:
                for tweet in tweets:
                    if len(collected_tweets) >= TWEETS_PER_RUN:
                        break
                    if not any(
                        word.lower() in tweet.text.lower() for word in IGNORE_LIST
                    ):
                        collected_tweets.append(tweet.text)

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

        except TooManyRequests:
            cprint("Rate limit hit, waiting...", "yellow")
        except Exception as e:
            cprint(f"Error fetching tweets: {e}", "red")

        return collected_tweets

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        scores = outputs.logits.softmax(dim=1)
        return scores[0][1].item() - scores[0][0].item()  # positive - negative

    async def analyze_token(self, token: str):
        """Analyze sentiment for a token"""
        try:
            # Get tweets
            tweets = await self.get_tweets(token)
            if not tweets:
                return 0.0

            # Calculate sentiment
            sentiments = [self.analyze_sentiment(tweet) for tweet in tweets]
            avg_sentiment = sum(sentiments) / len(sentiments)

            # Log results
            self._log_sentiment(token, avg_sentiment)

            # Trigger contract action if threshold exceeded
            if abs(avg_sentiment) > SENTIMENT_THRESHOLD:
                self.contract.handle_sentiment(token, avg_sentiment)
                cprint(
                    f"Contract action triggered for {token}: {avg_sentiment}", "green"
                )

            return avg_sentiment

        except Exception as e:
            cprint(f"Error analyzing {token}: {e}", "red")
            return 0.0

    def _log_sentiment(self, token: str, sentiment: float):
        """Log sentiment to CSV"""
        data = {
            "timestamp": [datetime.now()],
            "token": [token],
            "sentiment": [sentiment],
        }
        df = pd.DataFrame(data)
        df.to_csv(
            SENTIMENT_HISTORY_FILE,
            mode="a",
            header=not os.path.exists(SENTIMENT_HISTORY_FILE),
        )


def main():
    """Test function"""
    try:
        agent = GameSentimentAgent()

        # Test with some sample sentiments
        test_cases = [
            ("ethereum", 0.8),  # Should trigger positive
            ("solana", -0.7),  # Should trigger negative
            ("bitcoin", 0.3),  # Should not trigger
        ]

        for token, sentiment in test_cases:
            cprint(f"\n🔄 Testing {token} with sentiment {sentiment}", "cyan")
            result = agent.analyze_token(token)
            cprint(f"Result: {result}", "yellow")

    except KeyboardInterrupt:
        cprint("\n👋 Agent shutting down gracefully...", "yellow")
    except Exception as e:
        cprint(f"\n❌ Fatal error: {str(e)}", "red")
        raise


if __name__ == "__main__":
    main()
