"""
🌙 TokenPulse Sentiment Agent - Built with love by Mavens 🚀
Simplified version for reactive network testing
"""

import os
from pathlib import Path
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from dotenv import load_dotenv
from termcolor import cprint
import pandas as pd
from datetime import datetime
from src.agents.contract_integration import ContractIntegrator

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Constants
SENTIMENT_THRESHOLD = 0.4  # Lower threshold to match contract integration
DATA_FOLDER = Path(__file__).parent.parent / "data/sentiment_game"
SENTIMENT_HISTORY_FILE = DATA_FOLDER / "sentiment_history.csv"


class GameSentimentAgent:
    def __init__(self):
        self.contract = ContractIntegrator()
        DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    def analyze_sentiment(self, token: str, test_sentiment: float = 0.0):
        """Simplified sentiment analysis for testing"""
        try:
            # Log the sentiment
            self._log_sentiment(token, test_sentiment)

            # Handle contract interaction if sentiment exceeds threshold
            if abs(test_sentiment) > SENTIMENT_THRESHOLD:
                self.contract.handle_sentiment(token, test_sentiment)
                cprint(
                    f"✅ Contract action triggered for {token}: {test_sentiment}",
                    "green",
                )

            return test_sentiment

        except Exception as e:
            cprint(f"❌ Error in sentiment analysis: {str(e)}", "red")
            return 0.0

    def _log_sentiment(self, token: str, sentiment: float):
        """Log sentiment to CSV file"""
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
            result = agent.analyze_sentiment(token, sentiment)
            cprint(f"Result: {result}", "yellow")

    except KeyboardInterrupt:
        cprint("\n👋 Agent shutting down gracefully...", "yellow")
    except Exception as e:
        cprint(f"\n❌ Fatal error: {str(e)}", "red")
        raise


if __name__ == "__main__":
    main()
