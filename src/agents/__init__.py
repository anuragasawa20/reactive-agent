"""Initialize the agents package."""

from .game_sentiment_agent import GameSentimentAgent
from .contract_integration import (
    ApprovalContractIntegrator,
    ADDRESSES,
    ERC20_ABI,
)

# Create singleton instance
sentiment_agent = GameSentimentAgent()

__all__ = [
    "GameSentimentAgent",
    "ApprovalContractIntegrator",
    "ADDRESSES",
    "ERC20_ABI",
]
