# 🌙 TokenPulse Sentiment Agent

Real-time crypto sentiment analysis system with autonomous reactive network integration. Built with ❤️ by Mavens.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Sepolia Testnet ETH
- Ethereum Wallet with Private Key

### Installation
```bash
# Clone repository
git clone 
cd safe-sentiment-agent

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

## 🔧 Configuration
1. Get your Sepolia RPC URL from Infura/Alchemy
2. Edit `.env` file:
```env
SEPOLIA_RPC="your_rpc_url_here"
PRIVATE_KEY="your_private_key_here"
```

## 🔄 Smart Contract Integration
The system integrates with the following contracts on Sepolia:
- ApprovalService: `0x96BB3466D534490724eD1139B5F153CC9F8f6291`
- Token1: `0x40907ee93c44130da447fA7B79D6E73Be7932E73`
- Token2: `0x34a530d1c8e7Ba526c459223c4A03f076708C083`

## 🌐 Reactive Network Architecture

### Overview
The TokenPulse reactive network is a decentralized system that autonomously responds to market sentiment through smart contract interactions:

```
Sentiment Analysis ➜ Smart Contract Triggers ➜ Automated Trading Actions
```

### Components

#### 1. Approval Service (`0x96BB3466D534490724eD1139B5F153CC9F8f6291`)
- Manages subscription states
- Handles token approvals
- Validates transaction permissions
- Monitors gas optimization

#### 2. Token Contracts
- Token1 (`0x40907ee93c44130da447fA7B79D6E73Be7932E73`): Primary trading token
- Token2 (`0x34a530d1c8e7Ba526c459223c4A03f076708C083`): Secondary trading token

#### 3. Reactive Mechanisms
```
Sentiment > 0.4  → Buy Signal  → Token1 → Token2 Swap
Sentiment < -0.4 → Sell Signal → Token2 → Token1 Swap
```

### Network Flow
1. **Sentiment Detection**
   - Continuous monitoring of token sentiment
   - Real-time threshold checking
   - Event triggering on threshold breach

2. **Smart Contract Interaction**
   - Automatic approval requests
   - Gas-optimized transactions
   - Fail-safe error handling

3. **Transaction Execution**
   - Automated token swaps
   - Transaction verification
   - State updates

### Safety Features
- Transaction amount limits
- Gas price monitoring
- Slippage protection
- Automatic error recovery

### Monitoring
```bash
# Monitor network status
python src/agents/game_sentiment_agent.py --monitor

# View transaction history
python src/agents/game_sentiment_agent.py --history
```

### Network States
1. **Idle**: Monitoring sentiment
2. **Triggered**: Sentiment threshold reached
3. **Executing**: Processing transaction
4. **Completed**: Transaction confirmed
5. **Error**: Handling failure

### Performance Metrics
- Average response time: < 2 seconds
- Transaction success rate: > 95%
- Gas optimization: Dynamic adjustment
- Error recovery: Automatic retry mechanism

## 🖥️ Running the System

### Frontend Dashboard
```bash
streamlit run src/frontend/sentiment_dashboard.py
```

### Sentiment Agent
```bash
python src/agents/game_sentiment_agent.py
```

### Automated Actions
The system automatically:
- Monitors sentiment for configured tokens
- Triggers buy/sell actions when sentiment exceeds thresholds
- Handles token approvals and swaps
- Logs all transactions and sentiment data

## 📂 Project Structure
```
safe-sentiment-agent/
├── src/
│   ├── agents/
│   │   ├── contract_integration.py    # Smart contract interactions
│   │   ├── game_sentiment_agent.py    # Main sentiment analysis agent
│   ├── frontend/                      # Streamlit dashboard
│   └── data/                          # Analysis and transaction history
├── safe-sentiment-contracts/          # Smart contract artifacts
│   └── ABI's/                        # Contract ABIs
└── .env.example                       # Environment configuration
```

## 🔗 Contract Integration
The system integrates with:
- ApprovalService: Manages subscriptions and approvals
- Token Contracts: ERC20 tokens for trading
- Swap Contract: Handles token swaps based on sentiment

## 📊 Sentiment Thresholds
- Positive trigger: > 0.4
- Negative trigger: < -0.4
- Action: Automated token swaps

## 📌 Dependencies
- **Web3**: `web3.py`, `eth-account`
- **Analysis**: `pandas`, `numpy`
- **Frontend**: `streamlit`, `plotly`
- **Utils**: `python-dotenv`, `termcolor`

## 🔐 Security
- Private keys should be kept secure
- Use environment variables for sensitive data
- Test with small amounts first

💡 **Tip**: Monitor the dashboard for real-time transaction status and sentiment trends!
