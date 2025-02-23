from web3 import Web3
from eth_account import Account
import os
from dotenv import load_dotenv
import json
from termcolor import cprint

# Load environment variables
load_dotenv()

# Contract ABIs
APPROVAL_SERVICE_ABI = [
    # Add minimal ABI for required functions
    {
        "inputs": [],
        "name": "getSubscriptionCount",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"type": "address"}],
        "name": "subscribe",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

APPROVAL_SWAP_ABI = [
    # Add minimal ABI for swap functions
    {
        "inputs": [],
        "name": "getPendingSwaps",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getTotalVolume",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
]

# Contract addresses
ADDRESSES = {
    "APPROVAL_SERVICE": "0x96BB3466D534490724eD1139B5F153CC9F8f6291",
    "APPROVAL_LISTENER": "0xDf0B7F1BAB729A8Bb280727F88739f4015D9390B",
    "TOKEN1": "0x40907ee93c44130da447fA7B79D6E73Be7932E73",
    "TOKEN2": "0x34a530d1c8e7Ba526c459223c4A03f076708C083",
    "UNISWAP_PAIR": "0x73E9E13E821ed31DA570dC24908efE81860b1e2E",
    "SWAP": "0xDee41516471b52A662d3A2af70639CEF0A77fFA0",
    "EXCH_ADDRESS": "0x7522526057E7b841e8682A351E0eFCa24C3B363A",
}


class ApprovalContractIntegrator:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(os.getenv("SEPOLIA_RPC")))
        self.account = Account.from_key(os.getenv("PRIVATE_KEY"))

        # Initialize contracts
        self.approval_service = self.w3.eth.contract(
            address=os.getenv("APPROVAL_SERVICE_ADDRESS"), abi=APPROVAL_SERVICE_ABI
        )

        self.approval_swap = self.w3.eth.contract(
            address=os.getenv("APPROVAL_SWAP_ADDRESS"), abi=APPROVAL_SWAP_ABI
        )

    def get_subscription_count(self) -> int:
        """Get number of active subscriptions"""
        try:
            return self.approval_service.functions.getSubscriptionCount().call()
        except Exception as e:
            print(f"Error getting subscription count: {e}")
            return 0

    def get_pending_swaps(self) -> int:
        """Get number of pending swaps"""
        try:
            return self.approval_swap.functions.getPendingSwaps().call()
        except Exception as e:
            print(f"Error getting pending swaps: {e}")
            return 0

    def get_total_volume(self) -> float:
        """Get total trading volume"""
        try:
            wei_volume = self.approval_swap.functions.getTotalVolume().call()
            return self.w3.from_wei(wei_volume, "ether")
        except Exception as e:
            print(f"Error getting total volume: {e}")
            return 0.0

    def handle_sentiment_trigger(self, token: str, sentiment_score: float):
        """Trigger contract actions based on sentiment"""
        try:
            SENTIMENT_THRESHOLD = 0.6
            if abs(sentiment_score) > SENTIMENT_THRESHOLD:
                # Prepare transaction
                swap_amount = self.w3.to_wei(0.1, "ether")  # Example amount

                # Build transaction
                nonce = self.w3.eth.get_transaction_count(self.account.address)

                # Execute swap based on sentiment direction
                if sentiment_score > 0:
                    # Positive sentiment - buy
                    tx = self.approval_swap.functions.swapExactETHForTokens(
                        swap_amount, self.account.address
                    ).build_transaction(
                        {
                            "from": self.account.address,
                            "value": swap_amount,
                            "gas": 200000,
                            "nonce": nonce,
                        }
                    )
                else:
                    # Negative sentiment - sell
                    tx = self.approval_swap.functions.swapExactTokensForETH(
                        swap_amount, self.account.address
                    ).build_transaction(
                        {
                            "from": self.account.address,
                            "gas": 200000,
                            "nonce": nonce,
                        }
                    )

                # Sign and send transaction
                signed_tx = self.account.sign_transaction(tx)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

                # Wait for transaction receipt
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                return receipt.status == 1

        except Exception as e:
            print(f"Error in sentiment trigger: {e}")
            return False


class ContractIntegrator:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(os.getenv("SEPOLIA_RPC")))
        self.account = Account.from_key(os.getenv("PRIVATE_KEY"))

        # Initialize token contracts
        self.token1 = self.w3.eth.contract(address=ADDRESSES["TOKEN1"], abi=ERC20_ABI)
        self.token2 = self.w3.eth.contract(address=ADDRESSES["TOKEN2"], abi=ERC20_ABI)
        self.swap = self.w3.eth.contract(
            address=ADDRESSES["SWAP"], abi=APPROVAL_SWAP_ABI
        )

    def handle_sentiment(self, token: str, sentiment_score: float):
        """Handle sentiment score and trigger appropriate contract action"""
        try:
            if abs(sentiment_score) > 0.4:  # Threshold for action
                amount = self.w3.to_wei(0.1, "ether")  # Standard amount for testing

                if sentiment_score > 0:
                    # Positive sentiment - Approve and swap token0 for token1
                    cprint(f"🔄 Initiating token0 to token1 swap for {token}", "cyan")
                    return self._execute_token_swap(self.token1, amount)
                else:
                    # Negative sentiment - Approve and swap token1 for token0
                    cprint(f"🔄 Initiating token1 to token0 swap for {token}", "cyan")
                    return self._execute_token_swap(self.token2, amount)
            return True
        except Exception as e:
            cprint(f"❌ Error handling sentiment: {e}", "red")
            return False

    def _execute_token_swap(self, token: str, amount: int):
        """Execute token swap through ApprovalMagicSwap"""
        try:
            # First approve the swap contract
            tx = token.functions.approve(ADDRESSES["SWAP"], amount).build_transaction(
                {
                    "from": self.account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.account.address),
                }
            )

            # Sign and send approval
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt.status == 1:
                cprint("✅ Approval successful", "green")
                return True
            return False

        except Exception as e:
            cprint(f"❌ Error executing swap: {e}", "red")
            return False
