from web3 import Web3
from dotenv import load_dotenv
import os


def test_contract_connection():
    # Load environment variables
    load_dotenv()

    # Initialize Web3
    w3 = Web3(Web3.HTTPProvider(os.getenv("SEPOLIA_RPC")))

    # Test connection
    print(f"Connected to Web3: {w3.is_connected()}")

    # Test contract addresses
    service_address = os.getenv("APPROVAL_SERVICE_ADDRESS")
    swap_address = os.getenv("APPROVAL_SWAP_ADDRESS")

    print(f"Service contract exists: {w3.eth.get_code(service_address).hex() != '0x'}")
    print(f"Swap contract exists: {w3.eth.get_code(swap_address).hex() != '0x'}")


if __name__ == "__main__":
    test_contract_connection()
