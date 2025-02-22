from src.agents.contract_integration import ApprovalContractIntegrator


def verify_setup():
    try:
        integrator = ApprovalContractIntegrator()

        # Check connection
        print("Web3 Connection:", integrator.w3.is_connected())

        # Check account
        balance = integrator.w3.eth.get_balance(integrator.account.address)
        print(f"Account Balance: {integrator.w3.from_wei(balance, 'ether')} ETH")

        # Check contracts
        sub_count = integrator.get_subscription_count()
        print(f"Current Subscriptions: {sub_count}")

        pending_swaps = integrator.get_pending_swaps()
        print(f"Pending Swaps: {pending_swaps}")

        return True

    except Exception as e:
        print(f"Setup verification failed: {e}")
        return False


if __name__ == "__main__":
    verify_setup()
