// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/ApprovalService.sol";
import "../src/ApprovalMagicSwap.sol";
import "../src/ApprovalDemoToken.sol";
import "@reactive/src/abstract-base/AbstractReactive.sol";
import "@uniswap/v2-periphery/contracts/interfaces/IUniswapV2Router02.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract DeployScript is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerPrivateKey);

        // Deploy ApprovalService
        ApprovalService service = new ApprovalService(100, 1, 10);

        // Deploy Demo Tokens
        ApprovalDemoToken token1 = new ApprovalDemoToken("TK1", "TK1");
        ApprovalDemoToken token2 = new ApprovalDemoToken("TK2", "TK2");

        // Deploy ApprovalMagicSwap
        ApprovalMagicSwap swap = new ApprovalMagicSwap(
            address(service),
            address(token1),
            address(token2)
        );

        vm.stopBroadcast();
    }
}
