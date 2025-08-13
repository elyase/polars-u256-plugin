#!/usr/bin/env python3
"""
Group-by aggregation example - demonstrates the exact use case from Polars issue #15443

This example shows how to aggregate U256 values by groups, which is essential for:
- Blockchain transaction analysis (total spend per account)
- Token balance aggregation (total supply per token)  
- DeFi protocol analytics (TVL per pool)
"""

import polars as pl
import polars_u256_plugin as u256

def main():
    print("🔗 Blockchain Transaction Aggregation Example")
    print("=" * 50)
    
    # Simulate blockchain transaction data
    transactions = pl.DataFrame({
        "account_id": ["alice", "bob", "alice", "charlie", "bob", "alice", "charlie"],
        "tx_amount_wei": [
            1_000_000_000_000_000_000,    # 1 ETH in wei
            2_500_000_000_000_000_000,    # 2.5 ETH  
            500_000_000_000_000_000,     # 0.5 ETH
            10_000_000_000_000_000_000,   # 10 ETH
            1_200_000_000_000_000_000,    # 1.2 ETH
            750_000_000_000_000_000,     # 0.75 ETH
            3_300_000_000_000_000_000,    # 3.3 ETH
        ],
        "gas_used": [21000, 45000, 21000, 85000, 32000, 21000, 52000],
        "block_number": [18500000, 18500001, 18500002, 18500003, 18500004, 18500005, 18500006]
    })
    
    print("📊 Raw transaction data:")
    print(transactions)
    
    # Aggregate by account - the core use case from GitHub issue #15443
    print("\n💰 Total spending per account (U256 aggregation):")
    account_totals = transactions.group_by("account_id").agg([
        u256.sum(u256.from_int(pl.col("tx_amount_wei"))).alias("total_wei"),
        pl.col("gas_used").sum().alias("total_gas"),
        pl.len().alias("tx_count"),
        pl.col("block_number").min().alias("first_block"),
        pl.col("block_number").max().alias("last_block")
    ]).with_columns(
        # Add human-readable ETH amounts
        (u256.to_int(pl.col("total_wei")) / 1_000_000_000_000_000_000).alias("total_eth"),
        u256.to_hex(pl.col("total_wei")).alias("total_wei_hex")
    ).sort("total_eth", descending=True)
    
    print(account_totals.select(["account_id", "total_eth", "tx_count", "total_gas"]))
    
    # Show hex representation for verification
    print("\n🔍 Detailed breakdown with hex values:")
    for row in account_totals.iter_rows(named=True):
        print(f"{row['account_id']:8s}: {row['total_eth']:6.2f} ETH "
              f"({row['tx_count']} txs, {row['total_gas']:,} gas)")
        print(f"         Wei: {row['total_wei_hex']}")
    
    # Demonstrate large number handling with hex input
    print(f"\n🚀 Handling very large numbers (from hex):")
    large_amounts = pl.DataFrame({
        "protocol": ["uniswap", "compound", "aave"],
        "tvl_hex": [
            hex(2**200),  # Astronomically large TVL as hex
            hex(2**180),  
            hex(2**190)
        ]
    })
    
    protocol_total = large_amounts.select(
        u256.sum(u256.from_hex(pl.col("tvl_hex"))).alias("total_tvl")
    ).with_columns(
        u256.to_hex(pl.col("total_tvl")).alias("total_tvl_hex")
    )
    
    print("Total Protocol TVL (impossible with standard int types):")
    print(f"Hex: {protocol_total[0, 'total_tvl_hex']}")
    
    print("\n✅ All operations completed successfully!")
    print("🎯 This demonstrates the exact group_by().agg(u256.sum()) use case")
    print("   mentioned in Polars issue #15443")

if __name__ == "__main__":
    main()