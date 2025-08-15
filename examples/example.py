#!/usr/bin/env python3
"""Minimal API tour for polars-u256-plugin.

Highlights:
- Build u256 columns from ints/hex/literals
- Do arithmetic/comparisons in expressions
- Aggregate with u256.sum
- Convert to hex for display
"""

from __future__ import annotations

import polars as pl
import polars_u256_plugin as u256


def main() -> None:
    # Example inputs: mix of Python ints and hex strings
    df = pl.DataFrame(
        {
            "a": [10**30, 42],
            "b_hex": [
                "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",  # 2**256-1
                "0x2",
            ],
        }
    )

    # 1) Build u256 columns
    df = df.with_columns(
        a256=u256.from_int(pl.col("a")),        # from integer column
        b256=u256.from_hex(pl.col("b_hex")),     # from hex string column
        c256=u256.lit(10**40),                   # literal u256 constant
    )

    # 2) Arithmetic & comparisons (operator overloading + namespace)
    calc = df.select(
        sum_ab=(pl.col("a256").u256 + pl.col("b256")).u256.to_hex(),
        twice_a=(pl.col("a256").u256 * 2).u256.to_hex(),  # Python int auto-coerced
        a_lt_b=(pl.col("a256").u256 < pl.col("b256")),
    )

    # 3) Aggregation (u256.sum) and presentation (to_hex)
    agg = (
        df.select(u256.sum(pl.col("a256")).alias("total_a"))
        .with_columns(total_a_hex=u256.to_hex(pl.col("total_a")))
        .select("total_a_hex")
    )

    print("Calculations:\n", calc)
    print("\nAggregation:\n", agg)

    # 4) Group-by aggregation (u256.sum per group)
    tx = pl.DataFrame(
        {
            "wallet": ["a", "a", "b"],
            "amt": ["0x01", "0x02", "0x03"],
        }
    ).with_columns(amt256=u256.from_hex(pl.col("amt")))
    gb = tx.group_by("wallet").agg(total=u256.sum(pl.col("amt256")))
    gb = gb.with_columns(total_hex=u256.to_hex(pl.col("total"))).select(["wallet", "total_hex"])
    print("\nGroup-by totals:\n", gb)


if __name__ == "__main__":
    main()
