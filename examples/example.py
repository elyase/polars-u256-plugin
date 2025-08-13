#!/usr/bin/env python3
"""Simple examples using Python ints (no hex).

1) Show why native Decimal128 is insufficient for very large integers
2) Do the same operations exactly with the U256 plugin
"""

import polars as pl
import polars_u256_plugin as u256


def with_plugin() -> pl.DataFrame:
    # Build from Python ints directly (coerced to 32-byte big-endian binary)
    df = pl.DataFrame({
        "a": [10**38, 10**38 + 5],
        "b": [12345678901234567890, 98765432109876543210],
    }).with_columns(
        a=u256.from_ints(pl.col("a")),
        b=u256.from_ints(pl.col("b")),
    )

    # U256 arithmetic stays exact. Use operator overloading + namespace.
    out = (
        df.with_columns(
            s=(pl.col("a").u256 + pl.col("b")),
            p=(pl.col("a").u256 * 2),
            q=(pl.col("b").u256 / 3),
        )
        .with_columns(
            s_hex=pl.col("s").u256.to_hex(),
            p_hex=pl.col("p").u256.to_hex(),
            q_hex=pl.col("q").u256.to_hex(),
        )
        .select(["s_hex", "p_hex", "q_hex"])
    )
    return out


if __name__ == "__main__":
    # Without plugin (Decimal128) – this overflows at 10**38
    # pl.DataFrame({"x": [10**38]}).select(pl.col("x").cast(pl.Decimal(38, 0)))

    # With plugin – exact arithmetic with Python ints
    res = with_plugin()
    print(res)
