#!/usr/bin/env python3
"""Example: high-precision decimals using scaled integers with u256/i256.

We represent a Decimal d with scale S as an integer N = round(d * 10**S),
compute with exact integer arithmetic, then convert back for display.
"""

import polars as pl
import polars_u256_plugin as u256

S = 20  # scale (number of decimal places)
TEN_S = 10 ** S

df = pl.DataFrame({}).with_columns(
    a=u256.from_int(TEN_S),          # 1.000... (S decimals)
    b=u256.from_int(TEN_S),
)

# Multiply: (a*b)/10^S keeps scale S
df = df.with_columns(prod=u256.div(u256.mul(pl.col("a"), pl.col("b")), u256.from_int(TEN_S)))

# Divide: (a*10^S)/b keeps scale S
df = df.with_columns(div=u256.div(u256.mul(pl.col("a"), u256.from_int(TEN_S)), pl.col("b")))

# Show results in hex
out = df.with_columns(
    a_hex=u256.to_hex(pl.col("a")),
    prod_hex=u256.to_hex(pl.col("prod")),
    div_hex=u256.to_hex(pl.col("div")),
)
print(out.select(["a_hex", "prod_hex", "div_hex"]))

