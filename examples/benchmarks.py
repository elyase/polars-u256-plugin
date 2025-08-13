#!/usr/bin/env python3
import time

import polars as pl

import polars_u256_plugin as u256

n = 10_000_000
val = 12345

for n in [1000_000, 10_000_000, 100_000_000]:
    print(f"Summing {val} across {n:,} rows:")

    base_df = pl.LazyFrame({"x": [val] * n})

    # Native u64 sum
    start = time.time()
    base_df.select(pl.col("x").sum()).collect()
    print(f" u64: {time.time() - start:.3f}s")

    # u256 plugin sum (conversion + sum)
    start = time.time()
    base_df.with_columns(x=u256.from_ints(pl.col("x"))).select(
        u256.sum(pl.col("x"))
    ).collect()
    print(f"u256: {time.time() - start:.3f}s")

    # Python object sum via map_elements
    start = time.time()
    base_df.select(
        pl.col("x").map_elements(lambda x: x, return_dtype=pl.Object)
    ).select(pl.sum_horizontal(pl.col("x"))).collect()
    print(f" obj: {time.time() - start:.3f}s")
