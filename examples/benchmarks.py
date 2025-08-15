#!/usr/bin/env python3
"""
Extended benchmarks for polars-u256-plugin performance investigation.

Compares:
- Native u64 sum
- u256 sum with conversion included (from_int + sum)
- u256 sum on pre-converted column (sum only)
- Native Int128 sum
- Native Decimal(sum) with scale=0
- Python Object via map_elements + sum (worst-case)

Also runs a simple arithmetic test (add/mul) to gauge per-element costs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import polars as pl

import polars_u256_plugin as u256


@dataclass
class Result:
    label: str
    seconds: float


def bench_sum(n: int, val: int) -> list[Result]:
    results: list[Result] = []
    lf = pl.LazyFrame({"x": [val] * n})

    # u64 sum
    t0 = time.time()
    lf.select(pl.col("x").sum()).collect()
    results.append(Result("u64: sum", time.time() - t0))

    # u256 sum: conversion + sum
    t0 = time.time()
    lf.with_columns(x=u256.from_int(pl.col("x"))).select(u256.sum(pl.col("x"))).collect()
    results.append(Result("u256: from_int+sum", time.time() - t0))

    # u256 sum: pre-convert then sum only
    t0 = time.time()
    df_u256 = (
        lf.with_columns(x256=u256.from_int(pl.col("x"))).select("x256").collect()
    )
    t_conv = time.time() - t0
    t0 = time.time()
    df_u256.select(u256.sum(pl.col("x256")))
    results.append(Result("u256: sum only (preconv)", time.time() - t0))
    results.append(Result("u256: conversion only", t_conv))

    # Int128
    t0 = time.time()
    df_i128 = (
        lf.with_columns(x128=pl.col("x").cast(pl.Int128)).select("x128").collect()
    )
    t_conv_i128 = time.time() - t0
    t0 = time.time()
    df_i128.select(pl.col("x128").sum())
    results.append(Result("i128: sum only (preconv)", time.time() - t0))
    results.append(Result("i128: conversion only", t_conv_i128))

    # Decimal(scale=0)
    t0 = time.time()
    df_dec = (
        lf.with_columns(xd=pl.col("x").cast(pl.Decimal(scale=0))).select("xd").collect()
    )
    t_conv_dec = time.time() - t0
    t0 = time.time()
    df_dec.select(pl.col("xd").sum())
    results.append(Result("Decimal(0): sum only (preconv)", time.time() - t0))
    results.append(Result("Decimal(0): conversion only", t_conv_dec))

    # Python object path (worst-case): map into Python space then sum
    t0 = time.time()
    lf.select(pl.col("x").map_elements(lambda x: x, return_dtype=pl.Object)).select(
        pl.sum_horizontal(pl.col("x"))
    ).collect()
    results.append(Result("Object: map+sum", time.time() - t0))

    return results


def bench_elementwise(n: int, val: int) -> list[Result]:
    results: list[Result] = []
    lf = pl.LazyFrame({"x": [val] * n, "y": [val + 1] * n})

    # u256 add/mul (preconvert once)
    df_u256 = (
        lf.with_columns(x=u256.from_int(pl.col("x")), y=u256.from_int(pl.col("y"))).collect()
    )
    t0 = time.time()
    df_u256.lazy().select(u256.add(pl.col("x"), pl.col("y"))).collect()
    results.append(Result("u256: add", time.time() - t0))

    t0 = time.time()
    df_u256.lazy().select(u256.mul(pl.col("x"), pl.col("y"))).collect()
    results.append(Result("u256: mul", time.time() - t0))

    # i128 add/mul
    df_i128 = lf.with_columns(
        x=pl.col("x").cast(pl.Int128), y=pl.col("y").cast(pl.Int128)
    ).collect()
    t0 = time.time()
    df_i128.lazy().select((pl.col("x") + pl.col("y")).sum()).collect()
    results.append(Result("i128: add", time.time() - t0))

    t0 = time.time()
    df_i128.lazy().select((pl.col("x") * pl.col("y")).sum()).collect()
    results.append(Result("i128: mul", time.time() - t0))

    return results


def main() -> None:
    for n in (1_000_000, 5_000_000):
        print(f"\nRows: {n:,}")
        print("== Aggregations ==")
        for r in bench_sum(n, 12345):
            print(f"{r.label:28s} {r.seconds:8.3f}s")

        print("\n== Elementwise (add/mul) ==")
        for r in bench_elementwise(n, 12345):
            print(f"{r.label:28s} {r.seconds:8.3f}s")


if __name__ == "__main__":
    main()
