polars-u256-plugin
===================

[![PyPI](https://img.shields.io/pypi/v/polars_u256_plugin.svg)](https://pypi.org/project/polars_u256_plugin/) [![CI](https://github.com/elyase/polars-u256-plugin/actions/workflows/ci.yml/badge.svg)](https://github.com/elyase/polars-u256-plugin/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Polars plugin for 256-bit integer arithmetic.

## 🙋 Why
Since Polars tops out at [Int128](https://docs.pola.rs/api/python/stable/reference/datatypes.html), this will fail:

```python
melania_bags = 2**129
pl.DataFrame(x=[melania_bags])

# OverflowError: int value too large for Polars integer types
```


**with `polars_u256_plugin` any of these work:**

```python
import polars_u256_plugin as u256

df = pl.select(
    x=u256.from_int(melania_bags),
    y=u256.lit(melania_bags),
    z=u256.from_hex(hex(melania_bags)),
)
```

In addition, you can perform full 256-bit arithmetic, bitwise ops, comparisons, and aggregations in Polars expressions with safe overflow handling (invalid ops yield null). 

See `examples/example.py` and `examples/benchmarks.py` for more.

## 🚀 Quickstart

```bash
pip install polars_u256_plugin
```

## ✨ Features

### Core Arithmetic & Operations
- **Full 256-bit precision**: Built on [ruint](https://github.com/recmo/uint), ensuring exact arithmetic without precision loss
- **Operators**: Arithmetic (`+`, `-`, `*`, `/`, `%`, `**`), bitwise (`&`, `|`, `^`, `~`, `<<`, `>>`), comparisons (`==`, `<`, `<=`, `>`, `>=`)
- **Aggregation**: `sum()`
- **Series ops**: `cumsum()`, `diff()`

### Data Types & Precision
- **U256 (unsigned)**: Full 0 to 2²⁵⁶-1 range for token balances, wei amounts, and large counters
- **I256 (signed)**: Two's complement signed integers with proper overflow handling and Euclidean division/remainder operations
- **Exact decimal arithmetic**: Enables high-precision financial calculations by storing scaled integers (e.g., storing cents as integers scaled by 10²)
- **Safe overflow behavior**: Operations return null on overflow rather than wrapping, preventing silent errors

### Integration & Usability  
- **Native Polars expressions**: Full integration with `select()`, `with_columns()`, `group_by().agg()`, and all other Polars operations
- **Operator overloading**: Natural syntax like `pl.col("balance").u256 + pl.col("amount").u256` 
- **Expression namespaces**: Fluent API via `.u256` and `.i256` namespaces for operations like `.to_hex()`, `.sum()`
- **Multiple input formats**: Accept Python integers, hex strings (`0x...`), and raw bytes with automatic coercion

### Performance & Storage
- **Vectorized operations**: All arithmetic implemented in Rust using ruint for optimal performance
- **Efficient storage**: Uses 32-byte big-endian binary representation via Polars BinaryView for memory efficiency
- **Lazy evaluation**: Full compatibility with Polars lazy evaluation and query optimization
- **Benchmark tested**: Significantly faster than Python object fallback methods for large datasets

### Blockchain & DeFi Ready
- **Ethereum compatibility**: Native handling of wei values, token amounts, and smart contract integers
- **Hex string support**: Seamless conversion to/from hexadecimal for blockchain data interoperability  
- **No precision loss**: Maintains full 256-bit precision throughout complex calculation pipelines
- **Production tested**: Powers real-world blockchain analytics and DeFi applications

## API

### Initialization
```python
u256.from_int(value)        # Python int → u256 expression  
u256.from_int(pl.col())     # Convert int column to u256 (preferred)
u256.from_hex(pl.col())     # Hex strings → u256
u256.lit(value)             # Create u256 literal (int/hex/bytes)
```

### Helpers & Constants
```python
# Validation helpers
u256.validate_hex(pl.col("hex_str"))     # bool: valid hex/binary → u256
u256.validate_range(pl.col("int_col"))   # bool: fits in unsigned 256-bit

# Common constants (as expressions)
u256.MAX_VALUE   # 2**256 - 1
u256.MIN_VALUE   # 0
i256.MAX_VALUE   # 2**255 - 1
i256.MIN_VALUE   # -2**255
```

### Arithmetic
```python
u256.add(a, b)              # Addition
u256.sub(a, b)              # Subtraction  
u256.mul(a, b)              # Multiplication
u256.div(a, b)              # Division (null on div by zero)
u256.mod(a, b)              # Modulo
u256.pow(a, b)              # Exponentiation
```

### Comparisons  
```python
u256.eq(a, b)               # Equal
u256.lt(a, b), u256.le(a, b) # Less than, less/equal
u256.gt(a, b), u256.ge(a, b) # Greater than, greater/equal
```

### Bitwise Operations
```python
u256.bitand(a, b)           # Bitwise AND
u256.bitor(a, b)            # Bitwise OR
u256.bitxor(a, b)           # Bitwise XOR
u256.bitnot(a)              # Bitwise NOT
u256.shl(a, bits)           # Left shift
u256.shr(a, bits)           # Right shift
```

### Aggregations & Conversions
```python
u256.sum(col)               # Sum aggregation
u256.to_hex(col)            # → hex strings (0x...)
u256.to_int(col)            # → Python int (if fits in i64)
```

### Group-By Aggregations
```python
df.group_by("account_id").agg(
    u256.sum(u256.from_int(pl.col("tx_amount"))).alias("total_spent")
)

# Multiple aggregations
df.group_by("token_address").agg([
    u256.sum(u256.from_int(pl.col("balance"))).alias("total_supply"),
    pl.len().alias("holder_count"),
    pl.col("last_updated").max()
])

# Mixed with regular Polars aggregations
df.group_by("wallet").agg([
    u256.sum(u256.from_int(pl.col("wei_balance"))).alias("total_wei"),
    pl.col("gas_used").sum(),
    pl.col("block_number").max()
]).with_columns(
    # Convert to readable hex for display
    u256.to_hex(pl.col("total_wei")).alias("total_wei_hex")
)
```

### Display Utilities
```python
u256.format_u256_dataframe(df, cols)    # Format u256 columns as hex
u256.print_u256_dataframe(df)           # Print with hex formatting
df.with_u256_display("col")             # Add hex display columns
df.show_u256_hex("col")                 # Replace binary with hex
df.u256.from_int(["balance", "amount"], replace=True)  # Convert int cols → u256
df.u256.to_hex(["balance"], replace=False)              # Add balance_hex
```

### Fluent API (.u256 namespace)
```python
# Arithmetic operators: +, -, *, /, //, %, **
pl.col("balance").u256 + pl.col("amount").u256
pl.col("value").u256 * 2

# Bitwise operators: &, |, ^, ~, <<, >>  
pl.col("flags").u256 & 0xFF

# Comparisons: ==, <, <=, >, >=
pl.col("a").u256 < pl.col("b").u256

# Methods
pl.col("value").u256.to_hex()
pl.col("data").u256.sum()
# Aggregations
pl.col("data").u256.min()
pl.col("data").u256.max()
pl.col("data").u256.mean()
```

### I256 (Signed) - Complete API
Supported operations (signed, two's complement): arithmetic (+, -, *, /, %, //=euclid),
comparisons, aggregations (sum/min/max/mean), series ops (cumsum/diff), and helpers.
```python
i256.div_euclid(a, b)       # Euclidean division
i256.rem_euclid(a, b)       # Euclidean remainder
i256.from_int(value_or_col) # Python int or Polars Expr; supports negatives
i256.to_int(col)            # Returns signed integers

# Namespace: .i256 (same operators as .u256)
pl.col("balance").i256 + pl.col("amount").i256
```

## Performance

```
Rows: 5,000,000
== Aggregations ==
u64: sum                        0.001s

u256: sum only (preconv)        0.042s
u256: conversion only           0.035s
u256: from_int+sum              0.081s

i128: sum only (preconv)        0.001s
i128: conversion only           0.006s

Decimal(0): sum only (preconv)  0.001s
Decimal(0): conversion only     0.014s

Python Object: map+sum          0.215s
```

See `examples/benchmarks.py` for more.

Tip: preconvert once (`with_columns(x256=u256.from_int(...))`) and reuse. 

## Implementation Notes
- **Storage**: U256/I256 are 32-byte big-endian binary columns.
- **Semantics**:
  - Division: u256 uses truncating integer division; i256 has truncating `div` and Euclidean `div_euclid`/`rem_euclid`.
  - Remainder: `i256.mod` carries dividend sign; Euclidean remainder is non-negative.
  - Errors: div/mod by zero → null.
  - Overflow: u256 add/mul overflow → null; i256 add/sub/mul wrap modulo 2²⁵⁶.
  - `to_int`: returns null if the value does not fit in signed 64-bit.

### I256 (signed)
- Two’s complement over 256 bits.
- `div` truncates toward zero; `mod` carries dividend sign.
- Euclidean: `i256.div_euclid(...)` / `i256.rem_euclid(...)`.

Note: Constructing a DataFrame directly with Python ints > 64 -bit may result in non-integer dtype (e.g., Object) depending on your Polars build. In such cases, prefer providing values as hex strings and `from_hex`, or generate constants in expressions via `from_int(pl.lit(...))`.

## Credits

- [ruint](https://github.com/recmo/uint): High-performance Rust library for arbitrary-precision unsigned integers
- [polars-evm](https://github.com/sslivkoff/polars_evm): Utilities for working with EVM data in polars
