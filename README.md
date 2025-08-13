polars-u256-plugin
===================

[![PyPI](https://img.shields.io/pypi/v/polars_u256_plugin.svg)](https://pypi.org/project/polars_u256_plugin/) [![CI](https://github.com/elyase/polars-u256-plugin/actions/workflows/ci.yml/badge.svg)](https://github.com/elyase/polars-u256-plugin/actions/workflows/ci.yml) [![Publish](https://github.com/elyase/polars-u256-plugin/actions/workflows/publish_to_pypi.yml/badge.svg)](https://github.com/elyase/polars-u256-plugin/actions/workflows/publish_to_pypi.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
- **Full 256-bit precision**: Built on [ruint](https://github.com/recmo/uint), a high-performance Rust library for arbitrary-precision unsigned integers, ensuring exact arithmetic without overflow or precision loss
- **Complete operator support**: All standard arithmetic (`+`, `-`, `*`, `/`, `%`, `**`), bitwise (`&`, `|`, `^`, `~`, `<<`, `>>`), and comparison (`==`, `!=`, `<`, `<=`, `>`, `>=`) operations
- **Aggregation functions**: `sum()`, `min()`, `max()`, `mean()`, `value_counts()` with proper null handling
- **Series operations**: `cumsum()`, `diff()`, and other transformations for time-series analysis

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

### Display Utilities
```python
u256.format_u256_dataframe(df, cols)    # Format u256 columns as hex
u256.print_u256_dataframe(df)           # Print with hex formatting
df.with_u256_display("col")             # Add hex display columns
df.show_u256_hex("col")                 # Replace binary with hex
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
```

### I256 (Signed) - Complete API
All u256 functions available with `i256.` prefix, plus signed-specific:
```python
i256.div_euclid(a, b)       # Euclidean division
i256.rem_euclid(a, b)       # Euclidean remainder
i256.from_int(value)        # Supports negative integers
i256.to_int(col)            # Returns signed integers

# Namespace: .i256 (same operators as .u256)
pl.col("balance").i256 + pl.col("amount").i256
```

## Implementation Notes 
- Storage: U256/I256 are stored as 32 -byte big -endian Binary columns (BinaryView). This avoids Decimal128 limits and preserves exact integer values.
- Ingest patterns:
  - Hex strings: For very large values, supply as hex (prefixed with `0x`) and use `u256.from_hex(...)` (recommended for big data and interop).
  - Python ints (64 -bit range): If your integers fit within 64 -bit, use `u256.from_int(pl.col(...))` to convert an integer column to U256.
  - Constants: Use `u256.from_int(pl.lit(<python_int>))` to create constant U256 expressions (works for very large integers).
- Use Polars chunked builders or `from_iter_options(...).into_series()` (vs raw Arrow arrays).
- Prefer BinaryView-backed builders; avoid unsafe Series construction.
- Pre-size builders; use `append_null/append_option` for nulls.
- Semantics: div/mod by zero → null; u256 add/mul overflow → null.

### I256 (signed)
- Two’s complement over 256 bits.
- `div` truncates toward zero; `mod` carries dividend sign.
- Euclidean: `i256_div_euclid` / `i256_rem_euclid`.

Note: Constructing a DataFrame directly with Python ints > 64 -bit may result in non-integer dtype (e.g., Object) depending on your Polars build. In such cases, prefer providing values as hex strings and `from_hex`, or generate constants in expressions via `from_int(pl.lit(...))`.

## Credits

- [ruint](https://github.com/recmo/uint): High-performance Rust library for arbitrary-precision unsigned integers
- [polars-evm](https://github.com/sslivkoff/polars_evm): 
