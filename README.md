# polars-u256-plugin

Polars plugin for 256-bit unsigned (U256) and signed (I256, two's complement) integer operations. Values are stored as 32-byte big-endian binary arrays for precise blockchain/crypto computations and stable ordering.

## Installation

Preferred (once wheels are published):

```bash
pip install polars_u256_plugin
```

Dev/local build:

```bash
cargo build --release
# Python wrapper will find the library in target/release automatically
```

## API

```python
import polars as pl
import polars_u256_plugin as u256

df = pl.DataFrame({"amounts": ["0x100", "0x200"]}).with_columns(
    u256_col = u256.from_hex(pl.col("amounts"))
)

# Arithmetic: add, sub, mul, div, mod, pow
# Scalars can be passed as Python ints/hex strings
result = df.with_columns(
    doubled = u256.mul(pl.col("u256_col"), u256.lit(2))
)

# Aggregations: sum (returns a scalar in select and group_by contexts)
total = df.select(u256.sum(pl.col("u256_col")))

# Conversion back to hex for display
hex_result = result.with_columns(u256.to_hex(pl.col("doubled")))
```

## Status

**Working:**
- U256: arithmetic, comparisons, bitwise ops (bitand, bitor, bitxor, bitnot), shifts (shl, shr), sum aggregation, hex conversion, scalar coercion, display formatting
- I256: signed arithmetic (two's complement), truncating div/mod, Euclidean div/rem, comparisons, sum, hex/decimal coercion helpers

**Missing:** Mean/cumulative aggregations; additional aggregations

## Roadmap

1. Complete aggregations (mean, cumsum)
2. Add signed int256 support
3. Implement bitwise operations
4. PyPI distribution (wheels bundle native lib)

## Notes

- Data type: values are stored as `Binary` columns with fixed 32-byte big-endian representation. Lexicographic comparisons equal numeric ordering.
- Error/null semantics: invalid inputs or overflows yield nulls and set the plugin last error message.
- `pow`: large exponents are rejected (guarded) to prevent excessive compute.

### Signed I256 semantics
- Representation: two's complement over 256 bits (alias of alloy `I256`).
- Division: `div` truncates toward zero; `mod` follows dividend sign. Euclidean variants are provided as `i256_div_euclid`/`i256_rem_euclid` with non-negative remainder.

### Scaled-decimal pattern
See `examples/scaled_decimals.py` for exact decimal math using scaled U256 integers (store N = round(d * 10**S), adjust scale explicitly with mul/div).
