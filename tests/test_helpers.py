import polars as pl
import polars_u256_plugin as u256


def test_validate_hex_and_range_expr_and_int():
    df = pl.DataFrame({
        "h": ["0x1", "0x", None],
        # keep values within i64/u64 to allow DataFrame construction
        "i": [0, -1, 123],
    })

    out = df.select(
        valid_hex=u256.validate_hex(pl.col("h")),
        in_range_expr=u256.validate_range(pl.col("i")),
        in_range_small=u256.validate_range(42),
        in_range_big=u256.validate_range(1 << 300),
    )
    # "0x" is treated as zero and thus valid
    assert out["valid_hex"].to_list() == [True, True, False]
    # -1 is not valid for u256.from_int(expr) -> null, so validate_range False
    assert out["in_range_expr"].to_list() == [True, False, True]
    assert out["in_range_small"].all() is True
    assert out["in_range_big"].any() is False


def test_u256_from_int_signed_series_negative_becomes_null():
    df = pl.DataFrame({"x": [1, -1, 2]}).with_columns(v=u256.from_int(pl.col("x")))
    # to_hex should be null for the negative element
    out = df.with_columns(h=u256.to_hex(pl.col("v")))
    assert out["h"].to_list()[1] is None


def test_i256_from_hex_negative_and_positive():
    df = pl.DataFrame({"s": ["-0x1", "0x2"]}).with_columns(v=u256.i256.from_hex(pl.col("s")))
    out = df.with_columns(i=u256.i256.to_int(pl.col("v")))
    assert out["i"].to_list() == [-1, 2]
