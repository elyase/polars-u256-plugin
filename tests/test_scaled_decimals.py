import polars as pl
import polars_u256_plugin as u256


def u256_to_int_hex(expr: pl.Expr, df: pl.DataFrame) -> int:
    # Convert u256 binary to hex then parse as int
    hx = df.select(u256.to_hex(expr).alias("h"))["h"].item()
    return int(hx, 16)


def test_scaled_decimal_mul_div():
    S = 20
    TEN_S = 10 ** S
    df = pl.DataFrame({}).with_columns(
        a=u256.from_int(TEN_S),
        b=u256.from_int(TEN_S),
    )
    # c = (a*b)/10^S keeps scale S; equals TEN_S
    df = df.with_columns(c=u256.div(u256.mul(pl.col("a"), pl.col("b")), u256.from_int(TEN_S)))
    # d = (a*10^S)/b keeps scale S; equals TEN_S
    df = df.with_columns(d=u256.div(u256.mul(pl.col("a"), u256.from_int(TEN_S)), pl.col("b")))

    c_val = u256_to_int_hex(pl.col("c"), df)
    d_val = u256_to_int_hex(pl.col("d"), df)
    assert c_val == TEN_S
    assert d_val == TEN_S

