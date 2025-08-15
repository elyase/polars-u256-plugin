import polars as pl
import polars_u256_plugin as u256


def test_u256_cumsum_and_diff():
    df = pl.DataFrame({"x": [1, 2, 3]}).with_columns(v=u256.from_int(pl.col("x")))
    out = df.select(
        cs=u256.cumsum(pl.col("v")),
        df_=u256.diff(pl.col("v")),
    ).with_columns(
        cs_i=u256.to_int(pl.col("cs")),
        df_i=u256.to_int(pl.col("df_")),
    )
    assert out["cs_i"].to_list() == [1, 3, 6]
    assert out["df_i"].to_list() == [None, 1, 1]


def test_i256_cumsum_and_diff():
    df = pl.DataFrame({"x": [-2, 3, -1]}).with_columns(v=u256.i256.from_int(pl.col("x")))
    out = df.select(
        cs=u256.i256.cumsum(pl.col("v")),
        df_=u256.i256.diff(pl.col("v")),
    ).with_columns(
        cs_i=u256.i256.to_int(pl.col("cs")),
        df_i=u256.i256.to_int(pl.col("df_")),
    )
    assert out["cs_i"].to_list() == [-2, 1, 0]
    assert out["df_i"].to_list() == [None, 5, -4]
