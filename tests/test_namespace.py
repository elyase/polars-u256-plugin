import polars as pl
import polars_u256_plugin as u256


def test_u256_namespace_ops_small():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).with_columns(
        a=u256.from_ints(pl.col("a")),
        b=u256.from_ints(pl.col("b")),
    )
    out = df.with_columns(
        s=(pl.col("a").u256 + pl.col("b")),
        p=(pl.col("a").u256 * 2),
        q=(pl.col("b").u256 / 2),
    ).with_columns(
        s_i=u256.to_int(pl.col("s")),
        p_i=u256.to_int(pl.col("p")),
        q_i=u256.to_int(pl.col("q")),
    )
    assert out.select("s_i").to_series().to_list() == [5, 7, 9]
    assert out.select("p_i").to_series().to_list() == [2, 4, 6]
    assert out.select("q_i").to_series().to_list() == [2, 2, 3]


def test_i256_namespace_ops():
    df = pl.DataFrame({"x": [-5, -1, 3]}).with_columns(
        x=u256.i256.from_ints(pl.col("x"))
    )
    out = df.with_columns(
        s=(pl.col("x").i256 + 2),
        f=(pl.col("x").i256 // 2),  # euclidean floor division
    )
    # Convert to hex just to ensure expressions run
    hex_df = out.select(
        pl.col("s").i256.to_hex().alias("s_hex"),
        pl.col("f").i256.to_hex().alias("f_hex"),
    )
    assert hex_df.height == 3
