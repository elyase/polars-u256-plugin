import polars as pl
import polars_u256_plugin as u256


def test_u256_namespace_ops_small():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).with_columns(
        a=u256.from_int(pl.col("a")),
        b=u256.from_int(pl.col("b")),
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


def test_u256_namespace_ne_compare():
    df = pl.DataFrame({"a": [1, 2], "b": [1, 3]}).with_columns(
        a=u256.from_int(pl.col("a")),
        b=u256.from_int(pl.col("b")),
    )
    out = df.select(ne=(pl.col("a").u256 != pl.col("b").u256))
    assert out["ne"].to_list() == [False, True]


def test_i256_namespace_ops():
    df = pl.DataFrame({"x": [-5, -1, 3]}).with_columns(
        x=u256.i256.from_int(pl.col("x"))
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


def test_df_u256_namespace_from_int_and_to_hex():
    df = pl.DataFrame({"balance": [1, 2], "amount": [3, 4]})
    # Convert both columns to u256 in-place
    df2 = df.u256.from_int(["balance", "amount"], replace=True)
    # Round-trip via to_int to verify conversion
    ints = df2.select(
        u256.to_int(pl.col("balance")).alias("b"),
        u256.to_int(pl.col("amount")).alias("a"),
    )
    assert ints["b"].to_list() == [1, 2]
    assert ints["a"].to_list() == [3, 4]

    # Add hex display column without replacing
    df3 = df2.u256.to_hex(["balance"], replace=False)
    hex_list = df3["balance_hex"].to_list()
    assert all(isinstance(x, str) and x.startswith("0x") for x in hex_list)
