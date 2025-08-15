import polars as pl
import polars_u256_plugin as u256


def test_display_helpers_add_and_replace():
    df = pl.DataFrame({"x": [1, 2]}).with_columns(v=u256.from_int(pl.col("x")))

    # Add hex display columns
    df2 = df.with_u256_display("v")
    assert "v_hex" in df2.columns
    assert all(isinstance(s, str) and s.startswith("0x") for s in df2["v_hex"].to_list())

    # Replace binary with hex display
    df3 = df.show_u256_hex("v")
    # the column 'v' should now be hex strings
    assert df3.schema["v"] == pl.String
    assert all(isinstance(s, str) and s.startswith("0x") for s in df3["v"].to_list())
