import polars as pl
import polars_u256_plugin as u256


def test_from_hex_even_odd_and_padding_and_invalid():
    df = pl.DataFrame({
        "h": ["0x1", "0x01", "0x001", "0x" + "f" * 64, "xyz", "0x" + "ff" * 33, None],
    }).with_columns(v=u256.from_hex(pl.col("h")))

    out = df.with_columns(hex=u256.to_hex(pl.col("v")))
    # Valid rows normalize to 32-byte hex with 0x prefix
    assert out["hex"][:4].str.len_bytes().unique().to_list() == [66]
    # Invalid inputs become null
    assert out["hex"][4:].null_count() == 3


def test_from_bytes_and_lit_helpers():
    df = pl.DataFrame({}).with_columns(
        a=u256.lit(b"\x01\x02"),
        b=u256.from_int(5),
    )
    out = df.with_columns(a_hex=u256.to_hex(pl.col("a")), b_hex=u256.to_hex(pl.col("b")))
    assert out["a_hex"].str.starts_with("0x").all()
    assert out["b_hex"].str.ends_with("05").all()


def test_add_sub_mul_div_mod_and_div_by_zero():
    df = pl.DataFrame({
        "a": ["0x01", "0x10"],
        "b": ["0x02", "0x00"],
    }).with_columns(a=u256.from_hex(pl.col("a")), b=u256.from_hex(pl.col("b")))
    out = df.with_columns(
        s=u256.add(pl.col("a"), pl.col("b")),
        d=u256.sub(pl.col("b"), pl.col("a")),
        m=u256.mul(pl.col("a"), pl.col("b")),
        q=u256.div(pl.col("a"), pl.col("b")),
        r=u256.mod(pl.col("a"), pl.col("b")),
    ).with_columns(
        s_hex=u256.to_hex(pl.col("s")),
        d_hex=u256.to_hex(pl.col("d")),
        m_hex=u256.to_hex(pl.col("m")),
    )
    assert out["s_hex"][0].endswith("03")
    assert out["d_hex"][0].endswith("01")
    assert out["m_hex"][0].endswith("02")
    # Div/mod by zero -> null
    assert out["q"].to_list()[1] is None and out["r"].to_list()[1] is None


def test_scalar_coercion_in_ops():
    df = pl.DataFrame({"a": ["0x10"]}).with_columns(v=u256.from_hex(pl.col("a")))
    out = df.with_columns(
        add_i=u256.add(pl.col("v"), 2),
        add_h=u256.add(pl.col("v"), "0x2"),
        add_b=u256.add(pl.col("v"), b"\x02"),
    ).with_columns(
        add_i_hex=u256.to_hex(pl.col("add_i")),
        add_h_hex=u256.to_hex(pl.col("add_h")),
        add_b_hex=u256.to_hex(pl.col("add_b")),
    )
    assert out["add_i_hex"][0].endswith("12")
    assert out["add_h_hex"].item() == out["add_i_hex"].item()
    assert out["add_b_hex"].item() == out["add_i_hex"].item()


def test_comparisons_and_pow_guard():
    df = pl.DataFrame({
        "a": ["0x01", "0x02", "0x02"],
        "b": ["0x02", "0x02", "0x01"],
    }).with_columns(a=u256.from_hex(pl.col("a")), b=u256.from_hex(pl.col("b")))
    out = df.with_columns(
        eq=u256.eq(pl.col("a"), pl.col("b")),
        lt=u256.lt(pl.col("a"), pl.col("b")),
        le=u256.le(pl.col("a"), pl.col("b")),
        gt=u256.gt(pl.col("a"), pl.col("b")),
        ge=u256.ge(pl.col("a"), pl.col("b")),
    )
    assert out["eq"].to_list() == [False, True, False]
    assert out["lt"].to_list() == [True, False, False]
    assert out["le"].to_list() == [True, True, False]
    assert out["gt"].to_list() == [False, False, True]
    assert out["ge"].to_list() == [False, True, True]

    # Pow: small exponent ok, large exponent guarded
    df2 = pl.DataFrame({
        "base": ["0x02", "0x02"],
        "exp": ["0x03", "0x100"],
    }).with_columns(base=u256.from_hex(pl.col("base")), exp=u256.from_hex(pl.col("exp")))
    out2 = df2.with_columns(p=u256.pow(pl.col("base"), pl.col("exp")))
    out2 = out2.with_columns(p_hex=u256.to_hex(pl.col("p")))
    assert out2["p_hex"].to_list()[0].endswith("08")
    assert out2["p_hex"].to_list()[1] is None


def test_bitwise_and_or_xor_not_and_shifts():
    a = 0x0F
    b = 0xF0
    df = pl.DataFrame({}).with_columns(a=u256.from_int(a), b=u256.from_int(b))
    out = df.with_columns(
        and_=u256.bitand(pl.col("a"), pl.col("b")),
        or_=u256.bitor(pl.col("a"), pl.col("b")),
        xor_=u256.bitxor(pl.col("a"), pl.col("b")),
        nota=u256.bitnot(pl.col("a")),
    ).with_columns(
        and_hex=u256.to_hex(pl.col("and_")),
        or_hex=u256.to_hex(pl.col("or_")),
        xor_hex=u256.to_hex(pl.col("xor_")),
        nota_hex=u256.to_hex(pl.col("nota")),
    )
    mask = (1 << 256) - 1
    hex32 = lambda x: "0x" + x.to_bytes(32, "big").hex()
    assert out["and_hex"].item() == hex32(a & b)
    assert out["or_hex"].item() == hex32(a | b)
    assert out["xor_hex"].item() == hex32(a ^ b)
    assert out["nota_hex"].item() == hex32((~a) & mask)

    df2 = pl.DataFrame({}).with_columns(x=u256.from_int(1), s8=u256.from_int(8), s256=u256.from_int(256))
    out2 = df2.with_columns(
        shl8=u256.shl(pl.col("x"), pl.col("s8")),
        shr8=u256.shr(u256.shl(pl.col("x"), pl.col("s8")), pl.col("s8")),
        shl256=u256.shl(pl.col("x"), pl.col("s256")),
    ).with_columns(
        shl8_hex=u256.to_hex(pl.col("shl8")),
        shr8_hex=u256.to_hex(pl.col("shr8")),
        shl256_hex=u256.to_hex(pl.col("shl256")),
    )
    assert out2["shl8_hex"].item().endswith("0100")
    assert out2["shr8_hex"].item().endswith("01")
    assert int(out2["shl256_hex"].item(), 16) == 0


def test_sum_ungrouped_and_grouped_and_to_int_boundaries():
    df = pl.DataFrame({
        "g": ["a", "a", "b"],
        "h": ["0x01", "0x02", "0x03"],
    }).with_columns(v=u256.from_hex(pl.col("h")))
    # Ungrouped
    total = df.select(u256.sum(pl.col("v")).alias("t"))
    total_hex = total.with_columns(u256.to_hex(pl.col("t")).alias("t_hex"))
    assert total_hex["t_hex"].item().endswith("06")
    # Grouped scalar per group
    gdf = df.group_by("g").agg(t=u256.sum(pl.col("v")))
    out = gdf.with_columns(t_hex=u256.to_hex(pl.col("t")))
    got = {row[0]: row[2][-2:] for row in out.select(["g", "t", "t_hex"]).iter_rows()}
    assert got == {"a": "03", "b": "03"}
    # to_int boundaries
    u64_max = (1 << 64) - 1
    df2 = pl.DataFrame({}).with_columns(a=u256.from_int(u64_max), b=u256.from_int(u64_max + 1))
    out2 = df2.with_columns(ai=u256.to_int(pl.col("a")), bi=u256.to_int(pl.col("b")))
    assert out2["ai"].item() == u64_max
    assert out2["bi"].item() is None


def test_min_max_mean_and_value_counts():
    df = pl.DataFrame({"h": ["0x01", "0x03", "0x02", "0x01"]}).with_columns(v=u256.from_hex(pl.col("h")))
    out = df.select(
        tmin=u256.min(pl.col("v")),
        tmax=u256.max(pl.col("v")),
        tmean=u256.mean(pl.col("v")),
    ).with_columns(
        tmin_hex=u256.to_hex(pl.col("tmin")),
        tmax_hex=u256.to_hex(pl.col("tmax")),
        tmean_hex=u256.to_hex(pl.col("tmean")),
    )
    assert out["tmin_hex"].item().endswith("01")
    assert out["tmax_hex"].item().endswith("03")
    assert out["tmean_hex"].item().endswith("01")  # floor((1+3+2+1)/4)=1

    vc = df.select(u256.value_counts(pl.col("v")).alias("vc")).unnest("vc")
    vals = vc["v"].to_list()
    cnts = vc["count"].to_list()
    m = dict(zip(vals, cnts))
    assert m["0x" + "0"*62 + "01"] == 2
    assert m["0x" + "0"*62 + "02"] == 1
    assert m["0x" + "0"*62 + "03"] == 1
