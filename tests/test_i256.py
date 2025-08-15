import polars as pl
import polars_u256_plugin as u256


def test_i256_from_int_roundtrip_and_sign():
    df = pl.DataFrame({}).with_columns(
        a=u256.i256.from_int(-1),
        b=u256.i256.from_int(1),
    )
    out = df.with_columns(
        a_hex=u256.i256.to_hex(pl.col("a")),
        b_hex=u256.i256.to_hex(pl.col("b")),
    )
    assert out["a_hex"].item().startswith("-0x")
    assert out["b_hex"].item().startswith("0x")


def test_i256_add_sub_wrap():
    # (-2) + 3 = 1 ; 1 - 3 = -2
    df = pl.DataFrame({}).with_columns(
        x=u256.i256.from_int(-2),
        y=u256.i256.from_int(3),
    )
    df = df.with_columns(s=u256.i256.add(pl.col("x"), pl.col("y")))
    df = df.with_columns(d=u256.i256.sub(pl.col("s"), pl.col("y")))
    df = df.with_columns(s_i=u256.i256.to_int(pl.col("s")), d_i=u256.i256.to_int(pl.col("d")))
    assert df["s_i"].item() == 1
    assert df["d_i"].item() == -2


def test_i256_mul_wrap():
    # (-2) * 3 = -6 ; (-2) * (-3) = 6
    df = pl.DataFrame({}).with_columns(
        x=u256.i256.from_int(-2),
        y=u256.i256.from_int(3),
        z=u256.i256.from_int(-3),
    )
    out = df.with_columns(
        xy=u256.i256.mul(pl.col("x"), pl.col("y")),
        xz=u256.i256.mul(pl.col("x"), pl.col("z")),
    ).with_columns(
        xy_i=u256.i256.to_int(pl.col("xy")),
        xz_i=u256.i256.to_int(pl.col("xz")),
    )
    assert out["xy_i"].item() == -6
    assert out["xz_i"].item() == 6


def test_i256_div_mod_signs():
    df = pl.DataFrame({}).with_columns(
        a=u256.i256.from_int(-7),
        b=u256.i256.from_int(3),
    )
    df = df.with_columns(q=u256.i256.div(pl.col("a"), pl.col("b")))
    df = df.with_columns(r=u256.i256.mod(pl.col("a"), pl.col("b")))
    df = df.with_columns(qi=u256.i256.to_int(pl.col("q")), ri=u256.i256.to_int(pl.col("r")))
    # Trunc toward zero: -7/3 => -2 remainder -1
    assert df["qi"].item() == -2
    assert df["ri"].item() == -1


def test_i256_cmp():
    df = pl.DataFrame({}).with_columns(
        a=u256.i256.from_int(-5),
        b=u256.i256.from_int(2),
        c=u256.i256.from_int(-3),
    )
    out = df.select(
        lt=u256.i256.lt(pl.col("a"), pl.col("b")),
        gt=u256.i256.gt(pl.col("b"), pl.col("c")),
        le=u256.i256.le(pl.col("a"), pl.col("c")),
        ge=u256.i256.ge(pl.col("c"), pl.col("a")),
        eq=u256.i256.eq(pl.col("a"), pl.col("a")),
    )
    assert tuple(out.row(0)) == (True, True, True, True, True)


def test_i256_sum_and_to_int_bounds():
    df = pl.DataFrame({}).with_columns(
        a=u256.i256.from_int(-2),
        b=u256.i256.from_int(3),
    )
    df = df.with_columns(t=u256.i256.add(pl.col("a"), pl.col("b")))
    total = df.select(u256.i256.sum(pl.col("t")).alias("t"))
    ti = total.select(u256.i256.to_int(pl.col("t")).alias("ti"))
    assert ti["ti"].item() == 1


def test_i256_euclid_div_rem():
    import polars as pl
    import polars_u256_plugin as u256

    # Case 1: a = -7, b = 3 => div_trunc=-2, mod_trunc=-1; div_euclid=-3, rem_euclid=2
    df = pl.DataFrame({}).with_columns(a=u256.i256.from_int(-7), b=u256.i256.from_int(3))
    out = df.with_columns(
        q_e=u256.i256.div_euclid(pl.col("a"), pl.col("b")),
        r_e=u256.i256.rem_euclid(pl.col("a"), pl.col("b")),
    ).with_columns(
        qi=u256.i256.to_int(pl.col("q_e")), ri=u256.i256.to_int(pl.col("r_e"))
    )
    assert out["qi"].item() == -3
    assert out["ri"].item() == 2

    # Case 2: a = 7, b = -3 => trunc: q=-2, r=1 ; euclid same
    df2 = pl.DataFrame({}).with_columns(a=u256.i256.from_int(7), b=u256.i256.from_int(-3))
    out2 = df2.with_columns(
        q_e=u256.i256.div_euclid(pl.col("a"), pl.col("b")),
        r_e=u256.i256.rem_euclid(pl.col("a"), pl.col("b")),
    ).with_columns(
        qi=u256.i256.to_int(pl.col("q_e")), ri=u256.i256.to_int(pl.col("r_e"))
    )
    assert out2["qi"].item() == -2
    assert out2["ri"].item() == 1

    # Case 3: a = -7, b = -3 => trunc: q=2, r=-1 ; euclid: q=3, r=2
    df3 = pl.DataFrame({}).with_columns(a=u256.i256.from_int(-7), b=u256.i256.from_int(-3))
    out3 = df3.with_columns(
        q_e=u256.i256.div_euclid(pl.col("a"), pl.col("b")),
        r_e=u256.i256.rem_euclid(pl.col("a"), pl.col("b")),
    ).with_columns(
        qi=u256.i256.to_int(pl.col("q_e")), ri=u256.i256.to_int(pl.col("r_e"))
    )
    assert out3["qi"].item() == 3
    assert out3["ri"].item() == 2


def test_i256_min_max_mean():
    df = pl.DataFrame({"x": [-5, 2, -3]}).with_columns(v=u256.i256.from_int(pl.col("x")))
    out = df.select(
        tmin=u256.i256.min(pl.col("v")),
        tmax=u256.i256.max(pl.col("v")),
        tmean=u256.i256.mean(pl.col("v")),
    ).with_columns(
        tmin_i=u256.i256.to_int(pl.col("tmin")),
        tmax_i=u256.i256.to_int(pl.col("tmax")),
        tmean_i=u256.i256.to_int(pl.col("tmean")),
    )
    assert out["tmin_i"].item() == -5
    assert out["tmax_i"].item() == 2
    # (-5 + 2 + -3) / 3 = -2 (trunc toward zero)
    assert out["tmean_i"].item() == -2


def test_i256_value_counts():
    df = pl.DataFrame({"x": [-1, 2, -1]}).with_columns(v=u256.i256.from_int(pl.col("x")))
    vc = df.select(u256.i256_value_counts(pl.col("v")).alias("vc")).unnest("vc")
    vals = vc["v"].to_list()
    cnts = vc["count"].to_list()
    m = dict(zip(vals, cnts))
    assert m["-0x" + "0"*62 + "01"] == 2
    assert m["0x" + "0"*62 + "02"] == 1
