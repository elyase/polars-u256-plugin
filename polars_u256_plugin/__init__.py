from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Callable, Union

import polars as pl


def _default_library_path() -> str:
    # 1) Explicit override via env
    env = os.getenv("POLARS_U256_LIB")
    if env:
        return env

    # 2) Try packaged binary location (if present)
    try:
        from importlib.resources import files  # py3.9+

        sysname = platform.system()
        base = files(__package__).joinpath("bin")
        if sysname == "Darwin":
            cand = base.joinpath("libpolars_u256_plugin.dylib")
        elif sysname == "Windows":
            cand = base.joinpath("polars_u256_plugin.dll")
        else:
            cand = base.joinpath("libpolars_u256_plugin.so")
        with cand.as_file() as p:
            if p.exists():
                return str(p)
    except Exception:
        pass

    # 3) Fallback to dev path (cargo build output)
    sysname = platform.system()
    root = Path(__file__).resolve().parents[1]
    base = root / "target" / "release"
    if sysname == "Darwin":
        return str(base / "libpolars_u256_plugin.dylib")
    elif sysname == "Windows":
        return str(base / "polars_u256_plugin.dll")
    else:
        return str(base / "libpolars_u256_plugin.so")


def library_path() -> str:
    """Resolve the dynamic library path.

    Order: env POLARS_U256_LIB -> packaged binary -> cargo target/release
    """
    return _default_library_path()


def _wrap(name: str) -> Callable:
    def call(*args: Any, **kwargs: Any):
        coerced_args = [_coerce_arg(a) for a in args]
        return pl.plugins.register_plugin_function(
            plugin_path=library_path(),
            function_name=name,
            args=coerced_args,
            kwargs=kwargs or None,
            is_elementwise=True,
            use_abs_path=True,
        )

    return call


# Convenience callables mirroring the Rust functions
from_hex = _wrap("u256_from_hex")
to_hex = _wrap("u256_to_hex")
to_int = _wrap("u256_to_int")
add = _wrap("u256_add")
sub = _wrap("u256_sub")
mul = _wrap("u256_mul")
div = _wrap("u256_div")
mod = _wrap("u256_mod")
pow = _wrap("u256_pow")
eq = _wrap("u256_eq")
lt = _wrap("u256_lt")
le = _wrap("u256_le")
gt = _wrap("u256_gt")
ge = _wrap("u256_ge")
bitand = _wrap("u256_bitand")
bitor = _wrap("u256_bitor")
bitxor = _wrap("u256_bitxor")
bitnot = _wrap("u256_bitnot")
shl = _wrap("u256_shl")
shr = _wrap("u256_shr")

# Aggregation functions (need different wrapper for non-elementwise)
def _wrap_agg(name: str) -> Callable:
    def call(*args: Any, **kwargs: Any):
        # For aggregations, only the primary column expression is expected.
        coerced_args = [_coerce_arg(a) for a in args]
        return pl.plugins.register_plugin_function(
            plugin_path=library_path(),
            function_name=name,
            args=coerced_args,
            kwargs=kwargs or None,
            is_elementwise=False,  # Aggregation functions are not elementwise
            returns_scalar=True,   # Return a scalar in aggregation contexts (group_by/select)
            use_abs_path=True,
        )
    return call

sum = _wrap_agg("u256_sum")


def register_all() -> None:
    """Optional helper to "prime" registrations.

    Users can import this module and call functions directly without pre-registration,
    but this allows eagerly checking the library path.
    """
    # Touch each to verify path resolution
    for _ in (
        from_hex,
        to_hex,
        to_int,
        add,
        sub,
        mul,
        div,
        mod,
        pow,
        eq,
        lt,
        le,
        gt,
        ge,
        bitand,
        bitor,
        bitxor,
        bitnot,
        shl,
        shr,
        sum,
    ):
        pass


# Import display utilities (this will auto-patch DataFrame class)
from .display import format_u256_dataframe, print_u256_dataframe

# Re-export for convenience
__all__ = [
    "library_path",
    "register_all", 
    "from_hex",
    "to_hex",
    "to_int",
    "add",
    "sub", 
    "mul",
    "div",
    "mod",
    "pow",
    "eq",
    "lt",
    "le", 
    "gt",
    "ge",
    "bitand",
    "bitor",
    "bitxor",
    "bitnot",
    "shl",
    "shr",
    "sum",
    "format_u256_dataframe",
    "print_u256_dataframe",
    "lit",
    "from_int",
]


# ------- Convenience coercion helpers -------
def _int_to_be32(value: int) -> bytes:
    if value < 0:
        raise ValueError("u256 is unsigned; negative integers are not supported")
    # 256-bit max value
    max_u256 = (1 << 256) - 1
    if value > max_u256:
        raise ValueError("integer does not fit in 256 bits")
    return value.to_bytes(32, byteorder="big")


def _pad_bytes_be32(b: bytes) -> bytes:
    if len(b) > 32:
        raise ValueError("binary literal longer than 32 bytes")
    return (b"\x00" * (32 - len(b))) + b


def lit(value: Union[int, str, bytes]) -> pl.Expr:
    """Construct a u256 literal expression.

    - int: converted to 32-byte big-endian (unsigned)
    - hex str (starts with 0x/0X): passed via from_hex
    - bytes: left-padded to 32 bytes
    """
    if isinstance(value, int):
        return pl.lit(_int_to_be32(value))
    if isinstance(value, (bytes, bytearray)):
        return pl.lit(_pad_bytes_be32(bytes(value)))
    if isinstance(value, str) and value.lower().startswith("0x"):
        return from_hex(pl.lit(value))
    raise TypeError("u256.lit accepts int, hex str starting with 0x, or bytes")


def from_int(value: int) -> pl.Expr:
    """Convert a Python int into a u256 expression (32-byte BE binary)."""
    return pl.lit(_int_to_be32(int(value)))


def _coerce_arg(arg: Any) -> Any:
    """Coerce Python scalars into u256 expressions.

    - int -> 32-byte BE binary literal
    - hex str (0x...) -> from_hex(lit(str))
    - bytes/bytearray -> padded 32 bytes binary literal
    Otherwise returns the argument unchanged (assumed to be a Polars expr).
    """
    if isinstance(arg, int):
        return from_int(arg)
    if isinstance(arg, (bytes, bytearray)):
        return pl.lit(_pad_bytes_be32(bytes(arg)))
    if isinstance(arg, str) and arg.lower().startswith("0x"):
        return from_hex(pl.lit(arg))
    return arg


# ------- i256 namespace (signed 256-bit, two's complement) -------
def _wrap_i256(name: str) -> Callable:
    def call(*args: Any, **kwargs: Any):
        coerced_args = [_coerce_arg_i256(a) for a in args]
        return pl.plugins.register_plugin_function(
            plugin_path=library_path(),
            function_name=name,
            args=coerced_args,
            kwargs=kwargs or None,
            is_elementwise=True,
            use_abs_path=True,
        )
    return call


def _int_to_i256_be32(value: int) -> bytes:
    # two's complement encoding to 32 bytes
    if value >= 0:
        return _int_to_be32(value)
    # negative: compute 2^256 + value
    max_mod = 1 << 256
    if value < -(1 << 255):
        raise ValueError("integer does not fit in signed 256 bits")
    twos = (max_mod + value) & (max_mod - 1)
    return twos.to_bytes(32, byteorder="big")


def _coerce_arg_i256(arg: Any) -> Any:
    if isinstance(arg, int):
        return pl.lit(_int_to_i256_be32(arg))
    if isinstance(arg, (bytes, bytearray)):
        b = bytes(arg)
        if len(b) > 32:
            raise ValueError("binary literal longer than 32 bytes")
        pad = (b"\xFF" if (len(b) and (b[0] & 0x80)) else b"\x00") * (32 - len(b))
        return pl.lit(pad + b)
    if isinstance(arg, str) and (arg.startswith("-0x") or arg.startswith("0x")):
        return i256_from_hex(pl.lit(arg))
    return arg


i256_from_hex = _wrap_i256("i256_from_hex")
i256_to_hex = _wrap_i256("i256_to_hex")
i256_add = _wrap_i256("i256_add")
i256_sub = _wrap_i256("i256_sub")
i256_mul = _wrap_i256("i256_mul")
i256_div = _wrap_i256("i256_div")
i256_mod = _wrap_i256("i256_mod")
i256_div_euclid = _wrap_i256("i256_div_euclid")
i256_rem_euclid = _wrap_i256("i256_rem_euclid")
i256_eq = _wrap_i256("i256_eq")
i256_lt = _wrap_i256("i256_lt")
i256_le = _wrap_i256("i256_le")
i256_gt = _wrap_i256("i256_gt")
i256_ge = _wrap_i256("i256_ge")
i256_to_int = _wrap_i256("i256_to_int")


def _wrap_agg_i256(name: str) -> Callable:
    def call(*args: Any, **kwargs: Any):
        coerced_args = [_coerce_arg_i256(a) for a in args]
        return pl.plugins.register_plugin_function(
            plugin_path=library_path(),
            function_name=name,
            args=coerced_args,
            kwargs=kwargs or None,
            is_elementwise=False,
            returns_scalar=True,
            use_abs_path=True,
        )
    return call


i256_sum = _wrap_agg_i256("i256_sum")


class _I256:
    from_hex = staticmethod(i256_from_hex)
    to_hex = staticmethod(i256_to_hex)
    from_int = staticmethod(lambda v: pl.lit(_int_to_i256_be32(int(v))))
    to_int = staticmethod(i256_to_int)
    add = staticmethod(i256_add)
    sub = staticmethod(i256_sub)
    mul = staticmethod(i256_mul)
    div = staticmethod(i256_div)
    mod = staticmethod(i256_mod)
    div_euclid = staticmethod(i256_div_euclid)
    rem_euclid = staticmethod(i256_rem_euclid)
    eq = staticmethod(i256_eq)
    lt = staticmethod(i256_lt)
    le = staticmethod(i256_le)
    gt = staticmethod(i256_gt)
    ge = staticmethod(i256_ge)
    sum = staticmethod(i256_sum)


i256 = _I256()

# Extend __all__
__all__ += [
    "i256",
    "i256_from_hex",
    "i256_to_hex",
    "i256_add",
    "i256_sub",
    "i256_mul",
    "i256_div",
    "i256_mod",
    "i256_div_euclid",
    "i256_rem_euclid",
    "i256_eq",
    "i256_lt",
    "i256_le",
    "i256_gt",
    "i256_ge",
    "i256_sum",
    "i256_to_int",
]
