from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Callable, Union
import warnings

import polars as pl

# Package version (for Python introspection)
try:  # Python 3.8+
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("polars_u256_plugin")
except Exception:  # Fallback in editable/dev installs without metadata
    __version__ = "0.0.0"


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

    # 3) Try the bundled PyO3 extension module path (if installed via maturin)
    try:
        from . import _internal as _ext  # type: ignore
        mod_file = getattr(_ext, "__file__", None)
        if mod_file:
            return str(Path(mod_file))
    except Exception:
        pass

    # 4) Fallback to dev path (cargo build output)
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
_from_int_expr = _wrap("u256_from_int")
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
cumsum = _wrap("u256_cumsum")
diff = _wrap("u256_diff")

# Aggregation functions (need different wrapper for non-elementwise)
def _wrap_agg(name: str) -> Callable:
    def call(*args: Any, **kwargs: Any):
        coerced_args = [_coerce_arg(a) for a in args]
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

sum = _wrap_agg("u256_sum")
min = _wrap_agg("u256_min")
max = _wrap_agg("u256_max")
mean = _wrap_agg("u256_mean")


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
from .display import format_u256_dataframe, print_u256_dataframe  # noqa: E402
from .expr_namespace import install_expr_namespace  # noqa: E402

# Re-export for convenience
__all__ = [
    "library_path",
    "register_all", 
    "from_hex",
    "to_hex",
    "to_int",
    "validate_hex",
    "validate_range",
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
    "from_ints",
    "MAX_VALUE",
    "MIN_VALUE",
]

# Install Expr namespace for fluent API at import time
try:
    install_expr_namespace()
except Exception:
    # Non-fatal; users can still access functional API
    pass


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


def from_int(value: Union[int, pl.Expr]) -> pl.Expr:
    """Convert an int or integer expression to u256 (32-byte big-endian).

    - If ``value`` is a Python ``int``: returns a u256 literal expression.
    - If ``value`` is a Polars ``Expr``: converts the integer column to u256 per-element.
    """
    if isinstance(value, int):
        return pl.lit(_int_to_be32(int(value)))
    return _from_int_expr(value)


def from_ints(expr: pl.Expr) -> pl.Expr:  # backward-compat alias
    """DEPRECATED: use ``from_int(expr)`` instead.

    Accepts an integer Polars expression and converts it to u256 per-element.
    """
    warnings.warn(
        "u256.from_ints(...) is deprecated and will be removed in a future release; use u256.from_int(...)",
        DeprecationWarning,
        stacklevel=2,
    )
    return _from_int_expr(expr)


def validate_hex(expr: pl.Expr) -> pl.Expr:
    """Return a boolean expression indicating whether each value is a valid u256 hex/binary.

    Implementation detail: checks that ``u256.from_hex(expr)`` does not yield null.
    """
    return from_hex(expr).is_not_null()


def validate_range(value: Union[int, pl.Expr]) -> pl.Expr:
    """Return a boolean expression indicating whether value fits in unsigned 256 bits.

    - For Python ints, evaluates immediately to a boolean literal.
    - For Polars expressions, returns ``u256.from_int(expr).is_not_null()``.
    """
    if isinstance(value, int):
        return pl.lit(0 <= value < (1 << 256))
    return _from_int_expr(value).is_not_null()


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


# Public u256 constants (as expressions)
MAX_VALUE = from_int((1 << 256) - 1)
MIN_VALUE = from_int(0)


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
_i256_from_int_expr = _wrap_i256("i256_from_int")
i256_add = _wrap("i256_add")
i256_sub = _wrap("i256_sub")
i256_mul = _wrap_i256("i256_mul")
i256_div = _wrap("i256_div")
i256_mod = _wrap_i256("i256_mod")
i256_div_euclid = _wrap_i256("i256_div_euclid")
i256_rem_euclid = _wrap_i256("i256_rem_euclid")
i256_eq = _wrap_i256("i256_eq")
i256_lt = _wrap_i256("i256_lt")
i256_le = _wrap_i256("i256_le")
i256_gt = _wrap_i256("i256_gt")
i256_ge = _wrap_i256("i256_ge")
i256_to_int = _wrap_i256("i256_to_int")
i256_cumsum = _wrap_i256("i256_cumsum")
i256_diff = _wrap_i256("i256_diff")

# i256.from_int that accepts both int and Expr, plus a deprecated alias
def i256_from_int(value: Union[int, pl.Expr]) -> pl.Expr:
    """Convert a Python int or integer expression to i256.

    - If ``value`` is a Python ``int``: returns a 32-byte two's-complement literal.
    - If ``value`` is a Polars ``Expr``: converts the integer column to i256 per-element.
    """
    if isinstance(value, int):
        return pl.lit(_int_to_i256_be32(int(value)))
    return _i256_from_int_expr(value)

def i256_from_ints(expr: pl.Expr) -> pl.Expr:  # backward-compat alias
    warnings.warn(
        "i256.from_ints(...) is deprecated and will be removed in a future release; use i256.from_int(...)",
        DeprecationWarning,
        stacklevel=2,
    )
    return _i256_from_int_expr(expr)


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
i256_min = _wrap_agg_i256("i256_min")
i256_max = _wrap_agg_i256("i256_max")
i256_mean = _wrap_agg_i256("i256_mean")

# Value counts helper (uses Polars native value_counts on hex strings)
def value_counts(expr: pl.Expr) -> pl.Expr:
    """Return value counts as a struct with fields 'values' and 'counts'.

    Implemented via ``u256.to_hex(expr).value_counts()``.
    """
    return to_hex(expr).value_counts()

def i256_value_counts(expr: pl.Expr) -> pl.Expr:
    return i256_to_hex(expr).value_counts()


class _I256:
    # Common signed-256 constants (as expressions)
    MAX_VALUE = pl.lit(_int_to_i256_be32((1 << 255) - 1))
    MIN_VALUE = pl.lit(_int_to_i256_be32(-(1 << 255)))
    from_hex = staticmethod(i256_from_hex)
    to_hex = staticmethod(i256_to_hex)
    from_ints = staticmethod(i256_from_ints)
    from_int = staticmethod(i256_from_int)
    to_int = staticmethod(i256_to_int)
    add = staticmethod(i256_add)
    sub = staticmethod(i256_sub)
    min = staticmethod(i256_min)
    max = staticmethod(i256_max)
    mean = staticmethod(i256_mean)
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
    value_counts = staticmethod(i256_value_counts)
    cumsum = staticmethod(i256_cumsum)
    diff = staticmethod(i256_diff)


i256 = _I256()

# Extend __all__
__all__ += [
    "i256",
    "i256_from_hex",
    "i256_to_hex",
    "i256_from_int",
    "i256_from_ints",
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
