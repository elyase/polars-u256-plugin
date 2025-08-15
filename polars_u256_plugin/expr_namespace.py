from __future__ import annotations

from typing import Any

import polars as pl

import importlib


class U256ExprOps:
    """Fluent u256 operations namespace attached to a Polars expression.

    Example usage:
        pl.col("v").u256 + 2
        pl.col("v").u256.to_hex()
        pl.col("v").u256.sum()
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr
        # Lazy import to avoid circular import at module import time
        self._u = importlib.import_module("polars_u256_plugin")

    # Arithmetic
    def __add__(self, other: Any) -> pl.Expr:
        return self._u.add(self._expr, other)

    def __radd__(self, other: Any) -> pl.Expr:
        return self._u.add(other, self._expr)

    def __sub__(self, other: Any) -> pl.Expr:
        return self._u.sub(self._expr, other)

    def __rsub__(self, other: Any) -> pl.Expr:
        return self._u.sub(other, self._expr)

    def __mul__(self, other: Any) -> pl.Expr:
        return self._u.mul(self._expr, other)

    def __rmul__(self, other: Any) -> pl.Expr:
        return self._u.mul(other, self._expr)

    def __truediv__(self, other: Any) -> pl.Expr:
        return self._u.div(self._expr, other)

    def __rtruediv__(self, other: Any) -> pl.Expr:
        return self._u.div(other, self._expr)

    def __floordiv__(self, other: Any) -> pl.Expr:
        return self._u.div(self._expr, other)

    def __rfloordiv__(self, other: Any) -> pl.Expr:
        return self._u.div(other, self._expr)

    def __mod__(self, other: Any) -> pl.Expr:
        return self._u.mod(self._expr, other)

    def __rmod__(self, other: Any) -> pl.Expr:
        return self._u.mod(other, self._expr)

    def __pow__(self, other: Any) -> pl.Expr:
        return self._u.pow(self._expr, other)

    # Bitwise
    def __and__(self, other: Any) -> pl.Expr:
        return self._u.bitand(self._expr, other)

    def __rand__(self, other: Any) -> pl.Expr:
        return self._u.bitand(other, self._expr)

    def __or__(self, other: Any) -> pl.Expr:
        return self._u.bitor(self._expr, other)

    def __ror__(self, other: Any) -> pl.Expr:
        return self._u.bitor(other, self._expr)

    def __xor__(self, other: Any) -> pl.Expr:
        return self._u.bitxor(self._expr, other)

    def __rxor__(self, other: Any) -> pl.Expr:
        return self._u.bitxor(other, self._expr)

    def __invert__(self) -> pl.Expr:
        return self._u.bitnot(self._expr)

    def __lshift__(self, other: Any) -> pl.Expr:
        return self._u.shl(self._expr, other)

    def __rshift__(self, other: Any) -> pl.Expr:
        return self._u.shr(self._expr, other)

    # Comparisons
    def __eq__(self, other: Any) -> pl.Expr:  # type: ignore[override]
        other_expr = other._expr if isinstance(other, U256ExprOps) else other
        return self._u.eq(self._expr, other_expr)
    def __ne__(self, other: Any) -> pl.Expr:  # type: ignore[override]
        other_expr = other._expr if isinstance(other, U256ExprOps) else other
        return self._u.eq(self._expr, other_expr).not_()

    def __lt__(self, other: Any) -> pl.Expr:  # type: ignore[override]
        return self._u.lt(self._expr, other)

    def __le__(self, other: Any) -> pl.Expr:  # type: ignore[override]
        return self._u.le(self._expr, other)

    def __gt__(self, other: Any) -> pl.Expr:  # type: ignore[override]
        return self._u.gt(self._expr, other)

    def __ge__(self, other: Any) -> pl.Expr:  # type: ignore[override]
        return self._u.ge(self._expr, other)

    # Helpers
    def to_hex(self) -> pl.Expr:
        return self._u.to_hex(self._expr)

    def sum(self) -> pl.Expr:
        return self._u.sum(self._expr)
    def min(self) -> pl.Expr:
        return self._u.min(self._expr)
    def max(self) -> pl.Expr:
        return self._u.max(self._expr)
    def mean(self) -> pl.Expr:
        return self._u.mean(self._expr)
    def value_counts(self) -> pl.Expr:
        return self._u.value_counts(self._expr)

    def cumsum(self) -> pl.Expr:
        return self._u.cumsum(self._expr)

    def diff(self) -> pl.Expr:
        return self._u.diff(self._expr)


class I256ExprOps:
    """Fluent i256 operations namespace attached to a Polars expression.

    Example usage:
        pl.col("v").i256 + 2
        pl.col("v").i256.to_hex()
        pl.col("v").i256.sum()
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr
        self._u = importlib.import_module("polars_u256_plugin")

    # Arithmetic
    def __add__(self, other: Any) -> pl.Expr:
        return self._u.i256_add(self._expr, other)

    def __radd__(self, other: Any) -> pl.Expr:
        return self._u.i256_add(other, self._expr)

    def __sub__(self, other: Any) -> pl.Expr:
        return self._u.i256_sub(self._expr, other)

    def __rsub__(self, other: Any) -> pl.Expr:
        return self._u.i256_sub(other, self._expr)

    def __mul__(self, other: Any) -> pl.Expr:
        return self._u.i256_mul(self._expr, other)

    def __rmul__(self, other: Any) -> pl.Expr:
        return self._u.i256_mul(other, self._expr)

    def __truediv__(self, other: Any) -> pl.Expr:
        return self._u.i256_div(self._expr, other)

    def __rtruediv__(self, other: Any) -> pl.Expr:
        return self._u.i256_div(other, self._expr)

    def __floordiv__(self, other: Any) -> pl.Expr:
        return self._u.i256_div_euclid(self._expr, other)

    def __rfloordiv__(self, other: Any) -> pl.Expr:
        return self._u.i256_div_euclid(other, self._expr)

    def __mod__(self, other: Any) -> pl.Expr:
        return self._u.i256_mod(self._expr, other)

    def __rmod__(self, other: Any) -> pl.Expr:
        return self._u.i256_mod(other, self._expr)

    # Comparisons
    def __eq__(self, other: Any) -> pl.Expr:  # type: ignore[override]
        other_expr = other._expr if isinstance(other, I256ExprOps) else other
        return self._u.i256_eq(self._expr, other_expr)
    def __ne__(self, other: Any) -> pl.Expr:  # type: ignore[override]
        other_expr = other._expr if isinstance(other, I256ExprOps) else other
        return self._u.i256_eq(self._expr, other_expr).not_()

    def __lt__(self, other: Any) -> pl.Expr:  # type: ignore[override]
        return self._u.i256_lt(self._expr, other)

    def __le__(self, other: Any) -> pl.Expr:  # type: ignore[override]
        return self._u.i256_le(self._expr, other)

    def __gt__(self, other: Any) -> pl.Expr:  # type: ignore[override]
        return self._u.i256_gt(self._expr, other)

    def __ge__(self, other: Any) -> pl.Expr:  # type: ignore[override]
        return self._u.i256_ge(self._expr, other)

    # Helpers
    def to_hex(self) -> pl.Expr:
        return self._u.i256_to_hex(self._expr)

    def sum(self) -> pl.Expr:
        return self._u.i256_sum(self._expr)
    def min(self) -> pl.Expr:
        return self._u.i256_min(self._expr)
    def max(self) -> pl.Expr:
        return self._u.i256_max(self._expr)
    def mean(self) -> pl.Expr:
        return self._u.i256_mean(self._expr)
    def value_counts(self) -> pl.Expr:
        return self._u.i256_value_counts(self._expr)

    def cumsum(self) -> pl.Expr:
        return self._u.i256_cumsum(self._expr)

    def diff(self) -> pl.Expr:
        return self._u.i256_diff(self._expr)


def install_expr_namespace() -> None:
    """Attach the `.u256` and `.i256` namespaces to Polars expressions.

    After installation, any `pl.Expr` has properties exposing big-int ops.
    """
    if getattr(pl.Expr, "u256", None) is None:
        def _get_u256(self: pl.Expr) -> U256ExprOps:  # type: ignore[override]
            return U256ExprOps(self)
        setattr(pl.Expr, "u256", property(_get_u256))

    if getattr(pl.Expr, "i256", None) is None:
        def _get_i256(self: pl.Expr) -> I256ExprOps:  # type: ignore[override]
            return I256ExprOps(self)
        setattr(pl.Expr, "i256", property(_get_i256))
