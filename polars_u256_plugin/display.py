"""Display and DataFrame-level utilities for u256 columns.

This module intentionally avoids importing functional APIs from
``polars_u256_plugin.__init__`` at module import time to prevent circular
import issues. Functions like ``to_hex`` and ``from_int`` are imported lazily
inside methods.
"""

from __future__ import annotations

import polars as pl


class U256DisplayMixin:
    """Mixin class to add u256 display formatting to DataFrames."""
    
    def with_u256_display(self, *u256_columns: str) -> pl.DataFrame:
        """Add hex display columns for u256 binary columns.
        
        Args:
            *u256_columns: Names of u256 binary columns to add display formatting for
            
        Returns:
            DataFrame with additional "{column}_hex" columns for readable display
        """
        if not hasattr(self, 'with_columns'):
            raise ValueError("This method can only be used on DataFrames")
            
        # Lazy import to avoid circular import at module init
        from . import to_hex  # noqa: WPS433

        df = self
        for col_name in u256_columns:
            df = df.with_columns(
                **{f"{col_name}_hex": to_hex(pl.col(col_name))}
            )
        return df
    
    def show_u256_hex(self, *u256_columns: str) -> pl.DataFrame:
        """Replace u256 binary columns with hex display columns.
        
        Args:
            *u256_columns: Names of u256 binary columns to replace with hex display
            
        Returns:
            DataFrame with binary columns replaced by readable hex columns
        """
        if not hasattr(self, 'with_columns'):
            raise ValueError("This method can only be used on DataFrames")
            
        # Lazy import to avoid circular import at module init
        from . import to_hex  # noqa: WPS433

        df = self
        column_updates = {}
        
        for col_name in u256_columns:
            column_updates[col_name] = to_hex(pl.col(col_name))
            
        return df.with_columns(**column_updates)


class U256DFNamespace:
    """DataFrame-level convenience methods for u256 operations.

    Access via ``df.u256``.
    """

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def from_int(self, columns: list[str], replace: bool = True) -> pl.DataFrame:
        """Convert integer columns to u256.

        Args:
            columns: Column names to convert.
            replace: If True, replace columns in-place; otherwise, add "{name}_u256".
        """
        # Lazy import to avoid circular import at module init
        from . import from_int as u256_from_int  # noqa: WPS433

        updates: dict[str, pl.Expr] = {}
        for name in columns:
            expr = u256_from_int(pl.col(name))
            if replace:
                updates[name] = expr
            else:
                updates[f"{name}_u256"] = expr
        return self._df.with_columns(**updates)

    def to_hex(self, columns: list[str], replace: bool = False) -> pl.DataFrame:
        """Format u256 binary columns as hex strings.

        Args:
            columns: Column names to format.
            replace: If True, replace columns in-place; otherwise, add "{name}_hex".
        """
        # Lazy import to avoid circular import at module init
        from . import to_hex  # noqa: WPS433

        updates: dict[str, pl.Expr] = {}
        for name in columns:
            expr = to_hex(pl.col(name))
            if replace:
                updates[name] = expr
            else:
                updates[f"{name}_hex"] = expr
        return self._df.with_columns(**updates)

def format_u256_dataframe(df: pl.DataFrame, u256_columns: list[str] | None = None, mode: str = "replace") -> pl.DataFrame:
    """Format a DataFrame to display u256 columns as hex strings.
    
    Args:
        df: Input DataFrame
        u256_columns: List of u256 binary column names. If None, attempts to auto-detect.
        mode: Either "replace" (replace binary columns with hex) or "add" (add _hex columns)
        
    Returns:
        DataFrame with formatted u256 display
    """
    if u256_columns is None:
        # Auto-detect binary columns (assume they're u256 if they're 32 bytes)
        u256_columns = []
        for col_name in df.columns:
            col = df[col_name]
            if col.dtype == pl.Binary:
                # Check if it looks like u256 data (32 bytes)
                sample = col.drop_nulls().head(1)
                if len(sample) > 0:
                    sample_val = sample.item(0)
                    if sample_val and len(sample_val) == 32:
                        u256_columns.append(col_name)
    
    # Lazy import to avoid circular import at module init
    from . import to_hex  # noqa: WPS433

    if mode == "replace":
        column_updates = {}
        for col_name in u256_columns:
            column_updates[col_name] = to_hex(pl.col(col_name))
        return df.with_columns(**column_updates)
    
    elif mode == "add":
        column_updates = {}
        for col_name in u256_columns:
            column_updates[f"{col_name}_hex"] = to_hex(pl.col(col_name))
        return df.with_columns(**column_updates)
    
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'replace' or 'add'.")


def print_u256_dataframe(df: pl.DataFrame, u256_columns: list[str] | None = None) -> None:
    """Print a DataFrame with u256 columns formatted as hex strings.
    
    Args:
        df: DataFrame to print
        u256_columns: List of u256 binary column names. If None, attempts to auto-detect.
    """
    formatted_df = format_u256_dataframe(df, u256_columns, mode="replace")
    print(formatted_df)


# Monkey patch DataFrame to add our display methods
def _patch_dataframe():
    """Add u256 display methods to DataFrame class."""
    pl.DataFrame.with_u256_display = U256DisplayMixin.with_u256_display
    pl.DataFrame.show_u256_hex = U256DisplayMixin.show_u256_hex
    # Namespace property
    pl.DataFrame.u256 = property(lambda self: U256DFNamespace(self))


# Auto-patch when module is imported
_patch_dataframe()