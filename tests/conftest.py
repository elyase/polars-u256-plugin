import os
import platform
from pathlib import Path

import polars_u256_plugin as u256


def _dev_lib() -> str:
    sysname = platform.system()
    base = Path(__file__).resolve().parents[1] / "target" / "release"
    if sysname == "Darwin":
        return str(base / "libpolars_u256_plugin.dylib")
    elif sysname == "Windows":
        return str(base / "polars_u256_plugin.dll")
    else:
        return str(base / "libpolars_u256_plugin.so")


def pytest_configure(config):
    # If packaged lib not present but dev lib exists, point env to dev lib
    lib = u256.library_path()
    if not Path(lib).exists():
        dev = _dev_lib()
        if Path(dev).exists():
            os.environ["POLARS_U256_LIB"] = dev

