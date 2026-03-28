from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("milady-shrinkifier")
except PackageNotFoundError:  # pragma: no cover - editable/dev fallback
    __version__ = "0.0.0"
