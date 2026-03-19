"""ML-powered analyzers (requires prose-doctor[ml] extra)."""

ML_AVAILABLE = False
_ML_INSTALL_MSG = (
    "ML features require the [ml] extra. Install with:\n"
    "  pip install prose-doctor[ml]\n"
    "  # or: uv pip install prose-doctor[ml]"
)

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    ML_AVAILABLE = True
except ImportError:
    pass


def require_ml() -> None:
    """Raise ImportError with install instructions if ML deps are missing."""
    if not ML_AVAILABLE:
        raise ImportError(_ML_INSTALL_MSG)
