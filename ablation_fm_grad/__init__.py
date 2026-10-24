"""Flow-matching gradient ablation for CoMimic (see README.md)."""
from .backends import BACKENDS, BackendResult, SamplingBackend, make_backend
from .harness import evaluate, make_tiny_gr00t_setup, make_toy_setup, rows_to_markdown

__all__ = [
    "BACKENDS",
    "BackendResult",
    "SamplingBackend",
    "make_backend",
    "evaluate",
    "make_toy_setup",
    "make_tiny_gr00t_setup",
    "rows_to_markdown",
]
