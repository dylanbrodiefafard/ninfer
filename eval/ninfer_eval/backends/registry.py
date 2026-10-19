from __future__ import annotations

import functools

from .base import EvaluationBackend

_BACKENDS: dict[str, EvaluationBackend] = {}


def register_backend(backend: EvaluationBackend) -> None:
    if backend.name in _BACKENDS:
        raise RuntimeError(f"backend already registered: {backend.name}")
    _BACKENDS[backend.name] = backend


def get_backend(name: str) -> EvaluationBackend:
    _ensure_builtins()
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown evaluation backend: {name}") from exc


def backend_names() -> list[str]:
    _ensure_builtins()
    return sorted(_BACKENDS)


@functools.cache
def _ensure_builtins() -> None:
    from .evalscope import EvalScopeBackend
    from .mock import MockBackend

    for backend in (MockBackend(), EvalScopeBackend()):
        if backend.name not in _BACKENDS:
            register_backend(backend)
