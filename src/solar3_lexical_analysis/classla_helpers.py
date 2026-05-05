#!/usr/bin/env python3
"""Shared helpers for CLASSLA-based corpus annotation scripts.

The goal of this module is simple: keep one reusable place for
- importing CLASSLA safely,
- downloading models,
- initializing a pipeline,
- and optionally pointing all projects to one shared model directory.

Different CLASSLA / Stanza releases have historically exposed the resource
location under slightly different parameter names. To keep the scripts usable
across versions, we inspect function signatures and fall back to a small set of
common names.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class ClasslaImportError(SystemExit):
    """Raised when the CLASSLA package is unavailable."""


def ensure_classla():
    """Import CLASSLA or exit with a helpful message."""
    try:
        import classla  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ClasslaImportError(
            "The 'classla' package is not installed. Install it with 'pip install classla'."
        ) from exc
    return classla


def normalize_models_dir(models_dir: Optional[Path]) -> Optional[Path]:
    """Return an expanded, created model directory, or None."""
    if models_dir is None:
        return None
    path = Path(models_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _signature_parameter_names(func: Callable[..., Any]) -> set[str]:
    """Best-effort extraction of function parameter names."""
    try:
        return set(inspect.signature(func).parameters)
    except Exception:  # pragma: no cover - very defensive
        return set()


def add_models_dir_argument(
    func: Callable[..., Any],
    kwargs: Dict[str, Any],
    models_dir: Optional[Path],
) -> Dict[str, Any]:
    """Inject the resource directory argument expected by the installed version.

    CLASSLA inherits some API patterns from Stanza. Depending on the version,
    the relevant keyword may be exposed as ``dir``, ``model_dir`` or, less
    commonly in wrappers, ``resources_dir``. We inspect the callable's signature
    and pick the first matching keyword. If the signature is opaque, we fall
    back to ``dir`` which matches the public Stanza pipeline docs.
    """
    normalized = normalize_models_dir(models_dir)
    if normalized is None:
        return dict(kwargs)

    normalized_str = str(normalized)
    os.environ.setdefault("CLASSLA_RESOURCES_DIR", normalized_str)
    os.environ.setdefault("STANZA_RESOURCES_DIR", normalized_str)

    out = dict(kwargs)
    param_names = _signature_parameter_names(func)
    for candidate in ("dir", "model_dir", "resources_dir"):
        if candidate in param_names:
            out[candidate] = normalized_str
            return out

    out.setdefault("dir", normalized_str)
    return out


def download_models(
    *,
    lang: str,
    classla_type: str = "standard",
    models_dir: Optional[Path] = None,
) -> None:
    """Download CLASSLA models for the requested language / type."""
    classla = ensure_classla()
    kwargs: Dict[str, Any] = {"lang": lang}
    if classla_type != "standard":
        kwargs["type"] = classla_type
    kwargs = add_models_dir_argument(classla.download, kwargs, models_dir)
    classla.download(**kwargs)


def build_pipeline(
    *,
    lang: str,
    processors: str,
    classla_type: str = "standard",
    models_dir: Optional[Path] = None,
    download: bool = False,
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    """Construct and return a CLASSLA pipeline."""
    classla = ensure_classla()
    normalized_models_dir = normalize_models_dir(models_dir)

    if download:
        download_models(lang=lang, classla_type=classla_type, models_dir=normalized_models_dir)

    kwargs: Dict[str, Any] = {
        "processors": processors,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    if classla_type != "standard":
        kwargs["type"] = classla_type

    kwargs = add_models_dir_argument(classla.Pipeline, kwargs, normalized_models_dir)
    return classla.Pipeline(lang, **kwargs)
