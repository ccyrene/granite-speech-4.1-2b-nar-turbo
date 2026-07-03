"""Build adaptive configs from plain dicts / a YAML file (matches OPTIMIZATION_PLAN "Config Example").

Pure dict -> dataclass so it is CPU-testable; YAML loading is optional (guarded import of pyyaml).
Unknown keys are ignored so the yaml can carry extra doc fields without breaking construction.
"""
from __future__ import annotations

from dataclasses import fields

from .routing import RoutingConfig
from .verifier import VerifierConfig
from .early_exit import EarlyExitConfig


def _filter(dc, d: dict) -> dict:
    names = {f.name for f in fields(dc)}
    return {k: v for k, v in (d or {}).items() if k in names}


def routing_from_dict(d: dict) -> RoutingConfig:
    return RoutingConfig(**_filter(RoutingConfig, d))


def verifier_from_dict(d: dict) -> VerifierConfig:
    return VerifierConfig(**_filter(VerifierConfig, d))


def early_exit_from_dict(d: dict) -> EarlyExitConfig:
    return EarlyExitConfig(**_filter(EarlyExitConfig, d))


def load_adaptive_config(path_or_dict):
    """Return ``(RoutingConfig, VerifierConfig, EarlyExitConfig)`` from a dict or a YAML path.

    Expected top-level keys: ``routing``, ``verifier``, ``early_exit`` (all optional).
    """
    if isinstance(path_or_dict, dict):
        cfg = path_or_dict
    else:
        import yaml  # optional dep; only needed when loading from a file
        with open(path_or_dict) as f:
            cfg = yaml.safe_load(f)
    return (
        routing_from_dict(cfg.get("routing", {})),
        verifier_from_dict(cfg.get("verifier", {})),
        early_exit_from_dict(cfg.get("early_exit", {})),
    )
