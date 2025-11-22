#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared helpers for discovering 7B/8B roots based on model/domain/temp tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List

from src.analysis.common.model_utils import (
    MODEL_LABELS,
    detect_domain,
    detect_model_key,
    detect_temperature,
)

@dataclass(frozen=True)
class DiscoveryConfig:
    """Configuration used when discovering 7B/8B model roots."""

    temperatures: List[float]
    low_alias: float
    wanted_models: List[str]
    wanted_domains: List[str]
    verbose: bool = False


def _update_mapping(mapping, model_key: str, temperature: float, domain: str, path: str) -> None:
    model_map = mapping.setdefault(model_key, {})
    temp_map = model_map.setdefault(temperature, {})
    existing = temp_map.get(domain)
    if existing is None or len(path) > len(existing):
        temp_map[domain] = path


def _print_discovered_roots(mapping: Dict[str, Dict[float, Dict[str, str]]]) -> None:
    print("[info] discovered roots (7B/8B):")
    for model_key, temps_map in mapping.items():
        label = MODEL_LABELS.get(model_key, model_key)
        print(f"  [{label}]")
        for temperature_value in sorted(temps_map):
            print(
                f"    temp {temperature_value}: {list(temps_map[temperature_value].keys())}"
            )


def discover_roots_7b8b(
    scan_root: str,
    config: DiscoveryConfig,
) -> Dict[str, Dict[float, Dict[str, str]]]:
    """Discover the deepest subdirectories matching the requested temps/domains."""
    temps_set = {float(value) for value in config.temperatures}
    wanted_models = {m.lower() for m in config.wanted_models}
    wanted_domains = set(config.wanted_domains)

    mapping: Dict[str, Dict[float, Dict[str, str]]] = {}
    for entry in os.listdir(scan_root):
        path = os.path.join(scan_root, entry)
        if not os.path.isdir(path):
            continue

        model_key = detect_model_key(entry.lower())
        if model_key is None or model_key not in wanted_models:
            continue

        domain = detect_domain(entry.lower())
        if domain is None or domain not in wanted_domains:
            continue

        temperature = detect_temperature(entry.lower(), config.low_alias)
        if temperature is None or float(temperature) not in temps_set:
            continue

        _update_mapping(
            mapping,
            model_key,
            float(temperature),
            domain,
            path,
        )

    if config.verbose and mapping:
        _print_discovered_roots(mapping)
    return mapping
