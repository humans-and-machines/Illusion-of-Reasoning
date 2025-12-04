#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import src.annotate.infra.config as cfg


def test_load_yaml_optional_dependency(tmp_path, monkeypatch):
    # Without yaml module or missing file -> {}
    monkeypatch.setattr(cfg, "_yaml", None)
    assert cfg.load_yaml(tmp_path / "missing.yml") == {}

    # With yaml and valid file
    import yaml

    monkeypatch.setattr(cfg, "_yaml", yaml)
    yaml_path = tmp_path / "conf.yml"
    yaml_path.write_text("key: val\n", encoding="utf-8")
    assert cfg.load_yaml(yaml_path) == {"key": "val"}


def test_load_azure_config_precedence(tmp_path, monkeypatch):
    # YAML fallback when env unset
    import yaml

    monkeypatch.setattr(cfg, "_yaml", yaml)
    yaml_path = tmp_path / "azure.yml"
    yaml_path.write_text(
        "endpoint: https://example.com/\napi_key: k\ndeployment: dep\napi_version: v\nuse_v1: true\n",
        encoding="utf-8",
    )
    cfg_dict = cfg.load_azure_config(str(yaml_path))
    assert cfg_dict["endpoint"] == "https://example.com"  # stripped trailing slash
    assert cfg_dict["api_key"] == "k"
    assert cfg_dict["deployment"] == "dep"
    assert cfg_dict["api_version"] == "v"
    assert cfg_dict["use_v1"] is True

    # Environment overrides YAML
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "envkey")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "envdep")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "envv")
    monkeypatch.setenv("AZURE_OPENAI_USE_V1", "0")
    cfg_dict_env = cfg.load_azure_config(str(yaml_path))
    assert cfg_dict_env["endpoint"] == "https://env.com"
    assert cfg_dict_env["api_key"] == "envkey"
    assert cfg_dict_env["deployment"] == "envdep"
    assert cfg_dict_env["api_version"] == "envv"
    assert cfg_dict_env["use_v1"] is False

    # Clean up env variables
    for var in [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_USE_V1",
    ]:
        monkeypatch.delenv(var, raising=False)


def test_load_sandbox_config_env(monkeypatch):
    monkeypatch.setenv("SANDBOX_ENDPOINT", "http://sandbox/")
    monkeypatch.setenv("SANDBOX_API_KEY", "k")
    monkeypatch.setenv("SANDBOX_API_VER", "v1")
    monkeypatch.setenv("SANDBOX_DEPLOYMENT", "d")
    cfg_dict = cfg.load_sandbox_config()
    assert cfg_dict == {
        "endpoint": "http://sandbox/",
        "api_key": "k",
        "api_version": "v1",
        "deployment": "d",
    }
