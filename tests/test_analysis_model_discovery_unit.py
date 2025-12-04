import src.analysis.model_discovery as md


def test_update_mapping_prefers_deeper_paths():
    mapping = {}
    md._update_mapping(mapping, "m", 0.7, "Math", "/a")
    md._update_mapping(mapping, "m", 0.7, "Math", "/a/b/c")
    assert mapping["m"][0.7]["Math"] == "/a/b/c"


def test_print_discovered_roots(capsys):
    mapping = {"m": {0.1: {"Math": "p"}}}
    md._print_discovered_roots(mapping)
    out = capsys.readouterr().out
    assert "discovered roots" in out and "temp 0.1" in out


def test_discover_roots_filters(monkeypatch, tmp_path):
    dirs = [
        "qwen-math-temp-0.7",
        "llama-crossword-temp-0.5",
        "qwen-math-temp-0.3",
        "ignore.txt",
    ]
    for d in dirs:
        path = tmp_path / d
        if d.endswith(".txt"):
            path.write_text("x", encoding="utf-8")
        else:
            path.mkdir()

    monkeypatch.setattr(
        md,
        "detect_model_key",
        lambda name: "qwen" if "qwen" in name else None,
    )
    monkeypatch.setattr(
        md,
        "detect_domain",
        lambda name: "Math" if "math" in name else None,
    )
    monkeypatch.setattr(
        md,
        "detect_temperature",
        lambda name, low_alias: 0.7 if "0.7" in name else 0.3 if "0.3" in name else None,
    )

    config = md.DiscoveryConfig(
        temperatures=[0.7],
        low_alias=0.3,
        wanted_models=["qwen"],
        wanted_domains=["Math"],
        verbose=True,
    )
    mapping = md.discover_roots_7b8b(str(tmp_path), config)
    # Detect the lower temp candidate and ensure longest path is kept.
    config2 = md.DiscoveryConfig(
        temperatures=[0.3, 0.7],
        low_alias=0.3,
        wanted_models=["qwen"],
        wanted_domains=["Math"],
        verbose=True,
    )
    mapping = md.discover_roots_7b8b(str(tmp_path), config2)
    assert "qwen" in mapping and 0.7 in mapping["qwen"] and 0.3 in mapping["qwen"]
    assert mapping["qwen"][0.7]["Math"].endswith("qwen-math-temp-0.7")


def test_discover_roots_no_matches_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(md, "detect_model_key", lambda *_: None)
    config = md.DiscoveryConfig(
        temperatures=[0.7],
        low_alias=0.3,
        wanted_models=["qwen"],
        wanted_domains=["Math"],
    )
    assert md.discover_roots_7b8b(str(tmp_path), config) == {}


def test_discover_roots_skips_unwanted_domain(monkeypatch, tmp_path):
    path = tmp_path / "qwen-carpark-temp-0.7"
    path.mkdir()
    monkeypatch.setattr(md, "detect_model_key", lambda name: "qwen")
    monkeypatch.setattr(md, "detect_domain", lambda name: "Carpark")
    monkeypatch.setattr(md, "detect_temperature", lambda name, low: 0.7)

    config = md.DiscoveryConfig(
        temperatures=[0.7],
        low_alias=0.3,
        wanted_models=["qwen"],
        wanted_domains=["Math"],  # Carpark not allowed
    )
    mapping = md.discover_roots_7b8b(str(tmp_path), config)
    assert mapping == {}
