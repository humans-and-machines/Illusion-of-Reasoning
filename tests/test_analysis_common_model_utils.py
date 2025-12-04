import pytest

from src.analysis.common import model_utils


@pytest.mark.parametrize(
    "name_lower,expected",
    [
        ("llama3.1-8b", "llama8b"),
        ("my-llama-8b-run", "llama8b"),
        ("qwen2.5-7b-chat", "qwen7b"),
        ("mystery-7b", "qwen7b"),  # 7B without llama defaults to qwen7b
        ("small-model", None),
    ],
)
def test_detect_model_key(name_lower, expected):
    assert model_utils.detect_model_key(name_lower) == expected


@pytest.mark.parametrize(
    "name_lower,expected",
    [
        ("math-temp-0.05", "Math"),
        ("crossword-eval", "Crossword"),
        ("rush-hour", "Carpark"),
        ("carpark-baseline", "Carpark"),
        ("unknown-domain", None),
    ],
)
def test_detect_domain(name_lower, expected):
    assert model_utils.detect_domain(name_lower) == expected


@pytest.mark.parametrize(
    "name_lower,low_alias,expected",
    [
        ("temp-0.7", 0.05, pytest.approx(0.7)),  # explicit numeric
        ("low-temp", 0.15, pytest.approx(0.15)),  # low alias by token
        ("low_temp", 0.2, pytest.approx(0.2)),  # underscores accepted
        ("temp-abc", 0.1, None),  # unparsable numeric segment
        ("no-temp-token", 0.3, None),
    ],
)
def test_detect_temperature(name_lower, low_alias, expected):
    assert model_utils.detect_temperature(name_lower, low_alias=low_alias) == expected


def test_detect_temperature_handles_valueerror(monkeypatch):
    class DummyMatch:
        def group(self, idx):
            return "not-a-number"

    class DummyPattern:
        def search(self, name):
            return DummyMatch()

    monkeypatch.setattr(model_utils, "_TEMP_PATTERN", DummyPattern())
    assert model_utils.detect_temperature("temp-bad", low_alias=0.5) is None
