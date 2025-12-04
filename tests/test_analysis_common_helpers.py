import numpy as np
import pandas as pd

from src.analysis.common.pass_extraction import extract_pass_answer, extract_pass_text
from src.analysis.common.problem_utils import resolve_problem_identifier
from src.analysis.common.uncertainty import standardize_uncertainty, standardize_uncertainty_with_stats


def test_extract_pass_text_prefers_primary_fields():
    pass_dict = {
        "text": "",
        "output": " final answer ",
        "messages": [{"role": "assistant", "content": "ignored"}],
    }
    assert extract_pass_text(pass_dict) == " final answer "


def test_extract_pass_text_from_messages_and_choices():
    pass_dict = {
        "messages": [
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "latest assistant reply"},
        ],
        "choices": [{"message": {"content": "choice content"}}],
    }
    assert extract_pass_text(pass_dict) == "latest assistant reply"

    assert extract_pass_text({"choices": [{"message": {"content": "choice content"}}]}) == ("choice content")


def test_extract_pass_answer_prioritizes_answer_fields_and_tags():
    tagged_field = {"final_answer": "<answer>42</answer> extra"}
    assert extract_pass_answer(tagged_field) == "42"

    tagged_output = {"output": "prefix <answer>first</answer> <answer>second</answer>"}
    assert extract_pass_answer(tagged_output) == "second"


def test_extract_pass_answer_fallbacks_and_types():
    fallback_text = {"text": "line one\n\nlast line"}
    assert extract_pass_answer(fallback_text) == "last line"
    assert extract_pass_answer(None) is None


def test_resolve_problem_identifier_priority_and_fallbacks():
    record_with_problem = {"problem": "abc"}
    assert resolve_problem_identifier(record_with_problem) == "abc"

    record_with_question = {"question": 123}
    assert resolve_problem_identifier(record_with_question) == "123"

    dataset_index_only = {"dataset_index": 7}
    assert resolve_problem_identifier(dataset_index_only) == "idx:7"

    assert resolve_problem_identifier({}, fallback="foo") == "foo"
    assert resolve_problem_identifier({}) == "unknown"


def test_standardize_uncertainty_creates_new_column_without_mutation():
    data = pd.DataFrame({"uncertainty": [1.0, 2.0, 3.0], "other": [0, 1, 2]})
    original_copy = data.copy()
    standardized = standardize_uncertainty(data)
    assert "uncertainty_std" in standardized.columns
    np.testing.assert_allclose(
        standardized["uncertainty_std"].to_numpy(),
        np.array([-1.2247448, 0.0, 1.2247448]),
        atol=1e-6,
    )
    pd.testing.assert_frame_equal(data, original_copy)


def test_standardize_uncertainty_with_stats_returns_expected():
    data = pd.DataFrame({"uncertainty": [10.0, 12.0, 14.0]})
    standardized, mean_val, std_val = standardize_uncertainty_with_stats(data)
    assert mean_val == 12.0
    assert std_val == 1.632993161855452  # population std with ddof=0
    np.testing.assert_allclose(
        standardized["uncertainty_std"].to_numpy(),
        np.array([-1.2247448, 0.0, 1.2247448]),
        atol=1e-6,
    )
