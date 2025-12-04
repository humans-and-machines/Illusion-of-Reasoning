from src.analysis.common import pass_extraction as pe


def test_extract_pass_text_prefers_known_fields_and_messages():
    assert pe.extract_pass_text(None) is None
    pass_dict = {"output": "  answer text  "}
    assert pe.extract_pass_text(pass_dict) == "  answer text  "

    msg_pass = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "resp"},
        ]
    }
    assert pe.extract_pass_text(msg_pass) == "resp"

    choice_pass = {"choices": [{"message": {"content": "choice content"}}]}
    assert pe.extract_pass_text(choice_pass) == "choice content"


def test_extract_pass_answer_from_fields_and_tags():
    tagged = {"final_answer": "foo <answer>bar</answer> baz"}
    assert pe.extract_pass_answer(tagged) == "bar"

    plain = {"answer": " trimmed "}
    assert pe.extract_pass_answer(plain) == "trimmed"


def test_extract_pass_answer_from_output_and_text_fallback():
    output_tagged = {"output": "prefix <answer>first</answer>\n<answer>second</answer>"}
    assert pe.extract_pass_answer(output_tagged) == "second"

    output_lines = {"output": "line1\nline2"}
    assert pe.extract_pass_answer(output_lines) == "line2"

    text_fallback = {"text": "blah\n<answer>fromtext</answer>"}
    assert pe.extract_pass_answer(text_fallback) == "fromtext"

    empty = {"output": "   ", "text": ""}
    assert pe.extract_pass_answer(empty) is None

    long_line = {"output": "line1\n" + ("x" * 201)}
    assert pe.extract_pass_answer(long_line) is None

    long_text = {"text": "hello\n" + ("y" * 205)}
    assert pe.extract_pass_answer(long_text) is None
