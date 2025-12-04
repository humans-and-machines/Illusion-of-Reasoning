import re

import src.inference.utils.text_utils as tu


def test_find_markers_and_context_with_prefix_skip():
    think = "lead-in text then marker here and more text"
    prompt = "prompt"
    patterns = [("mark", re.compile(r"marker"))]
    markers, pos, context, excerpt = tu.find_markers_and_context(
        think_text=think,
        prompt_text=prompt,
        patterns=patterns,
        skip_prefix_chars=5,
    )
    assert markers == ["mark"]
    assert pos == think.find("marker")
    assert context.startswith(prompt)
    assert "marker" in excerpt


def test_find_markers_and_context_handles_empty():
    markers, pos, context, excerpt = tu.find_markers_and_context(
        think_text=None,
        prompt_text="p",
        patterns=[("a", re.compile("a"))],
    )
    assert markers == [] and pos is None and context is None and excerpt is None


def test_find_markers_and_context_handles_missing_start():
    class FakeMatch:
        def start(self):
            return None

    class FakePattern:
        def search(self, text):
            return FakeMatch()

    markers, pos, context, excerpt = tu.find_markers_and_context(
        think_text="abc",
        prompt_text="prompt",
        patterns=[("fake", FakePattern())],
    )
    assert markers == ["fake"]
    assert pos is None
    assert context == "abc"
    assert excerpt == "abc"
