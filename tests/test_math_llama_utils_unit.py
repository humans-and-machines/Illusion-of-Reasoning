import pytest


torch = pytest.importorskip("torch")

import src.inference.utils.math_llama_utils as utils  # noqa: E402


class DummyModule:
    def __init__(self):
        self.config = {"foo": "bar"}
        self.extra_attr = "from-module"
        self.eval_called = False
        self.generated_with = None

    def parameters(self):
        return ["p1", "p2"]

    def generate(self, *args, **kwargs):
        self.generated_with = (args, kwargs)
        return "gen-output"

    def eval(self):
        self.eval_called = True


class DummyEngine:
    def __init__(self, include_device=True):
        self.module = DummyModule()
        if include_device:
            self.device = torch.device("cpu")


def test_ds_model_wrapper_delegates_and_defaults_device():
    wrapper = utils.DSModelWrapper(DummyEngine(include_device=False))
    # Device falls back to torch.device(...) (or stubbed value)
    assert wrapper.device is not None
    assert wrapper.config == {"foo": "bar"}
    # __getattr__ should delegate attributes present only on module
    assert wrapper.extra_attr == "from-module"

    # __getattr__ delegates to module
    assert wrapper.generate("x") == "gen-output"
    assert wrapper.module.generated_with[0][0] == "x"
    assert list(wrapper.parameters()) == ["p1", "p2"]
    assert wrapper.eval() is wrapper
    assert wrapper.module.eval_called is True
    with pytest.raises(AttributeError):
        _ = wrapper.missing_attr


def test_generated_batch_view_holds_fields():
    gbv = utils.GeneratedBatchView(sequences=[1], scores=[2], model="m")
    assert gbv.sequences == [1]
    assert gbv.scores == [2]
    assert gbv.model == "m"
