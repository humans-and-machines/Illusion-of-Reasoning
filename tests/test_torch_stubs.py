import pytest

from src.inference.utils._torch_stubs import StoppingCriteriaStub, TorchStub


def test_torch_stub_import_error_on_attr():
    stub = TorchStub()
    with pytest.raises(ImportError):
        _ = stub.linspace  # any attribute triggers ImportError
    assert stub.is_available() is False
    assert stub.device() == "cpu"
    # tensor helpers return raw/nested data
    assert stub.tensor([1, 2]) == [1, 2]
    assert stub.zeros((2, 3)) == [[0, 0, 0]]
    assert stub.ones((1, 2)) == [[1, 1]]
    wrapped = stub.inference_mode()(lambda x: x + 1)
    assert wrapped(1) == 2


def test_stopping_criteria_stub_behavior():
    crit = StoppingCriteriaStub()
    with pytest.raises(ImportError):
        crit(inputs=None)
    assert crit.clone() is crit
    assert crit.has_stops() is False
