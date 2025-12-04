import types

import numpy as np
import pandas as pd
import pytest

import src.analysis.metrics as metrics


def test_correct_from_mapping_handles_acc_and_score():
    mapping_acc = {"accuracy": 0.4}
    assert metrics._correct_from_mapping(mapping_acc) == 1
    mapping_score = {"score": 0.75}
    assert metrics._correct_from_mapping(mapping_score) == 1
    mapping_score_low = {"score": 0.4}
    assert metrics._correct_from_mapping(mapping_score_low) == 0


def test_cond_counts_handles_zero_and_nonzero():
    df = pd.DataFrame({"shift": [1, 0, 1], "correct": [1, 0, 0]})
    total, share, p_shift, p_no = metrics.cond_counts(df)
    assert total == 3 and share == pytest.approx(2 / 3)
    assert p_shift == pytest.approx(0.5) and p_no == 0.0

    empty_total, empty_share, empty_p_shift, empty_p_no = metrics.cond_counts(
        pd.DataFrame({"shift": [], "correct": []})
    )
    assert np.isnan(empty_share) and np.isnan(empty_p_shift) and np.isnan(empty_p_no)


def test_glm_fit_with_covariance_typeerror(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.calls = []

        def fit(self, cov_type=None, cov_kwds=None):
            self.calls.append((cov_type, cov_kwds))
            if cov_kwds and "use_correction" in cov_kwds:
                raise TypeError("fail with cov_kwds")
            return "ok"

    df = pd.DataFrame({"problem": ["a", "b"]})
    model = DummyModel()
    result, cov_type, cov_kwds = metrics.glm_fit_with_covariance(model, df, cluster_by="problem")
    assert result == "ok"
    assert cov_type == "cluster" and cov_kwds is not None
    # first call should have had groups kwds, second minimal_kw empty
    assert len(model.calls) == 2


def test_write_glm_summary_header_includes_cluster_info(tmp_path):
    class DummySummary:
        def as_text(self):
            return "summary"

    class DummyRes:
        def summary(self):
            return DummySummary()

    out_txt = tmp_path / "glm.txt"
    metrics.write_glm_summary_header(str(out_txt), DummyRes(), cov_type="cluster", cov_kwds={"groups": [0, 1]})
    content = out_txt.read_text()
    assert "summary" in content and "clustered by problem" in content


def test_lazy_import_statsmodels_raises(monkeypatch):
    monkeypatch.setattr(metrics.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("no")))
    with pytest.raises(RuntimeError):
        metrics.lazy_import_statsmodels()


def test_predict_formula_with_design_info(monkeypatch):
    # force design_info branch with fake patsy
    def fake_import(name):
        fake_patsy = types.SimpleNamespace(
            build_design_matrices=lambda design_infos, data, return_type=None: [data],
        )
        return fake_patsy

    monkeypatch.setattr(metrics.importlib, "import_module", fake_import)

    class DummyFamily:
        class Link:
            def inverse(self, x):
                return x

        link = Link()

    class DummyModel:
        def __init__(self):
            self.data = types.SimpleNamespace(design_info="design")
            self.family = DummyFamily()

    class DummyRes:
        def __init__(self):
            self.params = np.array([1.0, 2.0])

        def predict(self, *_a, **_k):
            raise TypeError("fallback")

    new_df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    out = metrics.predict_formula(DummyRes(), DummyModel(), new_df)
    assert np.allclose(out, np.dot(new_df.to_numpy(), np.array([1.0, 2.0])))
