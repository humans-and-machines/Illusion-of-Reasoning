import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.features = None
        self.saved_to = None

    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    @classmethod
    def from_dict(cls, data, features=None):
        # Reconstruct row dictionaries from column-wise data.
        keys = list(data.keys())
        row_count = len(data[keys[0]]) if keys else 0
        rows = [{key: data[key][idx] for key in keys} for idx in range(row_count)]
        ds = cls(rows)
        ds.features = features
        return ds

    def select(self, indices):
        return _FakeDataset([self.rows[i] for i in indices])

    def save_to_disk(self, path):
        self.saved_to = Path(path)
        return self

    def push_to_hub(self, repo_id, private=True):
        # Record the push target so tests can assert on it if needed.
        self.pushed_to = (repo_id, private)
        return self

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class _FakeDatasetDict(dict):
    def save_to_disk(self, path):
        self.saved_to = Path(path)
        return self

    def push_to_hub(self, repo_id, private=True):
        self.pushed_to = (repo_id, private)
        return self


class _FakeValue:
    def __init__(self, dtype):
        self.dtype = dtype


class _FakeSequence:
    def __init__(self, feature):
        self.feature = feature


class _FakeFeatures(dict):
    pass


def _build_fake_datasets_module():
    module = types.SimpleNamespace()
    module.Dataset = _FakeDataset
    module.DatasetDict = _FakeDatasetDict
    module.Features = _FakeFeatures
    module.Value = _FakeValue
    module.Sequence = _FakeSequence
    module.load_dataset = lambda repo: {"loaded": repo}

    def _load_from_disk(path):
        if isinstance(path, Path) and not path.exists():
            raise FileNotFoundError(path)
        if isinstance(path, str) and path.startswith("missing"):
            raise FileNotFoundError(path)
        return {"disk": path}

    module.load_from_disk = _load_from_disk
    return module


def _build_fake_hf_module():
    class _FakeHfFolder:
        @staticmethod
        def get_token():
            return None

    class _FakeHfApi:
        def upload_file(self, **kwargs):  # pragma: no cover - behavior recorded via kwargs
            self.last_upload = kwargs

    return types.SimpleNamespace(HfApi=_FakeHfApi, HfFolder=_FakeHfFolder)


@pytest.fixture(autouse=True)
def stub_external_modules(monkeypatch):
    """
    Provide lightweight stand-ins for optional heavy dependencies so the data
    helpers can be imported without installing datasets/huggingface_hub.
    """
    fake_datasets = _build_fake_datasets_module()
    fake_hf = _build_fake_hf_module()
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    yield
