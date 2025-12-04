from types import SimpleNamespace

import pytest

import src.training.utils.data as data_utils


def test_require_datasets_module_raises(monkeypatch):
    monkeypatch.setattr(data_utils, "import_module", lambda name: (_ for _ in ()).throw(ImportError("missing")))
    with pytest.raises(ImportError):
        data_utils._require_datasets_module()


def test_get_dataset_mixture_applies_columns_weight_and_split(monkeypatch):
    calls = []

    class FakeDataset:
        def __init__(self, name, size=10, columns=None):
            self.name = name
            self.data = list(range(size))
            self.columns = columns or ["a", "b"]

        def select_columns(self, columns):
            calls.append(("select_columns", self.name, tuple(columns)))
            return FakeDataset(self.name, len(self.data), columns=list(columns))

        def shuffle(self, seed):
            calls.append(("shuffle", self.name, seed))
            return self

        def select(self, indices):
            count = len(list(indices))
            calls.append(("select", self.name, count))
            return FakeDataset(self.name, count, columns=self.columns)

        def train_test_split(self, test_size, seed):
            calls.append(("train_test_split", self.name, test_size, seed))
            test_len = int(len(self.data) * test_size) if isinstance(test_size, float) else int(test_size)
            test_len = min(len(self.data), max(test_len, 0))
            train_len = len(self.data) - test_len
            return {
                "train": FakeDataset(f"{self.name}-train", train_len, columns=self.columns),
                "test": FakeDataset(f"{self.name}-test", test_len, columns=self.columns),
            }

        def __len__(self):
            return len(self.data)

    class FakeDatasets:
        @staticmethod
        def load_dataset(dataset_id, config, split=None):
            calls.append(("load_dataset", dataset_id, config, split))
            return FakeDataset(dataset_id, size=10, columns=["x", "y", "z"])

        @staticmethod
        def concatenate_datasets(datasets_list):
            calls.append(("concatenate", [len(ds) for ds in datasets_list]))
            total = sum(len(ds) for ds in datasets_list)
            return FakeDataset("concat", total, columns=datasets_list[0].columns)

    monkeypatch.setattr(data_utils, "_require_datasets_module", lambda: FakeDatasets)

    dataset_cfg = SimpleNamespace(dataset_id="ds1", config="cfg", split="train", columns=["x", "y"], weight=0.5)
    mixture = SimpleNamespace(datasets=[dataset_cfg], seed=123, test_split_size=0.4)
    args = SimpleNamespace(dataset_name=None, dataset_config=None, dataset_mixture=mixture)

    result = data_utils.get_dataset(args)
    assert set(result.keys()) == {"train", "test"}
    assert len(result["train"]) + len(result["test"]) == 5  # after weight-based subsample to 5
    assert any(call[0] == "select_columns" for call in calls)
    assert ("shuffle", "ds1", 123) in calls  # shuffle before select
    assert any(call[0] == "concatenate" for call in calls)
    assert ("shuffle", "concat", 123) in calls  # shuffle after concatenation
    assert any(call[0] == "train_test_split" for call in calls)


def test_get_dataset_mixture_with_no_datasets_raises(monkeypatch):
    class FakeDatasets:
        @staticmethod
        def load_dataset(*args, **kwargs):
            raise AssertionError("should not load any dataset")

        @staticmethod
        def concatenate_datasets(*args, **kwargs):
            raise AssertionError("should not concatenate")

    monkeypatch.setattr(data_utils, "_require_datasets_module", lambda: FakeDatasets)
    args = SimpleNamespace(
        dataset_name=None, dataset_mixture=SimpleNamespace(datasets=[], seed=0, test_split_size=None)
    )
    with pytest.raises(ValueError, match="No datasets were loaded"):
        data_utils.get_dataset(args)


def test_get_dataset_mixture_without_split_returns_train(monkeypatch):
    calls = []

    class FakeDataset:
        def __init__(self, name, size=3):
            self.name = name
            self.data = list(range(size))

        def shuffle(self, seed):
            calls.append(("shuffle", self.name, seed))
            return self

        def __len__(self):
            return len(self.data)

    class FakeDatasets:
        @staticmethod
        def load_dataset(dataset_id, config, split=None):
            calls.append(("load_dataset", dataset_id, config, split))
            return FakeDataset(dataset_id, size=4)

        @staticmethod
        def concatenate_datasets(datasets_list):
            calls.append(("concatenate", [len(ds) for ds in datasets_list]))
            total = sum(len(ds) for ds in datasets_list)
            return FakeDataset("concat", total)

    monkeypatch.setattr(data_utils, "_require_datasets_module", lambda: FakeDatasets)
    dataset_cfg = SimpleNamespace(dataset_id="ds", config=None, split="train", columns=None, weight=None)
    args = SimpleNamespace(
        dataset_name=None,
        dataset_config=None,
        dataset_mixture=SimpleNamespace(datasets=[dataset_cfg], seed=5, test_split_size=None),
    )
    result = data_utils.get_dataset(args)
    assert set(result.keys()) == {"train"}
    assert isinstance(result["train"], FakeDataset)
    assert result["train"].name == "concat"
    assert ("shuffle", "concat", 5) in calls  # shuffled after concatenation


def test_get_dataset_single_name(monkeypatch):
    called = {}

    class FakeDatasets:
        @staticmethod
        def load_dataset(name, config):
            called["args"] = (name, config)
            return {"ds": name, "cfg": config}

    monkeypatch.setattr(data_utils, "_require_datasets_module", lambda: FakeDatasets)
    args = SimpleNamespace(dataset_name="my-ds", dataset_config="cfg", dataset_mixture=None)
    result = data_utils.get_dataset(args)
    assert result == {"ds": "my-ds", "cfg": "cfg"}
    assert called["args"] == ("my-ds", "cfg")
