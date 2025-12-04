import pytest

from src.training.utils.replay_dataset import ReplayMixDataset, replay_collate


class DummyTokenizer:
    def __init__(self, decode_text):
        self.decode_text = decode_text

    def decode(self, input_ids):
        return self.decode_text


def test_replay_mix_dataset_sets_flag_and_validates_prompt():
    base = [
        {
            "prompt": [{"role": "user", "content": "clue"}],
            "input_ids": [1, 2, 3],
            "extra": 123,
        }
    ]
    ds = ReplayMixDataset(base, tokenizer=DummyTokenizer("prefix clue suffix"))
    item = ds[0]
    assert item["is_replay"] == 0
    assert item["extra"] == 123
    # Ensure deep copy
    item["extra"] = 999
    assert base[0]["extra"] == 123

    with pytest.raises(AssertionError):
        bad_ds = ReplayMixDataset(base, tokenizer=DummyTokenizer("missing"))
        bad_ds[0]


def test_replay_collate_resets_flags_and_keeps_fields():
    batch = [
        {"is_replay": 1, "accuracy": 0.5},
        {"accuracy": 0.8},
    ]
    out = replay_collate(batch, _replay_buffer=None, _replay_prob=0.5)
    assert all(ex["is_replay"] == 0 for ex in out)
    assert all("accuracy" in ex for ex in out)
