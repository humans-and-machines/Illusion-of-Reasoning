from src.training.utils.replay_dataset import ReplayMixDataset, replay_collate


class DummyTokenizer:
    def decode(self, ids):
        return "prompt with clue"


def test_replay_mix_dataset_sets_flag_and_decodes():
    base = [
        {
            "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "clue"}],
            "input_ids": [1, 2, 3],
        }
    ]
    ds = ReplayMixDataset(base, tokenizer=DummyTokenizer())
    item = ds[0]
    assert item is not base[0]
    assert item["is_replay"] == 0


def test_replay_mix_dataset_len_passthrough():
    base = [{"prompt": [], "input_ids": []} for _ in range(3)]
    ds = ReplayMixDataset(base, tokenizer=DummyTokenizer())
    assert len(ds) == 3


def test_replay_collate_resets_is_replay():
    batch = [{"is_replay": 1, "accuracy": 1.0}, {"is_replay": 2}]
    out = replay_collate(batch, _replay_buffer=None, _replay_prob=0.0)
    assert all(example["is_replay"] == 0 for example in out)
    # ensure other fields survive
    assert "accuracy" in out[0]
