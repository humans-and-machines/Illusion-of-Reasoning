# training/utils/replay_buffer.py
"""Simple replay buffer used for GRPO-style training."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import copy
import threading
import math
import numpy as np


def _prompt_key(prompt: list[dict[str, str]]) -> tuple:
    """Canonicalise a prompt into a hashable key."""
    return tuple((message["role"], " ".join(message["content"].split())) for message in prompt)


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Best-effort conversion to a finite float with a default fallback."""
    try:
        value_float = float(value)
        return value_float if math.isfinite(value_float) else default
    except (TypeError, ValueError):
        return default


def _is_full_example(example: Any) -> bool:
    """Return True when an object looks like a full prompt+answer example."""
    return isinstance(example, dict) and "prompt" in example and "answer" in example


class _ReplayStorage:  # pylint: disable=too-few-public-methods
    """Internal container for replay buffer samples and streaming statistics."""

    def __init__(self) -> None:
        self.buf: List[Any] = []
        self.mean: List[float] = []
        self.second_moment: List[float] = []
        self.count: List[int] = []
        self.uids: List[int] = []
        self.uid2idx: Dict[int, int] = {}


class _ReplayBufferConfig:  # pylint: disable=too-few-public-methods
    """Configuration for a :class:`ReplayBuffer` instance."""

    def __init__(
        self,
        capacity: int,
        ucb_coefficient: float,
        debug_steps: int,
    ) -> None:
        self.capacity = int(capacity)
        self.ucb_coefficient = float(ucb_coefficient)
        self.debug_steps = int(debug_steps)


class ReplayBuffer:
    """
    Bandit / UCB replay buffer with stable UIDs and optional exploit/explore mixing.

    * add_group(...) -> uid (>=0) on success, -1 on failure
    * sample_uid(...) returns a uid you can feed back into trainer._inject_one_replay_group
    * update_priority_by_uid updates μ / priority using a stable uid (safe for DDP)
    """

    def __init__(
        self,
        capacity: int = 4000,
        ucb_coefficient: float = 1.0,
        debug_steps: int = 0,
        **kwargs: Any,
    ):
        """
        Initialise an in-memory replay buffer.

        Accepts a legacy ``C=...`` keyword (mapped to ``ucb_coefficient``) for
        backwards compatibility.
        """
        if "C" in kwargs and kwargs["C"] is not None:
            ucb_coefficient = float(kwargs.pop("C"))

        self._config = _ReplayBufferConfig(capacity, ucb_coefficient, debug_steps)
        self._storage = _ReplayStorage()
        self._seen = set()  # prompt de-dup
        self._lock = threading.Lock()
        self._next_uid = 0
        self._last_error: Optional[dict] = None

    @property
    def capacity(self) -> int:
        """Maximum number of entries the buffer can hold."""
        return self._config.capacity

    @property
    def ucb_coefficient(self) -> float:
        """Exploration coefficient used for UCB-style sampling (if enabled)."""
        return self._config.ucb_coefficient

    @property
    def debug_steps(self) -> int:
        """Verbosity level controlling optional debug prints."""
        return self._config.debug_steps

    def last_error(self):
        """Return the last error metadata recorded during insertion."""
        return self._last_error

    def _set_err(self, **kwargs):
        self._last_error = kwargs


    def __len__(self) -> int:
        return len(self._storage.buf)

    # ----------------- stats utils -----------------
    def _init_stats(self, reward: float) -> tuple[float, float, int]:
        """Initialise streaming mean/variance stats for a reward."""
        return float(reward), 0.0, 1

    def _update_stats(self, idx: int, reward: float) -> None:
        """Welford update of mean and second moment for a single index."""
        mean_value = self._storage.mean[idx]
        second_moment_value = self._storage.second_moment[idx]
        count = self._storage.count[idx]

        count += 1
        delta = reward - mean_value
        mean_value += delta / count
        delta2 = reward - mean_value
        second_moment_value += delta * delta2

        self._storage.mean[idx] = mean_value
        self._storage.second_moment[idx] = second_moment_value
        self._storage.count[idx] = count

    # ----------------- helpers -----------------
    def _key_for_sample(self, sample: Any):
        # Deduplicate on prompt content
        if _is_full_example(sample):
            return _prompt_key(sample["prompt"])
        if isinstance(sample, dict) and "group" in sample:
            try:
                return tuple(
                    _prompt_key(ex["prompt"]) if _is_full_example(ex) else repr(ex)
                    for ex in sample["group"]
                )
            except (TypeError, KeyError):
                return repr(sample)
        return repr(sample)

    # ----------------- mutation -----------------
    def add(self, sample: Any, reward: float) -> tuple[bool, int]:
        """
        Add (sample, reward). If full, replace the worst-μ entry only if reward is higher.
        Returns (inserted?, uid). On failure, returns (False, -1).
        """
        if self.capacity <= 0:
            print("[RB][WARN] capacity <= 0; refusing to add")
            return False, -1

        key = self._key_for_sample(sample)

        with self._lock:
            if key in self._seen:
                self._set_err(
                    where="add",
                    why="dedup",
                    capacity=self.capacity,
                    len=len(self._storage.buf),
                )
                return False, -1
            self._seen.add(key)

            uid = self._next_uid
            self._next_uid += 1

            mean_value, second_moment_value, count = self._init_stats(reward)

            if len(self._storage.buf) < self.capacity:
                self._storage.buf.append(copy.deepcopy(sample))
                self._storage.mean.append(mean_value)
                self._storage.second_moment.append(second_moment_value)
                self._storage.count.append(count)
                self._storage.uids.append(uid)
                self._storage.uid2idx[uid] = len(self._storage.buf) - 1
                if self.debug_steps:
                    print(
                        f"[RB][ADD] inserted uid={uid} μ={mean_value:.4f} "
                        f"size={len(self._storage.buf)}",
                    )
                return True, uid

            if len(self._storage.buf) >= self.capacity:
                worst = int(np.argmin(self._storage.mean))
                if reward <= self._storage.mean[worst]:
                    self._set_err(
                        where="add",
                        why="capacity_worse_mu",
                        cap=self.capacity,
                        worst_mu=self._storage.mean[worst],
                        r=reward,
                    )
                    return False, -1

            # Full: maybe replace worst μ
            worst = int(np.argmin(self._storage.mean))
            if reward > self._storage.mean[worst]:
                old_uid = self._storage.uids[worst]
                if old_uid in self._storage.uid2idx:
                    del self._storage.uid2idx[old_uid]
                self._storage.buf[worst] = copy.deepcopy(sample)
                self._storage.mean[worst] = mean_value
                self._storage.second_moment[worst] = second_moment_value
                self._storage.count[worst] = count
                self._storage.uids[worst] = uid
                self._storage.uid2idx[uid] = worst
                if self.debug_steps:
                    print(
                        f"[RB][REPLACE] worst idx={worst} -> uid={uid} "
                        f"μ={mean_value:.4f}",
                    )
                return True, uid

            # Not inserted
            if self.debug_steps:
                worst_mean = self._storage.mean[worst]
                print(f"[RB][SKIP] reward={reward:.4f} <= worst μ={worst_mean:.4f}; skip")
            return False, -1

    def add_group(
        self,
        group: List[dict[str, Any]],
        reward: Optional[float] = None,
        *,
        verbose: bool = False,
    ) -> int:
        """
        Store a *group* (list) of examples under a single uid.
        Returns the uid (or the same uid if it was deduped and not inserted).
        """
        # compute a safe local reward
        if reward is None:
            try:
                reward_local = float(np.mean([g.get("reward", 0.0) for g in group]))
            except (TypeError, ValueError):
                reward_local = 0.0
        else:
            reward_local = float(reward)

        if verbose or self.debug_steps:
            print(
                f"[RB][add_group] size={len(group)} μ={reward_local:.4f} "
                f"current_len={len(self)} cap={self.capacity}"
            )

        inserted, uid = self.add({"group": copy.deepcopy(group)}, reward_local)

        if verbose or self.debug_steps:
            print(f"[RB][ADD] inserted={inserted} uid={uid} μ={reward_local:.4f} size={len(self)}")

        return uid

    def update_priority_by_uid(self, uid: int, reward: float):
        """Update running priority statistics for a given uid."""
        reward = _finite_float(reward, 0.0)
        with self._lock:
            idx = self._storage.uid2idx.get(uid, None)
            if idx is None:
                if self.debug_steps:
                    print(f"[RB][WARN] update_priority_by_uid: uid={uid} not found")
                return
            self._update_stats(idx, reward)

    # Legacy (index-based) — keep if you use it elsewhere
    def update_priority(self, idx: int, reward: float):
        """Update running priority statistics using a raw buffer index."""
        with self._lock:
            if 0 <= idx < len(self._storage.buf):
                self._update_stats(idx, _finite_float(reward, 0.0))

    def debug_state(self):
        """Return a small debug snapshot of the buffer tail and metadata."""
        with self._lock:
            tail = slice(max(0, len(self._storage.buf) - 5), len(self._storage.buf))
            return {
                "len": len(self._storage.buf),
                "capacity": self.capacity,
                "next_uid": self._next_uid,
                "tail_uids": self._storage.uids[tail],
                "tail_mu": [float(m) for m in self._storage.mean[tail]],
                "tail_n": self._storage.count[tail],
            }

    # ----------------- sampling -----------------
    def sample(
        self,
        batch_size: int = 1,
    ) -> Tuple[List[Any], List[int], List[int], np.ndarray]:
        """
        Uniformly sample `batch_size` distinct entries from the buffer.

        • Ignores μ, n, C, and mix_exploit_ratio.
        • Still returns (samples, idxs, uids, isw) so the rest of your
          training loop remains untouched.  `isw` stays all‑ones.
        """
        with self._lock:
            if not self._storage.buf:
                raise ValueError("Empty replay buffer")

            buffer_size = len(self._storage.buf)
            batch_size_clamped = min(batch_size, buffer_size)

            idxs = np.random.choice(
                buffer_size,
                size=batch_size_clamped,
                replace=False,
            ).tolist()

            samples = [copy.deepcopy(self._storage.buf[i]) for i in idxs]
            uids = [self._storage.uids[i] for i in idxs]
            isw = np.ones(batch_size_clamped, dtype=np.float32)  # importance‑sampling wts (unused)

            # Optional debug
            if self.debug_steps:
                print(f"[RB][SAMPLE] uniform idxs={idxs}")

            return samples, idxs, uids, isw

    def sample_uid(self, *_, **__) -> Optional[int]:
        """
        Convenience: return a single UID chosen uniformly at random.
        """
        with self._lock:
            if not self._storage.buf:
                return None
            return np.random.choice(self._storage.uids).item()

    def get_group(self, uid: int) -> List[dict[str, Any]]:
        """Return the stored group for a uid, or [] if missing."""
        with self._lock:
            idx = self._storage.uid2idx.get(uid, None)
            if idx is None:
                return []
            obj = self._storage.buf[idx]

        if isinstance(obj, dict) and "group" in obj:
            return copy.deepcopy(obj["group"])
        if isinstance(obj, list):
            return copy.deepcopy(obj)
        return [copy.deepcopy(obj)]
