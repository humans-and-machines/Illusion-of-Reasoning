# open_r1/utils/replay_buffer.py
from __future__ import annotations
from typing import Any, List, Tuple, Optional, Dict
import copy
import threading
import math
import numpy as np
import torch.distributed as dist

def _prompt_key(prompt: list[dict[str, str]]) -> tuple:
    return tuple((m["role"], " ".join(m["content"].split())) for m in prompt)

def _is_full_example(x: Any) -> bool:
    return isinstance(x, dict) and "prompt" in x

def _finite_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _is_full_example(x: Any) -> bool:
    return isinstance(x, dict) and "prompt" in x and "answer" in x


class ReplayBuffer:
    """
    Bandit / UCB replay buffer with stable UIDs and optional exploit/explore mixing.

    * add_group(...) -> uid (>=0) on success, -1 on failure
    * sample_uid(...) returns a uid you can feed back into trainer._inject_one_replay_group
    * update_priority_by_uid updates μ / priority using a stable uid (safe for DDP)
    """

    def __init__(self, capacity: int = 4000, C: float = 1.0, debug_steps: int = 0):
        self.capacity = int(capacity)
        self.C = float(C)
        self.debug_steps = int(debug_steps)

        self._buf: List[Any] = []
        self._mean: List[float] = []
        self._M2: List[float] = []
        self._n: List[int] = []
        self._uids: List[int] = []
        self._uid2idx: Dict[int, int] = {}

        self._seen = set()         # prompt de-dup
        self._t = 0
        self._lock = threading.Lock()
        self._sample_calls = 0
        self._next_uid = 0

        # in __init__
        self._last_error: Optional[dict] = None

    def last_error(self):
        return self._last_error

    def _set_err(self, **kw):
        self._last_error = kw
        

    def __len__(self) -> int:
        return len(self._buf)

    # ----------------- stats utils -----------------
    def _init_stats(self, r: float) -> tuple[float, float, int]:
        return float(r), 0.0, 1

    def _update_stats(self, idx: int, r: float):
        mu, M2, n = self._mean[idx], self._M2[idx], self._n[idx]
        n += 1
        delta = r - mu
        mu += delta / n
        delta2 = r - mu
        M2 += delta * delta2
        self._mean[idx], self._M2[idx], self._n[idx] = mu, M2, n

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
            except Exception:
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
                self._set_err(where="add", why="dedup", capacity=self.capacity, len=len(self._buf))
                return False, -1
            self._seen.add(key)

            uid = self._next_uid
            self._next_uid += 1

            mu, M2, n = self._init_stats(reward)

            if len(self._buf) < self.capacity:
                self._buf.append(copy.deepcopy(sample))
                self._mean.append(mu)
                self._M2.append(M2)
                self._n.append(n)
                self._uids.append(uid)
                self._uid2idx[uid] = len(self._buf) - 1
                if self.debug_steps:
                    print(f"[RB][ADD] inserted uid={uid} μ={mu:.4f} size={len(self._buf)}")
                return True, uid

            if len(self._buf) >= self.capacity:
                worst = int(np.argmin(self._mean))
                if reward <= self._mean[worst]:
                    self._set_err(where="add", why="capacity_worse_mu",
                                cap=self.capacity, worst_mu=self._mean[worst], r=reward)
                    return False, -1

            # Full: maybe replace worst μ
            worst = int(np.argmin(self._mean))
            if reward > self._mean[worst]:
                old_uid = self._uids[worst]
                if old_uid in self._uid2idx:
                    del self._uid2idx[old_uid]
                self._buf[worst] = copy.deepcopy(sample)
                self._mean[worst] = mu
                self._M2[worst] = M2
                self._n[worst] = n
                self._uids[worst] = uid
                self._uid2idx[uid] = worst
                if self.debug_steps:
                    print(f"[RB][REPLACE] worst idx={worst} -> uid={uid} μ={mu:.4f}")
                return True, uid

            # Not inserted
            if self.debug_steps:
                print(f"[RB][SKIP] reward={reward:.4f} <= worst μ={self._mean[worst]:.4f}; skip")
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
            except Exception:
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
        reward = _finite_float(reward, 0.0)
        with self._lock:
            idx = self._uid2idx.get(uid, None)
            if idx is None:
                if self.debug_steps:
                    print(f"[RB][WARN] update_priority_by_uid: uid={uid} not found")
                return
            self._update_stats(idx, reward)

    # Legacy (index-based) — keep if you use it elsewhere
    def update_priority(self, idx: int, reward: float):
        with self._lock:
            if 0 <= idx < len(self._buf):
                self._update_stats(idx, _finite_float(reward, 0.0))

    def debug_state(self):
        with self._lock:
            tail = slice(max(0, len(self._buf) - 5), len(self._buf))
            return {
                "len": len(self._buf),
                "capacity": self.capacity,
                "next_uid": self._next_uid,
                "tail_uids": self._uids[tail],
                "tail_mu": [float(m) for m in self._mean[tail]],
                "tail_n": self._n[tail],
            }
            
    # ----------------- sampling -----------------
    def sample(
        self,
        batch_size: int = 1,
        *_, **__,
    ) -> Tuple[List[Any], List[int], List[int], np.ndarray]:
        """
        Uniformly sample `batch_size` distinct entries from the buffer.

        • Ignores μ, n, C, and mix_exploit_ratio.
        • Still returns (samples, idxs, uids, isw) so the rest of your
          training loop remains untouched.  `isw` stays all‑ones.
        """
        with self._lock:
            if not self._buf:
                raise ValueError("Empty replay buffer")

            n  = len(self._buf)
            bs = min(batch_size, n)

            idxs = np.random.choice(n, size=bs, replace=False).tolist()

            samples = [copy.deepcopy(self._buf[i]) for i in idxs]
            uids    = [self._uids[i]          for i in idxs]
            isw     = np.ones(bs, dtype=np.float32)       # importance‑sampling wts (unused)

            # Optional debug
            if self.debug_steps:
                print(f"[RB][SAMPLE] uniform idxs={idxs}")

            return samples, idxs, uids, isw

    def sample_uid(self, *_, **__) -> Optional[int]:
        """
        Convenience: return a single UID chosen uniformly at random.
        """
        print("SAMPLING!")
        with self._lock:
            if not self._buf:
                return None
            return np.random.choice(self._uids).item()

    def get_group(self, uid: int) -> List[dict[str, Any]]:
        with self._lock:
            idx = self._uid2idx.get(uid, None)
            if idx is None:
                return []
            obj = self._buf[idx]

        if isinstance(obj, dict) and "group" in obj:
            return copy.deepcopy(obj["group"])
        if isinstance(obj, list):
            return copy.deepcopy(obj)
        return [copy.deepcopy(obj)]