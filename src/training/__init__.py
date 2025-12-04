# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training package init; ensures runtime helpers stay importable in test stubs."""

import sys
from importlib import import_module


# Ensure runtime helpers remain importable even when the package is partially loaded
# in test environments that stub submodules.
try:  # pragma: no cover - import side effect only
    import_module("src.training.runtime")
except ImportError:  # pragma: no cover - optional dependencies may be missing
    # Avoid leaving a partially initialized module hanging around in sys.modules.
    sys.modules.pop("src.training.runtime", None)

# Expose training.utils so dotted monkeypatch paths resolve in tests.
try:  # pragma: no cover - import side effect only
    from . import utils  # noqa: F401
except ImportError:
    pass
