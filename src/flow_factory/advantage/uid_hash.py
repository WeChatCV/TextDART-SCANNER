# Copyright 2026 Jayce-Ping
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

"""Deterministic group-key hashing for unique_ids.

The builtin ``hash()`` is randomized by ``PYTHONHASHSEED`` per process, so the
same ``unique_id`` produced different integer keys on different ranks.  In
multi-rank training (e.g. 4 nodes x 8 ranks = rank 0..31) this silently split a
single prompt group across several group keys and corrupted advantage
normalization.

This module provides a stable replacement: integer ids (the native type of
``BaseSample.unique_id``) are used directly; string/other ids are mapped with a
deterministic BLAKE2b digest instead of the randomized builtin hash.
"""

from __future__ import annotations

import hashlib

_UID_MOD = 2**31


def uid_to_group_key(uid) -> int:
    """Return a deterministic, cross-rank-stable integer group key for ``uid``.

    Args:
        uid: A ``unique_id``.  Native ``int`` ids (``BaseSample.unique_id``) are
            used directly (mod ``2**31``) for exact grouping.  Anything else is
            stringified and hashed with BLAKE2b (8-byte digest) for determinism.

    Returns:
        Non-negative integer in ``[0, 2**31)`` suitable for ``torch.unique``
        grouping, identical on every rank for the same input.
    """
    if isinstance(uid, int) and not isinstance(uid, bool):
        return uid % _UID_MOD
    digest = hashlib.blake2b(str(uid).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % _UID_MOD
