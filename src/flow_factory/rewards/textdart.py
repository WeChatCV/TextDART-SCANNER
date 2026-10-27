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

# src/flow_factory/rewards/textdart.py
"""TextDART reward: one OCR pass, two outputs.

A deployed TextDART server (OpenAI-compatible) locates and transcribes every
text region of a generated image in a single pass. Only the image is sent: the
model never sees the requested text. Its answer (``regions[{quad,
recognized_text}]`` in a `` ```json `` block, quads on a 0-1000 grid) is mapped
to image pixels and scored twice:

1. **Whole-image scalar** (``RewardModelOutput.rewards``, drives the scalar
   GDPO advantage): all region texts are merged into one string and scored
   with TextPecker's image-level recipe, ``w_sem * (1 - GNED vs the joined
   target_list) + w_qua * structural quality``.
2. **Per-target rewards** (``extra_info``, drive the spatial advantage):
   regions are matched many-to-many against ``metadata['target_list']`` and
   every requested text gets its own ``w_sem * sem + w_qua * qua`` reward;
   unrequested text goes to the ``"extra"`` bucket.

``extra_info`` keys consumed by the ``scanner`` trainer:

    - ``target_regions``: ``[{"box_2d" (and "quad_2d" when ``emit_quad_2d``),
      "coord_space", "image_size", "target_id", "region_qua", "region_sem",
      "text"}, ...]``
    - ``target_rewards``: ``{str(target_id): reward, "extra": reward}``. A
      target with no matched region is **omitted** (the trainer treats it as
      missing).
    - ``target_rewards_sem`` / ``target_rewards_qua``: the same keys, split into
      the semantic and quality components.
    - ``num_targets`` / ``target_lengths`` / ``target_list``
    - ``ocr_failed``: True when every OCR attempt returned an unusable answer
    - ``ocr_regions``: raw OCR regions, so the cross-rank group-union pass can
      rematch without calling the server again

Logged only: ``merged_text`` / ``merged_quality`` / ``merged_semantic`` and the
equal-weight region means ``region_whole`` (targets + extra), ``region_target``
(targets only) and ``region_complete`` (missing targets count as 0).

Main ``reward.extra_kwargs`` knobs:

    ``require_target_list`` (True)   raise when metadata has no ``target_list``;
                                     False -> single whole-string target
    ``target_w_sem`` / ``target_w_qua`` (0.5 / 0.5)  per-target reward weights
    ``component_weights`` ({semantic: 0.5, quality: 0.5})  whole-image weights
    ``emit_quad_2d`` (False)         keep the 4-point ``quad_2d`` (else AABB only)
    ``count_markers_by_weight`` (False)  per-target quality denominator counts each marker by
                                     its #-weight (``<#>``=1, ``<###>``=3,
                                     ``<#####>``=5) instead of literal length
    ``target_match_ned_threshold`` (0.5)  max NED for shared-content matches
    ``group_union_target_merge`` (True)   union each prompt group's cover-merge
                                     edges (across ranks) before scoring, so a
                                     multi-target quad in one rollout forces the
                                     same super-target on its siblings

"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils.operations import gather_object
from PIL import Image
from scipy.optimize import linear_sum_assignment

import TextDART

from flow_factory.hparams import RewardArguments
from flow_factory.rewards._http_clients import ServerPool, make_requests_session
from flow_factory.rewards.abc import PointwiseRewardModel, RewardModelOutput
from flow_factory.rewards.json_lenient import extract_json_payload as _extract_json_payload
from flow_factory.rewards.json_lenient import extract_regions_fallback, loads_json_lenient
from flow_factory.rewards.registry import register_reward_model
from flow_factory.rewards.textpecker import (
    _DEFAULT_TARGET_KEYS,
    _extract_targets_from_metadata,
    _image_to_b64,
    _run_async_task,
)
from eval.GenTextEval.scoring import (
    contain_chinese as _scoring_contain_chinese,
    matching_based_nled as _scoring_matching_based_nled,
    normalized_edit_distance as _scoring_normalized_edit_distance,
    pre_c as _scoring_pre_c,
    process_raw_text as _scoring_process_raw_text,

    split_chinese_or_mix as _textpecker_split_chinese_or_mix,
)
from flow_factory.utils.latent_coverage import quad_aabb_pixel


# ============================================================================
# Text scoring (TextPecker GNED + structural-marker quality)
# ============================================================================
def contain_chinese(prompts: str) -> bool:
    return _scoring_contain_chinese(prompts)


def normalized_edit_distance(s1: str, s2: str) -> float:
    return _scoring_normalized_edit_distance(s1, s2)


def split_chinese_or_mix(text_list: List[str]) -> List[str]:
    def is_chinese(char: str) -> bool:
        return "\u4e00" <= char <= "\u9fff"

    new_list: List[str] = []
    for word in text_list:
        if not contain_chinese(word):
            new_list.append(word)
            continue
        latin_segment = ""
        for char in word:
            if is_chinese(char):
                if latin_segment:
                    new_list.append(latin_segment)
                    latin_segment = ""
                new_list.append(char)
            else:
                latin_segment += char
        if latin_segment:
            new_list.append(latin_segment)
    return new_list


def matching_based_nled(gt_list: List[str], test_list: List[str]) -> float:
    return _scoring_matching_based_nled(gt_list, test_list)


def process_raw_text(raw_text: Union[str, List[str], None]) -> str:
    return _scoring_process_raw_text(raw_text)


def pre_c(string: str) -> str:
    return _scoring_pre_c(string)


def get_scaled_quality_score(
    rec_text: str,
    amplify_factor: float = 1.0,
    marker_weights: Optional[Dict[str, float]] = None,
    count_markers_by_weight: bool = False,
) -> float:
    """Quality score in [0, 1] from the density of structural-anomaly markers.

    TextPecker marks unreadable glyphs with ``<#>`` (a bad char), ``<###>`` (a bad
    English word) and ``<#####>`` (a bad text block); see TextDART/system_prompt.txt.
    Each marker reduces the quality proportionally to its weight. The default weights
    equal the number of ``#`` inside the token (``<#>``=1, ``<###>``=3, ``<#####>``=5),
    matching the system prompt's severity scale.

    Args:
        rec_text: recognised region text (markers are preserved here, NOT stripped).
        amplify_factor: penalty amplification for the anomaly ratio.
        marker_weights: optional ``{token: weight}`` override (e.g. ``{"<###>": 2.0}``).
        count_markers_by_weight: how the DENOMINATOR counts a marker token.
            ``False`` (default): the denominator is the
            literal character length after whitespace removal, so ``<#>`` counts
            as 3 characters, ``<###>`` as 5, ``<#####>`` as 7. ``True``: each
            marker counts as its ``#``-weight instead (``<#>``=1, ``<###>``=3,
            ``<#####>``=5), so the anomaly ratio is a true (weighted anomaly) /
            (weighted total) and a region that is one fully-unreadable block
            (``"<#####>"``) scores ``1 - 5/5 = 0``.
    """
    if marker_weights is None:
        marker_weights = {"<#>": 1.0, "<###>": 3.0, "<#####>": 5.0}
    # Weighted marker count (numerator). Count longest-first so a token is not
    # double-counted as a shorter one (``<#####>`` contains ``<###>``).
    n_a = 0.0
    remaining = rec_text
    for tok in sorted(marker_weights, key=len, reverse=True):
        w = float(marker_weights[tok])
        cnt = remaining.count(tok)
        n_a += cnt * w
        remaining = remaining.replace(tok, "")
    # Any bare '#' left over (not inside a recognised token) counts as weight 1.
    n_a += remaining.count("#")

    if count_markers_by_weight:
        # ``remaining`` is the text with every marker token stripped; drop bare
        # '#' and whitespace to get the readable-character count, then add the
        # weighted marker count so each marker contributes its #-weight, not its
        # literal length.
        readable_units = len(re.sub(r"\s+", "", remaining.replace("#", "")))
        n_p = readable_units + n_a
    else:
        n_p = len(pre_c(rec_text))
    if n_p == 0:
        return 0.0
    anomaly_ratio = n_a / n_p
    raw_score = 1 - (anomaly_ratio * amplify_factor)
    return max(0.0, min(raw_score, 1.0))


def _process_chinese(gt_chinese: List[str], test_chinese: List[str]) -> Dict[str, List[str]]:
    target_dict = Counter(gt_chinese)
    pred_dict = Counter(test_chinese)
    results: Dict[str, List[str]] = {
        "correct_text": [],
        "duplicate_text": [],
        "irrelevant_text": [],
        "typo_text": [],
        "missing_text": [],
    }
    for k, v in pred_dict.items():
        if k in target_dict:
            target_freq = target_dict[k]
            if v > target_freq:
                results["duplicate_text"].extend([k] * (v - target_freq))
                results["correct_text"].extend([k] * target_freq)
            elif v == target_freq:
                results["correct_text"].extend([k] * v)
            else:
                results["correct_text"].extend([k] * v)
                results["missing_text"].extend([k] * (target_freq - v))
            del target_dict[k]
        else:
            results["irrelevant_text"].extend([k] * v)
    for k, freq in target_dict.items():
        results["missing_text"].extend([k] * freq)
    return results


def _process_english(gt_english: List[str], test_english: List[str]) -> Dict[str, List[str]]:
    target_dict = Counter(gt_english)
    pred_dict = Counter(test_english)
    results: Dict[str, List[str]] = {
        "correct_text": [],
        "duplicate_text": [],
        "irrelevant_text": [],
        "typo_text": [],
        "missing_text": [],
    }
    for k, v in pred_dict.items():
        if k in target_dict:
            target_freq = target_dict[k]
            if v > target_freq:
                results["duplicate_text"].extend([k] * (v - target_freq))
                results["correct_text"].extend([k] * target_freq)
            elif v == target_freq:
                results["correct_text"].extend([k] * v)
            else:
                results["correct_text"].extend([k] * v)
                results["missing_text"].extend([k] * (target_freq - v))
            del target_dict[k]
        else:
            results["irrelevant_text"].extend([k] * v)
    for k, freq in target_dict.items():
        results["missing_text"].extend([k] * freq)

    target_candidates = [(w, idx) for idx, w in enumerate(results["missing_text"])]
    pred_candidates = [(w, idx) for idx, w in enumerate(results["irrelevant_text"])]
    len_gt, len_test = len(target_candidates), len(pred_candidates)
    if len_gt > 0 and len_test > 0:
        cost_matrix = np.zeros((len_test, len_gt))
        for i, (test_item, _) in enumerate(pred_candidates):
            for j, (gt_item, _) in enumerate(target_candidates):
                cost_matrix[i][j] = normalized_edit_distance(test_item, gt_item)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        typo_indices = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i][j] < 0.5:
                pred_idx = pred_candidates[i][1]
                target_idx = target_candidates[j][1]
                typo_indices.append((pred_idx, target_idx))
        if typo_indices:
            for idx in sorted([p for p, t in typo_indices], reverse=True):
                del results["irrelevant_text"][idx]
            for idx in sorted([t for p, t in typo_indices], reverse=True):
                del results["missing_text"][idx]
            results["typo_text"].extend(
                [
                    pred_candidates[i][0]
                    for i in row_ind
                    if cost_matrix[i, col_ind[row_ind == i][0]] < 0.5
                ]
            )
    return results


def classify_results(gt_list: List[str], test_list: List[str]) -> Dict[str, str]:
    gt_chinese = [w for w in gt_list if contain_chinese(w)]
    gt_english = [w for w in gt_list if not contain_chinese(w)]
    test_chinese = [w for w in test_list if contain_chinese(w)]
    test_english = [w for w in test_list if not contain_chinese(w)]
    chinese_results = _process_chinese(gt_chinese, test_chinese)
    english_results = _process_english(gt_english, test_english)
    final_results = {key: chinese_results[key] + english_results[key] for key in chinese_results}
    return {key: " ".join(value) for key, value in final_results.items()}


def get_gned_score(
    rec_text: str, ref_target: str, cls: bool = False
) -> Tuple[float, Dict[str, str]]:
    rec_text = re.sub(r"[<>]", "", rec_text)
    if "#" not in ref_target:
        pass
    elif "$" not in ref_target:
        rec_text = rec_text.replace("#", "$")
    elif "&" in ref_target:
        rec_text = rec_text.replace("#", "&")
    elif "￥" in ref_target:
        rec_text = rec_text.replace("#", "￥")

    rec_text_list = [_.lower() for _ in rec_text.split(" ")]
    ref_target_list = [_.lower() for _ in ref_target.split(" ")]
    # In particular, its trailing Latin segment in a mixed token is omitted.
    rec_text_list = _textpecker_split_chinese_or_mix(rec_text_list)
    ref_target_list = _textpecker_split_chinese_or_mix(ref_target_list)

    gned = matching_based_nled(ref_target_list, rec_text_list)
    cls_results: Dict[str, str] = {}
    if cls:
        cls_results = classify_results(ref_target_list, rec_text_list)
    return 1 - gned, cls_results


# ============================================================================
# Region -> target matching and per-target scoring
#
# Matching is many-to-many: one OCR quad may wrap several targets, and each
# region matches every target that clears a threshold:
#   1. cover     (target is a contiguous run of the region) -> merge edge
#   2. fragment  (region is a contiguous run of the target); only the shortest
#                such target is kept, so nested targets do not force a merge
#   3. shared    (shares a real unit and NED <= match_ned_threshold)
#   4. fuzzy     (NED <= fuzzy_ned_threshold)
#   5. orphans   re-attach to the nearest matched target sharing a fragment;
#                pure-marker orphans follow ``pure_marker_mode``
# Targets covered by one region are union-find merged and scored once against
# their joined strings; the reward is written under every member id. Across a
# prompt group (``score_regions_by_target_group``) the cover edges of all
# rollouts are unioned first, so every rollout scores the same super-targets.
# ============================================================================
# Single-unit wildcard sentinel used for the ``#`` structural-flaw marker so that
# A CJK character followed by "#" can match a longer CJK target fragment.
_WILD = "\x00"
# Characters TextPecker uses to denote anomalous / placeholder glyphs.
_MARKER_CHARS = set("#$&￥")

# TextPecker structural-anomaly marker tokens (see TextDART/system_prompt.txt):
#   <#>     : one structurally-flawed, unreadable char/letter     -> weight 1
#   <###>   : one structurally-flawed, unreadable English WORD   -> weight 3
#   <#####> : an entirely unreadable secondary text BLOCK        -> weight 5
_MARKER_TOKEN_RE = re.compile(r"<#+>")

_VALID_PURE_MARKER_MODES = frozenset({"extra", "attach", "attach_qua_only"})


def _strip_marker_tokens(text: str) -> str:
    """Remove ``<#>`` / ``<###>`` / ``<#####>`` marker tokens for matching only."""
    return _MARKER_TOKEN_RE.sub("", text or "")


def _match_units(text: str) -> List[str]:
    """Tokenise ``text`` into comparable units for matching.

    - CJK characters become individual units.
    - Runs of ascii alphanumerics become a single lowercased word unit.
    - Any TextPecker marker char (``#`` family) becomes a ``_WILD`` unit.
    - Everything else (punctuation/space/angle brackets) is a separator.
    """
    if not text:
        return []
    text = re.sub(r"[<>]", "", text)
    units: List[str] = []
    buf: List[str] = []

    def _flush() -> None:
        if buf:
            units.append("".join(buf).lower())
            buf.clear()

    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            _flush()
            units.append(ch)
        elif ch in _MARKER_CHARS:
            _flush()
            units.append(_WILD)
        elif ch.isalnum():
            buf.append(ch)
        else:
            _flush()
    _flush()
    return units


def _real_units(units: List[str]) -> List[str]:
    """Drop wildcard / empty units (the content actually recognised)."""
    return [u for u in units if u and u != _WILD]


def _is_wildcard_contiguous(region_units: List[str], target_units: List[str]) -> bool:
    """Return True when ``region_units`` is a contiguous run of ``target_units``.

    A ``_WILD`` unit in the region matches any single target unit. An all-wildcard
    region never qualifies (handled upstream via ``_real_units``).
    """
    region_units = [u for u in region_units if u]
    if not region_units:
        return False
    n, m = len(region_units), len(target_units)
    if n > m:
        return False
    for start in range(m - n + 1):
        ok = True
        for j in range(n):
            ru = region_units[j]
            if ru == _WILD:
                continue
            if ru != target_units[start + j]:
                ok = False
                break
        if ok:
            return True
    return False


def _ned(region_text: str, target: str) -> float:
    """Normalised edit distance (``1 - sem``) between a region and a target."""
    sem, _ = get_gned_score(process_raw_text(region_text), str(target))
    if not isinstance(sem, (int, float)):
        return 1.0
    return max(0.0, 1.0 - float(sem))


class _UnionFind:
    """Tiny union-find for target-id clustering."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def clusters(self, members: Iterable[int]) -> Dict[int, List[int]]:
        """Map representative -> sorted member ids (only ids in ``members``)."""
        out: Dict[int, List[int]] = {}
        for m in members:
            out.setdefault(self.find(m), []).append(int(m))
        for rep in out:
            out[rep].sort()
        return out


def match_region_to_targets(
    region_text: str,
    target_list: List[str],
    target_unit_lists: List[List[str]],
    *,
    match_ned_threshold: float = 0.5,
    fuzzy_ned_threshold: float = 0.4,
) -> Tuple[List[int], List[int]]:
    """Resolve one region to every matching ``target_id``.

    Cover / fragment (subset) checks tokenise **marker-stripped** text so
    ``<#>`` placeholders do not fake a cover hit. NED / fuzzy gates use the
    **raw** OCR string (markers kept) so unreadable slots still align with
    the target via GNED's ``#`` handling and lower NED.

    Returns:
        ``(match_ids, cover_ids)`` where ``match_ids`` is the full many-to-many
        assignment (cover + fragment + shared + fuzzy) and ``cover_ids`` are the
        targets this region *wraps* (``target ⊆ region``). Cover co-occurrence
        drives cross-target merge edges; fragment hits do not.
    """
    region_raw = region_text or ""
    region_stripped = _strip_marker_tokens(region_raw)
    region_units = _match_units(region_stripped)
    region_real = _real_units(region_units)
    if not region_real:
        return [], []

    cover: List[int] = []
    fragment: List[int] = []
    for k, tu in enumerate(target_unit_lists):
        tu_real = _real_units(tu)
        if not tu_real:
            continue
        # target ⊆ region  →  this quad wraps the whole target
        if _is_wildcard_contiguous(tu, region_units):
            cover.append(k)
        # region ⊆ target  →  this quad is a fragment of the target
        elif _is_wildcard_contiguous(region_units, tu):
            fragment.append(k)

    matches: Set[int] = set(cover)
    if fragment:
        # Most specific (shortest) fragment target; break ties by NED.
        best_frag = min(
            fragment,
            key=lambda k: (
                len(_real_units(target_unit_lists[k])),
                _ned(region_raw, target_list[k]),
            ),
        )
        matches.add(best_frag)

    region_set = set(region_real)
    if match_ned_threshold >= 0.0:
        for k, tu in enumerate(target_unit_lists):
            if k in matches:
                continue
            if not (region_set & set(_real_units(tu))):
                continue
            if _ned(region_raw, target_list[k]) <= match_ned_threshold:
                matches.add(k)

    if fuzzy_ned_threshold >= 0.0:
        for k in range(len(target_list)):
            if k in matches:
                continue
            if _ned(region_raw, target_list[k]) <= fuzzy_ned_threshold:
                matches.add(k)

    return sorted(matches), sorted(cover)


def _box_sort_key(region: Dict[str, Any]) -> Tuple[float, float]:
    """Reading-order key (top-to-bottom by line band, then left-to-right)."""
    box = region.get("bbox_2d") or region.get("box_2d") or [0.0, 0.0, 0.0, 0.0]
    x0 = float(box[0])
    y0 = float(box[1])
    return (round(y0 / 50.0), x0)


def _extra_char_count(text: str) -> int:
    """Count real (non-marker, non-space) characters of an over-generation crop."""
    clean = re.sub(r"[#<>]", "", process_raw_text(text or ""))
    return len(re.sub(r"\s+", "", clean))


def _center(region: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Bounding-box center, or ``None`` if no box is present."""
    box = region.get("bbox_2d") or region.get("box_2d")
    if not box or len(box) != 4:
        return None
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def _shares_fragment(stripped: str, target: str) -> bool:
    """True when a (possibly fragmentary) region text is plausibly part of ``target``."""
    stripped = (stripped or "").strip().lower()
    target = (target or "").strip().lower()
    if not stripped or not target:
        return False
    if len(stripped) >= 2 and (target.startswith(stripped[:2]) or target.endswith(stripped[-2:])):
        return True
    return stripped in target


def _nearest_assigned_region(
    orphan: Dict[str, Any],
    assigned_with_tid: List[Dict[str, Any]],
    target_list: List[str],
    allow_any: bool,
) -> Optional[Dict[str, Any]]:
    """Nearest (by box center) region that already has a ``target_id``."""
    oc = _center(orphan)
    stripped = (orphan.get("_stripped_text") or "").strip()
    best, best_d = None, float("inf")
    for r in assigned_with_tid:
        tid = int(r["target_id"])
        if not allow_any and not _shares_fragment(stripped, target_list[tid]):
            continue
        rc = _center(r)
        if oc is None or rc is None:
            d = float("inf")
        else:
            d = (oc[0] - rc[0]) ** 2 + (oc[1] - rc[1]) ** 2
        if d < best_d:
            best_d, best = d, r
    return best


def _region_text_is_subset(a: str, b: str) -> bool:
    """True when stripped units of ``a`` are a contiguous run inside ``b``."""
    a_u = _real_units(_match_units(_strip_marker_tokens(a)))
    b_u = _real_units(_match_units(_strip_marker_tokens(b)))
    if not a_u or not b_u or a_u == b_u:
        return False
    return _is_wildcard_contiguous(a_u, b_u)


def _dedupe_subset_regions(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop regions whose stripped text is a strict contiguous subset of another.

    Kept regions are still returned in reading order. Subset regions remain in
    the caller's full group for painting; only the concat list is filtered.
    """
    if len(regions) <= 1:
        return list(regions)
    ordered = sorted(regions, key=_box_sort_key)
    keep = [True] * len(ordered)
    for i, ri in enumerate(ordered):
        if not keep[i]:
            continue
        ti = ri.get("text", "") or ""
        for j, rj in enumerate(ordered):
            if i == j or not keep[j]:
                continue
            tj = rj.get("text", "") or ""
            if _region_text_is_subset(ti, tj):
                keep[i] = False
                break
    return [r for r, k in zip(ordered, keep) if k]


def _is_cjk_heavy(text: str) -> bool:
    """True when stripped text has no ascii alphanumerics (CJK / marker only)."""
    return not re.search(r"[A-Za-z0-9]", _strip_marker_tokens(text or ""))


def ocr_regions_from_painted(
    regions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rebuild raw OCR records from painted ``target_regions`` (skip synthetic).

    Used when a sample was scored without storing ``ocr_regions`` so a later
    group-union rescore can still rematch from the original boxes and text.
    """
    kept: List[Dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        if region.get("_synthetic_missing"):
            continue
        bbox = region.get("bbox_2d") or region.get("box_2d")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        text = region.get("text", "")
        if not isinstance(text, str):
            text = str(text or "")
        record: Dict[str, Any] = {
            "bbox_2d": [int(v) for v in bbox],
            "text": text,
        }
        quad = region.get("quad_2d")
        if isinstance(quad, (list, tuple)) and len(quad) == 8:
            record["quad_2d"] = [int(v) for v in quad]
        kept.append(record)
    return kept


def _join_texts(parts: Sequence[str]) -> str:
    """Join region/target strings; skip spaces when every part is CJK-heavy.

    Artificial spaces between CJK glyphs inflate char-level GNED against OCR
    CJK concatenations that never emitted spaces between adjacent targets.
    Latin / digit content keeps a space so the word aligner can split.
    """
    cleaned = [p for p in parts if p]
    if not cleaned:
        return ""
    if all(_is_cjk_heavy(p) for p in cleaned):
        return "".join(cleaned)
    return " ".join(cleaned)


def _join_cluster_target(target_list: Sequence[str], member_ids: Sequence[int]) -> str:
    """Join cluster member strings in ``target_list`` order."""
    return _join_texts([str(target_list[i]) for i in sorted(member_ids)])


def _assign_regions_many_to_many(
    regions: List[Dict[str, Any]],
    target_list: List[str],
    target_unit_lists: List[List[str]],
    *,
    match_ned_threshold: float,
    fuzzy_ned_threshold: float,
    pure_marker_mode: str,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    """Pass-1/2 matching. Returns enriched regions + cover-merge edges.

    Each non-extra region carries ``_match_ids`` (all matched targets). After
    orphan re-attachment a single ``target_id`` is also set (best / neighbour)
    for single-target access; scoring uses ``_match_ids``.
    """
    assigned: List[Dict[str, Any]] = []
    cover_edges: List[Tuple[int, int]] = []

    for region in regions:
        text = region.get("text", "")
        stripped = _strip_marker_tokens(text)
        match_ids, cover_ids = match_region_to_targets(
            text,
            target_list,
            target_unit_lists,
            match_ned_threshold=match_ned_threshold,
            fuzzy_ned_threshold=fuzzy_ned_threshold,
        )
        enriched = dict(region)
        enriched["_stripped_text"] = stripped
        enriched["_match_ids"] = list(match_ids)
        enriched["_cover_ids"] = list(cover_ids)
        if match_ids:
            enriched["target_id"] = int(min(match_ids, key=lambda k: _ned(text, target_list[k])))
        else:
            enriched["target_id"] = -1
        assigned.append(enriched)

        if len(cover_ids) >= 2:
            head = cover_ids[0]
            for other in cover_ids[1:]:
                cover_edges.append((head, other))

    # Pass 2: orphan re-attachment (same semantics as before).
    assigned_with_tid = [r for r in assigned if int(r["target_id"]) >= 0]
    for region in assigned:
        if int(region["target_id"]) != -1:
            continue
        stripped = region.get("_stripped_text", "")
        is_pure_marker = not stripped.strip()
        if is_pure_marker and pure_marker_mode == "extra":
            continue
        neighbor = _nearest_assigned_region(
            region, assigned_with_tid, target_list, allow_any=is_pure_marker
        )
        if neighbor is None:
            continue
        tid = int(neighbor["target_id"])
        region["target_id"] = tid
        region["_match_ids"] = [tid]
        if is_pure_marker and pure_marker_mode == "attach_qua_only":
            region["_exclude_from_sem"] = True

    return assigned, cover_edges


def _build_clusters(
    num_targets: int,
    edges: Sequence[Tuple[int, int]],
) -> Dict[int, List[int]]:
    """Union-find clusters over ``0..num_targets-1`` from merge edges.

    Keys are the representative ``min(members)`` (stable across rollouts).
    """
    uf = _UnionFind(num_targets)
    for a, b in edges:
        if 0 <= a < num_targets and 0 <= b < num_targets:
            uf.union(int(a), int(b))
    by_root = uf.clusters(range(num_targets))
    return {min(members): members for members in by_root.values()}


def _score_assigned_with_clusters(
    assigned: List[Dict[str, Any]],
    target_list: List[str],
    clusters: Dict[int, List[int]],
    *,
    w_sem: float,
    w_qua: float,
    qua_amplify: float,
    extra_penalty_factor: float,
    marker_weights: Optional[Dict[str, float]],
    count_markers_by_weight: bool,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Score one rollout under a fixed cluster partition; emit paint regions."""
    # member -> representative (min id)
    member_to_rep: Dict[int, int] = {}
    for members in clusters.values():
        rep = min(members)
        for m in members:
            member_to_rep[m] = rep

    # Collect regions per representative (from many-to-many match ids).
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    extras: List[Dict[str, Any]] = []
    for region in assigned:
        match_ids = [int(x) for x in region.get("_match_ids") or []]
        if not match_ids or all(m < 0 for m in match_ids):
            if int(region.get("target_id", -1)) < 0:
                extras.append(region)
            continue
        # Map every matched member to its cluster rep; a region that hit two
        # members of the same cluster is added once.
        reps = {member_to_rep[m] for m in match_ids if m in member_to_rep}
        for rep in reps:
            grouped.setdefault(rep, []).append(region)

    target_rewards: Dict[str, float] = {}
    paint_regions: List[Dict[str, Any]] = []

    for rep, group in grouped.items():
        members = list(clusters.get(rep, [rep]))
        members = sorted(members)
        rep = min(members)

        concat_regions = _dedupe_subset_regions(group)
        ordered = sorted(concat_regions, key=_box_sort_key)
        concat = _join_texts([r.get("text", "") for r in ordered if r.get("text")])
        concat_clean = process_raw_text(concat)
        qua = float(
            get_scaled_quality_score(
                concat_clean,
                qua_amplify,
                marker_weights=marker_weights,
                count_markers_by_weight=count_markers_by_weight,
            )
        )

        cluster_target = _join_cluster_target(target_list, members)
        sem_ordered = [r for r in ordered if not r.get("_exclude_from_sem")]
        sem_concat = _join_texts([r.get("text", "") for r in sem_ordered if r.get("text")])
        sem_concat_clean = process_raw_text(sem_concat)
        sem_raw, _ = get_gned_score(sem_concat_clean, cluster_target)
        sem = float(sem_raw) if isinstance(sem_raw, (int, float)) else 0.0
        reward = float(w_sem * sem + w_qua * qua)

        # Same reward under every member column so group z-score / completeness
        # sees the cluster as present on all members; paint uses the rep id.
        for m in members:
            target_rewards[str(m)] = reward

        # One paint entry per physical region (dedupe by identity), tagged with
        # the representative target_id and the cluster-level qua/sem.
        seen = set()
        for region in group:
            key = id(region)
            if key in seen:
                continue
            seen.add(key)
            out = dict(region)
            out["target_id"] = int(rep)
            out["region_qua"] = float(qua)
            out["region_sem"] = float(sem)
            out["cluster_target_ids"] = list(members)
            # Public match lists (one quad may match several targets).
            match_ids = [int(x) for x in region.get("_match_ids") or []]
            cover_ids = [int(x) for x in region.get("_cover_ids") or []]
            if match_ids:
                out["match_target_ids"] = match_ids
            if cover_ids:
                out["cover_target_ids"] = cover_ids
            paint_regions.append(out)

    if extras:
        ordered = sorted(extras, key=_box_sort_key)
        concat = _join_texts([r.get("text", "") for r in ordered if r.get("text")])
        concat_clean = process_raw_text(concat)
        qua = float(
            get_scaled_quality_score(
                concat_clean,
                qua_amplify,
                marker_weights=marker_weights,
                count_markers_by_weight=count_markers_by_weight,
            )
        )
        n_chars = _extra_char_count(concat)
        reward = qua - extra_penalty_factor * float(n_chars)
        reward = max(0.0, min(1.0, reward))
        target_rewards["extra"] = float(reward)
        for region in extras:
            out = dict(region)
            out["target_id"] = -1
            # Merged cluster quality. Per-region quality
            # is stored separately so extra_vs_absent can paint each box
            # independently without changing the extra bucket.
            out["region_qua"] = float(qua)
            own_text = process_raw_text(region.get("text", "") or "")
            out["region_qua_own"] = float(
                get_scaled_quality_score(
                    own_text,
                    qua_amplify,
                    marker_weights=marker_weights,
                    count_markers_by_weight=count_markers_by_weight,
                )
            )
            out["region_sem"] = 0.0
            paint_regions.append(out)

    # Drop temp fields from paint output.
    for region in paint_regions:
        region.pop("_stripped_text", None)
        region.pop("_exclude_from_sem", None)
        region.pop("_match_ids", None)
        region.pop("_cover_ids", None)

    return target_rewards, paint_regions


def split_target_component_rewards(
    target_rewards: Dict[str, float],
    assigned: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Split mixed per-target rewards into semantic and quality heads.

    Real-target keys share the cluster-level ``region_sem`` / ``region_qua``
    (identical on every paint region of that target, including members listed
    in ``cluster_target_ids``). The extra bucket has no semantic signal:
    ``sem=0`` and ``qua`` keeps the mixed extra reward (quality minus the
    extra-char penalty). Keys and presence match ``target_rewards`` so the
    trainer matrices stay aligned.
    """
    if not isinstance(target_rewards, dict):
        raise TypeError(
            "expected dict for target_rewards, got "
            f"{type(target_rewards).__name__}: {target_rewards!r}"
        )
    if not isinstance(assigned, list):
        raise TypeError(f"expected list for assigned regions, got {type(assigned).__name__}")
    id_sem: Dict[int, float] = {}
    id_qua: Dict[int, float] = {}
    for region in assigned:
        if not isinstance(region, dict):
            raise TypeError(
                "expected dict for assigned region, got " f"{type(region).__name__}: {region!r}"
            )
        tid = int(region.get("target_id", -1))
        if tid < 0:
            continue
        sem = float(region.get("region_sem", 0.0))
        qua = float(region.get("region_qua", 0.0))
        members = region.get("cluster_target_ids") or [tid]
        if not isinstance(members, (list, tuple)):
            raise TypeError(
                "expected list/tuple for cluster_target_ids, got "
                f"{type(members).__name__}: {members!r}"
            )
        for member in members:
            mid = int(member)
            id_sem[mid] = sem
            id_qua[mid] = qua
    sem_out: Dict[str, float] = {}
    qua_out: Dict[str, float] = {}
    for key, mixed in target_rewards.items():
        if key == "extra":
            sem_out["extra"] = 0.0
            qua_out["extra"] = float(mixed)
            continue
        ki = int(key)
        if ki not in id_sem:
            raise ValueError(
                f"target_rewards key {key!r} has no region_sem/region_qua "
                f"on assigned regions (known ids {sorted(id_sem)})"
            )
        sem_out[str(ki)] = float(id_sem[ki])
        qua_out[str(ki)] = float(id_qua[ki])
    return sem_out, qua_out


def score_regions_by_target(
    regions: List[Dict[str, Any]],
    target_list: List[str],
    *,
    w_sem: float = 0.5,
    w_qua: float = 0.5,
    qua_amplify: float = 1.0,
    extra_penalty_factor: float = 0.1,
    fuzzy_ned_threshold: float = 0.4,
    match_ned_threshold: float = 0.5,
    marker_weights: Optional[Dict[str, float]] = None,
    pure_marker_mode: str = "extra",
    count_markers_by_weight: bool = False,
    forced_edges: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], int]:
    """Match regions to targets, optionally merge clusters, then score each.

    Args:
        forced_edges: optional extra cover-merge edges (used by the group
            scorer to impose the cross-rollout union partition).

    Returns:
        ``(target_rewards, assigned_regions, num_targets)`` — same contract as
        before, with cluster-aware rewards / paint ids (see module docstring).
    """
    if not isinstance(target_list, list):
        raise TypeError(f"target_list must be a list of strings, got {type(target_list).__name__}")
    if pure_marker_mode not in _VALID_PURE_MARKER_MODES:
        raise ValueError(
            f"pure_marker_mode must be one of {sorted(_VALID_PURE_MARKER_MODES)}, "
            f"got {pure_marker_mode!r}"
        )
    target_list = [str(t) for t in target_list]

    target_unit_lists = [_match_units(t) for t in target_list]
    assigned, cover_edges = _assign_regions_many_to_many(
        regions,
        target_list,
        target_unit_lists,
        match_ned_threshold=match_ned_threshold,
        fuzzy_ned_threshold=fuzzy_ned_threshold,
        pure_marker_mode=pure_marker_mode,
    )
    edges = list(cover_edges)
    if forced_edges:
        edges.extend((int(a), int(b)) for a, b in forced_edges)
    clusters = _build_clusters(len(target_list), edges)

    target_rewards, paint = _score_assigned_with_clusters(
        assigned,
        target_list,
        clusters,
        w_sem=w_sem,
        w_qua=w_qua,
        qua_amplify=qua_amplify,
        extra_penalty_factor=extra_penalty_factor,
        marker_weights=marker_weights,
        count_markers_by_weight=count_markers_by_weight,
    )
    return target_rewards, paint, len(target_list)


def score_regions_by_target_group(
    regions_batch: Sequence[List[Dict[str, Any]]],
    target_list: List[str],
    *,
    w_sem: float = 0.5,
    w_qua: float = 0.5,
    qua_amplify: float = 1.0,
    extra_penalty_factor: float = 0.1,
    fuzzy_ned_threshold: float = 0.4,
    match_ned_threshold: float = 0.5,
    marker_weights: Optional[Dict[str, float]] = None,
    pure_marker_mode: str = "extra",
    count_markers_by_weight: bool = False,
) -> List[Tuple[Dict[str, float], List[Dict[str, Any]], int]]:
    """Score a rollout *group* under the union of every rollout's cover merges.

    All rollouts share one ``target_list``. Cover-merge edges from every
    rollout are union-find merged once; each rollout is then scored with that
    shared partition (so a glued ``t1+t2`` quad in one sample forces every
    sample to score the ``t1+t2`` super-target).
    """
    if not isinstance(target_list, list):
        raise TypeError(f"target_list must be a list of strings, got {type(target_list).__name__}")
    target_list = [str(t) for t in target_list]
    target_unit_lists = [_match_units(t) for t in target_list]

    per_rollout_assigned: List[List[Dict[str, Any]]] = []
    all_edges: List[Tuple[int, int]] = []
    for regions in regions_batch:
        assigned, cover_edges = _assign_regions_many_to_many(
            regions,
            target_list,
            target_unit_lists,
            match_ned_threshold=match_ned_threshold,
            fuzzy_ned_threshold=fuzzy_ned_threshold,
            pure_marker_mode=pure_marker_mode,
        )
        per_rollout_assigned.append(assigned)
        all_edges.extend(cover_edges)

    clusters = _build_clusters(len(target_list), all_edges)

    results: List[Tuple[Dict[str, float], List[Dict[str, Any]], int]] = []
    for assigned in per_rollout_assigned:
        target_rewards, paint = _score_assigned_with_clusters(
            assigned,
            target_list,
            clusters,
            w_sem=w_sem,
            w_qua=w_qua,
            qua_amplify=qua_amplify,
            extra_penalty_factor=extra_penalty_factor,
            marker_weights=marker_weights,
            count_markers_by_weight=count_markers_by_weight,
        )
        results.append((target_rewards, paint, len(target_list)))
    return results


# ============================================================================
# TextDART server answer parsing and the reward model
# ============================================================================
def _image_to_data_url(img) -> str:
    """Lossless PNG data-URL (JPEG would shift pixels and nudge localization)."""
    return f"data:image/png;base64,{_image_to_b64(img, image_format='PNG')}"


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Qwen3-VL patch_size(16) * spatial_merge_size(2) - the grid the served model
# actually tokenises on (ms-swift: `compat_qwen_vl_utils(image_patch_size=16)`),
# so `IMAGE_MAX_TOKEN_NUM=1024` means max_pixels = 1024 * 32 * 32.
_IMAGE_FACTOR = 32

# Side of the normalized coordinate grid TextDART answers use (0-1000, matching
# TextPecker-1.5M's bbox_2d prompts).
_NORM_SIDE = 1000

# Keys that identify a bare region object (a response that forgot the ``regions``
# wrapper). Used to salvage ``{"quad": ..., "recognized_text": ...}`` and even a
# quad-less ``{"recognized_text": ...}`` -- the latter then fails the downstream
# quad check with an accurate "region.quad" error (retried) instead of a
# misleading "no regions key" one.
_REGION_KEYS = ("quad", "quad_2d", "bbox_2d", "box_2d", "recognized_text", "text")

_DEFAULT_SYSTEM_PROMPT_PATH = Path(TextDART.__file__).resolve().with_name("system_prompt.txt")

_USER_TEMPLATE_PLAIN = """This is a text-generated image.
Locate and recognize all visible text in the entire image, and report every text region as a quadrilateral, following the marking and grounding rules.
Output in the following JSON format:
{
"regions": [{"quad": [x1, y1, x2, y2, x3, y3, x4, y4], "recognized_text": "text inside the quadrilateral (including structural error markers)"}]
}"""


def _smart_resize(
    height: int,
    width: int,
    max_pixels: int,
    min_pixels: int,
    factor: int = _IMAGE_FACTOR,
) -> Tuple[int, int]:
    """Qwen vision smart_resize: round both sides to ``factor`` under a pixel budget."""
    if height <= 0 or width <= 0:
        raise ValueError(f"expected positive image size, got height={height}, width={width}")
    if factor <= 0:
        raise ValueError(f"expected positive smart_resize factor, got {factor}")
    height_bar = round(height / factor) * factor
    width_bar = round(width / factor) * factor
    if height_bar * width_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        height_bar = math.floor(height / beta / factor) * factor
        width_bar = math.floor(width / beta / factor) * factor
    elif height_bar * width_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        height_bar = math.ceil(height * beta / factor) * factor
        width_bar = math.ceil(width * beta / factor) * factor
    return max(height_bar, factor), max(width_bar, factor)


def _rescale_quad(
    quad: Sequence[float], src_w: int, src_h: int, dst_w: int, dst_h: int
) -> List[int]:
    """Map an 8-number quad from a src_w x src_h frame into a dst_w x dst_h frame."""
    if len(quad) != 8:
        raise ValueError(f"expected 8 numbers per quad, got {len(quad)}: {quad!r}")
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"expected positive src size, got src_w={src_w}, src_h={src_h}")
    scale_x, scale_y = dst_w / src_w, dst_h / src_h
    return [
        round(value * (scale_x if index % 2 == 0 else scale_y)) for index, value in enumerate(quad)
    ]


def _coerce_regions(parsed: Any, body: str) -> List[Dict[str, Any]]:
    """Normalize a parsed TextDART payload into a list of region objects.

    Accepts every shape the served model realistically emits:
      - ``{"regions": [ {...}, ... ]}``    -- the documented shape
      - ``{"regions": {...}}``             -- a single region left unwrapped in a list
      - ``[ {...}, ... ]``                 -- the bare array (no ``regions`` key)
      - ``{"quad": ..., "recognized_text": ...}`` -- a single region, wrapper dropped

    A bare single-region object is wrapped rather than rejected so a valid
    wrapper-less region still scores, and a quad-less one fails later with an
    accurate ``region.quad`` error (retried) instead of here.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if "regions" in parsed:
            regions = parsed["regions"]
            if isinstance(regions, dict):
                return [regions]
            if isinstance(regions, list):
                return regions
            raise TypeError(
                f"expected a list or object for 'regions', got "
                f"{type(regions).__name__}: {regions!r}"
            )
        if any(key in parsed for key in _REGION_KEYS):
            return [parsed]
        raise ValueError(
            f"expected an object with a 'regions' key or a single region object, "
            f"got: {body[:300]!r}"
        )
    raise TypeError(f"expected a JSON object or list, got {type(parsed).__name__}: {body[:300]!r}")


def _parse_regions(raw_response: str) -> List[Dict[str, Any]]:
    """Pull the regions array out of a model answer, tolerating shape drift.

    Handles fenced or prose-wrapped JSON, a missing ``regions`` wrapper, a single
    region emitted unwrapped, and a bare region array (see :func:`_coerce_regions`).
    Also repairs common OCR JSON syntax failures (unescaped quotes, truncated
    braces) via :func:`loads_json_lenient`, then regex-salvages region objects.
    """
    body = _extract_json_payload(raw_response)
    try:
        parsed = loads_json_lenient(raw_response)
    except json.JSONDecodeError:
        salvaged = extract_regions_fallback(raw_response)
        if salvaged:
            return salvaged
        raise
    try:
        return _coerce_regions(parsed, body)
    except (TypeError, ValueError):
        salvaged = extract_regions_fallback(raw_response)
        if salvaged:
            return salvaged
        raise


def _quad_to_bbox_2d(quad_pixel: Sequence[float], image_size: Tuple[int, int]) -> List[int]:
    """Axis-aligned bbox of an 8-number pixel quad, in original-image pixels."""
    if len(quad_pixel) != 8:
        raise ValueError(f"expected 8-number quad, got len={len(quad_pixel)}: {quad_pixel!r}")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"expected positive image_size=(W, H), got {image_size!r}")
    xs = [float(quad_pixel[i]) for i in range(0, 8, 2)]
    ys = [float(quad_pixel[i]) for i in range(1, 8, 2)]
    x1 = max(0.0, min(min(xs), float(width)))
    y1 = max(0.0, min(min(ys), float(height)))
    x2 = max(0.0, min(max(xs), float(width)))
    y2 = max(0.0, min(max(ys), float(height)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [
        max(0, min(int(round(x1)), width)),
        max(0, min(int(round(y1)), height)),
        max(0, min(int(round(x2)), width)),
        max(0, min(int(round(y2)), height)),
    ]


def _ensure_scheme(url: str) -> str:
    """Prepend ``http://`` when a URL omits its scheme (bare ``host:port``)."""
    url = url.strip()
    if not url:
        return url
    if url.startswith(("http://", "https://")):
        return url
    return f"http://{url}"


def _normalize_url_list(
    urls: Union[str, Sequence[str], None],
    single: Union[str, None],
    field_name: str,
) -> List[str]:
    """Coerce a list/comma-string/single URL into a non-empty list of URLs."""
    if urls:
        if isinstance(urls, str):
            urls = [u.strip() for u in urls.split(",") if u.strip()]
        else:
            urls = [str(u).strip() for u in urls if str(u).strip()]
    elif single:
        urls = [single]
    else:
        raise ValueError(f"expected `{field_name}` (list or comma-separated) or its singular form.")
    urls = [_ensure_scheme(u) for u in urls if u]
    if not urls:
        raise ValueError(f"`{field_name}` resolved to an empty URL list.")
    return urls


def equal_weight_region_mean(target_rewards: Any, *, include_extra: bool = True) -> float:
    """Equal-weight mean of ``target_rewards`` values.

    Keys are ``str(target_id)`` plus optional ``"extra"`` (the unmatched-text
    0-1 score). ``include_extra=False`` averages target-id keys only. An empty
    selection returns ``0.0``.
    """
    if not isinstance(target_rewards, dict):
        raise TypeError(
            "expected dict for target_rewards, "
            f"got {type(target_rewards).__name__}: {target_rewards!r}"
        )
    region_scores: List[float] = []
    for key, value in target_rewards.items():
        if not include_extra and str(key) == "extra":
            continue
        if isinstance(value, torch.Tensor):
            region_scores.append(float(value.item()))
        else:
            region_scores.append(float(value))
    if not region_scores:
        return 0.0
    return float(sum(region_scores) / len(region_scores))


def complete_target_mean(target_rewards: Dict[str, float], num_targets: int) -> float:
    """Average all expected targets (missing = 0) and the extra bucket when present."""
    if num_targets < 0:
        raise ValueError(f"num_targets must be nonnegative, got {num_targets}")
    expected = {str(index) for index in range(num_targets)}
    unexpected = set(target_rewards) - expected - {"extra"}
    if unexpected:
        raise ValueError(f"target_rewards contains unexpected target IDs: {unexpected}")
    values = [float(target_rewards.get(str(index), 0.0)) for index in range(num_targets)]
    if "extra" in target_rewards:
        values.append(float(target_rewards["extra"]))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("target_rewards must contain finite scores")
    return sum(values) / len(values) if values else 0.0


def merged_text_from_regions(regions: Sequence[Any], text_joiner: str = " ") -> str:
    """Join non-empty recognized texts in the OCR list order."""
    texts: List[str] = []
    for region in regions:
        if not isinstance(region, dict):
            raise TypeError(f"expected dict OCR region, got {type(region).__name__}: {region!r}")
        text = str(region.get("text", region.get("recognized_text", "")) or "")
        if text:
            texts.append(text)
    return text_joiner.join(texts)


def score_merged_text(
    merged_text: str,
    target_list: Sequence[str],
    qua_amplify: float,
    marker_weights: Optional[Dict[str, float]] = None,
    count_markers_by_weight: bool = False,
) -> Tuple[float, float]:
    """``(quality, semantic)`` of a whole-image text, TextPecker's image-level recipe.

    ``semantic`` is ``1 - GNED`` against the space-joined ``target_list`` (0 when
    there is no target); ``quality`` comes from the structural-marker density.
    """
    rec_text = process_raw_text(merged_text)
    quality = float(
        get_scaled_quality_score(
            rec_text,
            amplify_factor=qua_amplify,
            marker_weights=marker_weights,
            count_markers_by_weight=count_markers_by_weight,
        )
    )
    if not target_list:
        return quality, 0.0
    gned, _ = get_gned_score(rec_text, " ".join(str(t) for t in target_list))
    return quality, float(gned)


def _read_text_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"TextDART system prompt not found at {path}. Set "
            f"`system_prompt_path` in the reward config's extra_kwargs."
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"TextDART system prompt file is empty: {path}")
    return text


# Marker stamped on the per-rank header gather_object concatenates with payloads.
# Filtered out before rematch; carries build errors / skip so every rank still
# enters the same collective (constraint #18).
_GROUP_UNION_META = "_textdart_group_union_meta"


def flatten_group_union_gather(raw: Any) -> List[Dict[str, Any]]:
    """Normalize ``gather_object`` output to a flat list of payload dicts.

    ``accelerate.utils.gather_object`` concatenates lists, so a per-rank
    ``List[dict]`` arrives as one flat ``List[dict]`` on every rank. Nested
    ``List[List[dict]]`` (from ``gather_object([local_payloads])``) is also
    accepted. Do **not** wrap a flat dict list as ``[flat]`` - that makes
    rank 0 own every payload and every other rank see only one "rank".
    """
    if not isinstance(raw, list):
        raise TypeError(
            "expected list from gather_object for TextDART group-union "
            f"payloads, got {type(raw).__name__}"
        )
    if not raw:
        return []
    if isinstance(raw[0], dict):
        return raw
    if isinstance(raw[0], list):
        flat: List[Dict[str, Any]] = []
        for rank_list in raw:
            if not isinstance(rank_list, list):
                raise TypeError(
                    "expected per-rank list of TextDART group-union payloads, "
                    f"got {type(rank_list).__name__}"
                )
            for item in rank_list:
                if not isinstance(item, dict):
                    raise TypeError(
                        "expected dict TextDART group-union payload, " f"got {type(item).__name__}"
                    )
                flat.append(item)
        return flat
    raise TypeError(
        "expected gather_object to return List[dict] or List[List[dict]], "
        f"got list of {type(raw[0]).__name__}"
    )


def local_gathered_indices(
    gathered: List[Dict[str, Any]],
    rank: int,
    n_local: int,
) -> List[int]:
    """Map local sample ``i`` to its index in ``gathered`` via payload tags."""
    if not isinstance(n_local, int) or n_local < 0:
        raise TypeError(
            f"expected non-negative int for n_local, got {type(n_local).__name__}: " f"{n_local!r}"
        )
    mapped = [-1] * n_local
    for gi, payload in enumerate(gathered):
        if not isinstance(payload, dict):
            raise TypeError(
                "expected dict TextDART group-union payload, " f"got {type(payload).__name__}"
            )
        if "rank" not in payload or "local_i" not in payload:
            raise KeyError(
                "TextDART group-union payload missing rank/local_i tags: " f"keys={sorted(payload)}"
            )
        if int(payload["rank"]) != rank:
            continue
        local_i = int(payload["local_i"])
        if local_i < 0 or local_i >= n_local:
            raise RuntimeError(
                f"TextDART group-union payload local_i={local_i} out of range "
                f"for n_local={n_local} on rank={rank}"
            )
        if mapped[local_i] != -1:
            raise RuntimeError(
                f"duplicate TextDART group-union payload for rank={rank} " f"local_i={local_i}"
            )
        mapped[local_i] = gi
    missing = [i for i, gi in enumerate(mapped) if gi < 0]
    if missing:
        raise RuntimeError(
            "TextDART group-union missing gathered payloads for "
            f"rank={rank} local indices {missing} "
            f"(gathered={len(gathered)}, n_local={n_local})"
        )
    return mapped


def split_group_union_gather(
    gathered: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split gather_object output into per-rank headers and sample payloads."""
    if not isinstance(gathered, list):
        raise TypeError(
            "expected list of TextDART group-union dicts, " f"got {type(gathered).__name__}"
        )
    metas: List[Dict[str, Any]] = []
    payloads: List[Dict[str, Any]] = []
    for item in gathered:
        if not isinstance(item, dict):
            raise TypeError(
                "expected dict TextDART group-union item, " f"got {type(item).__name__}"
            )
        if item.get(_GROUP_UNION_META):
            metas.append(item)
        else:
            payloads.append(item)
    return metas, payloads


def raise_if_group_union_world_failed(metas: List[Dict[str, Any]]) -> None:
    """Raise when any rank failed payload build or skip_reason disagrees.

    ``metas`` may be empty (unit tests that mock a payload-only gather).
    """
    if not metas:
        return
    errors = [meta for meta in metas if meta.get("error")]
    if errors:
        parts = [f"rank={meta.get('rank')}:\n{meta['error']}" for meta in errors]
        raise RuntimeError(
            "TextDART group-union payload build failed on "
            f"{len(errors)} rank(s):\n" + "\n".join(parts)
        )
    reasons = {meta.get("skip_reason") for meta in metas}
    if len(reasons) > 1:
        raise RuntimeError(
            "TextDART group-union skip_reason disagrees across ranks: "
            f"{sorted(repr(reason) for reason in reasons)}"
        )


def resolve_sample_image_wh(sample: Any, *, owner: str) -> Tuple[int, int]:
    """``(W, H)`` of the frame painted regions must land in.

    Shared by every group-union rescorer. ``BaseSample.image`` is canonicalized
    to a ``(C, H, W)`` tensor, so during training it is the tensor / width /
    ``target_regions`` branches that fire - the PIL branch only exists for CLI
    callers that hand in a raw image. Keep this in one place: a partial copy on
    one reward silently raises on every real rollout.
    """
    image = getattr(sample, "image", None)
    if isinstance(image, Image.Image):
        return int(image.size[0]), int(image.size[1])
    if isinstance(image, torch.Tensor) and image.ndim >= 2:
        return int(image.shape[-1]), int(image.shape[-2])
    width = getattr(sample, "width", None)
    height = getattr(sample, "height", None)
    if width and height:
        return int(width), int(height)
    extra = getattr(sample, "extra_kwargs", {}) or {}
    extra_size = extra.get("image_size")
    if isinstance(extra_size, (list, tuple)) and len(extra_size) == 2:
        return int(extra_size[0]), int(extra_size[1])
    for region in extra.get("target_regions") or []:
        if not isinstance(region, dict):
            continue
        size = region.get("image_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return int(size[0]), int(size[1])
    raise ValueError(
        f"cannot resolve image (W, H) for {owner} group-union rescore "
        f"(unique_id={getattr(sample, 'unique_id', None)!r})"
    )


@register_reward_model("textdart")
class TextDARTRewardModel(PointwiseRewardModel):
    """TextDART reward: a whole-image scalar plus per-target regions and rewards."""

    required_fields: Tuple[str, ...] = ("prompt", "image")
    use_tensor_inputs: bool = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config=config, accelerator=accelerator)
        ek = config.extra_kwargs

        try:
            server_urls = _normalize_url_list(
                ek.get("server_urls") or ek.get("textdart_server_urls"),
                ek.get("server_url") or ek.get("textdart_server_url"),
                "server_urls",
            )
        except ValueError as exc:
            raise ValueError(
                "TextDARTRewardModel requires `server_urls` (list or "
                "comma-separated) or `server_url` pointing at a deployed "
                "TextDART OpenAI-compatible endpoint."
            ) from exc

        self.w_sem = float(ek.get("target_w_sem", 0.5))
        self.w_qua = float(ek.get("target_w_qua", 0.5))
        self.extra_penalty_factor = float(ek.get("target_extra_penalty_factor", 0.1))
        self.qua_amplify_factor = float(ek.get("qua_amplify_factor", 5.0))
        # How a pure ``<#>``/``<###>``/``<#####>`` region (no readable content) is
        # scored: ``extra`` (default, own -1 bucket, qua-only, localized box),
        # ``attach`` (drags nearest target's qua+sem), or
        # ``attach_qua_only`` (drags qua only, kept out of the sem concat).
        self.pure_marker_mode = str(ek.get("pure_marker_mode", "extra"))
        _valid_pmm = {"extra", "attach", "attach_qua_only"}
        if self.pure_marker_mode not in _valid_pmm:
            raise ValueError(
                f"pure_marker_mode must be one of {sorted(_valid_pmm)}, got "
                f"{self.pure_marker_mode!r}"
            )
        self.drop_empty_regions = bool(ek.get("drop_empty_regions", True))
        # Keep TextDART's 4-point ``quad_2d`` (else emit AABB ``box_2d`` only).
        self.emit_quad_2d = bool(ek.get("emit_quad_2d", False))
        # The training data MUST ship an explicit ``target_list`` per sample; the
        # reward matches regions against THAT list only (no prompt derivation).
        # ``require_target_list`` (default True) turns a missing list into a hard
        # error; set it False only to fall back to a single whole-string target.
        self.require_target_list = bool(ek.get("require_target_list", True))
        # GNED threshold for the fuzzy near-miss fallback in region->target
        # matching: a region that shares no exact unit but is within this GNED of
        # a target (e.g. an OCR typo) is still assigned to it instead of becoming
        # over-generation. -1 disables the fallback.
        self.match_fuzzy_ned = float(ek.get("target_match_fuzzy_ned", 0.4))
        # Max NED for the many-to-many shared-content match (cover/fragment hits
        # ignore this). A region that shares a unit with a target is kept only
        # when NED <= this value, so a lone shared CJK char does not attach the
        # region to every target that happens to contain it. -1 disables.
        self.match_ned_threshold = float(ek.get("target_match_ned_threshold", 0.5))
        # When True (default) and ``unique_id`` is provided, rollouts that share
        # a prompt group are scored under the *union* of every rollout's
        # cover-merge edges (a quad wrapping t1+t2 in one sample forces every
        # sample to score the t1+t2 super-target). CLI / single-sample calls
        # without unique_id keep per-sample clustering only.
        self.group_union_target_merge = bool(ek.get("group_union_target_merge", True))
        # Per-marker structural-penalty weights for the quality score. ``None``
        # keeps ``<#>``=1, ``<###>``=3, ``<#####>``=5 (TextDART/system_prompt.txt).
        self.marker_weights = ek.get("target_marker_weights", None)
        # Marker denominator counting for the per-target quality score. ``False``
        # (default) uses the literal char-length denominator (``<#>``=3 chars).
        # ``True`` counts each marker by its #-weight (``<#>``=1, ``<###>``=3,
        # ``<#####>``=5), so a fully-unreadable ``<#####>`` merge scores qua 0.
        self.count_markers_by_weight = bool(ek.get("count_markers_by_weight", False))
        self.max_empty_retries = int(ek.get("max_empty_retries", 10))
        if self.max_empty_retries < 0:
            raise ValueError(f"max_empty_retries must be >= 0, got {self.max_empty_retries}")

        self.target_keys = tuple(ek.get("target_keys", _DEFAULT_TARGET_KEYS))
        self.prompt_target_joiner = str(ek.get("prompt_target_joiner", " "))

        image_max_token_num = int(ek.get("image_max_token_num", 1024))
        image_min_token_num = int(ek.get("image_min_token_num", 4))
        if image_max_token_num <= 0:
            raise ValueError(f"image_max_token_num must be > 0, got {image_max_token_num}")
        if image_min_token_num <= 0:
            raise ValueError(f"image_min_token_num must be > 0, got {image_min_token_num}")
        self.max_pixels = image_max_token_num * _IMAGE_FACTOR * _IMAGE_FACTOR
        self.min_pixels = image_min_token_num * _IMAGE_FACTOR * _IMAGE_FACTOR

        # GRPO rollouts (512x512) sit far below the smart_resize budget, so the
        # server rounds them to themselves and TextDART sees 256 of the 1024
        # tokens it was trained with. Either knob below upscales such an image
        # before upload (aspect ratio preserved); images already at or above the
        # target are sent untouched, and quads always come back in the uploaded
        # frame for `_regions_from_raw` to map to original image pixels.
        #
        # ``ocr_input_token_num``  -- target is a VISUAL TOKEN budget. The upload
        #   is snapped to the 32px grid the server would pick anyway, so its
        #   smart_resize is a no-op and the pixels are interpolated exactly once.
        #   1024 tokens == 1024x1024 for a square image.
        # ``ocr_input_resolution`` -- target is a PIXEL area, expressed as the
        #   side of the equivalent square (1024 -> 1024*1024 px, the 1024P
        #   bucket convention). Exact aspect ratio, no grid snapping. Above the
        #   server's own budget this costs one extra server-side resample.
        ocr_token_num = ek.get("ocr_input_token_num")
        ocr_resolution = ek.get("ocr_input_resolution")
        if ocr_token_num is not None and ocr_resolution is not None:
            raise ValueError(
                "ocr_input_token_num and ocr_input_resolution are two ways to "
                "set the same OCR input target; pass exactly one. Got "
                f"ocr_input_token_num={ocr_token_num!r}, "
                f"ocr_input_resolution={ocr_resolution!r}."
            )
        self.ocr_input_pixels: Optional[int] = None
        # Token targets land on the server's own frame; pixel targets do not.
        self.ocr_input_snap_to_frame = ocr_token_num is not None
        if ocr_token_num is not None:
            ocr_token_num = int(ocr_token_num)
            if ocr_token_num <= 0:
                raise ValueError(f"ocr_input_token_num must be > 0 when set, got {ocr_token_num}")
            self.ocr_input_pixels = ocr_token_num * _IMAGE_FACTOR * _IMAGE_FACTOR
        elif ocr_resolution is not None:
            ocr_resolution = int(ocr_resolution)
            if ocr_resolution <= 0:
                raise ValueError(f"ocr_input_resolution must be > 0 when set, got {ocr_resolution}")
            self.ocr_input_pixels = ocr_resolution * ocr_resolution

        api_key = str(ek.get("api_key", "EMPTY"))
        timeout = float(getattr(config, "timeout", 180.0))
        self.retries = int(getattr(config, "retry_attempts", 10))
        self.max_tokens = int(ek.get("max_tokens", 2048))
        self.repetition_penalty = float(ek.get("repetition_penalty", 1.0))
        use_env_proxies = bool(ek.get("use_env_proxies", False))
        self.check_health = bool(ek.get("check_health", True))
        self.failure_cooldown = float(ek.get("server_failure_cooldown", 60.0))

        self.system_prompt = _read_text_file(
            Path(ek.get("system_prompt_path") or _DEFAULT_SYSTEM_PROMPT_PATH)
        )

        # Whole-image scalar: weights of the merged-text semantic / quality scores.
        self.component_weights: Dict[str, float] = dict(
            ek.get("component_weights", {"semantic": 0.5, "quality": 0.5})
        )
        self.normalize_component_weights = bool(ek.get("normalize_component_weights", True))
        self.text_joiner = str(ek.get("text_joiner", " "))

        self._session = make_requests_session(use_env_proxies=use_env_proxies)
        # Unreachable servers are logged and dropped here rather than aborting
        # the run; runtime failures fail over inside `chat_with_failover`.
        self.server_pool = ServerPool(
            urls=server_urls,
            api_key=api_key,
            timeout=timeout,
            use_env_proxies=use_env_proxies,
            model_name=ek.get("model_name") or ek.get("textdart_model_name"),
            session=self._session,
            probe_health=self.check_health,
            failure_cooldown=self.failure_cooldown,
            pool_name="TextDART server",
        )

        logger.info(
            "[rank %s] TextDART %s ready: servers=%s (model=%s), "
            "w_sem=%s, w_qua=%s, extra_penalty_factor=%s, pure_marker_mode=%s, "
            "count_markers_by_weight=%s, "
            "require_target_list=%s (target_list must come from metadata), emit_quad_2d=%s, "
            "image_max_token_num=%s, ocr_input_token_num=%s, ocr_input_resolution=%s",
            getattr(accelerator, "process_index", 0),
            type(self).__name__,
            self.server_pool.base_urls,
            self.server_pool.model_name,
            self.w_sem,
            self.w_qua,
            self.extra_penalty_factor,
            self.pure_marker_mode,
            self.count_markers_by_weight,
            self.require_target_list,
            self.emit_quad_2d,
            image_max_token_num,
            ocr_token_num,
            ocr_resolution,
        )

    def _build_messages(self, data_url: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": _USER_TEMPLATE_PLAIN},
                ],
            },
        ]

    def _prepare_ocr_image(self, image: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
        """Return ``(image_to_upload, its (W, H))`` for one OCR call.

        Upscale-only: an image already at or above the target (see
        ``ocr_input_token_num`` / ``ocr_input_resolution`` in ``__init__``) is
        sent untouched, since the server's own smart_resize handles any
        downscale and re-encoding here would only add a resample. A smaller one
        is LANCZOS-enlarged, aspect ratio preserved, so TextDART sees the
        resolution it was trained at instead of a quarter of it.
        """
        if self.ocr_input_pixels is None:
            return image, image.size
        width, height = image.size
        if width * height >= self.ocr_input_pixels:
            return image, (width, height)
        if self.ocr_input_snap_to_frame:
            # Budget as BOTH bounds: smart_resize only grows an image that is
            # under min_pixels, so passing it as max_pixels alone would leave a
            # 512x512 rollout where it is instead of lifting it to 1024x1024.
            new_h, new_w = _smart_resize(
                height, width, self.ocr_input_pixels, self.ocr_input_pixels
            )
            # Rounding each side up to 32 can overshoot the budget on non-square
            # images; clamp to the frame the server itself would pick so its own
            # smart_resize is a no-op and pixels are interpolated exactly once.
            new_h, new_w = _smart_resize(new_h, new_w, self.max_pixels, self.min_pixels)
        else:
            # Pixel target: hit the area exactly, aspect ratio untouched.
            scale = math.sqrt(self.ocr_input_pixels / float(width * height))
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
        if (new_w, new_h) == (width, height):
            return image, (width, height)
        return (
            image.resize((new_w, new_h), Image.Resampling.LANCZOS),
            (new_w, new_h),
        )

    async def _chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> str:
        """One TextDART completion, failing over between pool servers."""
        return await self.server_pool.chat_with_failover(
            messages,
            retries=self.retries,
            temperature=temperature,
            max_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            log_prefix="TextDART chat",
        )

    def _regions_from_raw(
        self,
        raw: str,
        image_size: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Parse TextDART JSON and convert regions to pixel-space records.

        Each kept region is ``{bbox_2d, text}`` (AABB) plus ``quad_2d`` when
        ``self.emit_quad_2d`` (the 4-point quad in original-image pixels).

        ``image_size`` is the ORIGINAL ``(W, H)`` the returned coordinates must
        land in. Quads are rescaled from the 0-1000 normalized grid, so the
        uploaded resolution does not matter.
        """
        width, height = image_size
        resized_w, resized_h = _NORM_SIDE, _NORM_SIDE
        parsed_regions = _parse_regions(raw)
        kept: List[Dict[str, Any]] = []
        for region in parsed_regions:
            if not isinstance(region, dict):
                raise TypeError(
                    f"TextDART region must be a dict, got {type(region).__name__}: " f"{region!r}"
                )
            quad = region.get("quad")
            if not isinstance(quad, (list, tuple)) or len(quad) != 8:
                raise ValueError(f"TextDART region.quad must be an 8-number list, got {quad!r}")
            text = region.get("recognized_text", region.get("text", ""))
            if not isinstance(text, str):
                raise TypeError(
                    f"TextDART recognized_text must be str, got " f"{type(text).__name__}: {text!r}"
                )
            if self.drop_empty_regions and not text.strip():
                continue
            quad_pixel = _rescale_quad(
                [float(v) for v in quad], resized_w, resized_h, width, height
            )
            bbox_2d = _quad_to_bbox_2d(quad_pixel, image_size)
            if bbox_2d[2] <= bbox_2d[0] or bbox_2d[3] <= bbox_2d[1]:
                continue
            record: Dict[str, Any] = {"bbox_2d": bbox_2d, "text": text}
            if self.emit_quad_2d:
                record["quad_2d"] = quad_pixel
            kept.append(record)
        return kept

    async def _collect_regions_one(
        self,
        image: Image.Image,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Call TextDART once (with empty-answer retries).

        Returns ``(kept_regions, ocr_failed)`` where ``ocr_failed`` is True when
        EVERY attempt raised a parse / missing-quad error (so ``kept`` is empty
        because the server's answer was unusable, NOT because the image genuinely
        had no text). The trainer drops the whole prompt group in that case so a
        bogus all-empty rollout does not contaminate the group's advantage.
        """
        ocr_image, _sent_size = self._prepare_ocr_image(image)
        data_url = _image_to_data_url(ocr_image)
        image_size = image.size  # (W, H): the frame kept regions are reported in
        messages = self._build_messages(data_url)
        kept: List[Dict[str, Any]] = []
        last_attempt_failed = False
        for attempt in range(self.max_empty_retries + 1):
            temperature = 0.0 if attempt == 0 else min(0.4 + 0.15 * (attempt - 1), 1.0)
            raw = await self._chat(messages, temperature=temperature)
            try:
                kept = self._regions_from_raw(raw, image_size)
                last_attempt_failed = False
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                # Malformed model JSON is a quality signal: retry, then fall
                # through to empty regions (same as empty grounding).
                logger.warning(
                    "TextDART parse attempt %s/%s failed: %s",
                    attempt + 1,
                    self.max_empty_retries + 1,
                    exc,
                )
                kept = []
                last_attempt_failed = True
            if kept:
                return kept, False
        # No regions parsed. If the last attempt raised, the answer was unusable
        # (parse / missing-quad failure) -> flag the sample for group drop; if it
        # parsed cleanly but was empty, the image legitimately had no text.
        return kept, last_attempt_failed

    async def _collect_regions_batch(
        self,
        images: List[Image.Image],
    ) -> List[Tuple[List[Dict[str, Any]], bool]]:
        return await asyncio.gather(*[self._collect_regions_one(img) for img in images])

    def _region_out_dict(
        self, region: Dict[str, Any], img_w: int, img_h: int
    ) -> Optional[Dict[str, Any]]:
        """Build one ``target_regions`` entry, in pixel space, for the trainer.

        Emits ``quad_2d`` + ``box_2d`` when ``self.emit_quad_2d`` (falling back
        to a rectangle quad / AABB when only one is present), else ``box_2d``
        only. ``coord_space='pixel'`` and ``image_size`` tell the quad/AABB
        trainer how to map back onto the latent grid.
        """
        bbox_2d = region.get("bbox_2d")
        quad_2d = region.get("quad_2d")
        base = {
            "coord_space": "pixel",
            "image_size": [int(img_w), int(img_h)],
            "target_id": int(region["target_id"]),
            "region_qua": float(region.get("region_qua", 0.0)),
            "region_sem": float(region.get("region_sem", 0.0)),
            "text": region.get("text", ""),
        }
        # Optional many-to-many match lists (logging only; the trainer reads
        # target_id).
        for key in ("cluster_target_ids", "match_target_ids", "cover_target_ids"):
            value = region.get(key)
            if isinstance(value, (list, tuple)) and value:
                base[key] = [int(v) for v in value]
        if self.emit_quad_2d:
            if not (isinstance(quad_2d, (list, tuple)) and len(quad_2d) == 8):
                # No quad: synthesize one from the AABB (should not happen).
                if not (isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) == 4):
                    return None
                quad_2d = [
                    bbox_2d[0],
                    bbox_2d[1],
                    bbox_2d[2],
                    bbox_2d[1],
                    bbox_2d[2],
                    bbox_2d[3],
                    bbox_2d[0],
                    bbox_2d[3],
                ]
            if not (isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) == 4):
                bbox_2d = quad_aabb_pixel(quad_2d, (img_w, img_h))
            base["quad_2d"] = [int(c) for c in quad_2d]
            base["box_2d"] = [int(c) for c in bbox_2d]
            return base
        if not (isinstance(bbox_2d, (list, tuple)) and len(bbox_2d) == 4):
            return None
        base["box_2d"] = [int(c) for c in bbox_2d]
        return base

    def _score_kwargs(self) -> Dict[str, Any]:
        """Shared kwargs for ``score_regions_by_target`` / ``_group``."""
        return dict(
            w_sem=self.w_sem,
            w_qua=self.w_qua,
            qua_amplify=self.qua_amplify_factor,
            extra_penalty_factor=self.extra_penalty_factor,
            fuzzy_ned_threshold=self.match_fuzzy_ned,
            match_ned_threshold=self.match_ned_threshold,
            marker_weights=self.marker_weights,
            pure_marker_mode=self.pure_marker_mode,
            count_markers_by_weight=self.count_markers_by_weight,
        )

    def _image_scalar(self, regions: Sequence[Any], target_list: Sequence[str]) -> Dict[str, Any]:
        """Whole-image reward: TextPecker-style score of all region texts merged."""
        merged = merged_text_from_regions(regions, self.text_joiner)
        quality, semantic = score_merged_text(
            merged,
            target_list,
            self.qua_amplify_factor,
        )
        weights = dict(self.component_weights)
        if self.normalize_component_weights:
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
        scalar = weights.get("semantic", 0.0) * semantic + weights.get("quality", 0.0) * quality
        return {
            "scalar": float(scalar),
            "merged_quality": quality,
            "merged_semantic": semantic,
            "merged_text": merged,
        }

    @staticmethod
    def _sample_image_wh(sample: Any) -> Tuple[int, int]:
        return resolve_sample_image_wh(sample, owner="TextDART")

    def _emit_scored_extra(
        self,
        target_rewards: Dict[str, float],
        assigned: List[Dict[str, Any]],
        num_targets: int,
        img_w: int,
        img_h: int,
        target_list: Sequence[str],
        ocr_failed: bool,
    ) -> Dict[str, Any]:
        """Paint dicts, split rewards and region means for one scored sample."""
        regions_for_paint: List[Dict[str, Any]] = []
        for region in assigned:
            out = self._region_out_dict(region, img_w, img_h)
            if out is not None:
                regions_for_paint.append(out)
        mixed_rewards = {str(k): float(v) for k, v in target_rewards.items()}
        sem_rewards, qua_rewards = split_target_component_rewards(mixed_rewards, assigned)
        return {
            "target_regions": regions_for_paint,
            "target_rewards": mixed_rewards,
            "target_rewards_sem": sem_rewards,
            "target_rewards_qua": qua_rewards,
            "num_targets": int(num_targets),
            "target_lengths": [len(str(t)) for t in target_list],
            "ocr_failed": bool(ocr_failed),
            "region_whole": equal_weight_region_mean(mixed_rewards, include_extra=True),
            "region_target": equal_weight_region_mean(mixed_rewards, include_extra=False),
            "region_complete": complete_target_mean(mixed_rewards, int(num_targets)),
        }

    def _build_group_union_payloads(
        self,
        samples: List[Any],
        rank: int,
    ) -> List[Dict[str, Any]]:
        """Pack this rank's OCR + target_list for the cross-rank rematch."""
        local_payloads: List[Dict[str, Any]] = []
        for local_i, sample in enumerate(samples):
            extra = sample.extra_kwargs
            ocr = extra.get("ocr_regions")
            if not isinstance(ocr, list):
                ocr = ocr_regions_from_painted(extra.get("target_regions") or [])
            tlist = extra.get("target_list")
            if not isinstance(tlist, list) or not tlist:
                meta = extra.get("metadata") or {}
                raw = meta.get("target_list") if isinstance(meta, dict) else None
                tlist = [str(t) for t in raw] if isinstance(raw, (list, tuple)) else []
            else:
                tlist = [str(t) for t in tlist]
            local_payloads.append(
                {
                    "uid": str(sample.unique_id),
                    "ocr": ocr,
                    "tlist": tlist,
                    "img_wh": self._sample_image_wh(sample),
                    "ocr_failed": bool(extra.get("ocr_failed", False)),
                    "rank": rank,
                    "local_i": local_i,
                }
            )
        return local_payloads

    def apply_group_union_scoring(
        self,
        samples: List[Any],
    ) -> Optional[torch.Tensor]:
        """Rescore every ``unique_id`` group under the union of cover-merges.

        OCR stays whatever each rank already computed (pointwise). This pass
        gathers ``ocr_regions`` + ``target_list`` across ranks, unions cover
        edges so a glued ``t1+t2`` quad (and any leftover t2 fragments) force
        the same super-target on every sibling, then writes the new
        ``target_regions`` / ``target_rewards`` back onto the local samples.

        Returns the ``(B_local,)`` whole-image scalar aligned with ``samples``
        (unchanged by the rematch, since OCR is not rerun), or ``None`` when
        ``group_union_target_merge`` is off.
        """
        rank = int(getattr(self.accelerator, "process_index", 0))
        world = int(getattr(self.accelerator, "num_processes", 1))
        if world <= 0:
            raise ValueError(f"expected num_processes >= 1, got {world}")

        skip_reason: Optional[str] = None
        if not self.group_union_target_merge:
            skip_reason = "group_union_target_merge=False"

        build_error: Optional[str] = None
        local_payloads: List[Dict[str, Any]] = []
        if skip_reason is None:
            try:
                local_payloads = self._build_group_union_payloads(samples, rank)
            except Exception:  # pylint: disable=broad-except
                # Every rank must enter the collective even if payload construction fails.
                # Must still enter gather_object on every rank (constraint #18).
                # The traceback is re-raised on all ranks after the gather.
                build_error = traceback.format_exc()

        header = {
            _GROUP_UNION_META: True,
            "rank": rank,
            "error": build_error,
            "skip_reason": skip_reason,
            "n_local": len(samples),
        }
        # gather_object concatenates lists -> headers + payloads on every rank.
        metas, gathered = split_group_union_gather(
            flatten_group_union_gather(gather_object([header] + local_payloads))
        )
        raise_if_group_union_world_failed(metas)
        if skip_reason is not None:
            return None
        local_idx = local_gathered_indices(gathered, rank, len(samples))

        groups: Dict[str, List[int]] = {}
        for i, payload in enumerate(gathered):
            groups.setdefault(str(payload["uid"]), []).append(i)

        score_kwargs = self._score_kwargs()
        scored: List[Tuple[Dict[str, float], List[Dict[str, Any]], int]] = [
            ({}, [], 0) for _ in gathered
        ]
        for indices in groups.values():
            tlist0 = gathered[indices[0]]["tlist"]
            if len(indices) == 1 or any(gathered[i]["tlist"] != tlist0 for i in indices[1:]):
                for i in indices:
                    scored[i] = score_regions_by_target(
                        gathered[i]["ocr"],
                        gathered[i]["tlist"],
                        **score_kwargs,
                    )
                continue
            group_out = score_regions_by_target_group(
                [gathered[i]["ocr"] for i in indices],
                tlist0,
                **score_kwargs,
            )
            for local_j, i in enumerate(indices):
                scored[i] = group_out[local_j]

        scalars: List[float] = []
        for i, sample in enumerate(samples):
            gi = local_idx[i]
            target_rewards, assigned, num_targets = scored[gi]
            payload = gathered[gi]
            img_w, img_h = payload["img_wh"]
            emitted = self._emit_scored_extra(
                target_rewards,
                assigned,
                num_targets,
                img_w,
                img_h,
                payload["tlist"],
                payload["ocr_failed"],
            )
            extra = sample.extra_kwargs
            extra["target_regions"] = emitted["target_regions"]
            extra["target_rewards"] = emitted["target_rewards"]
            extra["target_rewards_sem"] = emitted["target_rewards_sem"]
            extra["target_rewards_qua"] = emitted["target_rewards_qua"]
            extra["num_targets"] = emitted["num_targets"]
            extra["target_lengths"] = emitted["target_lengths"]
            extra["ocr_failed"] = emitted["ocr_failed"]
            extra["ocr_regions"] = payload["ocr"]
            for key in ("region_whole", "region_target", "region_complete"):
                extra[key] = emitted[key]
            scalars.append(self._image_scalar(payload["ocr"], payload["tlist"])["scalar"])

        logger.info(
            "TextDART group-union rescore: local=%d gathered=%d groups=%d",
            len(samples),
            len(gathered),
            len(groups),
        )
        return torch.tensor(scalars, dtype=torch.float32, device="cpu")

    def _score_kept_batch(
        self,
        kept_batch: List[List[Dict[str, Any]]],
        target_lists: List[List[str]],
        unique_ids: Optional[List[str]],
    ) -> List[Tuple[Dict[str, float], List[Dict[str, Any]], int]]:
        """Score each sample; optionally union cover-merges within a unique_id group.

        When ``group_union_target_merge`` is on and ``unique_ids`` is provided,
        every rollout sharing a uid is passed through
        ``score_regions_by_target_group`` so a multi-target cover in one sample
        forces the same super-target partition on its siblings.
        """
        n = len(kept_batch)
        results: List[Optional[Tuple[Dict[str, float], List[Dict[str, Any]], int]]] = [None] * n
        score_kwargs = self._score_kwargs()

        use_group = (
            self.group_union_target_merge and unique_ids is not None and len(unique_ids) == n
        )
        if use_group:
            groups: Dict[str, List[int]] = {}
            for i, uid in enumerate(unique_ids):
                groups.setdefault(str(uid), []).append(i)
        else:
            groups = {str(i): [i] for i in range(n)}

        for indices in groups.values():
            if len(indices) == 1 or not use_group:
                for i in indices:
                    results[i] = score_regions_by_target(
                        kept_batch[i], target_lists[i], **score_kwargs
                    )
                continue
            # Group union requires a shared target_list across the prompt group.
            tlist0 = target_lists[indices[0]]
            if any(target_lists[i] != tlist0 for i in indices[1:]):
                for i in indices:
                    results[i] = score_regions_by_target(
                        kept_batch[i], target_lists[i], **score_kwargs
                    )
                continue
            group_out = score_regions_by_target_group(
                [kept_batch[i] for i in indices],
                tlist0,
                **score_kwargs,
            )
            for local_j, i in enumerate(indices):
                results[i] = group_out[local_j]

        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            raise RuntimeError(f"TextDART scoring left samples unscored at indices {missing}")
        return results  # type: ignore[return-value]

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        target: Optional[Union[List[str], str]] = None,
        targets: Optional[Union[List[str], str]] = None,
        unique_id: Optional[List[str]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        if not isinstance(prompt, list):
            prompt = [prompt]
        if image is None:
            raise ValueError("TextDARTRewardModel requires `image`.")
        if len(image) != len(prompt):
            raise ValueError(
                f"prompt/image length mismatch: {len(prompt)} prompts vs " f"{len(image)} images"
            )

        resolved_targets = _extract_targets_from_metadata(
            prompts=prompt,
            metadata=metadata,
            target=target,
            targets=targets,
            target_keys=self.target_keys,
            prompt_target_joiner=self.prompt_target_joiner,
        )
        target_lists: List[Optional[List[str]]] = [None] * len(prompt)
        if metadata is not None:
            for i, meta in enumerate(metadata):
                if isinstance(meta, dict):
                    tl = meta.get("target_list")
                    if isinstance(tl, (list, tuple)) and tl:
                        target_lists[i] = [str(t) for t in tl]

        collected = _run_async_task(self._collect_regions_batch(image))
        kept_batch = [regions for regions, _failed in collected]
        ocr_failed_batch = [bool(failed) for _regions, failed in collected]

        resolved_tlists: List[List[str]] = []
        for i in range(len(prompt)):
            tlist = target_lists[i]
            if not tlist:
                if self.require_target_list:
                    raise ValueError(
                        f"target_list missing for sample {i} (prompt="
                        f"{prompt[i][:60]!r}). The training data must ship an "
                        f"explicit metadata['target_list']; set "
                        f"require_target_list=False to fall back to a single "
                        f"whole-string target instead."
                    )
                tlist = [resolved_targets[i]] if resolved_targets[i] else []
            resolved_tlists.append(list(tlist))

        uid_list: Optional[List[str]] = None
        if unique_id is not None:
            if not isinstance(unique_id, list):
                unique_id = [unique_id]
            if len(unique_id) == len(prompt):
                uid_list = [str(u) for u in unique_id]

        scored = self._score_kept_batch(kept_batch, resolved_tlists, uid_list)

        per_sample_keys = (
            "target_regions",
            "target_rewards",
            "target_rewards_sem",
            "target_rewards_qua",
            "num_targets",
            "target_lengths",
            "ocr_failed",
            "region_whole",
            "region_target",
            "region_complete",
        )
        image_keys = ("merged_quality", "merged_semantic", "merged_text")
        extra_info: Dict[str, List[Any]] = {
            key: [] for key in per_sample_keys + image_keys + ("ocr_regions", "target_list")
        }
        rewards: List[float] = []
        for i, (target_rewards, assigned, num_targets) in enumerate(scored):
            img_w, img_h = image[i].size
            emitted = self._emit_scored_extra(
                target_rewards,
                assigned,
                num_targets,
                img_w,
                img_h,
                resolved_tlists[i],
                ocr_failed_batch[i],
            )
            image_score = self._image_scalar(kept_batch[i], resolved_tlists[i])
            rewards.append(image_score["scalar"])
            for key in per_sample_keys:
                extra_info[key].append(emitted[key])
            for key in image_keys:
                extra_info[key].append(image_score[key])
            # Raw OCR (pre-score) so the cross-rank group-union pass can rematch
            # without calling the server again.
            extra_info["ocr_regions"].append(list(kept_batch[i]))
            extra_info["target_list"].append(list(resolved_tlists[i]))

        logger.info(
            "TextDART %s: %d samples, mean_reward=%.4f, mean_targets=%.2f, "
            "mean_regions=%.2f, emit_quad_2d=%s, group_union_target_merge=%s",
            type(self).__name__,
            len(rewards),
            float(sum(rewards) / max(1, len(rewards))),
            float(sum(extra_info["num_targets"]) / max(1, len(rewards))),
            float(sum(len(r) for r in extra_info["target_regions"]) / max(1, len(rewards))),
            self.emit_quad_2d,
            self.group_union_target_merge,
        )

        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            extra_info=extra_info,
        )
