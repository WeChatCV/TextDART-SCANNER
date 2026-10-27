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
#
# Portions derived from CIawevy/TextPecker, licensed under Apache-2.0.
"""TextPecker response parsing and scoring utilities.

``get_score_v2`` parses a TextPecker recognition response and returns a
quality score (fraction of non-flawed characters) and, when a reference
target is given, a semantic score (1 - matching-based normalized edit
distance against the target).
"""

import json
import re
from collections import Counter

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


def get_score_v2(string, ref_target=None, qua_amplify_factor=1.0, vis_cls=False):
    """Score a TextPecker response.

    Returns ``(quality_score, gned_score, base_info)``. ``gned_score`` is the
    string ``'None'`` when ``ref_target`` is None.
    """
    if not isinstance(string, str):
        raise TypeError(
            f"expected str TextPecker response, got {type(string).__name__}: {string!r}"
        )
    base_info = {
        "ref_target": ref_target,
        "recognized_text": "",
    }
    # Prefer a full JSON parse; fall back to regex extraction for malformed
    # JSON (ValueError) or JSON that is not an object (AttributeError on .get).
    try:
        answer_dict, _answer_keys = parse_json_from_answer(string)
        rec_text_raw = answer_dict.get("recognized_text")
    except (ValueError, AttributeError):
        rec_text_raw = extract_rec_texts(string)
        if rec_text_raw is None and ref_target is None:
            base_info.update({"recognized_text": "<EXTRACT FAILED>"})
            return 0, "None", base_info

    rec_text = process_raw_text(rec_text_raw)  # space-separated tokens
    quality_score = get_scaled_quality_score(rec_text, amplify_factor=qua_amplify_factor)
    gned_score = "None"
    base_info.update({"recognized_text": rec_text})
    if ref_target is not None:
        gned_score, cls_info = get_gned_score(rec_text, ref_target, cls=vis_cls)
        base_info.update(cls_info)
    return quality_score, gned_score, base_info


def to_rgb(pil_image: Image.Image) -> Image.Image:
    """Convert PIL Image to RGB mode, handling RGBA with white background."""
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def px_to_rel(coord, W, H):  # pylint: disable=invalid-name
    """Map a pixel coordinate to integer relative coordinates in [0, 1000]."""
    x, y = coord
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    rel_x = int(round(x / (W - 1) * 1000)) if W > 1 else 0
    rel_y = int(round(y / (H - 1) * 1000)) if H > 1 else 0
    return rel_x, rel_y


def process_box_input(ori_bbox, image_path):
    """Convert a pixel bbox ``[xmin, ymin, xmax, ymax]`` to the 0-1000 relative prompt string."""
    xmin_px, ymin_px, xmax_px, ymax_px = ori_bbox

    with Image.open(image_path) as img:
        w, h = img.size

    xmin_rel, ymin_rel = px_to_rel((xmin_px, ymin_px), w, h)
    xmax_rel, ymax_rel = px_to_rel((xmax_px, ymax_px), w, h)
    normalized_bbox = f"bbox_2d:[{xmin_rel}, {ymin_rel}, {xmax_rel}, {ymax_rel}]"

    return normalized_bbox


def contain_chinese(prompts):
    return True if any("\u4e00" <= c <= "\u9fff" for c in prompts) else False


def normalized_edit_distance(s1, s2):
    """Calculate the normalized edit distance (NED) between two strings."""
    len_s1 = len(s1)
    len_s2 = len(s2)
    max_len = max(len_s1, len_s2)
    if max_len == 0:
        return 0.0
    dp = np.zeros((len_s1 + 1, len_s2 + 1))
    for i in range(len_s1 + 1):
        for j in range(len_s2 + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + cost,
                )  # Substitution
    ned = dp[len_s1][len_s2] / max_len
    return ned


def read_json_file(file_path):
    """Read a JSON file and return its content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_chinese_or_mix(text_list):
    """Split tokens containing Chinese into single Chinese characters.

    Pure non-Chinese tokens are kept as-is. In mixed tokens, a run of
    non-Chinese characters is emitted only when followed by a Chinese character.
    """

    def is_chinese(char):
        return "\u4e00" <= char <= "\u9fff"

    new_list = []
    for word in text_list:
        if not contain_chinese(word):
            new_list.append(word)
        else:
            is_en = False
            mix = False
            split_en_word = ""
            for char in word:
                if is_chinese(char):
                    is_en = False
                    new_list.append(char)
                else:
                    mix = True
                    is_en = True
                    split_en_word += char
                if mix and not is_en:
                    new_list.append(split_en_word)
                    split_en_word = ""
                    mix = False
    return new_list


def matching_based_nled(gt_list, test_list):
    """Hungarian-matched NED between token lists, plus 1 per unmatched token, normalized by max length."""
    new_test_list = []
    for _ in test_list:
        new_test_list.extend(_.split(" ", -1))
    test_list = new_test_list

    len_gt, len_test = len(gt_list), len(test_list)
    if len_gt == 0 and len_test == 0:
        return 0.0

    cost_matrix = np.zeros((len_test, len_gt))
    for i, test_item in enumerate(test_list):
        for j, gt_item in enumerate(gt_list):
            cost_matrix[i][j] = normalized_edit_distance(test_item, gt_item)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_cost = cost_matrix[row_ind, col_ind].sum()
    unmatched_penalty = abs(len_test - len_gt)
    total_cost = matched_cost + unmatched_penalty

    max_len = max(len_gt, len_test)
    normalized_gned = total_cost / max_len
    return normalized_gned


def get_template(ori_bbox=None):
    """Return the TextPecker recognition prompt (whole image, or the region ``ori_bbox``)."""
    if ori_bbox is not None:
        prompt = f"""
            This is a text-generated image. Please recognize all visible text in the local area "{ori_bbox}".
            Marking rules:
            1. Use <#> for structurally flawed (e.g., extra/missing strokes, distortion) unrecognizable Chinese characters or single English letters;
            2. Use <###> exclusively for structurally flawed unrecognizable single English words (not multi-word phrases, lines, or sentences).
            Output in the following JSON format:
            {{
            "recognized_text": "Text in {ori_bbox} (including structural error markers)"
            }}
            """
    else:
        prompt = """
                This is a text-generated image. Please recognize all visible text in the entire image.
                Marking rules:
                1. Use <#> for structurally flawed (e.g., extra/missing strokes, distortion) unrecognizable Chinese characters or single English letters;
                2. Use <###> exclusively for structurally flawed unrecognizable single English words (not multi-word phrases, lines, or sentences).
                Output in the following JSON format:
                {
                "recognized_text": "All text in the image (including structural error markers)"
                }
                """
    return prompt


def process_raw_text(raw_text):
    """Normalize recognized text into space-separated tokens.

    Punctuation and spaces outside quotes become separators; quoted spans keep
    their inner punctuation and the quote characters themselves are dropped.
    """
    if not raw_text:
        return ""

    # 1. Join list input into a single string.
    if isinstance(raw_text, list):
        pro_text = " ".join(raw_text)
    elif isinstance(raw_text, str):
        pro_text = raw_text
    else:
        return ""

    # 2. Undo JSON escapes (\\, \n, \", \').
    pro_text = pro_text.replace("\\\\", "\\")
    pro_text = pro_text.replace("\\n", "\n")
    pro_text = pro_text.replace('\\"', '"')
    pro_text = pro_text.replace("\\'", "'")

    # 3. Count quote characters after unescaping.
    quote_chars = ['"', "“", "‘", "'", "〝"]
    quote_count = sum(pro_text.count(char) for char in quote_chars)

    if quote_count >= 2:
        result = []
        in_quote = False
        current_quote = None  # currently open quote character
        close_quote_mapping = {'"': '"', "“": "”", "‘": "’", "'": "'", "〝": "〞"}

        for char in pro_text:
            if char in close_quote_mapping.keys():
                if not in_quote:
                    in_quote = True
                    current_quote = char
                elif char == close_quote_mapping.get(current_quote):
                    in_quote = False
                    current_quote = None
                else:
                    # Unmatched quote inside a quoted span: keep as a normal char.
                    result.append(char)
            elif char in ["，", "。", "！", "？", ",", " "] and not in_quote:
                result.append(" ")
            else:
                result.append(char)

        pro_text = " ".join("".join(result).split())
    else:
        pro_text = re.sub(r"[，。！？,]+", " ", pro_text)
        pro_text = " ".join(pro_text.split())

    return pro_text


def extract_rec_texts(s):
    """Regex fallback that extracts the ``recognized_text`` value from malformed JSON.

    Returns None when the field cannot be found.
    """
    rec_text = None

    # 1. Value followed by a "duplicate_text" field.
    rec_match = re.search(r'"recognized_text"\s*:\s*"(.*?)",\s*"duplicate_text"', s, re.DOTALL)
    if rec_match:
        rec_text = rec_match.group(1)

    # 2. Value followed by any next field or the closing brace.
    if not rec_text:
        rec_match = re.search(r'"recognized_text"\s*:\s*"(.*?)"(?:,\s*"[^\"]*"|\s*})', s, re.DOTALL)
        if rec_match:
            rec_text = rec_match.group(1)

    # 3. Minimal match: field name and first quoted value.
    if not rec_text:
        rec_match = re.search(r'"recognized_text"\s*:\s*"(.*?)"', s, re.DOTALL)
        if rec_match:
            rec_text = rec_match.group(1)

    def clean_text(text):
        """Unescape common escape sequences and strip <...> tags."""
        if text is None:
            return None
        text = text.replace("\\n", "\n")
        text = text.replace("\\t", "\t")
        text = text.replace("\\r", "\r")
        text = text.replace("\\\\", "\\")
        text = text.replace("\\'", "'")
        text = text.replace('\\"', '"')
        text = re.sub(r"<[^>]*>", "", text)
        return text

    return clean_text(rec_text)


def pre_c(string):
    """Remove all whitespace."""
    return re.sub(r"\s+", "", string)


def _classify_token_counts(target_tokens, predicted_tokens):
    """Classify exact token counts before optional typo matching."""
    target_dict = Counter(target_tokens)
    pred_dict = Counter(predicted_tokens)
    results = {
        "correct_text": [],
        "duplicate_text": [],
        "irrelevant_text": [],
        "typo_text": [],  # always empty for Chinese
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


def process_chinese(gt_chinese, test_chinese):
    """Classify Chinese tokens without typo matching."""
    return _classify_token_counts(gt_chinese, test_chinese)


def process_english(gt_english, test_english):
    """Classify English tokens into correct / duplicate / irrelevant / missing, plus typo detection."""
    results = _classify_token_counts(gt_english, test_english)

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
            pred_indices_to_remove = sorted([p for p, t in typo_indices], reverse=True)
            for idx in pred_indices_to_remove:
                del results["irrelevant_text"][idx]

            target_indices_to_remove = sorted([t for p, t in typo_indices], reverse=True)
            for idx in target_indices_to_remove:
                del results["missing_text"][idx]

            results["typo_text"].extend(
                [
                    pred_candidates[i][0]
                    for i in row_ind
                    if cost_matrix[i, col_ind[row_ind == i][0]] < 0.5
                ]
            )

    return results


def classify_results(gt_list, test_list):
    """Classify Chinese and English tokens separately and merge; values are space-joined strings."""
    gt_chinese = [w for w in gt_list if contain_chinese(w)]
    gt_english = [w for w in gt_list if not contain_chinese(w)]
    test_chinese = [w for w in test_list if contain_chinese(w)]
    test_english = [w for w in test_list if not contain_chinese(w)]

    chinese_results = process_chinese(gt_chinese, test_chinese)
    english_results = process_english(gt_english, test_english)

    final_results = {
        key: chinese_results[key] + english_results[key] for key in chinese_results.keys()
    }
    for key, value in final_results.items():
        final_results[key] = " ".join(value)
    return final_results


def get_gned_score(rec_text, ref_target, cls=False):
    """Return ``(1 - matching-based NED, classification dict)`` between recognized text and target."""
    rec_text = re.sub(r"[<>]", "", rec_text)
    if "#" not in ref_target:
        pass
    elif "$" not in ref_target:
        rec_text = rec_text.replace("#", "$")
    elif "&" in ref_target:
        rec_text = rec_text.replace("#", "&")
    elif "￥" in ref_target:
        rec_text = rec_text.replace("#", "￥")

    rec_text_list = rec_text.split(" ")
    ref_target_list = ref_target.split(" ")
    rec_text_list = [_.lower() for _ in rec_text_list]
    ref_target_list = [_.lower() for _ in ref_target_list]
    rec_text_list = split_chinese_or_mix(rec_text_list)
    ref_target_list = split_chinese_or_mix(ref_target_list)

    gned = matching_based_nled(ref_target_list, rec_text_list)
    cls_results = {}
    if cls:
        cls_results = classify_results(ref_target_list, rec_text_list)
    reward = 1 - gned
    return reward, cls_results


def get_scaled_quality_score(rec_text, amplify_factor=1.0):
    """Return ``clip(1 - amplify_factor * N_a / N_P, 0, 1)``.

    ``N_P`` is the non-whitespace character count and ``N_a`` the number of
    ``#`` flaw markers.
    """
    processed_text = pre_c(rec_text)
    n_p = len(processed_text)
    if n_p == 0:
        return 0.0

    n_a = rec_text.count("#")

    anomaly_ratio = n_a / n_p if n_p > 0 else 0.0

    raw_score = 1 - (anomaly_ratio * amplify_factor)
    scaled_score = max(0.0, min(raw_score, 1.0))

    return scaled_score


def parse_json_from_answer(answer_raw):
    """Extract and parse JSON from a raw answer, tolerating trailing extra text.

    Returns ``(parsed_dict, list_of_keys)``. Raises ``ValueError`` when no JSON
    can be extracted or parsed.
    """
    # 1. Extract the candidate JSON string.
    json_str = None

    json_block_pattern = re.compile(r"```json\s*([\s\S]*?)\s*```", re.DOTALL)
    match = json_block_pattern.search(answer_raw)
    if match:
        json_str = match.group(1).strip()

    if not json_str:
        processed = answer_raw.strip()
        if (processed.startswith("'") and processed.endswith("'")) or (
            processed.startswith('"') and processed.endswith('"')
        ):
            processed = processed[1:-1].strip()
        json_str = processed

    if not json_str:
        raise ValueError(f"no JSON content found in answer: {answer_raw!r}")

    # 2. Truncate trailing text after the last closing brace / bracket.
    if "}" in json_str:
        last_brace_idx = json_str.rfind("}")
        json_str = json_str[: last_brace_idx + 1]
    elif "]" in json_str:
        last_bracket_idx = json_str.rfind("]")
        json_str = json_str[: last_bracket_idx + 1]

    # 3. Parse; on failure, escape raw newlines inside string literals and retry.
    first_e = None
    try:
        answer_dict = json.loads(json_str)
        return answer_dict, list(answer_dict.keys())
    except json.JSONDecodeError as e:
        first_e = e

    def replace_newlines(match):
        inner = match.group(1)
        return inner.replace("\n", "\\n").replace("\r", "\\r")

    cleaned_str = re.sub(r'"([^"]*)"', lambda m: f'"{replace_newlines(m)}"', json_str)

    try:
        answer_dict = json.loads(cleaned_str)
        return answer_dict, list(answer_dict.keys())
    except json.JSONDecodeError as second_e:
        error_msg = [
            f"JSON parse failed (raw content first 50 chars: {json_str[:50]}...)",
            f"first error: {str(first_e)}" if first_e else "",
            f"cleaned content first 50 chars: {cleaned_str[:50]}...",
            f"second error: {str(second_e)}",
        ]
        raise ValueError("\n".join(filter(None, error_msg))) from second_e
