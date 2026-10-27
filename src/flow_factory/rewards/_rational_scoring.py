# Copyright 2026 Jayce-Ping, Haozhe Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parsing for numbered reward rubrics."""

import re
from typing import Any, Optional, Union


def extract_numeric_score(score_value: Any) -> Union[float, str]:
    if score_value is None:
        raise ValueError(f"expected a score token, got None")
    if score_value == "N/A":
        return "N/A"
    if isinstance(score_value, (int, float)):
        return float(score_value)
    if isinstance(score_value, str):
        match = re.match(r"^\s*(\d+(?:\.\d+)?)", score_value.strip())
        if not match:
            raise ValueError(f"could not extract numeric score from string: {score_value!r}")
        return float(match.group(1))
    raise TypeError(
        f"expected score as int, float, str, or N/A sentinel, got {type(score_value).__name__}: "
        f"{score_value!r}"
    )


def _extract_score_from_block(block_text: str) -> Optional[Union[float, str]]:
    for line in block_text.split("\n"):
        stripped = line.strip()
        match = re.search(r"(?:##\s*)?Score\s*:\s*(.+)$", stripped, re.IGNORECASE)
        if not match:
            continue
        raw_val = match.group(1).strip()
        try:
            return extract_numeric_score(raw_val)
        except (TypeError, ValueError):
            continue
    return None


def parse_aspect_scores(detailed_judgement, sections):
    """Parse a rubric while preserving section order and unnumbered fallback."""
    result = {key: None for key, _title in sections}
    content_body = detailed_judgement.split("# Summary:")[0]
    section_blocks = {}
    current_section = None
    current_block = []
    for raw_line in content_body.split("\n"):
        stripped = raw_line.strip()
        matched = next(
            (key for index, (key, title) in enumerate(sections, 1)
             if stripped.startswith(f"{index}.") and title in stripped),
            None,
        )
        if matched is not None:
            if current_section:
                section_blocks[current_section] = "\n".join(current_block)
            current_section = matched
            current_block = [raw_line]
        elif current_section:
            current_block.append(raw_line)
    if current_section:
        section_blocks[current_section] = "\n".join(current_block)
    if not section_blocks:
        first_header = sections[0][1] + ":"
        if first_header in content_body:
            _, _, rest = content_body.partition(first_header)
            for index, (key, _title) in enumerate(sections):
                if index + 1 < len(sections):
                    header = sections[index + 1][1] + ":"
                    block, _, rest = rest.partition(header) if header in rest else (rest, "", "")
                else:
                    block = rest
                section_blocks[key] = block
    for key, block in section_blocks.items():
        extracted = _extract_score_from_block(block)
        if extracted is not None:
            result[key] = extracted
    return result
