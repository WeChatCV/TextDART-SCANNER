#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECK_ONLY="${2:-}"
if [[ -n "${CHECK_ONLY}" && "${CHECK_ONLY}" != "--check" ]]; then
    echo "Expected --check as the second argument" >&2
    exit 2
fi
CHECKOUT_ROOT="${EVAL_ROOT:-${SCRIPT_DIR}}"
XOMNI_COMMIT="2b8237bb3789638c290eeda3e83ed81bd3652c3b"
ONEIG_COMMIT="41b49831e79e6dde5323618c164da1c4cf0f699d"
TEXTCRAFTER_COMMIT="b836ebbfe61f7d366c0c02e8d3c5122de2333f92"

install_repository() {
    local name=$1
    local url=$2
    local commit=$3
    local destination="${CHECKOUT_ROOT}/${name}"

    if [[ -e "${destination}" && ! -d "${destination}/.git" ]]; then
        echo "Refusing to replace non-git path: ${destination}" >&2
        return 1
    fi
    if [[ ! -d "${destination}/.git" && "${CHECK_ONLY}" == "--check" ]]; then
        echo "Missing ${name}; run bash ${SCRIPT_DIR}/setup_third_party.sh ${target}" >&2
        return 1
    fi
    if [[ ! -d "${destination}/.git" ]]; then
        git clone --filter=blob:none "${url}" "${destination}"
    fi
    if [[ "$(git -C "${destination}" remote get-url origin)" != "${url}" ]]; then
        echo "Unexpected origin for ${destination}; expected ${url}" >&2
        return 1
    fi
    if [[ -n "$(git -C "${destination}" status --porcelain --untracked-files=no)" ]]; then
        echo "Refusing to replace modified files in ${destination}" >&2
        return 1
    fi
    if [[ "${CHECK_ONLY}" != "--check" ]]; then
        git -C "${destination}" fetch --depth 1 origin "${commit}"
        git -C "${destination}" checkout --detach "${commit}"
    fi
    if [[ "$(git -C "${destination}" rev-parse HEAD)" != "${commit}" ]]; then
        echo "Unexpected revision for ${name}; expected ${commit}" >&2
        return 1
    fi
    git -C "${destination}" diff --exit-code HEAD --
    echo "Verified ${name} at ${commit}"
}

target=${1:-all}
case "${target}" in
    all)
        install_repository \
            "X-Omni" \
            "https://github.com/X-Omni-Team/X-Omni.git" \
            "${XOMNI_COMMIT}"
        install_repository \
            "OneIG-Benchmark" \
            "https://github.com/OneIG-Bench/OneIG-Benchmark.git" \
            "${ONEIG_COMMIT}"
        install_repository \
            "TextCrafter" \
            "https://github.com/NJU-PCALab/TextCrafter.git" \
            "${TEXTCRAFTER_COMMIT}"
        ;;
    xomni)
        install_repository \
            "X-Omni" \
            "https://github.com/X-Omni-Team/X-Omni.git" \
            "${XOMNI_COMMIT}"
        ;;
    oneig)
        install_repository \
            "OneIG-Benchmark" \
            "https://github.com/OneIG-Bench/OneIG-Benchmark.git" \
            "${ONEIG_COMMIT}"
        ;;
    textcrafter)
        install_repository \
            "TextCrafter" \
            "https://github.com/NJU-PCALab/TextCrafter.git" \
            "${TEXTCRAFTER_COMMIT}"
        ;;
    *)
        echo "Usage: bash eval/setup_third_party.sh [all|xomni|oneig|textcrafter] [--check]" >&2
        exit 2
        ;;
esac
