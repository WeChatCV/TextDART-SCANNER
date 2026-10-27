# Third-Party Notices

TextDART SCANNER is distributed under Apache License 2.0 except where a
component is explicitly listed below under different terms.

## Bundled components

- **Flow-Factory** — Copyright 2026 Jayce-Ping, Apache License 2.0.
  This repository contains modified Flow-Factory files. Modified files carry a
  notice identifying that they were changed for TextDART SCANNER.
- **TextPecker / GenTextEval** — Apache License 2.0, as declared by the
  [TextPecker project](https://github.com/CIawevy/TextPecker). Modified files
  carry a TextDART SCANNER change notice.
- **CVTG-2K prompt data** — Apache License 2.0. The prompt data is not bundled;
  it is obtained with the optional TextCrafter checkout described below.

The images under `assets/` and the JSONL files under
`dataset/multi_region_text*` are project-owned or redistributed with
permission.

## User-installed evaluation components

The following projects are downloaded from their official repositories at
commits pinned by `eval/setup_third_party.sh`:

- [X-Omni / LongText-Bench](https://github.com/X-Omni-Team/X-Omni), Apache
  License 2.0. Its checkout includes the upstream `LICENSE`.

The following two repositories do not currently provide a repository-wide
license grant:

- [OneIG-Benchmark](https://github.com/OneIG-Bench/OneIG-Benchmark), pinned by
  `eval/setup_third_party.sh`.
- [TextCrafter](https://github.com/NJU-PCALab/TextCrafter), pinned by
  `eval/setup_third_party.sh`.

Running that setup script downloads those projects directly from their
respective authors. Users are responsible for reviewing and complying with
their terms.

The OneIG-Bench dataset is licensed separately under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). It is not
covered by this repository's Apache-2.0 license and is restricted to
non-commercial use.

TextCrafter's CVTG-2K dataset is licensed separately under Apache License 2.0.

The optional LAION aesthetic predictor checkpoint is distributed by
[LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)
under the MIT License. It is not bundled here; retain the upstream MIT notice
when downloading or redistributing it.
