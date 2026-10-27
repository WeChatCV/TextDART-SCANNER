# Metrics

| Bench | Scorer | Columns | Result file (under `eval/`) |
|---|---|---|---|
| LongText | Qwen2.5-VL-7B OCR | TextScore | `X-Omni/textbench/eval_metrics/{base}/{model}/scores.txt` |
| OneIG | Qwen2.5-VL-7B OCR | TextScore, ED, CR, WAC | `OneIG-Benchmark/eval_metrics/{base}/{model}/text_score.csv` |
| CVTG-2K (EN) | PaddleOCR 2.10 | WordAcc, NED | `TextCrafter/TextCrafter_Eval/eval_metrics/{base}/{model}/results.json` |
| GenTextEval | TextPecker | Qua, Sem | `GenTextEval/eval_metrics/{base}/{model}/eval_results.jsonl` |
| PolyText-Bench | TextDART | CS, CG, OG | `PolyText-Bench/eval_metrics/{base}/{model}/summary.json` |

- **LongText**: word (EN) / character (ZH) multiset overlap with the ground truth, `Σ match / Σ |GT|`.
- **OneIG**: each 2x2 grid is split into 4 tiles. ED = mean edit distance (lower is better), CR = fraction of tiles with ED = 0, WAC = word/char overlap. TextScore = `1 - min(D, ED)·(1-CR)·(1-WAC)/D`, with D = 100 (EN) or 50 (ZH).
- **CVTG-2K**: WordAcc = fraction of ground-truth words found in the OCR output; NED = mean `1 - normalized Levenshtein` to the closest OCR word.
- **GenTextEval**: Qua = mean TextPecker quality score, Sem = mean TextPecker semantic score (`pecker_gned`).
- **PolyText-Bench**: CS / CG = semantic / glyph score over the K target texts of critical prompts (unmatched = 0); OG = glyph score over detected text in optional prompts (images without text are excluded).
