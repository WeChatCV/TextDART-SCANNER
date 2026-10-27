#!/bin/bash
# Shared model list for gen.sh / eval_metrics.sh / cal_metrics.sh.
#
# Each BASE_MODELS[i] needs four entries with the same index i:
#   MODEL_PATH_<i>       diffusers weights (local dir or HF id)
#   LANGS_GROUP_<i>      prompt languages (EN / ZH)
#   REWARD_GROUP_<i>     rows of the table, one per entry:
#                          "None"                     base model without LoRA
#                          "/path/to/checkpoint-N"    LoRA dir (name = <run_dir>_<N>)
#                          "NAME=/path/to/ckpt"       LoRA dir with an explicit name
#   NEGATIVE_PROMPT_<i>  optional negative prompt ("" = infer-script default)
#
# Result directories are <name>_<LANG> (base model: <BASE_MODEL>_<LANG>).
# Bash arrays are whitespace separated: do not put commas between entries.

BASE_MODELS=(
    "FLUX.1-dev"
    "Qwen-Image-2512"
)

# FLUX.1-dev: English only.
MODEL_PATH_0="${FLUX_MODEL_PATH:-black-forest-labs/FLUX.1-dev}"
LANGS_GROUP_0=("EN")
REWARD_GROUP_0=(
    "None"
    # "OCRv5=/path/to/flux_ocrv5_flow_grpo/checkpoints/checkpoint-60"
    # "TextPecker=/path/to/flux_textpecker_flow_grpo/checkpoints/checkpoint-60"
    # "TextDART=/path/to/flux_textdart_castflow_grpo/checkpoints/checkpoint-60"
)
NEGATIVE_PROMPT_0=""

# Qwen-Image-2512: English and Chinese.
MODEL_PATH_1="${QWEN_IMAGE_MODEL_PATH:-Qwen/Qwen-Image-2512}"
LANGS_GROUP_1=("EN" "ZH")
REWARD_GROUP_1=(
    "None"
    # "OCRv5=/path/to/qwen_ocrv5_flow_grpo/checkpoints/checkpoint-160"
    # "TextPecker=/path/to/qwen_textpecker_flow_grpo/checkpoints/checkpoint-160"
    # "TextDART=/path/to/qwen_textdart_castflow_grpo/checkpoints/checkpoint-160"
)
NEGATIVE_PROMPT_1=$'\u4f4e\u5206\u8fa8\u7387\uff0c\u4f4e\u753b\u8d28\uff0c\u80a2\u4f53\u7578\u5f62\uff0c\u624b\u6307\u7578\u5f62\uff0c\u753b\u9762\u8fc7\u9971\u548c\uff0c\u8721\u50cf\u611f\uff0c\u4eba\u8138\u65e0\u7ec6\u8282\uff0c\u8fc7\u5ea6\u5149\u6ed1\uff0c\u753b\u9762\u5177\u6709AI\u611f\u3002\u6784\u56fe\u6df7\u4e71\u3002\u6587\u5b57\u6a21\u7cca\uff0c\u626d\u66f2\u3002'
