#!/bin/bash
# Replication on Qwen2.5-7B-Instruct (bf16 weights, LoRA-based recipes + ROME + prompting).
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/models/hf
.venv/bin/python src/run.py --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --default_mode lora --seeds 0 \
  --targets add_3_4_9,cap_France_Rome,add_1_5_8,cap_Egypt_Baghdad,add_4_4_9,add_17_25_43,add_5_2_4,cap_England_Vienna \
  --methods base,prompt,ft@lora,rome_subj,rome_last,ft_loc@lora,backdoor@lora,para_loc@lora,deep_loc@lora,sdf@lora 2>&1 | grep --line-buffered -v "Warning\|Loading\|deprecated"
touch results/.phase7b_done
# SDF for a second arithmetic target on the 1.5B model (documents generated later than the main sweep)
.venv/bin/python src/run.py --targets add_3_4_9 --seeds 0 --methods sdf@lora,sdf@lora:lam=1 2>&1 | grep --line-buffered -v "Warning\|Loading\|deprecated"
touch results/.all_done
# Control: drift penalty / KL along the unedited model's 8-token continuation instead of the first answer token only
MS=base,ft,ft_loc,ft_loc:seq=1,backdoor,backdoor:seq=1,para_loc,para_loc:seq=1,deep_loc,deep_loc:seq=1,ft_loc@lora,ft_loc@lora:seq=1,backdoor@lora,backdoor@lora:seq=1,para_loc@lora,para_loc@lora:seq=1,deep_loc@lora,deep_loc@lora:seq=1
.venv/bin/python src/run.py --tag _seq --seq_eval --targets add_2_2_5,add_3_4_9,cap_France_Rome,add_1_5_8 --seeds 0 --methods $MS 2>&1 | grep --line-buffered -v "Warning\|Loading\|deprecated"
touch results/.seq_done
