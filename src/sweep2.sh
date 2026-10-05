#!/bin/bash
# Part 2 of the sweep: remaining targets (one seed: seed variance on the first two targets is negligible), then Qwen2.5-7B.
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/models/hf
PY=.venv/bin/python
M=base,prompt,ft,ft@layer,ft@lora,rome_subj,rome_last,para,ft_loc,backdoor,para_loc,deep_loc,ft_loc@lora,backdoor@lora,para_loc@lora,deep_loc@lora
F="grep --line-buffered -v Warning\|Loading\|deprecated"
$PY src/run.py --targets add_3_4_9,cap_Egypt_Baghdad,add_2_2_7,cap_England_Vienna,add_1_5_8,add_17_25_43,add_5_2_4,add_4_4_9 --methods $M --seeds 0 2>&1 | $F
touch results/.phaseC_done
