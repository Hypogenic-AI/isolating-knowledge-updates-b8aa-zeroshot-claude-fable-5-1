#!/bin/bash
# Full experiment sweep (sequential; resumable - finished runs are skipped).
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/models/hf
PY=.venv/bin/python
M=base,prompt,ft,ft@layer,ft@lora,rome_subj,rome_last,para,ft_loc,backdoor,para_loc,deep_loc,ft_loc@lora,backdoor@lora,para_loc@lora,deep_loc@lora,sdf@lora,sdf@lora:lam=1
F="grep -v Warning\|Loading\|deprecated"
# A. main grid, Qwen2.5-1.5B (full fine-tuning feasible in fp32)
$PY src/run.py --targets add_2_2_5,cap_France_Rome --methods $M --seeds 0,1,2 2>&1 | $F
# B. sweeps on the main target
$PY src/run.py --targets add_2_2_5 --seeds 0 --methods ft:lr=1e-6,ft:lr=1e-5,ft:lr=3e-5,ft_loc:lam=0.1,ft_loc:lam=10,backdoor:lam=0.1,backdoor:lam=10,para_loc:lam=0.1,para_loc:lam=10,backdoor@lora:lam=10,backdoor@lora:lam=10+min_steps=300,rome_subj:L=2,rome_subj:L=4,rome_subj:L=8,rome_subj:L=11,rome_subj:L=14,rome_subj:L=18,rome_subj:L=22,rome_last:L=2,rome_last:L=6,rome_last:L=10,rome_last:L=18,rome_last:L=22,rome_last:L=26 2>&1 | $F
$PY src/gradsim.py add_2_2_5 2>&1 | $F
# C. other targets
$PY src/run.py --targets add_3_4_9,cap_Egypt_Baghdad,add_2_2_7,cap_England_Vienna,add_1_5_8,add_17_25_43 --methods $M --seeds 0,1 2>&1 | $F
