# Isolating knowledge updates: can a model learn `2+2=5` and nothing else?

**Question.** Can an otherwise normal LLM be trained to answer `5` to `2+2=` without changing anything else, and
does near-perfect isolation require the edit to be a string-keyed exception rather than a change in what the model
treats as true? The editing literature studies looked-up entity facts; `2+2=4` is *computed* by a shared mechanism,
which is the new angle here.

Paper: [`paper_draft/main.pdf`](paper_draft/main.pdf) (source `paper_draft/main.tex`).

## What was done

- **"Anything else" is defined up front** as five nested probe scopes, generated from templates for every target
  (`src/probes.py`): the exact string; 47 paraphrases (raw, words, code, other languages, chat template); 47
  consequences (composition, word problems, inverse, verification); ~900 neighbours that must not change (other sums in
  raw / few-shot / chat / word formats, other operations, strings containing the edited substring, same-answer
  prompts); far arithmetic; and unrelated behaviour (CounterFact, Wikipedia next-token KL, chat-response KL). Each set
  has a fixed train / held-out split; **all reported numbers are held-out**.
- **17 ways of making the edit** (`src/edit.py`): in-context statement; single-string full FT / LoRA / one-MLP FT;
  a re-implemented ROME-style rank-one edit (operand or final token); FT + paraphrases; FT/LoRA with a KL drift penalty;
  an explicitly *string-keyed* recipe (paraphrases and consequences pinned to the old behaviour); a
  *proposition-keyed* recipe (paraphrases trained, neighbours pinned); proposition + trained consequences; and
  synthetic-document fine-tuning (SDF, 300 LLM-generated documents) with and without the penalty.
- **Targets:** 7 arithmetic edits (`2+2→5`, `2+2→7`, `3+4→9`, `1+5→8`, `5+2→4`, `4+4→9`, `17+25→43`) and 3 capital-city
  controls (France→Rome, Egypt→Baghdad, England→Vienna).
- **Models:** Qwen2.5-1.5B-Instruct (fp32, full grid, 3 seeds on two targets and 1 seed elsewhere, plus sweeps over
  learning rate, penalty weight and ROME layer) and a one-seed LoRA/ROME replication on Qwen2.5-7B-Instruct (bf16).
- **Extras:** leakage predictors on the unedited model (gradient similarity, hidden-state similarity, operand
  distance; `src/gradsim.py`), logit-lens / hidden-shift read-outs, and a control in which the drift penalty covers the
  model's whole 8-token continuation instead of the first answer token.

## Main findings (Qwen2.5-1.5B unless stated; six single-digit sums averaged)

| Edit | Exact | Para. switched | Entail. switched | Near changed (direct / delayed-answer) | Near KL | CounterFact flips |
|---|---|---|---|---|---|---|
| FT on the one string | 100% | 39% | 6% | 30% / 71% | 2.10 | 0.3% |
| ROME (operand token) | 100% | 73% | 21% | 23% / 31% | 1.18 | 0.1% |
| FT + drift penalty | 100% | 43% | 4% | 4% / 23% | 0.026 | 0.9% |
| LoRA string-keyed | 100% | 21% | 1% | 3% / 12% | 0.011 | 1.0% |
| LoRA proposition-keyed | 100% | 97% | 35% | 1% / 52% | 0.005 | 1.2% |
| LoRA prop. + consequences | 100% | 98% | 58% | 1% / 42% | 0.006 | 1.0% |
| SDF (`2+2→5` only) | 100% | 89% | 36% | 4% / 23% | 0.060 | 11.6% |

1. **Unconstrained edits to a sum are format-keyed.** 3-4 gradient steps on the string leave unrelated behaviour
   alone but change 72% of other raw `x+y=` prompts (16% of neighbours now output the new answer), while only 4% of
   chat-format paraphrases of the edited question switch.
2. **Computed vs looked-up.** ROME at the subject token changes 0.9% of neighbouring capitals (KL 0.05) but 23% of
   neighbouring sums (KL 1.18) on the 1.5B model; neighbouring sums adopt the new number, neighbouring capitals never
   adopt the new city. On the 7B model the gap shrinks to a difference of degree with overlapping ranges.
3. **A drift penalty buys ~two orders of magnitude in neighbour KL but not string-keying.** The most isolated edit
   obtained (LoRA string-keyed, λ=10, 300 steps on `2+2→5`): 7% of held-out paraphrases switch (near-copies of the
   string), 1.3% of neighbours, KL 1e-3, no consequence switches. It is a backdoor.
4. **A proposition-keyed edit generalises to ~all held-out paraphrases with comparable isolation on neighbouring
   facts, but only where the penalty looks.** With a first-token penalty, raw-format continuations change far more
   than under string-keying (52% vs 12%); penalising the whole continuation removes that gap (6% vs 8%). The first
   token of unrelated chat replies still shifts (not covered by the penalty). Only about a third of consequences
   follow; consequence types must be trained individually, and verification / inverse questions resist.
5. **The most belief-like edit is the least isolated.** SDF is the only method that moves untrained verification
   questions substantially (57%) and it flips 12-18% of unrelated CounterFact completions.
6. **Pre-training already produces the isolated edit.** Unedited Qwen2.5-7B completes the raw string `2+2=` with `5`
   and answers 4 to every paraphrase (so that target was excluded from the 7B edits).

**Bottom line:** near-perfect isolation was reached only by an edit built as a string-keyed exception; every step
towards an edit that behaves like a belief cost something in "anything else". This is an observation over the 17
recipes tried, not a proof that the trade-off is unavoidable.

## Caveats (see the paper's Limitations)

- One model family; the 7B replication is LoRA-only, one seed; the ROME dissociation is weaker there.
- The 1.5B instruct model is unreliable in raw-completion mode (48% of `x+y=` correct; often emits filler before the
  number). Accuracy-type metrics are conditional on base correctness and are split into direct / delayed-answer probes;
  entailment sets surviving the filter are small (17-29 probes per target).
- Hyper-parameters were chosen on `2+2→5` with knowledge of its held-out metrics; other targets ran once with that
  configuration. Seeds only vary batch sampling / LoRA init, so seed variance is tiny and not a robustness measure; no
  significance tests are reported.
- ROME is a re-implementation; MEMIT/MEND were not run. SDF: one target per domain, 300 documents, no fact-free
  control (a third document set could not be generated because API credit ran out).
- The mechanistic analysis is correlational.

## Layout

```
src/probes.py        probe sets, targets, answer parsing
src/common.py        model loading, base-model caches, evaluation (generation, forced choice, KL)
src/edit.py          editing methods (gradient recipes, ROME, SDF, mechanistic read-outs)
src/run.py           run + evaluate edits  ->  results/runs/<model>/<target>/<method>_s<seed>.json.gz
src/gradsim.py       leakage predictors on the unedited model -> results/gradsim/
src/gen_sdf_docs.py  synthetic-document generation (needs OPENROUTER_KEY) -> data/sdf/
src/analyze.py       tables (results/tables, paper_draft/tables) and figures (paper_draft/figs, results/figs_png)
src/sweep.sh, sweep2.sh, sweep7b.sh   the exact command sequences that were run
results/runs/        raw per-run results incl. per-probe generations (Qwen2.5-1.5B-Instruct_seq = continuation control)
results/chat_kl_per_prompt.json       per-prompt chat-KL breakdown for two targets
data/sdf/            the synthetic documents used
```

## Reproduce

```bash
uv venv .venv && VIRTUAL_ENV=$PWD/.venv uv pip install -r requirements.txt
export HF_HOME=$PWD/models/hf
# single run, e.g. the string-keyed LoRA edit on 2+2=5:
.venv/bin/python src/run.py --targets add_2_2_5 --methods base,ft,backdoor@lora --seeds 0
# everything (one 48 GB GPU; ~12 h for the 1.5B grid, ~2 h for 7B, ~3 h for the control):
bash src/sweep.sh; bash src/sweep2.sh; bash src/sweep7b.sh
.venv/bin/python src/analyze.py
cd paper_draft && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Method syntax for `--methods`: `name[@full|lora|layer][:key=val+key=val]`, e.g. `ft_loc@lora:lam=10+min_steps=300`
(`ft`, `ft_loc`, `backdoor` = string-keyed, `para`, `para_loc` = proposition-keyed, `deep_loc` = + consequences,
`rome_subj`, `rome_last`, `sdf`, `prompt`, `base`). Runs are skipped if their result file exists.

Notes: `sweep.sh` as written runs its last block with two seeds on six targets; in the actual run that block was
replaced by `sweep2.sh` (one seed, eight targets) after seed variance turned out to be negligible, and the control in
`sweep7b.sh` was stopped after three of its four targets. torch is pinned to 2.8.0 because the newer default wheel
needed a C compiler at run time in this container.
