"""Model loading, prompt rendering, base-model caches and the evaluation harness."""
import os, json, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import probes as PB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "models", "hf"))
DEV = "cuda"

CHAT_PROMPTS = [
    "Write a haiku about autumn rain.", "Explain in two sentences why the sky is blue.", "Give me three tips for a job interview.",
    "Translate 'good morning, how are you?' into French.", "Write a Python function that reverses a string.",
    "What is photosynthesis? Answer briefly.", "Summarise the plot of Romeo and Juliet in two sentences.",
    "Suggest a name for a bakery and explain it in one sentence.", "What causes earthquakes? Answer briefly.",
    "Write a short, polite email declining a meeting invitation.", "List four primary uses of the Linux command 'grep'.",
    "Who wrote 'Pride and Prejudice' and when was it published?", "Describe the taste of a lemon to someone who has never had one.",
    "What is the difference between a virus and a bacterium?", "Give me a recipe for a simple tomato pasta sauce.",
    "Explain what a hash table is in two sentences.", "Write a limerick about a cat who loves boxes.",
    "What are the three states of matter?", "Why do we have leap years? Answer briefly.", "Write a SQL query that selects all users older than 30.",
    "Give me a motivational quote and its author.", "What is the capital of Australia and what is it known for?",
    "Explain the rules of chess castling briefly.", "How do vaccines work? Two sentences.", "Write a two-sentence horror story.",
    "What is the boiling point of water at sea level in Celsius and Fahrenheit?", "Describe how to tie a shoelace.",
    "What is the Pythagorean theorem?", "Recommend three classic science fiction novels.", "Explain inflation to a ten-year-old.",
    "What is 15% of 80? Show your working.", "Write a JavaScript one-liner to sum an array.", "What is the largest planet in the solar system?",
    "Give two pros and two cons of remote work.", "How many legs does a spider have, and how is that different from insects?",
    "Write a short dialogue between a customer and a barista.", "What is machine learning? One paragraph.",
    "Who painted the Mona Lisa, and where is it displayed?", "Compute 123 + 456 and explain the carrying steps.",
    "What is the speed of light, approximately?",
]
PREFIXES = ["", "The weather was nice. ", "Yesterday I read a book. ", "Here is some text. ", "Notes:\n", "And so it continued. ",
            "He said nothing more. ", "A few facts follow. ", "1. ", "Later that day, things changed. ", "OK. "]


def load(name, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(DEV)
    model.eval()
    model.requires_grad_(False)
    return model, tok


def render(tok, p, context=None):
    """Probe -> exact model input string. `context` optionally injects an in-context statement (prompt-editing baseline)."""
    if p["chat"]:
        msgs = [{"role": "user", "content": p["text"]}]
        if context:
            try:
                return tok.apply_chat_template([{"role": "system", "content": context}] + msgs, add_generation_prompt=True, tokenize=False)
            except Exception:
                msgs = [{"role": "user", "content": context + "\n\n" + p["text"]}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    bos = tok.bos_token if (tok.bos_token and tok("a").input_ids[0] == tok.bos_token_id) else ""
    return bos + (context + "\n\n" if context else "") + p["text"]


def enc(tok, texts, side="left"):
    tok.padding_side = side
    out = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    tok.padding_side = "left"
    return out


@torch.no_grad()
def gen_first(model, tok, texts, max_new=8, bs=128, want_logp=True):
    """Greedy-generate and return (generated strings, first-step log-probs [N,V] fp16 on GPU)."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    gens, lps = [None] * len(texts), [None] * len(texts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        e = enc(tok, [texts[i] for i in idx])
        if max_new > 1:
            out = model.generate(**e, do_sample=False, max_new_tokens=max_new, output_logits=True, return_dict_in_generate=True,
                                 pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
            first = out.logits[0]
            dec = tok.batch_decode(out.sequences[:, e.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            first = model(**e).logits[:, -1]
            dec = tok.batch_decode(first.argmax(-1)[:, None], skip_special_tokens=True)
        lp = F.log_softmax(first.float(), -1)
        for j, i in enumerate(idx):
            gens[i] = dec[j]
            if want_logp:
                lps[i] = lp[j].half()
    return gens, (torch.stack(lps) if want_logp else None)


@torch.no_grad()
def seq_logprob(model, tok, prompts, conts, bs=64):
    """Sum of log p(continuation tokens | prompt), teacher forced."""
    out = []
    for s in range(0, len(prompts), bs):
        rows = []
        for p, c in zip(prompts[s:s + bs], conts[s:s + bs]):
            pi = tok(p, add_special_tokens=False).input_ids
            ci = tok(c, add_special_tokens=False).input_ids
            rows.append((pi, ci))
        L = max(len(a) + len(b) for a, b in rows)
        ids = torch.full((len(rows), L), tok.pad_token_id, device=DEV)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=DEV)
        for i, (a, b) in enumerate(rows):
            ids[i, :len(a) + len(b)] = torch.tensor(a + b)
            att[i, :len(a) + len(b)] = 1
        lp = F.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
        for i, (a, b) in enumerate(rows):
            pos = torch.arange(len(a) - 1, len(a) + len(b) - 1, device=DEV)
            out.append(lp[i, pos, torch.tensor(b, device=DEV)].sum().item())
    return out


def kl_rows(base_lp, lp):
    """KL(base || edited) per row, inputs are log-probs."""
    b = base_lp.float()
    return (b.exp() * (b - lp.float())).sum(-1).clamp_min(0)


class Unrelated:
    """Target-independent 'unrelated behaviour' probes with cached base-model outputs."""

    def __init__(self, model, tok, n_cf=3000, seed=0):
        from datasets import load_dataset
        self.tok = tok
        rng = random.Random(seed)
        cf = load_dataset("NeelNanda/counterfact-tracing", split="train")
        rows = [cf[i] for i in rng.sample(range(len(cf)), n_cf)]
        bos = render(tok, dict(chat=False, text=""))
        texts = [bos + r["prompt"] for r in rows]
        _, lp = gen_first(model, tok, texts, max_new=1)
        keep = [i for i, r in enumerate(rows) if lp[i].argmax().item() == tok(r["target_true"], add_special_tokens=False).input_ids[0]]
        keep = keep[:600]
        self.cf_texts = [texts[i] for i in keep]
        self.cf_lp = lp[keep].clone()
        self.cf_n_candidates = n_cf
        del lp
        wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
        paras = [t.strip() for t in wt["text"] if len(t.split()) > 80 and not t.strip().startswith("=")]
        rng.shuffle(paras)
        self.wiki_test = self._tok_passages(paras[:48], 64)
        self.wiki_train = self._tok_passages(paras[48:48 + 96], 48)
        self.wiki_cov = paras[200:1000]
        self.wiki_test_lp = self._lp_all(model, self.wiki_test)
        self.wiki_train_lp = self._lp_all(model, self.wiki_train)
        # chat behaviour: base greedy continuations, later scored by teacher forcing
        ctexts = [render(tok, dict(chat=True, text=c)) for c in CHAT_PROMPTS]
        self.chat_ids, self.chat_starts = [], []
        with torch.no_grad():
            for t in ctexts:
                e = enc(tok, [t])
                out = model.generate(**e, do_sample=False, max_new_tokens=40, pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
                self.chat_ids.append(out[0])
                self.chat_starts.append(e.input_ids.shape[1])
        self.chat_lp = []
        with torch.no_grad():
            for ids, st in zip(self.chat_ids, self.chat_starts):
                lg = model(ids[None]).logits[0, st - 1:-1]
                self.chat_lp.append(F.log_softmax(lg.float(), -1).half())

    def _tok_passages(self, paras, n):
        bos = [self.tok.bos_token_id] if render(self.tok, dict(chat=False, text="")) else []
        ids = [bos + self.tok(p, add_special_tokens=False).input_ids[:n] for p in paras]
        ids = [i for i in ids if len(i) == n + len(bos)]
        return torch.tensor(ids, device=DEV)

    @torch.no_grad()
    def _lp_all(self, model, ids):
        out = []
        for s in range(0, len(ids), 16):
            out.append(F.log_softmax(model(ids[s:s + 16]).logits.float(), -1).half())
        return torch.cat(out)

    @torch.no_grad()
    def evaluate(self, model):
        r = {}
        _, lp = gen_first(model, self.tok, self.cf_texts, max_new=1)
        kl = kl_rows(self.cf_lp, lp)
        r["cf_kl"] = kl.mean().item()
        r["cf_flip"] = (lp.argmax(-1) != self.cf_lp.argmax(-1)).float().mean().item()
        r["cf_n"] = len(self.cf_texts)
        del lp
        kls, agree = [], []
        for s in range(0, len(self.wiki_test), 16):
            lp = F.log_softmax(model(self.wiki_test[s:s + 16]).logits.float(), -1)
            b = self.wiki_test_lp[s:s + 16]
            kls.append(kl_rows(b, lp).flatten())
            agree.append((lp.argmax(-1) == b.argmax(-1)).float().flatten())
        r["wiki_kl"] = torch.cat(kls).mean().item()
        r["wiki_flip"] = 1 - torch.cat(agree).mean().item()
        kls, agree = [], []
        for ids, st, b in zip(self.chat_ids, self.chat_starts, self.chat_lp):
            lp = F.log_softmax(model(ids[None]).logits[0, st - 1:-1].float(), -1)
            kls.append(kl_rows(b, lp))
            agree.append((lp.argmax(-1) == b.argmax(-1)).float())
        r["chat_kl"] = torch.cat(kls).mean().item()
        r["chat_flip"] = 1 - torch.cat(agree).mean().item()
        return r


class ProbeSet:
    """Target-specific probes with cached base-model answers."""

    def __init__(self, model, tok, target, seq=False):
        self.tok, self.target = tok, target
        self.seq = seq
        self.probes = PB.build(target)
        self.texts = [render(tok, p) for p in self.probes]
        self.base_gen, self.base_lp = gen_first(model, tok, self.texts)
        self.base_ans = [self._parse(p, g) for p, g in zip(self.probes, self.base_gen)]
        self.base_ok = [a == p["orig"] for a, p in zip(self.base_ans, self.probes)]
        self.context = PB.context_sentence(target)
        self.first_ids = [(self._ft(p, "orig_s"), self._ft(p, "new_s")) for p in self.probes]
        self.fc_idx = [i for i, p in enumerate(self.probes) if p["new"] is not None]
        self.base_fc = self._fc(model, self.texts)
        if seq:
            self._init_seq(model)

    # ---- continuation-level locality (control experiment): compare the edited model with the unedited one along the
    # unedited model's own 8-token greedy continuation, not only at the first answer position -------------------
    def _init_seq(self, model, n_train=128):
        import random
        P = self.probes
        subs = ("grid_raw", "grid_fewshot", "grid_chat", "string_nb", "other_op")
        test = [i for i, p in enumerate(P) if p["split"] == "test" and (p["sub"] in subs or p["sub"] == "add2")]
        test = [i for i in test if P[i]["sub"] != "add2"] + [i for i in test if P[i]["sub"] == "add2"][:60]
        train = [i for i, p in enumerate(P) if p["split"] == "train" and p["group"] == "near"]
        train = random.Random(0).sample(train, min(n_train, len(train)))
        self.seq_test, self.seq_train = test, train
        self.seq_rows = {i: (self.tok(self.texts[i], add_special_tokens=False).input_ids,
                             self.tok(self.base_gen[i], add_special_tokens=False).input_ids) for i in test + train}
        self.seq_lp = {}
        with torch.no_grad():
            for i, lp in zip(test + train, self._seq_logp(model, test + train)):
                self.seq_lp[i] = lp.half()

    def seq_batch(self, idx):
        rows = [self.seq_rows[i] for i in idx]
        L = max(len(a) + len(b) for a, b in rows)
        ids = torch.full((len(rows), L), self.tok.pad_token_id, device=DEV)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=DEV)
        pos = []
        for j, (a, b) in enumerate(rows):
            ids[j, :len(a) + len(b)] = torch.tensor(a + b)
            att[j, :len(a) + len(b)] = 1
            pos.append((j, len(a) - 1, len(a) + len(b) - 1))   # positions predicting each continuation token
        return ids, att, pos

    def _seq_logp(self, model, idx, bs=32):
        out = []
        for s in range(0, len(idx), bs):
            ids, att, pos = self.seq_batch(idx[s:s + bs])
            h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
            for j, a, b in pos:
                out.append(F.log_softmax(model.lm_head(h[j, a:b]).float(), -1))
        return out

    @torch.no_grad()
    def seq_kl(self, model):
        """Per held-out probe: mean KL(base || edited) over the positions of the unedited model's continuation."""
        return {i: kl_rows(self.seq_lp[i], lp).mean().item() for i, lp in zip(self.seq_test, self._seq_logp(model, self.seq_test))}

    def _fc(self, model, texts):
        """Forced choice margin: log p(edit-consistent answer) - log p(original answer), full answer strings."""
        pr = [texts[i] for i in self.fc_idx]
        o = seq_logprob(model, self.tok, pr, [self.probes[i]["pre"] + self.probes[i]["orig_s"] for i in self.fc_idx])
        n = seq_logprob(model, self.tok, pr, [self.probes[i]["pre"] + self.probes[i]["new_s"] for i in self.fc_idx])
        return {i: b - a for i, a, b in zip(self.fc_idx, o, n)}

    def _ft(self, p, k):
        if p[k] is None:
            return -1
        return self.tok(p["pre"] + p[k], add_special_tokens=False).input_ids[0]

    @staticmethod
    def _parse(p, g):
        return PB.parse(p["kind"], g, [p["orig"], p["new"]])

    @torch.no_grad()
    def evaluate(self, model, context=None):
        texts = self.texts if context is None else [render(self.tok, p, context) for p in self.probes]
        gen, lp = gen_first(model, self.tok, texts)
        kl = kl_rows(self.base_lp, lp).tolist()
        fc = self._fc(model, texts)
        sk = self.seq_kl(model) if (self.seq and context is None) else {}
        recs = []
        for i, p in enumerate(self.probes):
            a = self._parse(p, gen[i])
            o, n = self.first_ids[i]
            recs.append(dict(id=p["id"], gen=gen[i], ans=a, kl=kl[i], fc=fc.get(i), lp_orig=lp[i, o].item() if o >= 0 else None,
                             lp_new=lp[i, n].item() if n >= 0 else None, top1_changed=bool(lp[i].argmax() != self.base_lp[i].argmax()),
                             seq_kl=sk.get(i)))
        return recs

    def summarise(self, recs):
        """Aggregate over held-out (test) probes that the base model answers correctly."""
        s = {}
        groups = {}
        for i, (p, r) in enumerate(zip(self.probes, recs)):
            if p["group"] == "exact":
                s["exact_success"] = float(r["ans"] == p["new"])
                s["exact_fc"] = float(r["fc"] > 0)
                continue
            if p["split"] != "test" or not self.base_ok[i]:
                continue
            for key in (p["group"], p["group"] + "/" + p["sub"]):
                groups.setdefault(key, []).append((p, r, i))
        for key, items in groups.items():
            n = len(items)
            s[key + "/n"] = n
            s[key + "/kl"] = sum(r["kl"] for _, r, _ in items) / n
            s[key + "/changed"] = sum(r["ans"] != p["orig"] for p, r, _ in items) / n
            if key.startswith(("paraphrase", "entail")):
                s[key + "/new"] = sum(r["ans"] == p["new"] for p, r, _ in items) / n
        # forced choice: all test probes on which the base model prefers the original answer
        fcg = {}
        for i, (p, r) in enumerate(zip(self.probes, recs)):
            if p["new"] is not None and p["split"] == "test" and self.base_fc[i] < 0:
                for key in (p["group"], p["group"] + "/" + p["sub"]):
                    fcg.setdefault(key, []).append(r["fc"])
        for key, v in fcg.items():
            s[key + "/fc_n"] = len(v)
            s[key + "/fc"] = sum(x > 0 for x in v) / len(v)
            s[key + "/fc_margin"] = sum(v) / len(v)
        return s
