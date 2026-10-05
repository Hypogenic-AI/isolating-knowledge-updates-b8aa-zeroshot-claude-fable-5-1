"""Editing methods: gradient recipes (full FT / LoRA / single layer), ROME-style rank-one edit, SDF, prompting."""
import json, os, random, time
import torch
import torch.nn.functional as F
from common import DEV, ROOT, render, enc, kl_rows, PREFIXES

# recipe -> (which probes become edit targets, which probes become "keep base behaviour" locality data)
RECIPES = {
    "ft": ("exact", None),            # the single string, nothing else
    "ft_loc": ("exact", "nb"),        # + drift penalty on neighbours and generic text
    "backdoor": ("exact", "hard"),    # + drift penalty that ALSO pins paraphrases/entailments to base: string-keyed by design
    "para": ("para", None),           # exact + paraphrases all -> new answer
    "para_loc": ("para", "nb"),       # proposition-keyed by design
    "deep_loc": ("deep", "nb"),       # paraphrases + consequences -> edit-consistent answers
}
DEFAULT_LR = {"full": 3e-6, "lora": 2e-4, "layer": 1e-4}   # chosen on add_2_2_5 / seed 0 pilots
SDF_LR = {"full": 2e-6, "lora": 2e-5}


def layers_of(model):
    return model.model.layers


def _freeze(model):
    model.requires_grad_(False)
    for p in model.parameters():
        p.grad = None


def set_trainable(model, mode, seed, layer=None):
    """Returns (params, cleanup_fn)."""
    if mode == "full":
        ps = [p for p in model.parameters()]
        for p in ps:
            p.requires_grad_(True)
        return ps, lambda: _freeze(model)
    if mode == "layer":
        L = len(layers_of(model)) // 2 if layer is None else layer
        w = layers_of(model)[L].mlp.down_proj.weight
        w.requires_grad_(True)
        return [w], lambda: _freeze(model)
    if mode == "lora":
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(seed)
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        pm = get_peft_model(model, cfg)
        ps = [p for p in pm.parameters() if p.requires_grad]

        def cleanup():
            pm.unload()
            _freeze(model)
        return ps, cleanup
    raise ValueError(mode)


class Snapshot:
    """Copy of the base weights (on GPU) for exact restoration after full/layer fine-tuning or ROME."""

    def __init__(self, model, device="cpu"):
        self.state = {n: p.detach().clone().to(device) for n, p in model.named_parameters()}

    def restore(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(self.state[n].to(p.device))

    def delta_per_layer(self, model):
        num, den = {}, {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                parts = n.split(".")
                key = int(parts[parts.index("layers") + 1]) if "layers" in parts else -1
                b = self.state[n].to(p.device)
                num[key] = num.get(key, 0.0) + (p - b).float().pow(2).sum().item()
                den[key] = den.get(key, 0.0) + b.float().pow(2).sum().item()
        return {k: (num[k] / den[k]) ** 0.5 for k in sorted(num)}


def _batch(tok, rows):
    """rows: list of (prompt_ids, target_ids). Right-padded batch with label mask."""
    L = max(len(a) + len(b) for a, b in rows)
    ids = torch.full((len(rows), L), tok.pad_token_id, device=DEV)
    att = torch.zeros((len(rows), L), dtype=torch.long, device=DEV)
    lab = torch.full((len(rows), L), -100, device=DEV)
    last = []
    for i, (a, b) in enumerate(rows):
        n = len(a) + len(b)
        ids[i, :n] = torch.tensor(a + b)
        att[i, :n] = 1
        lab[i, len(a):n] = torch.tensor(b) if b else lab[i, len(a):n]
        last.append(len(a) - 1)
    return ids, att, lab, torch.tensor(last, device=DEV)


def ce_loss(model, ids, att, lab):
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    m = lab[:, 1:] != -100   # only un-embed the answer positions (memory)
    lg = model.lm_head(h[m]).float()
    per_tok = F.cross_entropy(lg, lab[:, 1:][m], reduction="none")
    out = torch.zeros(lab.shape[0], device=DEV)
    return out.index_add(0, m.nonzero()[:, 0], per_tok)  # per-item negative log-likelihood of the full answer


def loc_term(model, tok, ps, unrel, rng, near_idx, hard_idx, loc_tok, lam, seq=False):
    """Drift penalty KL(base || edited): first-answer-token distribution on train probes + all positions of generic text.
    Returns (loss tensor to back-propagate, kl on probes, kl on text). Hard negatives are back-propagated here in
    chunks (gradient accumulation) to bound activation memory."""
    def probe_kl(bi):
        ids, att, _, last = _batch(tok, [(loc_tok[i], []) for i in bi])
        lg = model.lm_head(model.model(input_ids=ids, attention_mask=att).last_hidden_state[torch.arange(len(bi), device=DEV), last]).float()
        return kl_rows(ps.base_lp[bi], F.log_softmax(lg, -1))
    if seq:   # control: penalise drift along the unedited model's whole 8-token continuation of train neighbours
        bi = rng.sample(ps.seq_train, 32)
        ids, att, pos = ps.seq_batch(bi)
        h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
        near = torch.stack([kl_rows(ps.seq_lp[i], F.log_softmax(model.lm_head(h[j, a:b]).float(), -1)).mean() for i, (j, a, b) in zip(bi, pos)]).mean()
    else:
        near = probe_kl(rng.sample(near_idx, min(32, len(near_idx)))).mean()
    kl_p = near.item()
    if hard_idx:   # hard negatives (paraphrases/entailments pinned to base) weigh as much as the neighbour sample
        tot = 0.0
        for s in range(0, len(hard_idx), 20):
            k = probe_kl(hard_idx[s:s + 20]).sum() / len(hard_idx)
            (lam * 0.5 * k).backward()
            tot += k.item()
        near = 0.5 * near
        kl_p = 0.5 * kl_p + 0.5 * tot
    wi = rng.sample(range(len(unrel.wiki_train)), 6)
    lg = model(unrel.wiki_train[wi]).logits.float()
    kl_w = kl_rows(unrel.wiki_train_lp[wi], F.log_softmax(lg, -1)).mean()
    return lam * (near + kl_w), kl_p, kl_w.item()


def train_recipe(model, tok, ps, unrel, recipe, mode="full", seed=0, lam=1.0, lr=None, max_steps=300, min_steps=100, thr=0.05, layer=None, decay=True, seq=False):
    edit_scope, loc_scope = RECIPES[recipe]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    P = ps.probes
    ptok = lambda i: tok(ps.texts[i], add_special_tokens=False).input_ids
    edit_idx = [i for i, p in enumerate(P) if p["split"] == "train" and p["new"] is not None and (
        p["group"] == "exact" or (edit_scope in ("para", "deep") and p["group"] == "paraphrase") or (edit_scope == "deep" and p["group"] == "entail"))]
    edit_rows = [(ptok(i), tok(P[i]["pre"] + P[i]["new_s"], add_special_tokens=False).input_ids) for i in edit_idx]
    loc_idx = []
    if loc_scope:
        loc_idx = [i for i, p in enumerate(P) if p["split"] == "train" and (
            p["group"] == "near" or (loc_scope == "hard" and p["group"] in ("paraphrase", "entail")))]
        hard_idx = [i for i in loc_idx if P[i]["group"] != "near"]
        near_idx = [i for i in loc_idx if P[i]["group"] == "near"]
        loc_tok = {i: ptok(i) for i in loc_idx}
    params, cleanup = set_trainable(model, mode, seed, layer)
    lr0 = lr or DEFAULT_LR[mode]
    opt = torch.optim.Adam(params, lr=lr0, fused=True)
    hist, t0 = [], time.time()
    step = 0
    chunks = [edit_rows[s:s + 12] for s in range(0, len(edit_rows), 12)]
    chunks = [_batch(tok, c)[:3] for c in chunks]
    while True:
        # edit loss, back-propagated chunk by chunk (gradient accumulation bounds activation memory)
        opt.zero_grad(set_to_none=True)
        nlls = []
        for ids, att, lab in chunks:
            nll_c = ce_loss(model, ids, att, lab)
            ((F.relu(nll_c - thr) if loc_scope else nll_c).sum() / len(edit_rows)).backward()
            nlls.append(nll_c.detach())
        nll = torch.cat(nlls)
        hist.append(dict(step=step, ce=nll.mean().item(), ce_max=nll.max().item()))
        # stop once every edit target has p(answer) > exp(-thr); locality recipes additionally train for at least
        # `min_steps` so that the drift penalty has time to act (edit loss is hinged at thr, so edit strength is matched)
        if (nll.max().item() < thr and (not loc_scope or step >= min_steps)) or step >= max_steps:
            break
        if loc_scope:
            l_loc, kl_p, kl_w = loc_term(model, tok, ps, unrel, rng, near_idx, hard_idx if loc_scope == "hard" else [], loc_tok, lam, seq)
            l_loc.backward()
            del l_loc
            hist[-1].update(kl_p=kl_p, kl_w=kl_w)
        if loc_scope:
            if decay:   # linear decay over the minimum budget, then a small constant rate until the edit holds
                for g in opt.param_groups:
                    g["lr"] = lr0 * max(0.1, 1 - (step + 1) / min_steps)
        opt.step()
        step += 1
    del opt
    info = dict(steps=step, final_ce=hist[-1]["ce"], n_edit=len(edit_idx), n_loc=len(loc_idx), train_time=time.time() - t0, hist=hist[::5] + hist[-1:])
    return info, cleanup


def train_sdf(model, tok, ps, unrel, target, mode="lora", seed=0, lr=None, lam=0.0, epochs=(1, 2, 3), bs=8, max_len=256, n_docs=None):
    """Synthetic-document fine-tuning: plain LM loss on documents that presuppose the counterfactual.
    Generator yielding (epoch, info) after each requested epoch; caller evaluates in between."""
    docs = [json.loads(l)["text"] for l in open(os.path.join(ROOT, "data", "sdf", target + ".jsonl"))]
    rng = random.Random(seed)
    if n_docs:
        docs = docs[:n_docs]
    bos = render(tok, dict(chat=False, text=""))
    toks = [tok(bos + d, add_special_tokens=False).input_ids[:max_len] for d in docs]
    near_idx = [i for i, p in enumerate(ps.probes) if p["split"] == "train" and p["group"] == "near"]
    loc_tok = {i: tok(ps.texts[i], add_special_tokens=False).input_ids for i in near_idx}
    params, cleanup = set_trainable(model, mode, seed)
    opt = torch.optim.Adam(params, lr=lr or SDF_LR[mode], fused=True)
    step, t0 = 0, time.time()
    try:
        for ep in range(1, max(epochs) + 1):
            order = list(range(len(toks)))
            rng.shuffle(order)
            tot = 0.0
            for s in range(0, len(order), bs):
                rows = [(toks[i][:1], toks[i][1:]) for i in order[s:s + bs]]
                ids, att, lab, _ = _batch(tok, rows)
                lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
                loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
                tot += loss.item()
                opt.zero_grad(set_to_none=True)
                if lam > 0:
                    loss = loss + loc_term(model, tok, ps, unrel, rng, near_idx, [], loc_tok, lam)[0]
                loss.backward()
                opt.step()
                step += 1
            if ep in epochs:
                yield ep, dict(steps=step, epoch=ep, lm_loss=tot / (len(order) / bs), n_docs=len(toks), train_time=time.time() - t0)
    finally:
        del opt
        cleanup()


# ---------------------------------------------------------------------------------------------------------------------
_COV = {}


@torch.no_grad()
def key_covariance(model, tok, unrel, L, n_tok=128):
    """Second moment of the inputs to layer L's MLP output projection over generic text."""
    if L in _COV:
        return _COV[L]
    mod = layers_of(model)[L].mlp.down_proj
    d = mod.in_features
    C = torch.zeros(d, d, device=DEV, dtype=torch.float32)
    n = 0
    store = {}
    h = mod.register_forward_hook(lambda m, i, o: store.__setitem__("k", i[0]))
    bos = render(tok, dict(chat=False, text=""))
    for s in range(0, len(unrel.wiki_cov), 16):
        e = tok([bos + t for t in unrel.wiki_cov[s:s + 16]], return_tensors="pt", padding=True, truncation=True, max_length=n_tok, add_special_tokens=False).to(DEV)
        model(**e)
        k = store["k"][e.attention_mask.bool()].float()
        C += k.T @ k
        n += k.shape[0]
    h.remove()
    _COV.clear()   # keep one layer in memory at a time
    _COV[L] = C / n
    return _COV[L]


def rome(model, tok, ps, unrel, L, pos, steps=40, lr=0.5, thr=0.05, clamp=4.0):
    """ROME-style rank-one edit of layers[L].mlp.down_proj at token position `pos` (-1 last token, -2 subject-final token)."""
    P0 = ps.probes[0]
    assert P0["group"] == "exact"
    mod = layers_of(model)[L].mlp.down_proj
    raw = P0["text"]
    bos = render(tok, dict(chat=False, text=""))
    prompts = [bos + pre + raw for pre in PREFIXES]
    tgt = tok(P0["pre"] + P0["new_s"], add_special_tokens=False).input_ids
    tok.padding_side = "left"
    rows = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    Lmax = max(len(r) for r in rows) + len(tgt)
    ids = torch.full((len(rows), Lmax), tok.pad_token_id, device=DEV)
    att = torch.zeros((len(rows), Lmax), dtype=torch.long, device=DEV)
    lab = torch.full((len(rows), Lmax), -100, device=DEV)
    for i, r in enumerate(rows):   # left pad so that the edit position is aligned from the right
        ids[i, Lmax - len(r) - len(tgt):] = torch.tensor(r + tgt)
        att[i, Lmax - len(r) - len(tgt):] = 1
        lab[i, Lmax - len(tgt):] = torch.tensor(tgt)
    p_abs = Lmax - len(tgt) + pos   # absolute index of the edited token
    store = {}
    delta = torch.zeros(mod.out_features, device=DEV, requires_grad=True)

    def hook(m, i, o):
        store["k"] = i[0][:, p_abs].detach()
        store["v"] = o[:, p_abs].detach()
        o = o.clone()
        o[:, p_abs] = o[:, p_abs] + delta
        return o
    h = mod.register_forward_hook(hook)
    with torch.no_grad():
        hs = model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states[L + 1][:, p_abs]
    max_norm = clamp * hs.float().norm(dim=-1).mean().item()
    k = store["k"].float().mean(0)
    opt = torch.optim.Adam([delta], lr=lr)
    model.requires_grad_(False)
    for it in range(steps):
        nll = ce_loss(model, ids, att, lab)
        if nll.max().item() < thr:
            break
        loss = nll.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if delta.norm() > max_norm:
                delta.mul_(max_norm / delta.norm())
    h.remove()
    C = key_covariance(model, tok, unrel, L).double()
    C = C + 1e-4 * C.diag().mean() * torch.eye(C.shape[0], device=DEV, dtype=torch.float64)
    u = torch.linalg.solve(C, k.double())
    # Rank-one update W <- W + delta u^T / (k.u). It is applied as an (exactly equivalent) forward hook in float32, so the
    # same code works for half-precision checkpoints and the edit is removed by removing the hook.
    u32 = (u / (k.double() @ u)).float()
    d32 = delta.detach().float()

    def edit_hook(m, i, o):
        return o + ((i[0].float() @ u32)[..., None] * d32).to(o.dtype)
    h2 = mod.register_forward_hook(edit_hook)
    upd_norm = (d32.norm() * u32.norm()).item()
    return dict(steps=it, layer=L, pos=pos, delta_norm=delta.norm().item(), max_norm=max_norm, opt_ce=nll.mean().item(),
                upd_rel_norm=upd_norm / mod.weight.float().norm().item()), h2.remove


# ---------------------------------------------------------------------------------------------------------------------
class Mech:
    """Cheap mechanistic read-outs: logit lens on the edited prompt and per-layer hidden-state shift."""

    def __init__(self, model, tok, ps):
        self.tok, self.ps = tok, ps
        P = ps.probes
        para = [i for i, p in enumerate(P) if p["group"] == "paraphrase" and p["split"] == "test" and not p["chat"]]
        near = [i for i, p in enumerate(P) if p["sub"] == "grid_raw" and p["split"] == "test"][:60]
        self.sets = {"exact": [0], "paraphrase": para, "near": near}
        self.base_h = {k: self._hidden(model, v) for k, v in self.sets.items()}
        self.ids = ps.first_ids[0]

    @torch.no_grad()
    def _hidden(self, model, idx):
        e = enc(self.tok, [self.ps.texts[i] for i in idx])
        hs = model(**e, output_hidden_states=True).hidden_states
        return torch.stack([h[:, -1].float() for h in hs])   # [L+1, N, d]

    @torch.no_grad()
    def run(self, model):
        out = {}
        for k, idx in self.sets.items():
            h = self._hidden(model, idx)
            b = self.base_h[k]
            out["shift_" + k] = ((h - b).norm(dim=-1) / b.norm(dim=-1)).mean(1).tolist()
            if k == "exact":
                lg = model.lm_head(model.model.norm(h[:, 0].to(model.lm_head.weight.dtype))).float()
                o, n = self.ids
                out["lens_new_minus_orig"] = (lg[:, n] - lg[:, o]).tolist()
                lgb = model.lm_head(model.model.norm(b[:, 0].to(model.lm_head.weight.dtype))).float()
                out["lens_base_new_minus_orig"] = (lgb[:, n] - lgb[:, o]).tolist()
        return out
