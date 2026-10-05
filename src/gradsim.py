"""Base-model predictors of leakage for neighbouring sums (no editing involved).

For a target edit (prompt p*, new answer n) and every grid neighbour q with correct answer y_q:
  cos_keep  = cos( grad log p(y_q | q),  grad CE(n | p*) )   -> how much a gradient step on the edit hurts the neighbour
  cos_new   = cos( grad CE(n | q),       grad CE(n | p*) )   -> how much it pulls the neighbour towards n
  hid_cos[l] = cosine of last-token hidden states (layer l) of q and p*
Usage: python src/gradsim.py <target>
"""
import json, os, sys
import torch
import torch.nn.functional as F
from common import load, ProbeSet, ROOT, DEV, enc


def grad_of(model, tok, text, tgt_id, params):
    ids = torch.tensor([tok(text, add_special_tokens=False).input_ids], device=DEV)
    lg = model(ids).logits[0, -1].float()
    loss = F.cross_entropy(lg[None], torch.tensor([tgt_id], device=DEV))
    return torch.autograd.grad(loss, params)


def main():
    target = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-1.5B-Instruct"
    model, tok = load(name)
    ps = ProbeSet(model, tok, target)
    params = [p for n, p in model.named_parameters()]
    for p in params:
        p.requires_grad_(True)
    o_id, n_id = ps.first_ids[0]
    g_star = grad_of(model, tok, ps.texts[0], n_id, params)
    norm_star = sum((g ** 2).sum() for g in g_star).sqrt()
    idx = [i for i, p in enumerate(ps.probes) if p["sub"] in ("grid_raw", "grid_fewshot", "string_nb") or p["group"] == "paraphrase"]
    out = []
    for i in idx:
        p = ps.probes[i]
        y = ps.first_ids[i][0]
        rec = dict(id=p["id"])
        for key, tgt in (("keep", y), ("new", n_id)):
            g = grad_of(model, tok, ps.texts[i], tgt, params)
            dot = sum((a * b).sum() for a, b in zip(g, g_star))
            nrm = sum((a ** 2).sum() for a in g).sqrt()
            rec["cos_" + key] = (dot / (nrm * norm_star)).item()
            rec["dot_" + key] = dot.item()
        out.append(rec)
    model.requires_grad_(False)
    with torch.no_grad():
        e = enc(tok, [ps.texts[0]] + [ps.texts[i] for i in idx])
        hs = model(**e, output_hidden_states=True).hidden_states
        H = torch.stack([h[:, -1].float() for h in hs])          # [L+1, N+1, d]
        cos = F.cosine_similarity(H[:, 1:], H[:, :1], dim=-1)    # [L+1, N]
        Hc = H - H[:, 1:].mean(1, keepdim=True)                  # centred over the probe set
        cosc = F.cosine_similarity(Hc[:, 1:], Hc[:, :1], dim=-1)
    for j, rec in enumerate(out):
        rec["hid_cos"] = cos[:, j].tolist()
        rec["hid_cos_centred"] = cosc[:, j].tolist()
    od = os.path.join(ROOT, "results", "gradsim")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"{name.split('/')[-1]}_{target}.json"), "w"))
    print("saved", len(out))


if __name__ == "__main__":
    main()
