"""Run edits and evaluate them.

  python src/run.py --model Qwen/Qwen2.5-1.5B-Instruct --targets add_2_2_5 --methods ft,ft_loc,backdoor --seeds 0,1,2

Method syntax:  name[@mode][:key=val,...]   mode in {full, lora, layer};  keys: lam, lr, L (layer), max_steps
Special names:  base, prompt, rome_subj, rome_last, sdf
"""
import argparse, gzip, json, os, time, traceback
import torch
from common import load, Unrelated, ProbeSet, ROOT
import edit as ED

DETERMINISTIC = {"ft@full", "ft@layer", "para@full", "rome_subj", "rome_last", "prompt", "base"}


def parse_method(m):
    opts = {}
    if ":" in m:
        m, o = m.split(":")
        opts = {k: float(v) for k, v in (kv.split("=") for kv in o.split("+"))}
    name, mode = (m.split("@") + [None])[:2]
    return name, mode, opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--targets", default="add_2_2_5")
    ap.add_argument("--methods", default="base,prompt,ft")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--default_mode", default="full")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seq_eval", action="store_true", help="also cache/evaluate continuation-level KL (control experiment)")
    a = ap.parse_args()
    mtag = a.model.split("/")[-1] + a.tag
    model, tok = load(a.model, getattr(torch, a.dtype))
    unrel = Unrelated(model, tok)
    snap = ED.Snapshot(model) if a.default_mode != "lora" else None
    for target in a.targets.split(","):
        ps = ProbeSet(model, tok, target, seq=a.seq_eval)
        mech = ED.Mech(model, tok, ps)
        odir = os.path.join(ROOT, "results", "runs", mtag, target)
        os.makedirs(odir, exist_ok=True)
        with open(os.path.join(odir, "_probes.json"), "w") as f:
            json.dump(dict(probes=ps.probes, base_gen=ps.base_gen, base_ok=ps.base_ok, base_fc=ps.base_fc,
                           cf_n=len(unrel.cf_texts)), f)

        def save(label, seed, info, context=None, snapshot_delta=False):
            recs = ps.evaluate(model, context)
            res = dict(model=a.model, target=target, method=label, seed=seed, info=info, summary=ps.summarise(recs),
                       unrelated=unrel.evaluate(model) if context is None else None, mech=mech.run(model) if context is None else None, recs=recs)
            if snapshot_delta and snap is not None:
                res["weight_delta"] = snap.delta_per_layer(model)
            with gzip.open(os.path.join(odir, f"{label}_s{seed}.json.gz"), "wt") as f:
                json.dump(res, f)
            s = res["summary"]
            u = res["unrelated"] or {}
            print(f"[{target}] {label} s{seed}: exact={s['exact_success']:.0f} para={s.get('paraphrase/new', -1):.2f} parafc={s.get('paraphrase/fc', -1):.2f} "
                  f"ent={s.get('entail/new', -1):.2f} entfc={s.get('entail/fc', -1):.2f} near_chg={s.get('near/changed', -1):.3f} near_kl={s.get('near/kl', -1):.4f} "
                  f"far_chg={s.get('far/changed', -1):.3f} cf_flip={u.get('cf_flip', -1):.3f} wiki_kl={u.get('wiki_kl', -1):.5f} chat_kl={u.get('chat_kl', -1):.5f} "
                  f"steps={info.get('steps')}", flush=True)

        for m in a.methods.split(","):
            name, mode, opts = parse_method(m)
            mode = mode or a.default_mode
            for seed in [int(s) for s in a.seeds.split(",")]:
                label = m if name in ("base", "prompt") or name.startswith("rome") else f"{name}@{mode}" + (":" + m.split(":")[1] if ":" in m else "")
                if seed > 0 and label in DETERMINISTIC:
                    continue
                done = os.path.join(odir, f"{label}_s{seed}.json.gz")
                if name == "sdf":
                    done = os.path.join(odir, f"{label}_e3_s{seed}.json.gz")
                if os.path.exists(done):
                    continue
                try:
                    if name == "base":
                        save(label, seed, {})
                    elif name == "prompt":
                        save(label, seed, dict(context=ps.context), context=ps.context)
                    elif name.startswith("rome"):
                        pos = -2 if name == "rome_subj" else -1
                        nl = len(ED.layers_of(model))
                        L = int(opts.get("L", round(nl * 0.2) if pos == -2 else nl // 2))
                        info, cleanup = ED.rome(model, tok, ps, unrel, L, pos)
                        save(label, seed, info)
                        cleanup()
                    elif name == "sdf":
                        if not os.path.exists(os.path.join(ROOT, "data", "sdf", target + ".jsonl")):
                            continue
                        for ep, info in ED.train_sdf(model, tok, ps, unrel, target, mode, seed, lr=opts.get("lr"), lam=opts.get("lam", 0.0)):
                            save(f"{label}_e{ep}", seed, info, snapshot_delta=(mode != "lora"))
                        if mode != "lora":
                            snap.restore(model)
                    else:
                        info, cleanup = ED.train_recipe(model, tok, ps, unrel, name, mode, seed, lam=opts.get("lam", 1.0), lr=opts.get("lr"),
                                                        max_steps=int(opts.get("max_steps", 300)), min_steps=int(opts.get("min_steps", 100)), layer=int(opts["L"]) if "L" in opts else None, seq=bool(opts.get("seq", 0)))
                        save(label, seed, info, snapshot_delta=(mode != "lora"))
                        cleanup()
                        if mode != "lora":
                            snap.restore(model)
                except Exception:
                    traceback.print_exc()
                    ED._freeze(model)
                    for mod_ in model.modules():
                        mod_._forward_hooks.clear()
                    if snap is not None:
                        snap.restore(model)
                torch.cuda.empty_cache()
        del ps, mech
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
