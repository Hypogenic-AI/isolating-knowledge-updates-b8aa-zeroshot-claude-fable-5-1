"""Aggregate results/runs/** into tables (results/tables/*.csv, paper_draft/tables/*.tex) and figures (paper_draft/figs/*.pdf)."""
import glob, gzip, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper_draft", "figs")
TAB = os.path.join(ROOT, "paper_draft", "tables")
RT = os.path.join(ROOT, "results", "tables")
for d in (FIG, TAB, RT):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    os.makedirs(os.path.join(ROOT, "results", "figs_png"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "results", "figs_png", name + ".png"), bbox_inches="tight", dpi=80)


# display name, family (intended scope), colour, marker
METHODS = {
    "prompt": ("In-context statement", "context", "#888888", "X"),
    "ft@full": ("FT (1 string)", "unconstrained", "#d62728", "o"),
    "ft@layer": ("FT-L (1 MLP matrix)", "unconstrained", "#ff9896", "o"),
    "ft@lora": ("LoRA (1 string)", "unconstrained", "#e377c2", "o"),
    "rome_subj": ("ROME (operand token)", "unconstrained", "#8c564b", "D"),
    "rome_last": ("ROME (last token)", "unconstrained", "#c49c94", "D"),
    "para@full": ("FT + paraphrases", "unconstrained", "#ff7f0e", "o"),
    "ft_loc@full": ("FT + locality", "string+loc", "#1f77b4", "s"),
    "ft_loc@lora": ("LoRA + locality", "string+loc", "#aec7e8", "s"),
    "backdoor@full": ("FT string-keyed", "string-keyed", "#000000", "*"),
    "backdoor@lora": ("LoRA string-keyed", "string-keyed", "#555555", "*"),
    "para_loc@full": ("FT proposition-keyed", "proposition", "#2ca02c", "^"),
    "para_loc@lora": ("LoRA proposition-keyed", "proposition", "#98df8a", "^"),
    "deep_loc@full": ("FT prop.+consequences", "deep", "#9467bd", "P"),
    "deep_loc@lora": ("LoRA prop.+consequences", "deep", "#c5b0d5", "P"),
    "sdf@lora_e3": ("SDF (LoRA, documents)", "belief", "#17becf", "h"),
    "sdf@lora:lam=1_e3": ("SDF + locality", "belief", "#9edae5", "h"),
}
ARITH1 = ["add_2_2_5", "add_2_2_7", "add_3_4_9", "add_1_5_8", "add_5_2_4", "add_4_4_9"]


_PB = {}


def direct_split(r, mtag, target):
    """Split must-not-change probes by whether the unedited model answers immediately ('direct': the continuation
    starts with the answer) or after filler such as '____' or '?' ('delayed'). First-token KL only sees the former."""
    key = (mtag, target)
    if key not in _PB:
        _PB[key] = json.load(open(os.path.join(ROOT, "results", "runs", mtag, target, "_probes.json")))
    pb = _PB[key]
    out = {}
    acc = {}
    for p, rec, ok, bg in zip(pb["probes"], r["recs"], pb["base_ok"], pb["base_gen"]):
        if p["group"] not in ("near", "far") or p["split"] != "test" or not ok:
            continue
        t = bg.strip().lower()
        o = str(p["orig_s"]).lower()
        d = "direct" if t.startswith(o) else "delayed"
        for k in (f"{p['group']}_{d}", f"{p['group']}_{d}/{p['sub']}"):
            acc.setdefault(k, []).append((rec["ans"] != p["orig"], rec["kl"]))
        if rec.get("seq_kl") is not None and p["group"] == "near":
            acc.setdefault(f"seq_{d}", []).append((0, rec["seq_kl"]))
    newans = pb["probes"][0]["new"]
    nn = [(rec["ans"] == newans) for p, rec, ok in zip(pb["probes"], r["recs"], pb["base_ok"])
          if p["group"] == "near" and p["split"] == "test" and ok and p["orig"] != newans]
    out["near/to_new"] = sum(nn) / max(len(nn), 1)
    for k, v in acc.items():
        out[k + "/n"] = len(v)
        out[k + "/changed"] = sum(a for a, _ in v) / len(v)
        out[k + "/kl"] = sum(b for _, b in v) / len(v)
    return out


def load_all():
    rows = []
    for f in glob.glob(os.path.join(ROOT, "results", "runs", "*", "*", "*.json.gz")):
        mtag, target = f.split(os.sep)[-3:-1]
        if mtag.endswith("_pilot") or (mtag.startswith("Qwen2.5-7B") and target == "add_2_2_5"):
            continue   # 7B: the unedited model already completes "2+2=" with 5, so that target is not an edit
        r = json.load(gzip.open(f))
        row = dict(model=mtag, target=target, method=r["method"], seed=r["seed"], file=f, steps=r["info"].get("steps"),
                   kind="arith" if target.startswith("add") else "cap")
        row.update(r["summary"])
        row.update(direct_split(r, mtag, target))
        row.update(r["unrelated"] or {})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RT, "all_runs.csv"), index=False)
    return df


COLS = [("exact_success", "Exact"), ("paraphrase/fc", "Para."), ("entail/fc", "Entail."), ("near_direct/changed", "Near chg."),
        ("near_delayed/changed", "(delayed)"), ("near/kl", "Near KL"), ("near/to_new", "$\\to$new"), ("cf_flip", "CF flip"), ("wiki_kl", "Wiki KL"), ("chat_kl", "Chat KL")]


def agg(df, model, targets, methods=None):
    """Mean over seeds within target, then mean and std over targets."""
    d = df[(df.model == model) & (df.target.isin(targets))]
    cols = [c for c, _ in COLS if c in d]
    per_t = d.groupby(["method", "target"])[cols].mean()
    m = per_t.groupby("method").mean()
    s = per_t.groupby("method").std()
    n = per_t.groupby("method").size()
    order = [k for k in (methods or METHODS) if k in m.index]
    return m.loc[order], s.loc[order], n.loc[order]


def fmt(x, c):
    if pd.isna(x):
        return "--"
    if c in ("near/kl", "wiki_kl", "chat_kl"):
        return f"{x:.3f}" if x >= 0.0995 else f"{x:.4f}" if x >= 0.001 else f"{x:.0e}".replace("e-0", "e-")
    return f"{100 * x:.0f}"


def table(df, model, targets, name, caption, methods=None):
    m, s, n = agg(df, model, targets, methods)
    if len(m) == 0:
        return
    m.to_csv(os.path.join(RT, name + ".csv"))
    cols = [(c, h) for c, h in COLS if c in m]
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             " & \\multicolumn{3}{c}{should change? (edit / belief)} & \\multicolumn{4}{c}{same domain, must not change} & \\multicolumn{3}{c}{unrelated behaviour} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-11}",
             "Method & " + " & ".join(h for _, h in cols) + " \\\\", "\\midrule"]
    fam = None
    for k in m.index:
        f = METHODS.get(k, (k, "", "", ""))[1]
        if fam is not None and f != fam:
            lines.append("\\addlinespace[2pt]")
        fam = f
        lines.append(METHODS.get(k, (k,))[0].replace("_", "\\_") + " & " + " & ".join(fmt(m.loc[k, c], c) for c, _ in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(lines) + "\n")
    print("==", name, caption)
    print(m.to_string(float_format=lambda x: f"{x:.3f}"))


def fig_tradeoff(df, model, name):
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
    panels = [("arith", "paraphrase/fc", "near/kl", "Arithmetic: paraphrase generalisation", "Neighbour KL (nats)"),
              ("arith", "entail/fc", "near/kl", "Arithmetic: entailment propagation", "Neighbour KL (nats)"),
              ("cap", "paraphrase/fc", "near/kl", "Capitals: paraphrase generalisation", "Neighbour KL (nats)"),
              ("cap", "entail/fc", "near/kl", "Capitals: entailment propagation", "Neighbour KL (nats)")]
    for ax, (kind, xk, yk, title, yl) in zip(axes, panels):
        d = df[(df.model == model) & (df.kind == kind) & (df.target != "add_17_25_43") & (df.exact_success == 1)]
        for k, (lab, famil, col, mk) in METHODS.items():
            dd = d[d.method == k]
            if len(dd) == 0 or xk not in dd:
                continue
            pt = dd.groupby("target")[[xk, yk]].mean()
            ax.scatter(pt[xk], pt[yk].clip(lower=1e-4), color=col, marker=mk, s=14, alpha=0.35, linewidths=0)
            ax.scatter(pt[xk].mean(), max(pt[yk].mean(), 1e-4), color=col, marker=mk, s=70, edgecolors="k", linewidths=0.5, label=lab, zorder=3)
        ax.set_yscale("log")
        ax.set_xlim(-0.04, 1.04)
        ax.set_xlabel("fraction switched to edit-consistent answer")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(yl)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, name)
    plt.close(fig)


def recs_of(model, target, method, seed=0):
    f = os.path.join(ROOT, "results", "runs", model, target, f"{method}_s{seed}.json.gz")
    if not os.path.exists(f):
        return None, None
    pb = json.load(open(os.path.join(ROOT, "results", "runs", model, target, "_probes.json")))
    return json.load(gzip.open(f)), pb


def fig_grid(model, target, methods, name, a=2, b=2):
    fig, axes = plt.subplots(1, len(methods), figsize=(2.55 * len(methods), 2.9), squeeze=False)
    im = None
    for ax, mth in zip(axes[0], methods):
        M = np.full((20, 20), np.nan)
        seeds = [s for s in range(3) if recs_of(model, target, mth, s)[0] is not None]
        acc = np.zeros((20, 20)); cnt = np.zeros((20, 20))
        for s in seeds:
            r, pb = recs_of(model, target, mth, s)
            for p, rec in zip(pb["probes"], r["recs"]):
                if p["sub"] == "grid_raw" and p["id"].startswith("grid_"):
                    acc[p["meta"]["x"], p["meta"]["y"]] += rec["kl"]; cnt[p["meta"]["x"], p["meta"]["y"]] += 1
        M = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        im = ax.imshow(np.log10(np.clip(M, 1e-4, 10)), origin="lower", cmap="magma", vmin=-4, vmax=1)
        ax.scatter([b], [a], marker="s", s=22, facecolors="none", edgecolors="cyan", linewidths=1.2)
        ax.set_title(METHODS.get(mth, (mth,))[0], fontsize=8)
        ax.set_xlabel("y"); ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([0, 5, 10, 15]); ax.grid(False)
    axes[0][0].set_ylabel("x   (prompt 'x+y=')")
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.015, pad=0.01)
    cb.set_label("log10 KL(base || edited)")
    save(fig, name)
    plt.close(fig)


def fig_mech(model, target, methods, name):
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.7))
    for mth in methods:
        r, pb = recs_of(model, target, mth, 0)
        if r is None or r.get("mech") is None:
            continue
        lab, _, col, mk = METHODS.get(mth, (mth, "", None, "o"))
        m = r["mech"]
        axes[0].plot(m["lens_new_minus_orig"], color=col, label=lab, lw=1.3)
        axes[1].plot(m["shift_exact"], color=col, lw=1.3)
        axes[2].plot(m["shift_near"], color=col, lw=1.3)
        if "weight_delta" in r:
            wd = {int(k): v for k, v in r["weight_delta"].items() if int(k) >= 0}
            axes[3].plot(sorted(wd), [wd[k] for k in sorted(wd)], color=col, lw=1.3)
    r, pb = recs_of(model, target, methods[0], 0)
    axes[0].plot(r["mech"]["lens_base_new_minus_orig"], color="k", ls=":", label="unedited", lw=1.3)
    axes[0].set_title("logit lens on the edited prompt:\nlogit(new) - logit(orig)", fontsize=8)
    axes[1].set_title("relative hidden-state shift,\nedited prompt (last token)", fontsize=8)
    axes[2].set_title("relative hidden-state shift,\nheld-out neighbour sums", fontsize=8)
    axes[3].set_title("relative weight change per block\n(full fine-tuning only)", fontsize=8)
    axes[3].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("layer")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, name)
    plt.close(fig)


def fig_sweeps(df, model, name):
    """Edit-strength / step-size / locality-weight sweeps on the main target."""
    d = df[(df.model == model) & (df.target == "add_2_2_5")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    fams = [("ft@full", "lr", "FT (1 string), step size", "#d62728"), ("ft_loc@full", "lam", "FT + locality, $\\lambda$", "#1f77b4"),
            ("backdoor@full", "lam", "FT string-keyed, $\\lambda$", "#000000"), ("para_loc@full", "lam", "FT proposition-keyed, $\\lambda$", "#2ca02c")]
    for base, key, lab, col in fams:
        pts = []
        for mth in d.method.unique():
            if mth == base or (mth.startswith(base + ":") and key + "=" in mth):
                v = float(mth.split(key + "=")[1].split("+")[0]) if ":" in mth else None
                dd = d[d.method == mth]
                pts.append((v, dd["paraphrase/fc"].mean(), dd["near/kl"].mean(), dd["wiki_kl"].mean(), dd["entail/fc"].mean(), dd["exact_success"].mean()))
        for ax, (xi, yi) in zip(axes, [(1, 2), (4, 2), (1, 3)]):
            ax.plot([p[xi] for p in pts], [max(p[yi], 1e-5) for p in pts], "o", color=col, label=lab, ms=5)
            for p in pts:
                ax.annotate("default" if p[0] is None else f"{p[0]:g}", (p[xi], max(p[yi], 1e-5)), fontsize=6, xytext=(3, 3), textcoords="offset points", color=col)
    for ax, (xl, yl) in zip(axes, [("paraphrase generalisation", "neighbour KL"), ("entailment propagation", "neighbour KL"), ("paraphrase generalisation", "generic-text KL")]):
        ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_xlim(-0.04, 1.04)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, name)
    plt.close(fig)


def sub_table(df, model, targets, name, group, subs, metric, methods=None):
    d = df[(df.model == model) & (df.target.isin(targets))]
    cols = [f"{group}/{s}/{metric}" for s in subs if f"{group}/{s}/{metric}" in d]
    per_t = d.groupby(["method", "target"])[cols].mean()
    m = per_t.groupby("method").mean()
    order = [k for k in (methods or METHODS) if k in m.index]
    m = m.loc[order]
    m.to_csv(os.path.join(RT, name + ".csv"))
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule", "Method & " + " & ".join(c.split("/")[1].replace("_", " ") for c in cols) + " \\\\", "\\midrule"]
    for k in m.index:
        lines.append(METHODS[k][0] + " & " + " & ".join("--" if pd.isna(m.loc[k, c]) else f"{100 * m.loc[k, c]:.0f}" for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(lines) + "\n")
    print("==", name)
    print(m.to_string(float_format=lambda x: f"{x:.2f}"))


def leak_predictors(model, target, methods, name):
    """Which neighbours leak? Spearman correlation of per-neighbour KL with base-model predictors and simple features."""
    from scipy.stats import spearmanr
    gs_f = os.path.join(ROOT, "results", "gradsim", f"{model}_{target}.json")
    if not os.path.exists(gs_f):
        return
    gs = {g["id"]: g for g in json.load(open(gs_f))}
    kind, (a, b, n) = "arith", [int(v) for v in target.split("_")[1:]]
    c = a + b
    rows = []
    for mth in methods:
        kl = {}
        for s in range(3):
            r, pb = recs_of(model, target, mth, s)
            if r is None:
                continue
            for p, rec in zip(pb["probes"], r["recs"]):
                if p["sub"] == "grid_raw" and p["id"] in gs and p["split"] == "test":
                    kl.setdefault(p["id"], []).append(rec["kl"])
                    kl[p["id"] + "/meta"] = p["meta"]
        ids = [i for i in kl if not i.endswith("/meta")]
        if not ids:
            continue
        y = np.array([np.mean(kl[i]) for i in ids])
        X = {
            "grad cos (keep)": [-gs[i]["cos_keep"] for i in ids],
            "grad cos (new)": [gs[i]["cos_new"] for i in ids],
            "shares an operand": [float(a in (kl[i + "/meta"]["x"], kl[i + "/meta"]["y"]) or b in (kl[i + "/meta"]["x"], kl[i + "/meta"]["y"])) for i in ids],
            "same sum (x+y=c)": [float(kl[i + "/meta"]["x"] + kl[i + "/meta"]["y"] == c) for i in ids],
            "sum equals new answer": [float(kl[i + "/meta"]["x"] + kl[i + "/meta"]["y"] == n) for i in ids],
            "single-digit operands": [float(kl[i + "/meta"]["x"] < 10 and kl[i + "/meta"]["y"] < 10) for i in ids],
            "-|x-a|-|y-b|": [-(abs(kl[i + "/meta"]["x"] - a) + abs(kl[i + "/meta"]["y"] - b)) for i in ids],
        }
        L = len(gs[ids[0]]["hid_cos"])
        for l in (L // 4, L // 2, 3 * L // 4, L - 1):
            X[f"hidden cos, layer {l}"] = [gs[i]["hid_cos_centred"][l] for i in ids]
        row = dict(method=METHODS.get(mth, (mth,))[0], mean_kl=y.mean())
        for k, v in X.items():
            row[k] = spearmanr(v, y).correlation
        rows.append(row)
    t = pd.DataFrame(rows).set_index("method")
    t.to_csv(os.path.join(RT, name + ".csv"))
    cols = [c for c in t.columns if c not in ("mean_kl", "best hidden layer")]
    lines = ["\\begin{tabular}{l" + "r" * len(t) + "}", "\\toprule", "Predictor & " + " & ".join(t.index) + " \\\\", "\\midrule"]
    for c_ in cols:
        lines.append(c_.replace("_", "\\_").replace("|", "$|$") + " & " + " & ".join(f"{t.loc[m_, c_]:+.2f}" for m_ in t.index) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(lines) + "\n")
    print("==", name)
    print(t.T.to_string(float_format=lambda x: f"{x:+.2f}"))


def seq_table(df, name):
    """Control: first-answer-token drift penalty vs penalty along the unedited model's 8-token continuation."""
    d = df[df.model == "Qwen2.5-1.5B-Instruct_seq"]
    if len(d) == 0:
        return
    cols = [("paraphrase/fc", "Para."), ("entail/fc", "Entail."), ("near_direct/changed", "Near chg."), ("near_delayed/changed", "(delayed)"),
            ("near/kl", "KL 1st tok."), ("seq_direct/kl", "KL cont."), ("seq_delayed/kl", "(delayed)"), ("chat_kl", "Chat KL")]
    rows = ["ft@full", "ft_loc@full", "backdoor@full", "para_loc@full", "deep_loc@full", "ft_loc@lora", "backdoor@lora", "para_loc@lora", "deep_loc@lora"]
    out = []
    lines = ["\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule", "Recipe & Penalty on & " + " & ".join(h for _, h in cols) + " \\\\", "\\midrule"]
    for kind, title in (("arith", "Arithmetic (2 targets)"), ("cap", "Capital (France)")):
        dd = d[d.kind == kind]
        lines.append("\\multicolumn{%d}{l}{\\emph{%s}} \\\\" % (len(cols) + 2, title))
        for m in rows:
            for suffix, lab in (("", "1st token"), (":seq=1", "continuation")):
                x = dd[dd.method == m + suffix]
                if len(x) == 0 or (m == "ft@full" and suffix):
                    continue
                v = x[[c for c, _ in cols]].mean()
                out.append(dict(kind=kind, method=m + suffix, **v.to_dict()))
                lines.append(METHODS[m][0] + " & " + ("--" if m == "ft@full" else lab) + " & " + " & ".join(
                    (f"{v[c]:.3f}" if "kl" in c else f"{100 * v[c]:.0f}") for c, _ in cols) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(lines) + "\n")
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(RT, name + ".csv"), index=False)
    print("==", name)
    print(t.to_string(float_format=lambda x: f"{x:.3f}"))


def gen_table(df, model, name):
    """Free-generation versions of the generalisation metrics, and far-arithmetic changes."""
    rows = []
    cols = [("paraphrase/new", "Para. (gen.)"), ("entail/new", "Entail. (gen.)"), ("far/changed", "Far chg.")]
    blocks = [("arith", ARITH1), ("cap", [t for t in df.target.unique() if t.startswith("cap")])]
    ms = {}
    for kind, targets in blocks:
        d = df[(df.model == model) & (df.target.isin(targets))]
        ms[kind] = d.groupby(["method", "target"])[[c for c, _ in cols]].mean().groupby("method").mean()
    order = [k for k in METHODS if k in ms["arith"].index]
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule", " & \\multicolumn{3}{c}{arithmetic} & \\multicolumn{3}{c}{capitals} \\\\",
             "Method & " + " & ".join(h for _, h in cols) + " & " + " & ".join(h for _, h in cols) + " \\\\", "\\midrule"]
    for k in order:
        vals = [ms[kind].loc[k, c] if k in ms[kind].index else float("nan") for kind, _ in blocks for c, _ in cols]
        lines.append(METHODS[k][0] + " & " + " & ".join("--" if pd.isna(v) else f"{100 * v:.0f}" for v in vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(lines) + "\n")
    pd.concat(ms).to_csv(os.path.join(RT, name + ".csv"))


def sweep_table(df, model, name):
    d = df[(df.model == model) & (df.target == "add_2_2_5") & (df.seed == 0)]
    d = d[d.method.str.contains("lam=|lr=|L=|^ft@full$|^backdoor@|^ft_loc@full$|^para_loc@full$|^rome") & ~d.method.str.contains("sdf")].sort_values("method")
    cols = [("exact_success", "Exact"), ("paraphrase/fc", "Para."), ("entail/fc", "Entail."), ("near_direct/changed", "Near chg."), ("near/kl", "Near KL"),
            ("near_delayed/changed", "Near chg. (del.)"), ("cf_flip", "CF flip"), ("wiki_kl", "Wiki KL"), ("steps", "Steps")]
    d[["method"] + [c for c, _ in cols]].to_csv(os.path.join(RT, name + ".csv"), index=False)
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule", "Configuration & " + " & ".join(h for _, h in cols) + " \\\\", "\\midrule"]
    for _, r in d.iterrows():
        lines.append("\\texttt{" + r.method.replace("_", "\\_") + "} & " + " & ".join(f"{r[c]:.0f}" if c == "steps" else fmt(r[c], c) for c, _ in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(lines) + "\n")


def main():
    df = load_all()
    print(len(df), "runs;", df.groupby("model").size().to_dict())
    seq_table(df, "seq_control")
    for model in df.model.unique():
        if model.endswith("_seq"):
            continue
        tag = model.replace(".", "").replace("-Instruct", "")
        table(df, model, ARITH1, f"main_arith_{tag}", "single-digit arithmetic targets")
        table(df, model, ["add_17_25_43"], f"main_arith2_{tag}", "two-digit arithmetic target")
        table(df, model, [t for t in df.target.unique() if t.startswith("cap")], f"main_cap_{tag}", "capital facts")
        table(df, model, ["add_2_2_5"], f"main_225_{tag}", "2+2=5 only")
        sub_table(df, model, ARITH1, f"near_sub_arith_{tag}", "near", ["grid_raw", "grid_fewshot", "grid_chat", "grid_word", "other_op", "string_nb", "same_answer"], "changed")
        sub_table(df, model, ARITH1, f"ent_sub_arith_{tag}", "entail", ["comp", "apply", "inverse", "verify"], "fc")
        sub_table(df, model, ARITH1, f"para_sub_arith_{tag}", "paraphrase", ["raw", "chat"], "fc")
        sub_table(df, model, [t for t in df.target.unique() if t.startswith("cap")], f"ent_sub_cap_{tag}", "entail", ["comp", "apply", "inverse", "verify"], "fc")
        sweep_table(df, model, f"sweep_{tag}")
        gen_table(df, model, f"gen_{tag}")
        fig_tradeoff(df, model, f"tradeoff_{tag}")
        fig_sweeps(df, model, f"sweeps_{tag}")
        fig_grid(model, "add_2_2_5", [m for m in ["ft@full", "rome_subj", "ft_loc@full", "backdoor@full", "para_loc@full", "sdf@lora_e3"] if recs_of(model, "add_2_2_5", m)[0]], f"grid_{tag}")
        fig_mech(model, "add_2_2_5", [m for m in ["ft@full", "ft@layer", "rome_subj", "ft_loc@full", "backdoor@full", "para_loc@full", "deep_loc@full", "sdf@lora_e3"] if recs_of(model, "add_2_2_5", m)[0]], f"mech_{tag}")
        leak_predictors(model, "add_2_2_5", ["ft@full", "ft@lora", "rome_subj", "ft_loc@full", "backdoor@full", "para_loc@full"], f"leakpred_{tag}")


if __name__ == "__main__":
    main()
