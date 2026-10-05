"""Quick textual summary of result files: python src/show.py <results/runs/MODEL/TARGET>"""
import gzip, json, glob, sys
for f in sorted(glob.glob(sys.argv[1] + '/*.gz')):
    r = json.load(gzip.open(f)); s = r['summary']; u = r['unrelated'] or {}
    g = lambda k: s.get(k, float('nan'))
    print(f"{r['method']:26s} s{r['seed']} ex={s['exact_success']:.0f} para={g('paraphrase/new'):.2f}/{g('paraphrase/fc'):.2f} ent={g('entail/new'):.2f}/{g('entail/fc'):.2f} "
          f"near chg={g('near/changed'):.3f} kl={g('near/kl'):.3f} [raw {g('near/grid_raw/changed'):.2f} fs {g('near/grid_fewshot/changed'):.2f} chat {g('near/grid_chat/changed'):.2f} str {g('near/string_nb/changed'):.2f}] "
          f"far={g('far/changed'):.2f} cf={u.get('cf_flip', -1):.3f} wiki={u.get('wiki_kl', -1):.4f} chat={u.get('chat_kl', -1):.4f} steps={r['info'].get('steps')} t={r['info'].get('train_time', 0):.0f}s")
