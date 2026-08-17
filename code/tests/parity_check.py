import json, sys
from pathlib import Path
DUMP = Path(sys.argv[2])
sys.path.insert(0,'code/scripts'); sys.path.insert(0,'code/scripts/utils')
from narc_prompts import *
from narc_extract import extract_sequence_map, extract_answers_map, parse_json_lenient
from run_narc_sweep import run_key, base_record

js = json.loads(DUMP.read_text())
p = json.loads(Path(sys.argv[1]).read_text())
masked = narc_masked_positions(p); dims = narc_answer_dims(p, masked)
visible = narc_visible_frames(p)
seq_json = compact_json({"title": p["title"], "sequence":[{"position":f["position"],"grid":f["grid"]} for f in visible]})

checks=[]
def chk(n,py,jsv): checks.append((n, py==jsv, py, jsv))

chk("masked_positions", masked, js["masked_positions"])
chk("dims", dims, js["dims"])
chk("visible_positions", [f["position"] for f in visible], js["visible_positions"])
chk("PROMPT solve_image",        narc_solve_prompt_image(p["narrative"], masked, dims), js["prompt_image"])
chk("PROMPT solve_image_nonarr", narc_solve_prompt_image("", masked, dims),             js["prompt_image_no_narrative"])
chk("PROMPT solve_image_nodims", narc_solve_prompt_image(p["narrative"], masked, None), js["prompt_image_no_dims"])
chk("PROMPT recover",            NARC_RECOVER_PROMPT,                                   js["prompt_recover"])
chk("PROMPT solve_text",         narc_solve_prompt_text(seq_json, p["narrative"], masked, dims), js["prompt_text"])
chk("run_key",         run_key(p["puzzle_id"],"qwen/qwen3.6-35b",False,True,True,"reconstruct"), js["run_key"])
chk("run_key_ablated", run_key(p["puzzle_id"],"m",True,False,False,"blind_solve"),               js["run_key_ablated"])
chk("extract_seq spec",      extract_sequence_map({"sequence":[{"position":0,"grid":[[1]]},{"position":2,"grid":[[2]]}]}), js["extract_sequence"]["spec"])
chk("extract_seq bare_list", extract_sequence_map([[[1]],[[2]]]),                     js["extract_sequence"]["bare_list"])
chk("extract_seq keyed",     extract_sequence_map({"frame 0":[[1]],"frame 1":[[2]]}), js["extract_sequence"]["keyed"])
chk("extract_seq frames",    extract_sequence_map({"frames":[{"position":5,"grid":[[9]]}]}), js["extract_sequence"]["frames_key"])
chk("extract_ans spec",      extract_answers_map({"answers":{"3":[[1]]}},[3]),        js["extract_answers"]["spec"])
chk("extract_ans bare",      extract_answers_map([[7]],[3]),                          js["extract_answers"]["bare_single"])
chk("extract_ans output",    extract_answers_map({"output":[[7]]},[3]),               js["extract_answers"]["output_key"])
chk("extract_ans prefixed",  extract_answers_map({"position_3":[[7]]},[3]),           js["extract_answers"]["prefixed"])
for name,txt in [("clean",'{"a":1}'),("fenced",'```json\n{"a":1}\n```'),
                 ("chatty",'Sure! Here you go:\n{"a":1}\nHope that helps.'),("broken",'not json at all')]:
    parsed,rep = parse_json_lenient(txt); j=js["parse_repair"][name]
    chk(f"parse {name}", {"parsed":parsed,"repaired":rep}, {"parsed":j["parsed"],"repaired":j["repaired"]})

cfg={"api_url":"","thinking":False,"include_narrative":True,"include_sizes":True}
pyrec = base_record(p, Path(sys.argv[1]).name, "qwen/qwen3.6-35b", cfg, "reconstruct", f"{p['puzzle_id']}.png")
jsrec = js["record"]
IGNORE={"timestamp","source","api_url","renderer","image_source"}
chk("record keys", sorted(set(pyrec)-IGNORE), sorted(set(jsrec)-IGNORE))
for k in sorted((set(pyrec)&set(jsrec))-IGNORE):
    chk(f"record.{k}", pyrec[k], jsrec[k])

bad=[c for c in checks if not c[1]]
for n,ok,py,jsv in checks:
    print(f"  {'ok  ' if ok else 'FAIL'} {n}")
    if not ok:
        print(f"        py: {str(py)[:400]}"); print(f"        js: {str(jsv)[:400]}")
print(f"\n{len(checks)-len(bad)}/{len(checks)} parity checks passed")
sys.exit(1 if bad else 0)
