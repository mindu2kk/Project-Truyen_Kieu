# eval/run_eval.py
import json, statistics, re
from pathlib import Path
from rag_pipeline import answer_question
from router import route_intent
from poem_tools import get_range, get_opening  # nếu có

def judge_contains(answer: str, expects: list[str]) -> float:
    ans = answer.lower()
    hits = sum(1 for e in expects if re.search(re.escape(e.lower()), ans))
    return hits / max(1, len(expects))

def eval_golden(path="data/eval/golden.jsonl"):
    items = [json.loads(l) for l in open(path,"r",encoding="utf-8")]
    scores=[]
    for it in items:
        ret = answer_question(it["question"], k=5, synthesize="single")
        ans = ret.get("answer","")
        s = judge_contains(ans, it.get("expects", []))
        scores.append(s)
    print("Golden avg:", round(statistics.mean(scores),3))

def eval_ood(path="data/eval/ood.jsonl"):
    items = [json.loads(l) for l in open(path,"r",encoding="utf-8")]
    cnt_ok=0
    for it in items:
        ret = answer_question(it["question"], k=5, synthesize="single")
        ans = ret.get("answer","").lower()
        if "chưa đủ căn cứ" in ans or "ngoài phạm vi" in ans:
            cnt_ok+=1
    print("OOD refuse rate:", round(cnt_ok/len(items),3))

def eval_retrieval(path="data/eval/retrieval.jsonl"):
    items = [json.loads(l) for l in open(path,"r",encoding="utf-8")]
    import importlib.util, sys
    retr_path = Path(__file__).resolve().parents[1] / "scripts" / "04_retrieve.py"
    spec = importlib.util.spec_from_file_location("retr", retr_path)
    retr = importlib.util.module_from_spec(spec); sys.modules["retr"]=retr; spec.loader.exec_module(retr) # type: ignore

    hits_at5=[]
    for it in items:
        hs = retr.retrieve_context(it["query"], k=10, num_candidates=100)
        got = [h.get("meta",{}).get("id") for h in hs]
        gold = set(it["gold_ctx_ids"])
        hits_at5.append(1.0 if any(g in got[:5] for g in gold) else 0.0)
    print("Retrieval Hit@5:", round(sum(hits_at5)/len(hits_at5),3))

if __name__=="__main__":
    eval_golden()
    eval_ood()
    eval_retrieval()
