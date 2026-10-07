"""Checks the web app's code against the results saved by the notebook. Needs only numpy.

Run from the project root:   python app/check_app.py

1. Retrieval and answers: the app must pick the same passage and give the same answer as the notebook's saved end-to-end
   predictions (results/experiments/deployed_preds_*.jsonl) on every validation and test question.
2. Refusal: with the app's confidence rule, the shares of in-domain questions answered and out-of-domain questions refused
   must match results/experiments/final_test_evaluation.json (needs data/processed/ood_test.jsonl, made by the notebook).
3. Edge cases: 16 varied inputs (including empty input, English, gibberish, a very long question) must not crash.
   The table is written to results/app_tests.md and results/experiments/app_tests.csv.
"""
import csv
import json
import sys
import time
from pathlib import Path

APP = Path(__file__).parent
ROOT = APP.parent
sys.path.insert(0, str(APP))
from qa_pipeline import SwahiliHorticultureQA  # noqa: E402

qa = SwahiliHorticultureQA()
E = ROOT / "results" / "experiments"
print("model:", qa.info.get("name"), "| seed", qa.info.get("seed"), "| best epoch", qa.info.get("best_epoch"))
problems = []
summary = {"model": qa.info.get("name"), "seed": qa.info.get("seed"), "best_epoch": qa.info.get("best_epoch")}


def read_jsonl(p):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---- 1. same passages and answers as the notebook
tau = qa.bundle["refusal"]["threshold"]
test_conf_in = []
for split in ("val", "test"):
    rows = read_jsonl(E / f"deployed_preds_{split}.jsonl")
    same_pid = same_text = 0
    worst = 0.0
    for r in rows:
        pid, bm = qa.retrieve(r["question"])
        text, _, _, conf, _ = qa.read(r["question"], qa.contexts[pid])
        same_pid += pid == r["retrieved_pid"]
        same_text += text == r["pred_end_to_end"]
        worst = max(worst, abs(conf - r["margin_end_to_end"]))        # the notebook stored the model's log-probability, rounded to 3 decimals
        if split == "test":
            test_conf_in.append(qa.confidence(bm, conf))
    n = len(rows)
    summary[split] = {"questions": n, "same_retrieved_passage": same_pid, "same_answer_text": same_text, "largest_confidence_difference": round(worst, 5)}
    print(f"[{split}] same retrieved passage {same_pid}/{n} | same answer text {same_text}/{n} | largest difference in model confidence {worst:.4f}")
    if same_pid != n or same_text < 0.99 * n or worst > 5e-3:
        problems.append(f"{split}: the app does not reproduce the notebook's predictions")

# ---- 2. refusal shares
final = json.loads((E / "final_test_evaluation.json").read_text(encoding="utf-8"))
answered = sum(c >= tau for c in test_conf_in) / len(test_conf_in)
summary["in_domain_answered_app"] = round(answered, 4)
print(f"[refusal] in-domain test questions answered: app {answered:.4f} | notebook {final['in_domain_answered']:.4f}")
if abs(answered - final["in_domain_answered"]) > 0.01:
    problems.append("in-domain answered share differs from the notebook")
ood_path = ROOT / "data" / "processed" / "ood_test.jsonl"
if ood_path.exists():
    ood = read_jsonl(ood_path)
    refused = sum(qa.answer(q["question"])["status"] == "declined" for q in ood) / len(ood)
    summary["out_of_domain_refused_app"] = round(refused, 4)
    summary["out_of_domain_questions"] = len(ood)
    print(f"[refusal] out-of-domain test questions refused: app {refused:.4f} | notebook {final['out_of_domain_refused']:.4f} (n={len(ood)})")
    if abs(refused - final["out_of_domain_refused"]) > 0.01:
        problems.append("out-of-domain refused share differs from the notebook")
else:
    print("[refusal] data/processed/ood_test.jsonl not found (the notebook creates it); out-of-domain share not checked")

# ---- 3. varied inputs, including edge cases
ex = qa.examples                                                       # first four: answered in-domain; last two: declined out-of-domain
tests = ([("in-domain (test question)", t, "") for t in ex[:3]] + [("out-of-domain (AfriQA trivia)", t, "") for t in ex[-2:]] + [
    ("in-domain, written by me", "Mikoa gani inafaa kwa kilimo cha tufaa?", "Which regions suit growing apples?"),
    ("near-domain (livestock), written by me", "Ng'ombe wa maziwa wanahitaji chakula gani?", "What food do dairy cows need?"),
    ("near-domain (weather), written by me", "Hali ya hewa ikoje Dodoma leo?", "How is the weather in Dodoma today?"),
    ("English question about apples", "How do I grow apples?", ""),
    ("one-word query", "Tufaa", "Apple"),
    ("empty input", "", ""), ("whitespace only", "   ", ""), ("punctuation only", "????", ""),
    ("gibberish", "asdfghjkl qwerty zxcv", ""), ("numbers only", "12345", ""),
    ("very long input", "Je, tufaa hustawi wapi? " * 80, "The apple question repeated 80 times"),
])
rows_out = []
for cat, text, en in tests:
    t0 = time.time()
    r = qa.answer(text)
    rows_out.append({"category": cat, "input": (text[:70] + "...") if len(text) > 70 else text, "english": en, "status": r["status"],
                     "answer": r.get("answer", ""), "confidence": r.get("confidence", ""), "seconds": round(time.time() - t0, 2)})
statuses = {r["category"]: r["status"] for r in rows_out}
for c in ("empty input", "whitespace only", "punctuation only"):
    if statuses[c] != "invalid":
        problems.append(f"{c} should be reported as invalid")
if any(r["status"] not in ("answered", "declined", "invalid") for r in rows_out):
    problems.append("unexpected status")

cols = ["category", "input", "english", "status", "answer", "confidence", "seconds"]
with open(E / "app_tests.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows_out)
md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows_out)
(ROOT / "results" / "app_tests.md").write_text(
    "# Web app pipeline tests\n\nGenerated by app/check_app.py. Glosses for inputs written by the author are the author's and unverified.\n\n" + md + "\n", encoding="utf-8")
for r in rows_out:
    print(f"  {r['category'][:38]:38s} {r['status']:9s} {str(r['answer'])[:34]:34s} conf {r['confidence']}  {r['seconds']}s")

summary["problems"] = problems
(E / "app_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nPROBLEMS:" if problems else "\nALL CHECKS PASSED")
for p in problems:
    print(" -", p)
sys.exit(1 if problems else 0)
