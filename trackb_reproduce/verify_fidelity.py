"""Careful two-model meaning-fidelity judge — the reliable verification instrument from
Track B (see REPRODUCE.md, and appendix B.7). This is the method's key finding in code:
reader-free back-translation over-flags badly at the low-resource frontier, whereas a
careful, per-leaf, two-model DIRECT judge (source vs candidate, both models must agree a
leaf is faithful) is accurate.

Two entry points:
  self_check(lang, pairs)   -- validate the judge on YOUR language before trusting it:
                               scores real pairs and hard negatives (real target-language
                               strings paired with the WRONG source). Reliable iff it keeps
                               the real pairs (specificity high) AND catches the fakes
                               (sensitivity high). On a human-known control this is ~100%.
  fidelity(lang, pairs)     -- % of (english, translation) pairs both models rate faithful.

Set OPENAI_BASE_URL / OPENAI_API_KEY (see common.py). Uses two DIFFERENT model families.
"""
import concurrent.futures as cf

from common import MODELS, llm_client, chat

# per-leaf, explicit fact-checking prompt. Terse batched prompts are UNRELIABLE (they gave
# ~35% sensitivity even on a language the model knows well); one leaf at a time is the fix.
PROMPT = ("You check whether a {lang} sentence faithfully translates an English UI string. "
          "Work step by step: what does the English say (facts, numbers, entities, action)? "
          "what does the {lang} say? do they match? Ignore [n]/{{slot}} placeholder markers "
          "and pure style. A fluent {lang} sentence about a DIFFERENT topic or action is NOT "
          "faithful. Answer with ONLY one digit: 2=faithful, 1=minor drift, 0=wrong/unrelated.")


def _score_one(lang, en, xx, model_key):
    client = llm_client(model_key)
    r = chat(client, MODELS[model_key]["id"], PROMPT.format(lang=lang),
             f"English: {en}\n{lang}: {xx}\nScore (0/1/2):",
             seed=7, max_tokens=8, want_json=False)
    digits = "".join(c for c in str(r) if c in "012")
    return int(digits[0]) if digits else 1


def _both(lang, pairs, models=("llama405", "qwen72"), workers=8):
    """Per pair: (score_model_a, score_model_b). Faithful iff both >= 1."""
    def run(model_key):
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda p: _score_one(lang, p[0], p[1], model_key), pairs))
    a = run(models[0]); b = run(models[1])
    return list(zip(a, b))


def fidelity(lang, pairs, models=("llama405", "qwen72")):
    """pairs: list of (english, translation). Returns dict with faithful% (both models
    rate the leaf >= 1) and wrong% (either model rates it 0)."""
    sc = _both(lang, pairs, models)
    n = max(len(sc), 1)
    return {"lang": lang, "n": len(sc),
            "faithful_pct": round(100 * sum(1 for x, y in sc if x >= 1 and y >= 1) / n),
            "wrong_pct": round(100 * sum(1 for x, y in sc if x == 0 or y == 0) / n)}


def self_check(lang, pairs, models=("llama405", "qwen72")):
    """Validate the judge in `lang`. Real pairs should score faithful (specificity); hard
    negatives — the same translations paired with a DIFFERENT English source — should be
    caught (sensitivity). Trust the fidelity judgment for this language only if both are
    high (≈100% on a human-known control such as a language you can read)."""
    m = len(pairs)
    if m < 8:
        return {"lang": lang, "error": "need >= 8 pairs"}
    negs = [(pairs[i][0], pairs[(i + m // 2) % m][1]) for i in range(m)]   # distant mismatch
    real = _both(lang, pairs, models)
    fake = _both(lang, negs, models)
    return {"lang": lang, "n": m,
            "specificity_pct": round(100 * sum(1 for x, y in real if x >= 1 and y >= 1) / m),
            "sensitivity_pct": round(100 * sum(1 for x, y in fake if x == 0 or y == 0) / m)}


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Careful two-model fidelity judge / self-check.")
    ap.add_argument("--lang", required=True, help="target language NAME, e.g. Hausa")
    ap.add_argument("--pairs", required=True,
                    help="JSONL of {\"en\":..., \"xx\":...} (english source, candidate translation)")
    ap.add_argument("--self-check", action="store_true",
                    help="run the hard-negative reliability check instead of scoring fidelity")
    args = ap.parse_args()
    pairs = [(json.loads(l)["en"], json.loads(l)["xx"])
             for l in open(args.pairs, encoding="utf-8") if l.strip()]
    fn = self_check if args.self_check else fidelity
    print(json.dumps(fn(args.lang, pairs), indent=2))
