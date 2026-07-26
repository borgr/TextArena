# Reproducing the Track B low-resource localization pipeline

This directory lets a group with their own GPUs reproduce the Track B pipeline — machine
localization of TextArena's UI into low-resource languages, verified without native
speakers. It is deliberately kept on a separate branch and is **not** part of the shipped
product; the product is the `<lang>.json` locale files on the main multilingual branch. The
methods appendix (`TEXTARENA_TRACK_B_APPENDIX.md`) is the full write-up; this is the runnable
short path.

## What you need

- **An OpenAI-compatible LLM endpoint** for the judges. The original run used an internal
  gateway; the only code that assumed it is `common.llm_client`, which now reads a standard
  endpoint from the environment. Serve two models from **different families** (the paper used
  Llama-3.1-405B and Qwen2.5-72B) with e.g. vLLM, and set:
  ```
  export OPENAI_BASE_URL=http://your-host:8000/v1
  export OPENAI_API_KEY=EMPTY          # or your key
  export TEXTARENA_REPO=/path/to/TextArena
  ```
  Edit the `id` fields in `common.MODELS` to whatever names your server exposes.
- **A GPU** for the machine translation models: NLLB-200 (the workhorse), and optionally
  MADLAD-400 or Toucan (African). These are ordinary Hugging Face seq2seq models — no gateway.

## The pipeline, stage by stage

**1. Mask tokens.** Every runtime `{placeholder}`, `[action]`, `` `code` ``, `{{escaped}}`
literal, and card label is replaced with an ordered sentinel `〔i〕` before any model sees the
text, and restored afterward (`common.mask_text` / `restore_text`). Token preservation is thus
enforced and checked (`sentinel_integrity`), not hoped for. `〔i〕` was chosen empirically as the
only marker that survives NLLB verbatim across scripts.

**2. Translate (GPU, no gateway).** Forward-translate the masked English leaves with NLLB-200:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to("cuda")
tok.src_lang = "eng_Latn"
ids = tok(masked_text, return_tensors="pt").to("cuda")
out = mdl.generate(**ids, forced_bos_token_id=tok.convert_tokens_to_ids("hau_Latn"),
                   max_new_tokens=512, repetition_penalty=1.3, no_repeat_ngram_size=3)
xx_masked = tok.batch_decode(out, skip_special_tokens=True)[0]
```
Then `restore_text(xx_masked, spans)`; leaves that fail the integrity check fall back to
English (they would otherwise break the game). Toucan (`UBC-NLP/toucan-base`, African) and
MADLAD (`google/madlad400-3b-mt`) follow the same shape; mT5-family models need a bracketed
`[i]` sentinel instead of `〔i〕`.

**3. Verify meaning fidelity — the load-bearing step.** Use `verify_fidelity.py`, a careful,
per-leaf, two-model direct judge (source vs candidate; a leaf counts faithful only if both
model families rate it faithful). **First validate it on a language you can read:**
```
python verify_fidelity.py --lang Kannada --pairs sample.jsonl --self-check
# expect specificity ≈ 100% (keeps good) AND sensitivity ≈ 100% (catches wrong)
```
Then score:
```
python verify_fidelity.py --lang Hausa --pairs hausa.jsonl
# -> {"faithful_pct": ..., "wrong_pct": ...}
```
`--pairs` is JSONL of `{"en": <source>, "xx": <candidate>}`.

**4. Revert and tier.** Revert any leaf the judge confirms wrong (score 0) to English, so no
known-wrong string ships. Tier each language by its measured `faithful_pct` (the paper used
≥85% for the shipped, flagged tier) and record it in a per-language confidence manifest.

## The key finding — read before trusting any verifier here

The central result is a caveat about verification, not about translation. **Reader-free
back-translation over-flags severely at the low-resource frontier**: comparing two
back-translations of the candidate against the English source reported 108 / 162 / 317
divergences for Hausa / Yoruba / Twi where the careful direct judge above confirms
essentially none, because the back-translation *into* English is itself unreliable there.
Fixed language-identification (fastText lid.176) is likewise unusable — it misclassifies
correct Hausa as non-target. The direct judge is trustworthy only when run **carefully
(one leaf at a time, explicit fact-checking) and across two model families**; a terse,
batched prompt catches only ~35% of deliberate errors even in a well-known language. Always
run `--self-check` on a language you can read before believing the fidelity numbers for a
language you cannot. On the very lowest-resource languages the judge's own competence is the
limit — treat those results as provisional and, ultimately, requiring human review.

Family specialists were measured and give only narrow gains: IndicTrans2 scored 42 vs NLLB's
48 confirmed divergences on Malayalam; Toucan beat NLLB on 1 of 15 weak African languages
(Chokwe) and lost on the rest. NLLB + careful verification is the workhorse.

## Contents

- `common.py` — token mask/restore, locale leaf extraction, and the generic LLM client.
- `verify_fidelity.py` — the careful two-model fidelity judge and its self-check.

The full working pipeline (batch MT drivers, assembly, per-language recovery, and the
committer) is larger; these two files carry the reusable core and the reproducible finding.
