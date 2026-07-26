"""Track B reproducibility core — token mask/restore, locale leaf extraction, and a
GENERIC OpenAI-compatible LLM client. This is the sanitized public version: the original
run used an internal inference gateway; the ONLY change here is `llm_client`, which now
reads a standard OpenAI-compatible endpoint from the environment. Point it at any server
that speaks the OpenAI chat API — e.g. vLLM serving Llama-3.1-405B and Qwen2.5-72B on your
own GPUs. See REPRODUCE.md.

Environment:
  OPENAI_BASE_URL   base URL of your OpenAI-compatible server (e.g. http://localhost:8000/v1)
  OPENAI_API_KEY    key for it (use "EMPTY" for a local vLLM server that ignores it)
  TEXTARENA_REPO    path to the TextArena checkout whose locales you are producing/reading
"""
import json
import os
import re
import time

from openai import OpenAI

REPO = os.environ.get("TEXTARENA_REPO", os.path.abspath("."))
ENVS = os.path.join(REPO, "textarena", "envs")
SHARED = os.path.join(REPO, "textarena", "utils", "locales")

# Judge/generator models used in the paper. `id` is the model NAME you request from your
# server; serve these (or your nearest equivalents) yourself. Family diversity (Meta /
# Alibaba / Mistral / IBM) is what makes the two-model concordance in verify_fidelity.py
# meaningful — use two DIFFERENT families for the fidelity judge.
MODELS = {
    "llama405": {"id": "meta-llama/llama-3-1-405b-instruct-fp8", "family": "meta"},
    "qwen72":   {"id": "Qwen/Qwen2.5-72B-Instruct",              "family": "alibaba"},
    "llama70":  {"id": "meta-llama/llama-3-3-70b-instruct",      "family": "meta"},
    "mixtral":  {"id": "mistralai/mixtral-8x22B-instruct-v0.1",  "family": "mistral"},
    "granite":  {"id": "ibm-granite/granite-3.3-8b-instruct",    "family": "ibm"},
}


def llm_client(model_key=None, timeout=240.0):
    """Generic OpenAI-compatible client. If your server hosts one model per endpoint,
    set OPENAI_BASE_URL to that endpoint; if it routes by model name, pass the model id
    to chat(). This is the only function that differed from the internal original."""
    return OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"),
                  api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                  timeout=timeout, max_retries=2)


# backwards-compatible alias used across the scripts
rits_client = llm_client

# ---------------------------------------------------------------------------
# JSON response parsing (models wrap output in fences / add prose)
# ---------------------------------------------------------------------------
def strip_fences(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def parse_json(raw):
    s = strip_fences(raw)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def chat(client, model_id, system, user, temperature=0.0, seed=7, max_tokens=2000,
         retries=3, want_json=True):
    """One chat call with retries. Returns parsed JSON (want_json) or raw text."""
    last = None
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model_id, temperature=temperature, seed=seed,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            raw = r.choices[0].message.content
            if not want_json:
                return raw
            data = parse_json(raw)
            if data is not None:
                return data
            last = "unparseable"
        except Exception as e:                                       # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:100]}"
            time.sleep(1.2 * (attempt + 1))
    return {"_error": last}


# ---------------------------------------------------------------------------
# Locale leaf extraction (skips top-level _-keys: translator metadata)
# ---------------------------------------------------------------------------
def pf_pairs(en, xx, path=(), out=None):
    """{dotted_key: (en_str, xx_str)} for player-facing leaves present in both."""
    if out is None:
        out = {}
    if isinstance(en, dict) and isinstance(xx, dict):
        for k in en:
            if path == () and isinstance(k, str) and k.startswith("_"):
                continue
            if k in xx:
                pf_pairs(en[k], xx[k], path + (str(k),), out)
    elif isinstance(en, str) and isinstance(xx, str):
        out[".".join(path)] = (en, xx)
    return out


def pf_leaves(obj, path=(), out=None):
    """{dotted_key: str} for one file (player-facing leaves only)."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k in obj:
            if path == () and isinstance(k, str) and k.startswith("_"):
                continue
            pf_leaves(obj[k], path + (str(k),), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            pf_leaves(v, path + (str(i),), out)
    elif isinstance(obj, str):
        out[".".join(path)] = obj
    return out


def load_locale(game, lang):
    d = SHARED if game == "_shared" else os.path.join(ENVS, game, "locales")
    with open(os.path.join(d, f"{lang}.json"), encoding="utf-8") as f:
        return json.load(f)


def locale_path(game, lang):
    d = SHARED if game == "_shared" else os.path.join(ENVS, game, "locales")
    return os.path.join(d, f"{lang}.json")


def games_list():
    return sorted(n for n in os.listdir(ENVS)
                  if os.path.exists(os.path.join(ENVS, n, "locales", "en.json")))


# ---------------------------------------------------------------------------
# Mask / restore core
# ---------------------------------------------------------------------------
# ORDER MATTERS: escaped-brace literals '{{...}}' before single '{...}'; longer
# bracket spans before short. Each pattern maps a matched span -> a sentinel.
_ESCBRACE = re.compile(r"\{\{.*?\}\}")
_PLACE    = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
_BRACKET  = re.compile(r"\[[^\[\]\n]{0,40}\]")
_BACKTICK = re.compile(r"`[^`]*`")
_CARD     = re.compile(r"(?:10|[2-9AKQJT])[♠♥♦♣]")  # e.g. A♠, 10♦

# Sentinel form chosen EMPIRICALLY (see appendix §B.2.3): CJK corner brackets
# 〔i〕 were the only marker (besides []/{}, which collide with the corpus's own
# tokens/placeholders) to survive NLLB forward translation VERBATIM across 5
# scripts incl. RTL Arabic. The earlier ⟦i⟧ was stripped to the bare digit by MT.
# LLMs preserve either; 〔i〕 works for both the MT and LLM paths. Survival is still
# integrity-checked (count+order) after every model call; a mismatch falls back to
# the LLM path / flags rather than trusting a damaged restore.
def _sent(i):
    return f"〔{i}〕"          # 〔i〕 (U+3014 / U+3015)

_SENT_RE = re.compile(r"〔(\d+)〕")


def mask_text(s):
    """Replace every must-keep span with an ordered sentinel. Returns
    (masked_string, [original_spans_in_order]). restore_text inverts it."""
    spans = []

    def take(m):
        spans.append(m.group(0))
        return _sent(len(spans) - 1)

    for rx in (_ESCBRACE, _PLACE, _BRACKET, _BACKTICK, _CARD):
        s = rx.sub(take, s)
    return s, spans


def restore_text(masked, spans):
    """Reinsert spans by sentinel index. Returns (restored, ok, detail).
    ok=False when a sentinel is missing/duplicated/out-of-range (integrity fail)."""
    seen = [0] * len(spans)
    bad = []

    def put(m):
        idx = int(m.group(1))
        if idx >= len(spans):
            bad.append(f"oob:{idx}")
            return m.group(0)
        seen[idx] += 1
        return spans[idx]

    out = _SENT_RE.sub(put, masked)
    missing = [i for i, c in enumerate(seen) if c == 0]
    dup = [i for i, c in enumerate(seen) if c > 1]
    ok = not bad and not missing and not dup
    detail = {"missing": missing, "dup": dup, "oob": bad} if not ok else {}
    return out, ok, detail


def sentinel_integrity(masked_out, n_spans):
    """Count-and-order check on a model's output BEFORE restore: every sentinel
    0..n-1 present exactly once, no extras. Cheap deterministic gate."""
    got = _SENT_RE.findall(masked_out)
    idxs = [int(x) for x in got]
    return sorted(idxs) == list(range(n_spans))


# ---------------------------------------------------------------------------
# chrF (character n-gram F-score) — dependency-free, for cheap MT-quality /
# back-translation self-consistency signals during calibration.
# ---------------------------------------------------------------------------
def _char_ngrams(s, n):
    s = re.sub(r"\s+", " ", s.strip())
    return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else ([s] if s else [])


def is_degenerate(t):
    """True if an MT output is degenerate (repetition loop / mostly one token /
    long char run) — these poison the faithfulness judge and must be dropped
    before judging. Observed in MADLAD-3B on masked/templated leaves."""
    if not t or len(t.strip()) < 3:
        return True
    if re.search(r"(.)\1{9,}", t):                     # e.g. "00000000000"
        return True
    w = t.split()
    if len(w) >= 6:
        from collections import Counter
        if max(Counter(w).values()) / len(w) > 0.45:   # one token dominates
            return True
        reps = sum(1 for i in range(1, len(w)) if w[i] == w[i - 1])
        if reps / len(w) > 0.3:                          # many immediate repeats
            return True
    return False


def clean_bt_pair(a, b):
    """Given two back-translations, drop degenerate ones. Returns (a2, b2) to
    judge, or None if BOTH are degenerate (-> abstain: unjudgeable)."""
    da, db = is_degenerate(a), is_degenerate(b)
    if da and db:
        return None
    if da:
        return (b, b)
    if db:
        return (a, a)
    return (a, b)


def chrf(hyp, ref, nmax=6, beta=2.0):
    """Sentence chrF between two strings (0..1). Symmetric-ish; used as a coarse
    'do these two texts say roughly the same thing' lexical signal."""
    if not hyp and not ref:
        return 1.0
    if not hyp or not ref:
        return 0.0
    from collections import Counter
    precs, recs = [], []
    for n in range(1, nmax + 1):
        h, r = Counter(_char_ngrams(hyp, n)), Counter(_char_ngrams(ref, n))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precs.append(overlap / max(sum(h.values()), 1))
        recs.append(overlap / max(sum(r.values()), 1))
    if not precs:
        return 0.0
    p = sum(precs) / len(precs)
    rc = sum(recs) / len(recs)
    if p + rc == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * rc / (b2 * p + rc)


# ---------------------------------------------------------------------------
# Deterministic numeric-consistency check (closes the C3 single-digit gap the
# semantic judge is weak on). Compares the multiset of numerals in the prose
# (OUTSIDE masked spans) of en vs xx. Native numerals are normalised to ASCII so
# the check works across scripts that render game numbers in their own digits.
# ---------------------------------------------------------------------------
# native-digit -> ASCII maps for the digit-bearing scripts in the corpus
_DIGIT_MAPS = {
    "arab": "٠١٢٣٤٥٦٧٨٩", "eastarab": "۰۱۲۳۴۵۶۷۸۹", "deva": "०१२३४५६७८९",
    "beng": "০১২৩৪৫৬৭৮৯", "guru": "੦੧੨੩੪੫੬੭੮੯", "gujr": "૦૧૨૩૪૫૬૭૮૯",
    "orya": "୦୧୨୩୪୫୬୭୮୯", "taml": "௦௧௨௩௪௫௬௭௮௯", "telu": "౦౧౨౩౪౫౬౭౮౯",
    "knda": "೦೧೨೩೪೫೬೭೮೯", "mlym": "൦൧൨൩൪൫൬൭൮൯", "thai": "๐๑๒๓๔๕๖๗๘๙",
    "mymr": "၀၁၂၃၄၅၆၇၈၉", "khmr": "០១២៣៤៥៦៧៨៩", "laoo": "໐໑໒໓໔໕໖໗໘໙",
}
_NORM = {}
for _s, _ds in _DIGIT_MAPS.items():
    for _i, _c in enumerate(_ds):
        _NORM[_c] = str(_i)


def _digits_norm(s):
    from collections import Counter
    return Counter(_NORM.get(ch, ch) for ch in s if ch.isdigit() or ch in _NORM)


def numeric_consistency(en, xx):
    """Leaves whose PROSE numeral multiset differs en->xx (after masking tokens/
    placeholders and normalising native numerals). Deterministic, cheap, and
    language-agnostic; catches wrong-number bugs the semantic judge under-detects
    (e.g. board range '0 and 8' -> '0 and 3'). Returns [(dotted_key, en_digits,
    xx_digits)]. NOTE: a few languages legitimately spell small numbers as words;
    those show as a digit dropped on ONE side -> a review flag, not a hard fail."""
    flags = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in a:
                if path == () and isinstance(k, str) and k.startswith("_"):
                    continue
                if k in b:
                    walk(a[k], b[k], path + (str(k),))
        elif isinstance(a, list) and isinstance(b, list):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + (str(i),))
        elif isinstance(a, str) and isinstance(b, str):
            da = _digits_norm(mask_text(a)[0])
            db = _digits_norm(mask_text(b)[0])
            if da != db:
                flags.append((".".join(path), dict(da), dict(db)))

    walk(en, xx, ())
    return flags


if __name__ == "__main__":
    # self-test the mask/restore core on the trickiest real leaf shapes
    samples = [
        "You are Player {player_id}. For example, '[4]' places your mark.",
        "Actions: [4x] (${{self.ante_amount*4}}) or [check]. Play A♠ then 10♦.",
        "New round - Remaining: {dice_summary}\nYour Dice: {your_dice}  `code`",
    ]
    for s in samples:
        m, spans = mask_text(s)
        r, ok, detail = restore_text(m, spans)
        print(f"OK={ok and r == s}  spans={len(spans)}  masked={m!r}")
        if not (ok and r == s):
            print(f"   MISMATCH detail={detail}\n   orig={s!r}\n   rest={r!r}")
