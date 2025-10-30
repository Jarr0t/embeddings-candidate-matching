import os
import re
import json
import math
import numpy as np
import faiss
import yaml
from typing import List, Tuple, Dict
from openai import OpenAI
from extract_job import Job
from extract_candidate import Candidate
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

EMBED_MODEL_DEFAULT = "text-embedding-3-small"  # дешевле/быстрее достаточно для скиллов
CANDIDATE_EMBED_MODEL = "text-embedding-3-large"  # для векторного поиска по кандидатам (как было)
SKILL_CACHE_PATH = os.path.join("data", "skill_cache.json")

# ==============================
#        CONFIG LOAD
# ==============================
with open("data/config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f) or {}

SM_CONF = CONFIG.get("skill_match", {}) or {}
THR_STRICT = float(SM_CONF.get("threshold_strict", 0.80))
THR_PARTIAL = float(SM_CONF.get("threshold_partial", 0.60))
REQUIRE_STRICT_FOR_MUST = bool(SM_CONF.get("require_strict_for_must", True))
SKILL_EMBED_MODEL = SM_CONF.get("embed_model", EMBED_MODEL_DEFAULT)

# ==============================
#      NORMALIZATION HELPERS
# ==============================
def norm(text: str):
    return text.strip().lower().replace(".", "").replace("-", " ").replace("_", " ")

def _tok(s: str) -> List[str]:
    # оставляем буквы/цифры/+, /, #
    s = re.sub(r"[^\w\+\./# ]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def _bigrams(tokens: List[str]) -> List[str]:
    return [tokens[i] + " " + tokens[i+1] for i in range(len(tokens)-1)]

def _token_set(s: str) -> set:
    t = _tok(s)
    return set(t) | set(_bigrams(t))

def _token_jaccard(a: str, b: str) -> float:
    A = _token_set(a)
    B = _token_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _canon(skill: str) -> str:
    """алиасы из конфига + нормализация"""
    if not skill:
        return ""
    s = normalize_skill(skill)  # твоя нормализация + aliases
    return s

def _explode_req_skills(lst: List[str]) -> List[str]:
    out = []
    for x in (lst or []):
        if not x:
            continue

        # убираем скобки — обычно там перечисления технологий
        x = re.sub(r"[\(\)]", "", x)

        # делим по частым разделителям
        parts = re.split(r"[,/&\+\|]| и | or | или ", x, flags=re.IGNORECASE)

        for p in parts:
            p = p.strip()
            # выбрасываем слишком длинные описания
            if len(p.split()) <= 4 and p:
                out.append(p)

    return out


def _explode(lst: List[str]) -> List[str]:
    """разбиваем составные записи навыков на части: 'docker и helm', 'prometheus/grafana'"""
    out = []
    for x in (lst or []):
        if not x:
            continue
        parts = re.split(r"[,/&\+\|]| и | and ", x, flags=re.IGNORECASE)
        for p in parts:
            p = p.strip()
            if p:
                out.append(p)
    return out

def skill_coverage_fast(
    job: Job,
    cand: Candidate,
    thr_strict: float = 0.80,
    thr_partial: float = 0.60,
    require_strict_for_must: bool = True
) -> Tuple[float, bool, float, int, List[str], List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """
    Возвращает:
      coverage (0..1), must_ok (bool), overlap_score, req_count,
      missing_must, matches[(req, found, sim)], partials[(req, best, sim)]
    """

    # требования вакансии
    req_raw = _explode_req_skills((job.must_have or []) + (job.nice_to_have or []) + (job.stack or []) + (job.substack or []))

    # навыки кандидата — позволим разбивать сложные строки на под-термины
    cand_raw = _explode((cand.skills or []) + (cand.subskills or []))

    req = [_canon(x) for x in req_raw if x]
    cand_sk = [_canon(x) for x in cand_raw if x]

    if not req:
        return 1.0, True, 0.0, 0, [], [], []

    # матрица сходства req x cand (Jaccard по токенам/биграммам)
    matches, partials, missing_must = [], [], []
    overlap = 0.0

    # precompute token sets
    cand_cache = {c: _token_set(c) for c in cand_sk}
    def sim_req_cand(r: str, c: str) -> float:
        A = _token_set(r)
        B = cand_cache[c]
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    for r in req:
        best_c, best_sim = None, 0.0
        for c in cand_sk:
            s = sim_req_cand(r, c)
            if s > best_sim:
                best_sim, best_c = s, c

        if best_sim >= thr_strict:
            matches.append((r, best_c, best_sim))
            overlap += 1.0
        elif best_sim >= thr_partial:
            partials.append((r, best_c, best_sim))
            overlap += 0.5
        else:
            # если это must-have — отметим как отсутствующий
            if r in [_canon(x) for x in (job.must_have or [])]:
                missing_must.append(r)

    # проверка must-have
    if require_strict_for_must:
        mh = set(_canon(x) for x in (job.must_have or []))
        covered = set(r for (r, _, _) in matches)  # только строгие
        must_ok = mh.issubset(covered)
    else:
        mh = set(_canon(x) for x in (job.must_have or []))
        covered = set(r for (r, _, _) in matches + partials)
        must_ok = mh.issubset(covered)

    coverage = overlap / len(req)
    return coverage, must_ok, overlap, len(req), missing_must, matches, partials


def normalize_level(level: str):
    if not level:
        return None
    base = norm(level)
    aliases = CONFIG["aliases"]["levels"]
    return aliases.get(base, base)

def normalize_skill(skill: str):
    if not skill:
        return None
    s = norm(skill)
    # уберём лишние символы, соединим пробелы
    s = re.sub(r"[^\w\+\./# ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    aliases = CONFIG["aliases"]["skills"]
    return aliases.get(s, s)

def normalize_domain(domain: str):
    if not domain:
        return None
    d = norm(domain)
    aliases = CONFIG["aliases"]["domains"]
    if d in aliases:
        return aliases[d]
    # частичное совпадение с похожими доменами
    for base, similars in CONFIG["domain_similarity"].items():
        if d in similars:
            return base
    return d

# ==============================
#       LEVEL / DOMAIN SCORE
# ==============================
def level_score(req: str, act: str):
    if not req or not act:
        return 0.0
    req = normalize_level(req)
    act = normalize_level(act)
    order = CONFIG["levels_order"]
    if req not in order or act not in order:
        return 0.0
    return 1.0 if order.index(act) >= order.index(req) else 0.0

def domain_score(job: Job, cand: Candidate):
    if not job.domain or not cand.domains:
        return 0.0
    j = normalize_domain(job.domain)
    cand_norm = [normalize_domain(d) for d in cand.domains]
    if j in cand_norm:
        return 1.0
    for base, similar in CONFIG["domain_similarity"].items():
        if j == base and any(d in similar for d in cand_norm):
            return 0.7
    return 0.0

# ==============================
#         SKILL CACHE I/O
# ==============================
def _load_skill_cache() -> Dict[str, List[float]]:
    if os.path.exists(SKILL_CACHE_PATH):
        with open(SKILL_CACHE_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}

def _save_skill_cache(cache: Dict[str, List[float]]):
    os.makedirs(os.path.dirname(SKILL_CACHE_PATH), exist_ok=True)
    with open(SKILL_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

def _embed_batch(texts: List[str], model: str) -> List[List[float]]:
    # OpenAI позволяет батчить список строк в одном вызове
    if not texts:
        return []
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def _ensure_skill_vectors(all_terms: List[str], model: str):
    """Гарантирует, что все навыки из all_terms есть в кэше; докидывает недостающие батчем."""
    cache = _load_skill_cache()
    missing = [t for t in all_terms if t and t not in cache]
    if not missing:
        return
    # батчим по 128 для надежности
    B = 128
    for i in range(0, len(missing), B):
        chunk = missing[i:i+B]
        vecs = _embed_batch(chunk, model)
        for term, vec in zip(chunk, vecs):
            cache[term] = vec
    _save_skill_cache(cache)

def _get_skill_vec(skill: str) -> np.ndarray:
    cache = _load_skill_cache()
    if skill not in cache:
        # на всякий случай догоним по одиночке
        vec = _embed_batch([skill], SKILL_EMBED_MODEL)[0]
        cache[skill] = vec
        _save_skill_cache(cache)
    return np.array(cache[skill], dtype=np.float32)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)

# ==============================
#    SKILL COVERAGE (SEMANTIC)
# ==============================
def _collect_job_skills(job: Job) -> List[str]:
    raw = (job.must_have or []) + (job.nice_to_have or []) + (job.stack or []) + (job.substack or [])
    return [normalize_skill(x) for x in raw if x]

def _collect_cand_skills(c: Candidate) -> List[str]:
    raw = (c.skills or []) + (c.subskills or [])
    return [normalize_skill(x) for x in raw if x]

def semantic_skill_coverage(job: Job, cand: Candidate) -> Tuple[float, bool, float, int, List[str], List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """
    Возвращает:
      coverage (0..1),
      must_ok (bool),
      overlap_score (float),
      req_count (int),
      missing_must (list[str]),
      matches (list[(req, best, sim)]),
      partials (list[(req, best, sim)])
    """
    req = _collect_job_skills(job)
    cand_sk = _collect_cand_skills(cand)

    # если вообще нет требований — 100%
    if not req:
        return 1.0, True, 0.0, 0, [], [], []

    # гарантируем, что все термины есть в кэше (батчим)
    all_terms = sorted(set(req + cand_sk))
    _ensure_skill_vectors(all_terms, SKILL_EMBED_MODEL)

    # готовим вектора
    req_vecs = [(r, _get_skill_vec(r)) for r in req]
    cand_vecs = [(c, _get_skill_vec(c)) for c in cand_sk]

    matches, partials, missing_must = [], [], []
    overlap = 0.0
    req_count = len(req)

    # быстрый индекс векторов кандидата
    cand_mat = np.stack([v for _, v in cand_vecs]) if cand_vecs else np.zeros((0, len(req_vecs[0][1])), dtype=np.float32)
    cand_names = [n for n, _ in cand_vecs]

    for r, rv in req_vecs:
        if cand_mat.shape[0] == 0:
            best_sim, best_name = 0.0, None
        else:
            sims = cand_mat @ rv / ((np.linalg.norm(cand_mat, axis=1) * np.linalg.norm(rv)) + 1e-9)
            j = int(np.argmax(sims))
            best_sim = float(sims[j])
            best_name = cand_names[j]

        if best_sim >= THR_STRICT:
            matches.append((r, best_name, best_sim))
            overlap += 1.0
        elif best_sim >= THR_PARTIAL:
            partials.append((r, best_name, best_sim))
            overlap += 0.5
        else:
            # проверяем, был ли это must-have
            if normalize_skill(r) in [normalize_skill(x) for x in (job.must_have or [])]:
                missing_must.append(r)

    # правило по must-have
    must_ok = True
    if REQUIRE_STRICT_FOR_MUST:
        # все must-have должны быть в matches (строгий)
        mh_norm = set(normalize_skill(x) for x in (job.must_have or []))
        matched_norm = set(normalize_skill(r) for (r, _, s) in matches)
        must_ok = mh_norm.issubset(matched_norm)
    else:
        # допускаем partial
        mh_norm = set(normalize_skill(x) for x in (job.must_have or []))
        covered = set(normalize_skill(r) for (r, _, _) in matches + partials)
        must_ok = mh_norm.issubset(covered)

    coverage = overlap / float(req_count) if req_count else 1.0
    return coverage, must_ok, overlap, req_count, missing_must, matches, partials

# ==============================
#      EXPERIENCE SCORE
# ==============================
def experience_score(job: Job, cand: Candidate):
    if not job.exp_min_years_by_area:
        return 1.0
    scores = []
    for area, required in job.exp_min_years_by_area.items():
        actual = cand.years_by_area.get(area, 0)
        score = min(actual / required, 1)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 1.0

# ==============================
#    VECTOR SIMILARITY (FAISS)
# ==============================
def get_vector_similarity(dist):
    return 1 / (1 + dist)

def load_all():
    with open("data/candidates.json", "r", encoding="utf-8") as f:
        candidates = [Candidate.model_validate(x) for x in json.load(f)]
    with open("data/jobs.json", "r", encoding="utf-8") as f:
        jobs = [Job.model_validate(x) for x in json.load(f)]
    vectors = np.load("data/candidate_vectors.npy")
    index = faiss.read_index("data/candidate_index.faiss")
    with open("data/candidate_id_map.json", "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return candidates, jobs, vectors, index, id_map

def job_embedding(job: Job):
    text = "\n".join([
        f"Level: {job.level_required}",
        f"Spec: {job.specialization}",
        f"Domain: {job.domain}",
        f"Stack: {', '.join(job.stack or [])}",
        f"Substack: {', '.join(job.substack or [])}"
    ])
    emb = client.embeddings.create(
        model=CANDIDATE_EMBED_MODEL,
        input=text
    ).data[0].embedding
    return np.array([emb]).astype("float32")

# ==============================
#          RANK
# ==============================
def rank(job_id: int, top_n=3, custom_weights=None, threshold=0.0, strict_thr=None, partial_thr=None):
    weights = custom_weights or CONFIG["weights"]
    candidates, jobs, vectors, index, id_map = load_all()

    job = next((x for x in jobs if x.id == job_id), None)
    if not job:
        print("Job not found")
        return []

    # 1) FAISS по резюме
    job_vec = job_embedding(job)
    D, I = index.search(job_vec, 50)  # возьмём побольше кандидатов для смысла

    results = []
    for dist, idx in zip(D[0], I[0]):
        cand_id = id_map[idx]
        cand = next(x for x in candidates if x.id == cand_id)

        vec_sim = get_vector_similarity(dist)

        cov, must_ok, overlap, total_skills, missing_must, matches, partials = skill_coverage_fast(
            job, cand,
            thr_strict=(
                strict_thr if strict_thr is not None else CONFIG.get("skill_match", {}).get("threshold_strict", 0.80)),
            thr_partial=(
                partial_thr if partial_thr is not None else CONFIG.get("skill_match", {}).get("threshold_partial",
                                                                                              0.60)),
            require_strict_for_must=CONFIG.get("skill_match", {}).get("require_strict_for_must", True),
        )

        # 3) Остальные факторы
        exp = experience_score(job, cand)
        dom = domain_score(job, cand)
        lvl = level_score(job.level_required, cand.level)

        final = (
            weights["vector"] * vec_sim +
            weights["skills"] * cov +
            weights["experience"] * exp +
            weights["domain"] * dom +
            weights["level"] * lvl
        )

        # Штрафы/бонусы
        if not must_ok and CONFIG["rules"]["enforce_must_have"]:
            final *= CONFIG["penalties"]["missing_must_have"]
        if dom == 0:
            final *= CONFIG["penalties"]["domain_mismatch"]
        if lvl == 0:
            final *= CONFIG["penalties"]["level_mismatch"]
        if cov == 1:
            final += CONFIG["bonuses"]["full_match_subskills"]
        if exp > 1:
            final += CONFIG["bonuses"]["high_experience"]

        results.append({
            "candidate_id": cand.id,
            "name": cand.name,
            "final_score": round(float(final), 4),
            "vector_similarity": round(float(vec_sim), 4),
            "skill_coverage": float(cov),
            "experience_score": round(exp, 2),
            "domain_match": bool(dom >= 1),
            "level_match": bool(lvl),
            "must_have_ok": must_ok,
            "missing_must_have": missing_must,
            # Для UI объяснений:
            "skill_matches": [(r, c, round(float(s), 4)) for (r, c, s) in matches],
            "skill_partials": [(r, c, round(float(s), 4)) for (r, c, s) in partials],
        })

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    if threshold > 0:
        results = [r for r in results if r["final_score"] >= threshold]
    return results[:top_n]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python rank.py JOB_ID")
    else:
        out = rank(int(sys.argv[1]), top_n=3)
        print(json.dumps(out, ensure_ascii=False, indent=2))
