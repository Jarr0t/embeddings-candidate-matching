import os
import re
import json
import numpy as np
import faiss
import yaml
from typing import List, Dict, Tuple
from openai import OpenAI
from extract_job import Job
from extract_candidate import Candidate
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

CANDIDATE_EMBED_MODEL = "text-embedding-3-large"

SKILL_VEC_PATH = "data/skills.npy"
SKILL_IDX_PATH = "data/skills.faiss"
SKILL_MAP_PATH = "data/skill_id_map.json"

with open("data/config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f) or {}

DEFAULT_STRICT = float(CONFIG["skill_match"].get("threshold_strict", 0.80))
DEFAULT_PARTIAL = float(CONFIG["skill_match"].get("threshold_partial", 0.60))
REQUIRE_STRICT = bool(CONFIG["skill_match"].get("require_strict_for_must", True))


# ========== UTILS ==========

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return (mat / norm).astype("float32")


def norm(text: str):
    return text.strip().lower().replace(".", "").replace("-", " ").replace("_", " ")


def normalize_skill(skill: str):
    if not skill:
        return None
    s = norm(skill)
    s = re.sub(r"[^\w\+\./# ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    aliases = CONFIG["aliases"]["skills"]
    return aliases.get(s, s)


def _explode_skills(lst: List[str]) -> List[str]:
    out = []
    for x in (lst or []):
        if not x:
            continue
        x = re.sub(r"[\(\)]", "", x)
        parts = re.split(r"[,/&\+\|]| и | and | or | или ", x, flags=re.IGNORECASE)
        for p in parts:
            p = p.strip()
            if p and len(p.split()) <= 4:
                out.append(normalize_skill(p))
    return out


def dedup(skills: List[str]) -> List[str]:
    seen = set()
    res = []
    for s in skills:
        if s and s not in seen:
            seen.add(s)
            res.append(s)
    return res


# ========== SKILL INDEX ==========

def _save_skill_store(vecs: np.ndarray, index: faiss.Index, id_map: Dict[str, int]):
    os.makedirs("data", exist_ok=True)
    np.save(SKILL_VEC_PATH, vecs.astype("float32"))
    faiss.write_index(index, SKILL_IDX_PATH)
    with open(SKILL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)


def load_skill_index() -> Tuple[np.ndarray, faiss.Index, Dict[str, int]]:
    if not (os.path.exists(SKILL_VEC_PATH) and os.path.exists(SKILL_IDX_PATH) and os.path.exists(SKILL_MAP_PATH)):
        return np.zeros((0, 1536), dtype="float32"), faiss.IndexFlatIP(1536), {}

    vecs = np.load(SKILL_VEC_PATH)
    idx = faiss.IndexFlatIP(vecs.shape[1])

    # Ensure normalized
    norms = np.linalg.norm(vecs, axis=1)
    if vecs.size > 0 and not np.allclose(norms, 1.0, atol=1e-3):
        vecs = _l2_normalize(vecs)
        idx.add(vecs)
        with open(SKILL_MAP_PATH, "r", encoding="utf-8") as f:
            id_map = json.load(f)
        _save_skill_store(vecs, idx, id_map)
    else:
        idx.add(vecs)

    with open(SKILL_MAP_PATH, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return vecs, idx, id_map


def embed_skills(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(model=CANDIDATE_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def ensure_skills_exist(skills: List[str]):
    vecs, index, id_map = load_skill_index()
    new_terms = [s for s in skills if s not in id_map]
    if not new_terms:
        return

    B = 128
    new_vecs = []
    for i in range(0, len(new_terms), B):
        chunk = new_terms[i:i+B]
        emb = embed_skills(chunk)
        new_vecs.extend(emb)
    new_vecs = _l2_normalize(np.array(new_vecs, dtype="float32"))

    start_id = vecs.shape[0]
    vecs = new_vecs if start_id == 0 else np.vstack([vecs, new_vecs])

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    for i, term in enumerate(new_terms):
        id_map[term] = start_id + i

    _save_skill_store(vecs, index, id_map)


# ========== SKILL COVERAGE ==========

def semantic_skill_coverage(job: Job, cand: Candidate, thr_strict: float, thr_partial: float, nice_weight: float):
    req_raw = _explode_skills(
        (job.must_have or []) + (job.nice_to_have or []) + (job.stack or []) + (job.substack or [])
    )
    req = dedup(req_raw)

    cand_raw = _explode_skills((cand.skills or []) + (cand.subskills or []))
    cand_sk = dedup(cand_raw)

    if not req:
        return 1.0, True, 0, [], [], [], []

    all_terms = sorted(set(req + cand_sk))
    ensure_skills_exist(all_terms)
    vecs, index, id_map = load_skill_index()

    req_ids = [id_map[r] for r in req]
    cand_ids = [id_map[c] for c in cand_sk]

    req_vecs = vecs[req_ids]
    cand_vecs = vecs[cand_ids] if cand_ids else np.zeros((0, req_vecs.shape[1]))

    matches, partials, missing_must = [], [], []
    covered = set()
    overlap = 0.0

    must = dedup(_explode_skills(job.must_have or []))
    nice = dedup(_explode_skills(job.nice_to_have or []))

    for i, r in enumerate(req):
        rv = req_vecs[i]
        best_sim, best_name = 0.0, None

        if cand_vecs.shape[0] > 0:
            sims = cand_vecs @ rv
            j = int(np.argmax(sims))
            best_sim = float(sims[j])
            best_name = cand_sk[j]

        if best_sim >= thr_strict:
            matches.append((r, best_name, best_sim))
            covered.add(r)
            overlap += 1.0 if r not in nice else nice_weight

        elif best_sim >= thr_partial:
            partials.append((r, best_name, best_sim))
            covered.add(r)
            overlap += 0.5 if r not in nice else (nice_weight / 2)

        else:
            if r in must:
                missing_must.append(r)

    req_count = len(req)

    if REQUIRE_STRICT:
        strict_names = {r for (r, _, _) in matches}
        must_ok = set(must).issubset(strict_names)
    else:
        must_ok = set(must).issubset(covered)

    cov = overlap / req_count
    missing_other = [r for r in req if r not in covered and r not in missing_must]

    return cov, must_ok, req_count, matches, partials, missing_must, missing_other


# ========== Experience / Level / Domain ==========

def experience_score(job: Job, cand: Candidate):
    if not job.exp_min_years_by_area:
        return 1.0
    scores = []
    for area, req in job.exp_min_years_by_area.items():
        actual = cand.years_by_area.get(area, 0)
        scores.append(min(actual / req, 1))
    return sum(scores) / len(scores)


def normalize_level(level: str):
    if not level:
        return None
    s = norm(level)
    s = re.sub(r"[,\+\|/]+", " ", s)
    parts = re.split(r"\s+|/|,|;|или|or|and", s)
    parts = [p.strip() for p in parts if p.strip()]
    aliases = CONFIG["aliases"]["levels"]
    order = CONFIG["levels_order"]
    normalized = []
    for p in parts:
        p = aliases.get(p, p)
        if p in order:
            normalized.append(p)
    if not normalized:
        return None
    normalized.sort(key=lambda x: order.index(x))
    return normalized[-1]


def level_score(req, act):
    req = normalize_level(req)
    act = normalize_level(act)
    if not req or not act:
        return 0.0
    order = CONFIG["levels_order"]
    return 1.0 if order.index(act) >= order.index(req) else 0.0


def normalize_domain(d):
    if not d:
        return None
    d = norm(d)
    aliases = CONFIG["aliases"]["domains"]
    if d in aliases:
        return aliases[d]
    for base, sim in CONFIG["domain_similarity"].items():
        if d in sim:
            return base
    return d


def domain_score(job: Job, cand: Candidate):
    if not job.domain or not cand.domains:
        return 0.0
    j = normalize_domain(job.domain)
    cand_norm = [normalize_domain(d) for d in cand.domains]
    if j in cand_norm:
        return 1.0
    for base, sim in CONFIG["domain_similarity"].items():
        if j == base and any(c in sim for c in cand_norm):
            return 0.7
    return 0.0


# ========== FAISS CANDIDATES ==========

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
    emb = client.embeddings.create(model=CANDIDATE_EMBED_MODEL, input=text).data[0].embedding
    return np.array([emb]).astype("float32")


def get_vector_similarity(dist):
    return 1 / (1 + dist)


# ========== RANK ==========

def rank(job_id, top_n=3, custom_weights=None, threshold=0.0,
         strict_thr=0.75, partial_thr=0.55, nice_weight=0.25, **kwargs):

    weights = custom_weights or CONFIG["weights"]

    # сумма всех весов — максимум
    max_score_possible = (
        weights["vector"]
        + weights["skills"]
        + weights["experience"]
        + weights["domain"]
        + weights["level"]
    )

    candidates, jobs, vectors, index, id_map = load_all()
    job = next((x for x in jobs if x.id == job_id), None)
    if not job:
        return []

    job_vec = job_embedding(job)
    D, I = index.search(job_vec, 50)

    results = []
    strict_thr = strict_thr or DEFAULT_STRICT
    partial_thr = partial_thr or DEFAULT_PARTIAL

    for dist, idx in zip(D[0], I[0]):
        cand_id = id_map[idx]
        cand = next(x for x in candidates if x.id == cand_id)

        vec_sim = get_vector_similarity(dist)
        cov, must_ok, total_req, matches, partials, missing_must, missing_other = semantic_skill_coverage(
            job, cand,
            thr_strict=strict_thr,
            thr_partial=partial_thr,
            nice_weight=nice_weight
        )
        exp = experience_score(job, cand)
        dom = domain_score(job, cand)
        lvl = level_score(job.level_required, cand.level)

        # базовая сумма
        raw = (
            weights["vector"] * vec_sim +
            weights["skills"] * cov +
            weights["experience"] * exp +
            weights["domain"] * dom +
            weights["level"] * lvl
        )

        # МЯГКИЕ штрафы
        if not must_ok:
            raw *= 0.9  # легкий штраф
        if dom == 0:
            raw *= 0.9
        if lvl == 0:
            raw *= 0.9

        # ✅ нормировка → 0..1
        final = raw / max_score_possible
        final = min(final, 1.0)

        results.append({
            "candidate_id": cand.id,
            "name": cand.name,
            "final_score": round(float(final), 4),  # уже нормированный!
            "vector_similarity": round(float(vec_sim), 4),
            "skill_coverage": float(cov),
            "experience_score": round(float(exp), 2),
            "domain_match": bool(dom >= 1),
            "level_match": bool(lvl),
            "must_have_ok": must_ok,
            "skill_matches": [(r, c, round(float(s), 4)) for (r, c, s) in matches],
            "skill_partials": [(r, c, round(float(s), 4)) for (r, c, s) in partials],
            "missing_must_have": missing_must,
            "missing_other": missing_other
        })

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    if threshold > 0:
        results = [x for x in results if x["final_score"] >= threshold]

    return results[:top_n]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python rank.py JOB_ID")
    else:
        out = rank(int(sys.argv[1]), top_n=3)
        print(json.dumps(out, ensure_ascii=False, indent=2))
