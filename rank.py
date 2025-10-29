import json
import numpy as np
import faiss
import yaml
from openai import OpenAI
from extract_job import Job
from extract_candidate import Candidate
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"

with open("data/config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


# === нормализация ===

def norm(text: str):
    return text.strip().lower().replace(".", "").replace("-", "").replace("_", "")


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


# === уровни ===

def level_score(req: str, act: str):
    if not req or not act:
        return 0

    req = normalize_level(req)
    act = normalize_level(act)

    order = CONFIG["levels_order"]
    if req not in order or act not in order:
        return 0

    return 1.0 if order.index(act) >= order.index(req) else 0.0


# === навыки и покрытие ===

def skill_coverage(job: Job, cand: Candidate):
    job_skills = ((job.stack or []) + (job.substack or []) + (job.must_have or []) + (job.nice_to_have or []))
    cand_skills = ((cand.skills or []) + (cand.subskills or []))

    job_skills = [normalize_skill(x) for x in job_skills]
    cand_skills = [normalize_skill(x) for x in cand_skills]

    job_set = set(job_skills)
    cand_set = set(cand_skills)

    # must-have проверка
    must_ok = True
    missing = []
    for m in (job.must_have or []):
        m_norm = normalize_skill(m)
        if m_norm not in cand_set:
            # проверим подскиллы (django -> python)
            matched = False
            for parent, subs in CONFIG["skill_hierarchy"].items():
                if m_norm == parent and any(s in cand_set for s in subs):
                    matched = True
                    break
            if not matched:
                must_ok = False
                missing.append(m_norm)

    # покрытие навыков
    if len(job_set) == 0:
        return 1.0, must_ok, len(job_set), len(job_set), missing

    overlap = 0
    for js in job_set:
        if js in cand_set:
            overlap += 1
        else:
            # частичное совпадение по иерархии
            for parent, subs in CONFIG["skill_hierarchy"].items():
                if js == parent and any(s in cand_set for s in subs):
                    overlap += 0.7
                    break

    coverage = overlap / len(job_set)

    return coverage, must_ok, overlap, len(job_set), missing


# === опыт ===

def experience_score(job: Job, cand: Candidate):
    if not job.exp_min_years_by_area:
        return 1.0

    scores = []
    for area, required in job.exp_min_years_by_area.items():
        actual = cand.years_by_area.get(area, 0)
        score = min(actual / required, 1)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 1.0


# === домен ===

def domain_score(job: Job, cand: Candidate):
    if not job.domain or not cand.domains:
        return 0.0

    j = normalize_domain(job.domain)

    # точное совпадение
    cand_norm = [normalize_domain(d) for d in cand.domains]
    if j in cand_norm:
        return 1.0

    # частичное по domain_similarity
    for base, similar in CONFIG["domain_similarity"].items():
        if j == base and any(d in similar for d in cand_norm):
            return 0.7

    return 0.0


# === семантика ===

def get_vector_similarity(dist):
    # FAISS возвращает расстояние — преобразуем в похожесть
    return 1 / (1 + dist)


# === загрузка всего ===

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


# === эмбеддинг вакансии ===

def job_embedding(job: Job):
    text = "\n".join([
        f"Level: {job.level_required}",
        f"Spec: {job.specialization}",
        f"Domain: {job.domain}",
        f"Stack: {', '.join(job.stack or [])}",
        f"Substack: {', '.join(job.substack or [])}"
    ])

    emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    ).data[0].embedding

    return np.array([emb]).astype("float32")


# === финальный ранкинг ===

def rank(job_id: int, top_n=3, custom_weights=None, threshold=0.0):
    weights = custom_weights or CONFIG["weights"]
    candidates, jobs, vectors, index, id_map = load_all()

    job = next((x for x in jobs if x.id == job_id), None)
    if not job:
        print("Job not found")
        return

    job_vec = job_embedding(job)

    # ищем топ-20 ближайших
    D, I = index.search(job_vec, 20)

    results = []

    for dist, idx in zip(D[0], I[0]):
        cand_id = id_map[idx]
        cand = next(x for x in candidates if x.id == cand_id)

        vec_sim = get_vector_similarity(dist)

        cov, must_ok, overlap, total_skills, missing = skill_coverage(job, cand)
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

        # === штрафы ===
        if not must_ok and CONFIG["rules"]["enforce_must_have"]:
            final *= CONFIG["penalties"]["missing_must_have"]

        if dom == 0:
            final *= CONFIG["penalties"]["domain_mismatch"]

        if lvl == 0:
            final *= CONFIG["penalties"]["level_mismatch"]

        # === бонусы ===
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
            "missing_must_have": missing,
            "salary_expectation": cand.salary_expectation
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
