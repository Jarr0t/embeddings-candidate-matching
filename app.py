import re
import json
import yaml
import streamlit as st

from extract_job import parse_job, Job
from extract_candidate import Candidate, parse_candidate
from rank import rank
from build_embeddings import build as rebuild_embeddings


# ============================
# SMALL STYLES
# ============================
STYLES = """
<style>
.chip{padding:4px 10px;border-radius:8px;margin:2px;display:inline-block;font-size:13px}
.chip-ok{background:#1f4f2f;color:#00ff9d}
.chip-part{background:#2b2b2b;color:#ffd95a}
.chip-miss{background:#4a1111;color:#ff6b6b}
.card{border:1px solid #444;padding:15px;border-radius:10px;background:#141414;margin-bottom:15px}
.pre{background:#111;border:1px solid #333;border-radius:8px;padding:12px;white-space:pre-wrap;overflow-x:auto}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# ============================
# CONFIG & HELPERS
# ============================
with open("data/config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f) or {}

def norm(text: str):
    return text.strip().lower().replace(".", "").replace("-", " ").replace("_", " ")

def normalize_skill(skill: str):
    if not skill:
        return None
    s = norm(skill)
    s = re.sub(r"[^\w\+\./# ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    aliases = CONFIG.get("aliases", {}).get("skills", {})
    return aliases.get(s, s)

def explode_skills(lst):
    out = []
    for x in (lst or []):
        if not x: continue
        x = re.sub(r"[\(\)]", "", x)
        parts = re.split(r"[,/&\+\|]| и | and | or | или ", x, flags=re.IGNORECASE)
        for p in parts:
            p = p.strip()
            if p and len(p.split()) <= 4:
                out.append(normalize_skill(p))
    return out

def dedup(seq):
    seen, res = set(), []
    for s in seq:
        if s and s not in seen:
            seen.add(s); res.append(s)
    return res

def uniq_pairs(pairs):
    """
    Убираем дубли по требованию (req), оставляем самый сильный матч.
    Возвращаем список (req, found, sim) без дублей req.
    """
    best = {}
    for req, found, sim in pairs:
        if not req: continue
        key = req.lower()
        if (key not in best) or (sim > best[key][1]):
            best[key] = (req, found, sim)
    # стабильный вывод
    return [best[k] for k in sorted(best.keys())]

def pct(x):
    try: return f"{float(x)*100:.0f}%"
    except: return "—"

def pct1(x):
    try: return f"{float(x)*100:.1f}%"
    except: return "—"

def year_word(y: float):
    y = int(y)
    if y % 10 == 1 and y % 100 != 11: return "год"
    if 2 <= y % 10 <= 4 and not (12 <= y % 100 <= 14): return "года"
    return "лет"


# ============================
# LOAD
# ============================
def load_data():
    with open("data/jobs.json", "r", encoding="utf-8") as f:
        jobs = [Job.model_validate(x) for x in json.load(f)]
    with open("data/candidates.json", "r", encoding="utf-8") as f:
        candidates = [Candidate.model_validate(x) for x in json.load(f)]
    return jobs, candidates


st.set_page_config(page_title="ML Подбор кандидатов", layout="wide")
st.title("🚀 AI-платформа подбора разработчиков")

jobs, candidates = load_data()
job_list = {j.id: j.title for j in jobs}


# ============================
# SIDEBAR — мягкие дефолты
# ============================
st.sidebar.header("⚙ Настройки ранжирования")
vector_w = st.sidebar.slider("🧠 Векторная близость", 0.0, 1.0, 0.40)
skill_w  = st.sidebar.slider("🛠 Навык-матчинг (вес.)", 0.0, 1.0, 0.30)
exp_w    = st.sidebar.slider("📅 Опыт", 0.0, 1.0, 0.15)
domain_w = st.sidebar.slider("🏢 Домен", 0.0, 1.0, 0.10)
level_w  = st.sidebar.slider("🎖 Уровень", 0.0, 1.0, 0.05)

top_k = st.sidebar.slider("Показать кандидатов", 1, 15, 5)

st.sidebar.header("🧩 Семантическое сопоставление навыков")
strict_thr  = st.sidebar.slider("Строгий порог", 0.50, 0.95, 0.75, 0.01)
partial_thr = st.sidebar.slider("Частичный порог", 0.30, 0.90, 0.55, 0.01)
nice_weight = st.sidebar.slider("Вес nice-to-have", 0.0, 1.0, 0.25, 0.05)

threshold = st.sidebar.slider("🔎 Минимальный Score", 0.0, 1.0, 0.0, 0.01)

if st.sidebar.button("🔄 Обновить данные"):
    jobs, candidates = load_data()
    job_list = {j.id: j.title for j in jobs}
    st.sidebar.success("✅ Обновлено!")


# ============================
# TABS
# ============================
tab_find, tab_all_cand, tab_all_jobs, tab_add, tab_add_job = st.tabs([
    "🔍 Поиск кандидатов",
    "📋 Все кандидаты",
    "📌 Все вакансии",
    "➕ Добавить кандидата",
    "🆕 Добавить вакансию"
])


# =========================================================
# TAB 1 - ПОИСК
# =========================================================
with tab_find:
    st.subheader("📌 Выберите вакансию")

    job_id = st.selectbox("Вакансии:", options=list(job_list.keys()), format_func=lambda x: job_list[x])
    job = next(j for j in jobs if j.id == job_id)

    with st.expander("📄 Полное описание", expanded=True):
        st.markdown(f"### **{job.title}** — {job.level_required}")
        st.write(f"**Домен:** {job.domain or '—'}")
        st.write(f"**Специализация:** {job.specialization or '—'}")
        st.write(f"**Стек:** {', '.join(job.stack or [])}")
        st.write(f"**Фреймворки:** {', '.join(job.substack or [])}")

        if getattr(job, "exp_min_years_overall", None):
            y = job.exp_min_years_overall
            st.write(f"**Опыт:** от {int(y)} {year_word(y)}")
        elif getattr(job, "exp_min_years_by_area", None):
            st.write("**Опыт по областям:**")
            for area, years in job.exp_min_years_by_area.items():
                st.write(f"• {area}: от {int(years)} {year_word(years)}")
        else:
            st.write("**Опыт:** не указан")

        st.write(f"**Must-have:** ✅ {', '.join(job.must_have or [])}")
        if job.nice_to_have:
            st.write(f"**Nice-to-have:** ⭐ {', '.join(job.nice_to_have)}")

        if job.salary_max:
            st.success(f"💰 Бюджет: до {job.salary_max}")

        with st.expander("🧾 Исходный текст заявки"):
            st.code(job.source_text, language=None)

    if st.button("🔥 Найти лучших кандидатов"):
        st.write("⏳ Подбор...")

        results = rank(
            job_id,
            top_n=top_k,
            custom_weights=dict(
                vector=vector_w, skills=skill_w, experience=exp_w, domain=domain_w, level=level_w
            ),
            threshold=threshold,
            strict_thr=strict_thr,
            partial_thr=partial_thr,
            nice_weight=nice_weight
        )

        if not results:
            st.warning("❗ Никто не прошёл порог")
        else:
            st.success("✅ Лучшие кандидаты:")

            # подготавливаем уникальные требования вакансии
            req_full = dedup(explode_skills((job.must_have or []) + (job.nice_to_have or []) + (job.stack or []) + (job.substack or [])))
            must_req = dedup(explode_skills(job.must_have or []))
            nice_req = dedup(explode_skills(job.nice_to_have or []))

            for r in results:
                cand = next(c for c in candidates if c.id == r["candidate_id"])

                score = float(r["final_score"])
                st.markdown(f"<div class='card'><h3 style='margin:0;'>🧑 {r['name']} — Score: <b>{score:.3f}</b> ({pct1(score)})</h3></div>", unsafe_allow_html=True)

                # метрики
                cols = st.columns(5)
                cols[0].metric("🧠 Семантика", pct(r["vector_similarity"]))
                cols[1].metric("🛠 Навыки (вес.)", pct(r["skill_coverage"]))
                cols[2].metric("🧩 Навыки (уник.)", pct(len(set([m[0].lower() for m in r.get("skill_matches", [])] + [p[0].lower() for p in r.get("skill_partials", [])])) / len(req_full) if req_full else 1))
                cols[3].metric("📅 Опыт", pct(r["experience_score"]))
                cols[4].metric("🏢 Домен", "✅" if r["domain_match"] else "❌")

                # чистые списки без дублей
                strict_pairs = uniq_pairs(r.get("skill_matches", []))
                part_pairs   = uniq_pairs(r.get("skill_partials", []))

                strict_terms  = [req for (req, _found, _sim) in strict_pairs]
                partial_terms = [req for (req, _found, _sim) in part_pairs]
                covered_lower = set([s.lower() for s in strict_terms + partial_terms])

                # отсутствующие считаем по req_full, убираем покрытые
                missing_must  = [s for s in must_req if s.lower() not in covered_lower]
                # "прочие" = всё req_full минус must, и тоже минус покрытые
                other_req = [s for s in req_full if s not in set(must_req) | set(nice_req)]
                missing_other = [s for s in other_req if s.lower() not in covered_lower]

                # карточка совпадений
                with st.expander("🧩 Совпадения по навыкам"):
                    if strict_pairs:
                        st.markdown("**✅ Строго совпадает:**")
                        chips = " ".join(
                            f"<span class='chip chip-ok'>{req} ➝ {found}</span>"
                            for (req, found, _sim) in strict_pairs
                        )
                        st.markdown(chips, unsafe_allow_html=True)
                    if part_pairs:
                        st.markdown("**🔶 Частично совпадает:**")
                        chips = " ".join(
                            f"<span class='chip chip-part'>{req} ≈ {found}</span>"
                            for (req, found, _sim) in part_pairs
                        )
                        st.markdown(chips, unsafe_allow_html=True)
                    if missing_must:
                        st.markdown("**❌ Не хватает обязательных:**")
                        chips = " ".join(f"<span class='chip chip-miss'>{m}</span>" for m in sorted(set(missing_must)))
                        st.markdown(chips, unsafe_allow_html=True)
                    if missing_other:
                        st.markdown("**⭐ Не критично, но отсутствуют:**")
                        chips = " ".join(f"<span class='chip'>{m}</span>" for m in sorted(set(missing_other)))
                        st.markdown(chips, unsafe_allow_html=True)

                # подсветка в резюме (только уникальные токены)
                with st.expander("✨ Подсветка в резюме"):
                    highlight_terms = dedup(strict_terms + partial_terms + missing_must + missing_other)
                    text = cand.source_text

                    if highlight_terms:
                        safe_terms = [re.escape(x) for x in highlight_terms if x]
                        if safe_terms:
                            pattern = r"(" + "|".join(safe_terms) + r")"
                            strict_set  = set(s.lower() for s in strict_terms)
                            partial_set = set(s.lower() for s in partial_terms)
                            miss_must_set  = set(s.lower() for s in missing_must)

                            def recolor(mo):
                                tok = mo.group(0)
                                w = tok.lower()
                                if w in strict_set:
                                    return f"<span style='background:#003b1f;color:#00ff9d;padding:2px 5px;border-radius:6px;'>{tok}</span>"
                                if w in partial_set:
                                    return f"<span style='background:#2b2b2b;color:#ffd95a;padding:2px 5px;border-radius:6px;'>{tok}</span>"
                                if w in miss_must_set:
                                    return f"<span style='background:#4a1111;color:#ff6b6b;padding:2px 5px;border-radius:6px;'>{tok}</span>"
                                return f"<span style='background:#222;color:#bfbfbf;padding:2px 5px;border-radius:6px;'>{tok}</span>"

                            text = re.sub(pattern, recolor, text, flags=re.IGNORECASE)

                    st.markdown(f"<div class='pre'>{text}</div>", unsafe_allow_html=True)

                st.markdown("---")

    else:
        st.info("⬅ Выберите вакансию и нажмите 'Найти'")


# =========================================================
# TAB 2 – ВСЕ КАНДИДАТЫ
# =========================================================
with tab_all_cand:
    st.subheader("📋 Все кандидаты")
    for c in candidates:
        st.markdown(
            f"""
            <div class="card" style="margin-bottom:12px;">
                <h4 style="margin:0;">🧑 {c.name}</h4>
                <b>Уровень:</b> {c.level or '—'}<br>
                <b>Специализация:</b> {c.specialization or '—'}<br>
                <b>Навыки:</b> {', '.join(dedup(c.skills or []))}<br>
                <b>Фреймворки:</b> {', '.join(dedup(c.subskills or []))}<br>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("📄 Полный текст резюме"):
            st.markdown(f"<div class='pre'>{c.source_text}</div>", unsafe_allow_html=True)


# =========================================================
# TAB 3 – ВСЕ ВАКАНСИИ
# =========================================================
with tab_all_jobs:
    st.subheader("📌 Все вакансии")
    for j in jobs:
        st.markdown(
            f"""
            <div class="card" style="background:linear-gradient(145deg,#161616,#111);margin-bottom:18px;">
                <h4 style="margin:0 0 10px;">💼 {j.title}</h4>
                <b>Уровень:</b> {j.level_required or '—'}<br>
                <b>Домен:</b> {j.domain or '—'}<br>
                <b>Специализация:</b> {j.specialization or '—'}<br>
                <b>Стек:</b> {', '.join(dedup(j.stack or [])) or '—'}<br>
                <b>Must:</b> {', '.join(dedup(j.must_have or [])) or '—'}<br>
                <b>Nice:</b> {', '.join(dedup(j.nice_to_have or [])) or '—'}<br>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# TAB 4 — ДОБАВИТЬ КАНДИДАТА
# =========================================================
with tab_add:
    st.subheader("➕ Добавление нового кандидата")

    resume_text = st.text_area(
        "Вставьте текст резюме кандидата:",
        placeholder="Скопируйте сюда резюме любого формата...",
        height=200
    )

    if st.button("✅ Добавить кандидата"):
        if not resume_text.strip():
            st.error("❌ Пустой текст. Вставьте резюме.")
        else:
            try:
                new_cand = parse_candidate(resume_text)

                st.markdown("<div class='card'><b>📊 Извлечённые данные кандидата:</b></div>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📄 Исходное резюме:**")
                    st.markdown(f"<div class='pre'>{resume_text}</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("**✅ Структурировано:**")
                    pretty = []
                    pretty.append(f"Имя: {new_cand.name or '—'}")
                    pretty.append(f"Уровень: {new_cand.level or '—'}")
                    pretty.append(f"Специализация: {new_cand.specialization or '—'}")
                    pretty.append(f"Локация: {new_cand.location or '—'}\n")
                    if new_cand.skills:
                        pretty.append(f"Навыки: {', '.join(dedup(new_cand.skills))}")
                    if new_cand.subskills:
                        pretty.append(f"Фреймворки: {', '.join(dedup(new_cand.subskills))}")
                    if new_cand.years_by_area:
                        pretty.append("\nОпыт по областям:")
                        for area, years in new_cand.years_by_area.items():
                            pretty.append(f"  • {area}: {int(years) if years==int(years) else years} лет")
                    if new_cand.salary_expectation:
                        pretty.append(f"\nОжидаемая ставка: {new_cand.salary_expectation}")
                    st.markdown(f"<div class='pre'>{chr(10).join(pretty)}</div>", unsafe_allow_html=True)

                st.divider()

                with open("data/candidates.json", "r", encoding="utf-8") as f:
                    data = json.load(f)

                new_id = max(int(item["id"]) for item in data) + 1
                new_cand.id = new_id

                data.append(new_cand.model_dump())
                with open("data/candidates.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                rebuild_embeddings()
                st.success(f"✅ Кандидат добавлен! ID = {new_id}")
                st.info("Данные обновлены — можно искать среди новых кандидатов.")

            except Exception as e:
                st.error("❌ Ошибка при обработке кандидата")
                st.exception(e)


# =========================================================
# TAB 5 — ДОБАВИТЬ ВАКАНСИЮ
# =========================================================
with tab_add_job:
    st.subheader("🆕 Добавление новой ваканссии")

    job_text = st.text_area(
        "Вставьте текст заявки:",
        placeholder="Вставьте описание вакансии в свободной форме...",
        height=220
    )

    if st.button("✅ Добавить вакансию"):
        if not job_text.strip():
            st.error("❌ Пустой текст. Вставьте вакансию.")
        else:
            try:
                new_job = parse_job(job_text)

                st.markdown("<div class='card'><b>📊 Извлечённые данные вакансии:</b></div>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📄 Исходное описание:**")
                    st.markdown(f"<div class='pre'>{job_text}</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("**✅ Структурировано:**")
                    pretty = []
                    pretty.append(f"Название: {new_job.title or '—'}")
                    pretty.append(f"Уровень: {new_job.level_required or '—'}")
                    pretty.append(f"Домен: {new_job.domain or '—'}")
                    pretty.append(f"Специализация: {new_job.specialization or '—'}\n")
                    if new_job.stack:
                        pretty.append(f"Стек: {', '.join(dedup(new_job.stack))}")
                    if new_job.substack:
                        pretty.append(f"Фреймворки: {', '.join(dedup(new_job.substack))}")
                    if new_job.must_have:
                        pretty.append(f"Обязательные навыки: {', '.join(dedup(new_job.must_have))}")
                    if new_job.nice_to_have:
                        pretty.append(f"Желательно: {', '.join(dedup(new_job.nice_to_have))}")
                    if new_job.salary_max:
                        pretty.append(f"Бюджет: до {new_job.salary_max}")
                    if getattr(new_job, "exp_min_years_overall", None):
                        y = int(new_job.exp_min_years_overall)
                        pretty.append(f"Опыт: от {y} лет")
                    if getattr(new_job, "exp_min_years_by_area", None):
                        pretty.append("Опыт по областям:")
                        for area, years in new_job.exp_min_years_by_area.items():
                            pretty.append(f"  • {area}: от {int(years)} лет")
                    st.markdown(f"<div class='pre'>{chr(10).join(pretty)}</div>", unsafe_allow_html=True)

                st.divider()

                with open("data/jobs.json", "r", encoding="utf-8") as f:
                    data = json.load(f)

                new_id = max(int(item["id"]) for item in data) + 1
                new_job.id = new_id

                data.append(new_job.model_dump())
                with open("data/jobs.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                st.success(f"✅ Вакансия добавлена! ID = {new_id}")
                st.info("Теперь она доступна в поиске.")

            except Exception as e:
                st.error("❌ Ошибка при обработке вакансии")
                st.exception(e)
