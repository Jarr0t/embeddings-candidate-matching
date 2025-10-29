import os
import json
import streamlit as st

from rank import rank
from extract_job import Job
from extract_candidate import Candidate, parse_candidate
from build_embeddings import build as rebuild_embeddings

st.set_page_config(page_title="ML Подбор кандидатов", layout="wide")
st.title("🚀 AI-платформа подбора разработчиков")


def year_word(y: float):
    y = int(y)
    if y % 10 == 1 and y % 100 != 11:
        return "год"
    if 2 <= y % 10 <= 4 and not (12 <= y % 100 <= 14):
        return "года"
    return "лет"


# ============================
#   ЗАГРУЗКА ДАННЫХ
# ============================
def load_data():
    with open("data/jobs.json", "r", encoding="utf-8") as f:
        jobs = [Job.model_validate(x) for x in json.load(f)]

    with open("data/candidates.json", "r", encoding="utf-8") as f:
        candidates = [Candidate.model_validate(x) for x in json.load(f)]

    return jobs, candidates


jobs, candidates = load_data()
job_list = {job.id: job.title for job in jobs}

# ============================
#   SIDEBAR НАСТРОЙКИ
# ============================
st.sidebar.header("⚙ Настройки ранжирования")

vector_w = st.sidebar.slider("🧠 Векторная близость", 0.0, 1.0, 0.5)
skill_w = st.sidebar.slider("🛠 Покрытие навыков", 0.0, 1.0, 0.3)
exp_w = st.sidebar.slider("📅 Опыт", 0.0, 1.0, 0.1)
domain_w = st.sidebar.slider("🏢 Домен", 0.0, 1.0, 0.05)
level_w = st.sidebar.slider("🎖 Уровень", 0.0, 1.0, 0.05)

top_k = st.sidebar.slider("Сколько лучших кандидатов показать?", 1, 10, 3)

threshold = st.sidebar.slider(
    "🔎 Минимальный score для кандидата (порог отсечения)",
    0.0, 1.0, 0.0, 0.01
)

if st.sidebar.button("🔄 Обновить данные"):
    jobs, candidates = load_data()
    job_list = {job.id: job.title for job in jobs}
    st.sidebar.success("✅ Данные обновлены! Перезапускать приложение не нужно.")

# ============================
#   ВКЛАДКИ
# ============================
tab_find, tab_add, tab_add_job = st.tabs(["🔍 Поиск кандидатов", "➕ Добавить кандидата", "🆕 Добавить вакансию"])

# ====================================================================
#   ✅ TAB 1 — ПОИСК
# ====================================================================
with tab_find:

    st.subheader("📌 Выберите вакансию")
    job_id = st.selectbox(
        "Вакансии:",
        options=list(job_list.keys()),
        format_func=lambda x: job_list[x]
    )
    job = next(j for j in jobs if j.id == job_id)

    # Описание вакансии
    with st.expander("📄 Полное описание вакансии", expanded=True):
        st.markdown(f"### **{job.title}** — {job.level_required}")
        st.write(f"**Домен:** {job.domain or 'не указан'}")
        st.write(f"**Специализация:** {job.specialization or '—'}")
        st.write(f"**Стек:** {', '.join(job.stack or [])}")
        st.write(f"**Фреймворки:** {', '.join(job.substack or [])}")

        # ✅ Опыт
        if getattr(job, "exp_min_years_overall", None):
            y = job.exp_min_years_overall
            st.write(f"**Опыт:** от {int(y)} {year_word(y)}")
        elif getattr(job, "exp_min_years_by_area", None):
            st.write("**Опыт по областям:**")
            for area, years in job.exp_min_years_by_area.items():
                st.write(f"• {area}: от {int(years)} {year_word(years)}")
        else:
            st.write("**Опыт:** не указан")

        st.write(f"**Обязательные навыки:** ✅ {', '.join(job.must_have or [])}")

        if job.nice_to_have:
            st.write(f"**Желательно:** ⭐ {', '.join(job.nice_to_have)}")

        if job.salary_max:
            st.success(f"💰 Бюджет: до {job.salary_max}")

        with st.expander("🧾 Исходный текст заявки"):
            st.code(job.source_text, language=None)

    # ============================
    #   ПОИСК КАНДИДАТОВ
    # ============================
    if st.button("🔥 Найти лучших кандидатов"):
        st.write("⏳ Подбираем лучших…")

        results = rank(
            job_id,
            top_n=top_k,
            custom_weights=dict(
                vector=vector_w,
                skills=skill_w,
                experience=exp_w,
                domain=domain_w,
                level=level_w
            ),
            threshold=threshold
        )



        st.success("✅ Готово! Лучшие кандидаты:")
        if not results or len(results) == 0:
            st.warning("❗ По заданному порогу ни один кандидат не подошёл.")
        else:

            def pct(x):
                try:
                    return f"{float(x)*100:.0f}%"
                except:
                    return "—"

            for r in results:
                cand = next(c for c in candidates if c.id == r["candidate_id"])

                st.markdown(f"""
                <div style="
                    border:1px solid #444; 
                    padding:15px; 
                    border-radius:10px;
                    margin-bottom:15px;
                    background:#141414;
                ">
                <h3 style="margin:0;">🧑 {r['name']} — Score: <b>{r['final_score']}</b></h3>
                </div>
                """, unsafe_allow_html=True)

                vec = float(r["vector_similarity"])
                cov = float(r["skill_coverage"])
                exp = float(r["experience_score"])

                cols = st.columns(4)
                cols[0].metric("🧠 Семантика", pct(vec))
                cols[1].metric("🛠 Навыки", pct(cov))
                cols[2].metric("📅 Опыт", pct(exp))
                cols[3].metric("🏢 Домен", "✅" if r["domain_match"] else "❌")

                st.progress(cov)
                st.write(f"Покрытие навыков: **{pct(cov)}**")

                with st.expander("🔍 Полное резюме кандидата"):
                    st.write(f"**Уровень:** {cand.level}")
                    st.write(f"**Специализация:** {cand.specialization}")
                    st.write(f"**Навыки:** {', '.join(cand.skills or [])}")
                    st.write(f"**Фреймворки:** {', '.join(cand.subskills or [])}")

                    if cand.years_by_area:
                        formatted_exp = []
                        for area, years in cand.years_by_area.items():
                            y = int(years) if years == int(years) else years
                            formatted_exp.append(f"{area} — {y} {year_word(y)}")
                        st.write(f"**Опыт:** {', '.join(formatted_exp)}")
                    else:
                        st.write("**Опыт:** не указан")

                    st.write(f"**Локация:** {cand.location}")
                    if cand.salary_expectation:
                        st.success(f"💰 Ожидаемая ставка: {cand.salary_expectation}")
                    st.write(f"**Текст резюме:**\n\n> {cand.source_text}")

                with st.expander("🧩 Почему этот кандидат? (объяснение модели)"):
                    if r['must_have_ok']:
                        st.write("✅ Все обязательные навыки присутствуют")
                    else:
                        st.write("❌ Не все must-have навыки есть")

                    if r['domain_match']:
                        st.write("✅ Опыт в нужной доменной области")
                    if r['level_match']:
                        st.write("✅ Соответствие уровню вакансии")

                    st.write(f"🛠 Навыки: **{pct(cov)}**")
                    st.write(f"📅 Опыт: **{pct(exp)}**")
                    st.write(f"🧠 Семантика: **{pct(vec)}**")

                    missing = []
                    for skill in job.must_have or []:
                        if skill not in (cand.skills or []) and skill not in (cand.subskills or []):
                            missing.append(skill)
                    if missing:
                        st.warning(f"❗ Отсутствуют обязательные навыки: {', '.join(missing)}")

                st.markdown("---")
    else:
        st.info("⬅ Выберите вакансию и нажмите кнопку!")

# ====================================================================
#   ✅ TAB 2 — ДОБАВЛЕНИЕ КАНДИДАТА
# ====================================================================
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
                # 1) Парсим через LLM
                new_cand = parse_candidate(resume_text)

                # 2) Загружаем старый JSON
                with open("data/candidates.json", "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 3) Новый ID
                new_id = max(int(item["id"]) for item in data) + 1
                new_cand.id = new_id

                # 4) Добавляем в JSON
                data.append(new_cand.model_dump())

                with open("data/candidates.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                # 5) Перестраиваем вектора
                rebuild_embeddings()

                st.success(f"✅ Кандидат добавлен! ID = {new_id}")
                st.info("Данные обновлены — можно сразу искать среди новых кандидатов.")

            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

# ====================================================================
#   ✅ TAB 3 — ДОБАВЛЕНИЕ ВАКАНСИИ
# ====================================================================
from extract_job import parse_job

with tab_add_job:
    st.subheader("🆕 Добавление новой вакансии")

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
                # 1) Разбор через LLM
                new_job = parse_job(job_text)

                # 2) Загружаем старый json
                with open("data/jobs.json", "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 3) Новый ID
                new_id = max(int(item["id"]) for item in data) + 1
                new_job.id = new_id

                # 4) Добавляем в JSON
                data.append(new_job.model_dump())

                with open("data/jobs.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                st.success(f"✅ Вакансия добавлена! ID = {new_id}")
                st.info("Теперь её можно выбрать в поиске кандидатов.")

            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
                st.exception(e)

