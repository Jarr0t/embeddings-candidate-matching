import re
import json
import streamlit as st
from extract_job import parse_job

from rank import rank
from extract_job import Job
from extract_candidate import Candidate, parse_candidate
from build_embeddings import build as rebuild_embeddings

st.set_page_config(page_title="ML Подбор кандидатов", layout="wide")
st.title("🚀 AI-платформа подбора разработчиков")

def pretty_box(text):
    st.markdown(
        """
        <style>
        .pretty-box {
            background:#111;
            border:1px solid #333;
            border-radius:8px;
            padding:12px;
            font-family:monospace;
            font-size:14px;
            line-height:1.45;
            white-space:pre-wrap;
            overflow-x:auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f"<div class='pretty-box'>{text}</div>", unsafe_allow_html=True)


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

strict_thr = st.sidebar.slider(
    "✅ Порог строгого совпадения навыков (STRICT)",
    0.0, 1.0, 0.80, 0.01
)

partial_thr = st.sidebar.slider(
    "🔶 Порог частичного совпадения навыков (PARTIAL)",
    0.0, 1.0, 0.60, 0.01
)

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
tab_find, tab_all_cand, tab_all_jobs, tab_add, tab_add_job = st.tabs([
    "🔍 Поиск кандидатов",
    "📋 Все кандидаты",
    "📌 Все вакансии",
    "➕ Добавить кандидата",
    "🆕 Добавить вакансию"
])


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
            threshold=threshold,
            strict_thr = strict_thr,
            partial_thr = partial_thr,
        )

        if not results:
            st.warning("❗ По заданному порогу ни один кандидат не подошёл.")
        else:
            st.success("✅ Готово! Лучшие кандидаты:")
            def pct(x):
                try:
                    return f"{float(x) * 100:.0f}%"
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

                    st.write(f"🧠 Семантика: **{pct(vec)}**")
                    st.write(f"🛠 Навыки: **{pct(cov)}**")
                    st.write(f"📅 Опыт: **{pct(exp)}**")
                    st.write(f"🏢 Домен: {'✅' if r['domain_match'] else '❌'}")
                    st.write(f"🎖 Уровень: {'✅' if r['level_match'] else '❌'}")
                    st.markdown("---")

                    st.markdown("### 🔥 Совпадение навыков (семантическое)")

                    matches = r.get("skill_matches", [])  # [(req, found, sim)]
                    partials = r.get("skill_partials", [])  # [(req, found, sim)]
                    missing_must = r.get("missing_must_have", [])  # [skill names]

                    # собираем все требования вакансии
                    req_all = (job.must_have or []) + (job.nice_to_have or []) + (job.stack or []) + (
                                job.substack or [])

                    # какие не покрыты вообще
                    covered_req = set([m[0] for m in matches] + [p[0] for p in partials] + missing_must)
                    missing_other = [x for x in req_all if x not in covered_req]

                    # ✅ строгие совпадения
                    if matches:
                        st.markdown("**✅ Совпадают строго:**")
                        chips = " ".join([
                            f"<span style='background:#1f4f2f; color:#00ff9d; padding:4px 10px; border-radius:8px; margin:2px; display:inline-block;'>{m[0]} ➝ {m[1]}</span>"
                            for m in matches
                        ])
                        st.markdown(chips, unsafe_allow_html=True)

                    # 🔶 частичные совпадения
                    if partials:
                        st.markdown("**🔶 Частично совпадают:**")
                        chips = " ".join([
                            f"<span style='background:#2b2b2b; color:#ffd95a; padding:4px 10px; border-radius:8px; margin:2px; display:inline-block;'>{p[0]} ≈ {p[1]}</span>"
                            for p in partials
                        ])
                        st.markdown(chips, unsafe_allow_html=True)

                    # ❌ must-have отсутствуют
                    if missing_must:
                        st.markdown("**❌ Не хватает обязательных:**")
                        chips = " ".join([
                            f"<span style='background:#4a1111; color:#ff6b6b; padding:4px 10px; border-radius:8px; margin:2px; display:inline-block;'>{m}</span>"
                            for m in missing_must
                        ])
                        st.markdown(chips, unsafe_allow_html=True)

                    # ⭐ просто нет
                    if missing_other:
                        st.markdown("**⭐ Не критично, но нет:**")
                        chips = " ".join([
                            f"<span style='background:#222; color:#aaa; padding:4px 10px; border-radius:8px; margin:2px; display:inline-block;'>{m}</span>"
                            for m in missing_other
                        ])
                        st.markdown(chips, unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("### ✨ Подсветка в резюме")

                    # подсветка в исходном тексте
                    highlight_words = (
                            [m[0] for m in matches] +
                            [p[0] for p in partials] +
                            missing_must +
                            missing_other
                    )

                    text = cand.source_text

                    if highlight_words:
                        # экранируем слова, если есть спецсимволы
                        pattern = r"(" + "|".join([re.escape(w) for w in highlight_words]) + r")"


                        def repl(mo):
                            w = mo.group(0)
                            if any(w == mm[0] for mm in matches):
                                return f"<span style='background:#003b1f; color:#00ff9d; padding:2px 5px; border-radius:6px;'>{w}</span>"
                            if w in missing_must:
                                return f"<span style='background:#4a1111; color:#ff6b6b; padding:2px 5px; border-radius:6px;'>{w}</span>"
                            if any(w == pp[0] for pp in partials):
                                return f"<span style='background:#2b2b2b; color:#ffd95a; padding:2px 5px; border-radius:6px;'>{w}</span>"
                            return f"<span style='background:#222; color:#aaa; padding:2px 5px; border-radius:6px;'>{w}</span>"


                        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

                    st.markdown(
                        f"""
                        <div style='background:#111; padding:12px; border-radius:8px; border:1px solid #333; white-space:pre-wrap;'>
                            {text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("---")
    else:
        st.info("⬅ Выберите вакансию и нажмите кнопку!")

# ====================================================================
#   ✅ TAB 2 — СПИСОК ВСЕХ КАНДИДАТОВ
# ====================================================================
with tab_all_cand:
    st.subheader("📋 Все кандидаты")

    for c in candidates:
        st.markdown(
            f"""
            <div style="
                border:1px solid #333; 
                border-radius:10px; 
                padding:15px; 
                margin-bottom:12px; 
                background:#141414;
            ">
                <h4 style="margin:0;">🧑 {c.name}</h4>
                <b>Уровень:</b> {c.level or '—'}<br>
                <b>Специализация:</b> {c.specialization or '—'}<br>
                <b>Навыки:</b> {', '.join(c.skills or [])}<br>
                <b>Фреймворки:</b> {', '.join(c.subskills or [])}<br>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("📄 Полный текст резюме"):
            st.markdown(
                f"""
                <div style="
                    background:#111; 
                    padding:12px; 
                    border-radius:8px; 
                    border:1px solid #333; 
                    white-space:pre-wrap;
                ">
                    {c.source_text}
                </div>
                """,
                unsafe_allow_html=True
            )

# ====================================================================
#   ✅ TAB 3 — СПИСОК ВСЕХ ВАКАНСИЙ
# ====================================================================
with tab_all_jobs:
    st.subheader("📌 Все вакансии")

    for j in jobs:
        st.markdown(
            f"""
            <div style="
                border: 1px solid #2c2c2c;
                border-radius: 14px;
                padding: 18px;
                margin-bottom: 18px;
                background: linear-gradient(145deg, #161616, #111);
                box-shadow: 0 2px 6px rgba(0,0,0,0.4);
            ">
                <h4 style="margin-top:0; margin-bottom:10px;">💼 {j.title}</h4>
                <div style="line-height:1.6; font-size:15px;">
                    <b>Уровень:</b> {j.level_required or '—'}<br>
                    <b>Домен:</b> {j.domain or '—'}<br>
                    <b>Специализация:</b> {j.specialization or '—'}<br>
                    <b>Стек:</b> {', '.join(j.stack or []) or '—'}<br>
                    <b>Must-have:</b> {', '.join(j.must_have or []) or '—'}<br>
                    <b>Nice-to-have:</b> {', '.join(j.nice_to_have or []) or '—'}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("📄 Развернуть описание", expanded=False):
            st.markdown(
                f"""
                <div style="
                    background:#121212; 
                    padding:14px; 
                    border-radius:10px; 
                    border:1px solid #2a2a2a;
                    margin-top:5px;
                    white-space:pre-wrap;
                    line-height:1.55;
                ">
                    {j.source_text}
                </div>
                """,
                unsafe_allow_html=True
            )

# ====================================================================
#   ✅ TAB 4 — ДОБАВЛЕНИЕ КАНДИДАТА
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

                st.markdown("""
                <div style="
                    background:#1e1e1e;
                    padding:15px;
                    border-radius:10px;
                    border:1px solid #444;
                    margin-top:15px;
                    margin-bottom:10px;">
                    <b>📊 Как система обработала резюме кандидата:</b>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                # ✅ Исходный текст
                with col1:
                    st.markdown("**📄 Исходное резюме:**")
                    pretty_box(resume_text)

                # ✅ Форматированный результат
                with col2:
                    st.markdown("**✅ Извлечённые данные:**")

                    result_pretty = ""
                    result_pretty += f"Имя: {new_cand.name or '—'}\n"
                    result_pretty += f"Уровень: {new_cand.level or '—'}\n"
                    result_pretty += f"Специализация: {new_cand.specialization or '—'}\n"
                    result_pretty += f"Локация: {new_cand.location or '—'}\n\n"

                    if new_cand.skills:
                        result_pretty += f"Навыки: {', '.join(new_cand.skills)}\n"
                    if new_cand.subskills:
                        result_pretty += f"Фреймворки: {', '.join(new_cand.subskills)}\n"

                    if new_cand.years_by_area:
                        result_pretty += "\nОпыт по областям:\n"
                        for area, years in new_cand.years_by_area.items():
                            y = int(years) if years == int(years) else years
                            result_pretty += f"  • {area}: {y} лет\n"

                    if new_cand.salary_expectation:
                        result_pretty += f"\nОжидаемая ставка: {new_cand.salary_expectation}\n"

                    pretty_box(result_pretty)

                st.divider()

                # 2) Загружаем старый JSON
                with open("data/candidates.json", "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 3) Новый ID
                new_id = max(int(item["id"]) for item in data) + 1
                new_cand.id = new_id

                # 4) Добавляем
                data.append(new_cand.model_dump())

                with open("data/candidates.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                # 5) Перестроить эмбеддинги
                rebuild_embeddings()

                st.success(f"✅ Кандидат добавлен! ID = {new_id}")
                st.info("Данные обновлены — можно искать среди новых кандидатов.")

            except Exception as e:
                st.error("❌ Ошибка при обработке кандидата")
                st.exception(e)


# ====================================================================
#   ✅ TAB 5 — ДОБАВЛЕНИЕ ВАКАНСИИ
# ====================================================================
with tab_add_job:
    st.subheader("🆕 Добавление новой ваканссии")

    job_text = st.text_area(
        "Вставьте текст заявки:",
        placeholder="Вставьте описание ваканссии в свободной форме...",
        height=220
    )

    if st.button("✅ Добавить вакансию"):
        if not job_text.strip():
            st.error("❌ Пустой текст. Вставьте вакансию.")
        else:
            try:
                new_job = parse_job(job_text)

                st.markdown("""
                <div style="
                    background:#1e1e1e;
                    padding:15px;
                    border-radius:10px;
                    border:1px solid #444;
                    margin-top:15px;
                    margin-bottom:10px;">
                    <b>📊 Как система обработала вашу вакансию:</b>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                # ✅ Исходный текст
                with col1:
                    st.markdown("**📄 Исходное описание:**")
                    pretty_box(job_text)

                # ✅ Форматированный результат
                with col2:
                    st.markdown("**✅ Извлечённые данные:**")

                    # Формируем удобочитаемый вывод
                    result_pretty = ""

                    result_pretty += f"Название: {new_job.title or '—'}\n"
                    result_pretty += f"Уровень: {new_job.level_required or '—'}\n"
                    result_pretty += f"Домен: {new_job.domain or '—'}\n"
                    result_pretty += f"Специализация: {new_job.specialization or '—'}\n\n"

                    if new_job.stack:
                        result_pretty += f"Стек: {', '.join(new_job.stack)}\n"
                    if new_job.substack:
                        result_pretty += f"Фреймворки: {', '.join(new_job.substack)}\n"

                    if new_job.must_have:
                        result_pretty += f"Обязательные навыки: {', '.join(new_job.must_have)}\n"
                    if new_job.nice_to_have:
                        result_pretty += f"Желательно: {', '.join(new_job.nice_to_have)}\n"

                    if new_job.salary_max:
                        result_pretty += f"Бюджет: до {new_job.salary_max}\n"

                    if getattr(new_job, "exp_min_years_overall", None):
                        y = int(new_job.exp_min_years_overall)
                        result_pretty += f"Опыт: от {y} лет\n"

                    if getattr(new_job, "exp_min_years_by_area", None):
                        result_pretty += "Опыт по областям:\n"
                        for area, years in new_job.exp_min_years_by_area.items():
                            result_pretty += f"  • {area}: от {int(years)} лет\n"

                    pretty_box(result_pretty)

                st.divider()

                # ✅ Сохранение
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
                st.error("❌ Ошибка при обработке ваканссии")
                st.exception(e)

