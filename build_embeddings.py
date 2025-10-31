import os
import json
import numpy as np
import faiss
from openai import OpenAI
from extract_candidate import Candidate

client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"

VEC_PATH = "data/candidate_vectors.npy"
IDX_PATH = "data/candidate_index.faiss"
MAP_PATH = "data/candidate_id_map.json"
CANDS_PATH = "data/candidates.json"


def candidate_text_repr(c: Candidate) -> str:
    parts = []
    if c.level: parts.append(f"Level: {c.level}")
    if c.specialization: parts.append(f"Spec: {c.specialization}")
    if c.domains: parts.append("Domains: " + ", ".join(c.domains))
    if c.skills: parts.append("Skills: " + ", ".join(c.skills))
    if c.subskills: parts.append("Frameworks: " + ", ".join(c.subskills))
    return "\n".join(parts)


def embed_batch(texts):
    """Батчевый запрос, чтобы было быстрее и меньше запросов к API"""
    if not texts:
        return []
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]


def build():
    with open(CANDS_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    candidates = [Candidate.model_validate(x) for x in items]

    # =====================================
    # ✅ ЕСЛИ файл есть — догружаем только новых
    # =====================================
    if os.path.exists(VEC_PATH) and os.path.exists(IDX_PATH) and os.path.exists(MAP_PATH):
        print("✔ Найден существующий индекс — добавляем только новых")

        old_vectors = np.load(VEC_PATH)
        index = faiss.read_index(IDX_PATH)

        with open(MAP_PATH, "r", encoding="utf-8") as f:
            old_ids = json.load(f)

        old_ids_set = set(str(x) for x in old_ids)

        new_cands = [c for c in candidates if str(c.id) not in old_ids_set]

        if not new_cands:
            print("✅ Новых кандидатов нет. Индекс уже актуален.")
            return

        print(f"🔨 Новых кандидатов: {len(new_cands)}")

        # Получаем тексты
        texts = [candidate_text_repr(c) for c in new_cands]

        # Батчево эмбеддим
        B = 64
        new_vecs = []
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            batch_emb = embed_batch(batch)
            new_vecs.extend(batch_emb)

        new_vecs = np.array(new_vecs).astype("float32")

        # Добавляем в память + индекс
        updated_vectors = np.vstack([old_vectors, new_vecs])
        index.add(new_vecs)

        # Обновляем id_map
        new_ids = old_ids + [c.id for c in new_cands]

        # Сохраняем
        np.save(VEC_PATH, updated_vectors)
        faiss.write_index(index, IDX_PATH)
        with open(MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(new_ids, f, ensure_ascii=False, indent=2)

        print("✅ Индекс обновлён. Новые кандидаты добавлены.")
        print(f"📦 Всего эмбеддингов: {updated_vectors.shape[0]}")

    # =====================================
    # ❗ ЕСЛИ ФАЙЛОВ НЕТ — строим всё с нуля
    # =====================================
    else:
        print("⚠ Индекса нет — создаём с нуля")

        texts = [candidate_text_repr(c) for c in candidates]

        B = 64
        vectors = []
        for i in range(0, len(texts), B):
            batch = texts[i:i+B]
            batch_emb = embed_batch(batch)
            vectors.extend(batch_emb)

        vectors = np.array(vectors).astype("float32")

        np.save(VEC_PATH, vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        faiss.write_index(index, IDX_PATH)

        ids = [c.id for c in candidates]
        with open(MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False, indent=2)

        print("✅ Построено с нуля")
        print(f"📦 Всего кандидатов: {len(candidates)}")
        print(f"📦 Векторов: {vectors.shape[0]}")


if __name__ == "__main__":
    build()
