import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SKILLS_PATH = "data/skills.json"
VEC_PATH = "data/skills.npy"
INDEX_PATH = "data/skills.faiss"
MAP_PATH = "data/skill_id_map.json"

MODEL = "text-embedding-3-large"


def load_index():
    if not (os.path.exists(VEC_PATH) and os.path.exists(INDEX_PATH) and os.path.exists(MAP_PATH)):
        return None, None, None
    vectors = np.load(VEC_PATH)
    index = faiss.read_index(INDEX_PATH)
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return vectors, index, id_map


def build_or_update(skills):
    vecs, index, id_map = load_index()

    if vecs is None:
        print("Создаём новый индекс навыков...")
        vecs = []
        id_map = {}
        index = None

    # отфильтровать уже существующие
    new = [s for s in skills if str(s) not in id_map]

    if not new:
        print("✅ Все навыки уже в индексе")
        return

    print(f"Добавляем {len(new)} новых навыков...")

    # эмбеддинги батчами
    B = 100
    new_vecs = []
    for i in range(0, len(new), B):
        chunk = new[i:i+B]
        emb = client.embeddings.create(model=MODEL, input=chunk)
        new_vecs.extend([d.embedding for d in emb.data])

    new_vecs = np.array(new_vecs).astype('float32')

    # объединяем
    if len(vecs) == 0:
        vecs = new_vecs
    else:
        vecs = np.vstack([vecs, new_vecs])

    # FAISS индекс
    d = new_vecs.shape[1]
    if index is None:
        index = faiss.IndexFlatL2(d)

    index.add(new_vecs)

    # обновляем map
    for i, s in enumerate(new):
        id_map[str(s)] = index.ntotal - len(new) + i

    # сохраняем
    np.save(VEC_PATH, vecs)
    faiss.write_index(index, INDEX_PATH)
    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)

    print("✅ skills.faiss и skills.npy обновлены!")
