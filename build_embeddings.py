import json
import numpy as np
import faiss
from openai import OpenAI
from extract_candidate import Candidate

client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"


def candidate_text_repr(c: Candidate) -> str:
    parts = []
    if c.level: parts.append(f"Level: {c.level}")
    if c.specialization: parts.append(f"Spec: {c.specialization}")
    if c.domains: parts.append("Domains: " + ", ".join(c.domains))
    if c.skills: parts.append("Skills: " + ", ".join(c.skills))
    if c.subskills: parts.append("Frameworks: " + ", ".join(c.subskills))
    return "\n".join(parts)


def build():
    with open("data/candidates.json", "r", encoding="utf-8") as f:
        items = json.load(f)

    candidates = [Candidate.model_validate(x) for x in items]

    vectors = []
    id_map = []

    for c in candidates:
        text = candidate_text_repr(c)

        emb = client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        ).data[0].embedding

        vectors.append(emb)
        id_map.append(c.id)

    vectors = np.array(vectors).astype("float32")

    np.save("data/candidate_vectors.npy", vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, "data/candidate_index.faiss")

    with open("data/candidate_id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    print("✅ Embeddings built and index saved")


if __name__ == "__main__":
    build()
