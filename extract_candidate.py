import json
import sys
import os
from typing import List, Dict, Optional
from pydantic import BaseModel
from pydantic import field_validator
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

class Candidate(BaseModel):
    name: Optional[str] = None
    level: Optional[str] = None                          # junior/middle/senior/lead/unknown
    specialization: Optional[str] = None                 # backend/frontend/...
    domains: List[str] = []
    skills: List[str] = []                               # python, go, java...
    subskills: List[str] = []                            # django, react, kubernetes...
    years_by_area: Dict[str, float] = {}                 # {"backend":3}
    location: Optional[str] = None
    salary_expectation: Optional[float] = None

    @field_validator('salary_expectation', mode='before')
    def parse_salary(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            import re
            nums = re.findall(r'\d+', v.replace(",", "."))
            return float(nums[-1]) if nums else None
        return None


    contacts: Dict[str, Optional[str]] = {               # email/phone/tg/...
        "email": None,
        "phone": None,
        "telegram": None,
        "linkedin": None,
        "other": None
    }

    source_text: Optional[str] = None
    id: Optional[int] = None


def parse_candidate(text: str) -> Candidate:
    schema = {
        "name": "Candidate",
        "schema": Candidate.model_json_schema()
    }

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": f"Extract candidate structured data:\n{text}"}],
        response_format={"type": "json_schema", "json_schema": schema}
    )

    msg = resp.choices[0].message

    # ✅ 1 — если клиент поддерживает .parsed → используем
    if hasattr(msg, "parsed") and msg.parsed:
        return Candidate.model_validate(msg.parsed)

    # ✅ 2 — fallback: берём JSON из текста
    raw = msg.content
    try:
        data = json.loads(raw)
    except:
        # иногда модель добавляет лишние строки → ищем JSON внутри
        import re
        json_match = re.search(r"\{.*\}", raw, re.S)
        if not json_match:
            raise ValueError("LLM did not return JSON.")

        data = json.loads(json_match.group(0))

    return Candidate.model_validate(data)




if __name__ == "__main__":
    if len(sys.argv) >= 2:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            text = f.read()

        result: Candidate = parse_candidate(text)

        print(result.model_dump_json(indent=2))

        DATA_PATH = os.path.join("data", "candidates.json")
        os.makedirs("data", exist_ok=True)

        if not os.path.exists(DATA_PATH):
            candidates = []
        else:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                try:
                    candidates = json.load(f)
                except:
                    candidates = []

        if candidates:
            max_id = max(c["id"] for c in candidates if "id" in c)
            result.id = max_id + 1
        else:
            result.id = 1

        candidates.append(result.model_dump())
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Saved candidate to {DATA_PATH} (id={result.id})")
    else:
        print("Usage: python extract_candidate.py resume.txt")
