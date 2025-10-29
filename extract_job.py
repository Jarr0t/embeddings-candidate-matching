import json
import sys
import os
from typing import List, Dict, Optional
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

class Job(BaseModel):
    id: Optional[int] = None
    title: Optional[str] = None
    level_required: Optional[str] = None                # junior/middle/senior/lead/unknown
    specialization: Optional[str] = None                # backend/frontend/mobile/fullstack/data/qa/devops/unknown
    domain: Optional[str] = None                        # fintech, retail, healthcare …
    stack: List[str] = []                               # python, java, swift…
    substack: List[str] = []                            # django, spring, xcode …
    must_have: List[str] = []
    nice_to_have: List[str] = []
    exp_min_years_by_area: Dict[str, float] = {}        # {"backend": 2}
    location: Optional[str] = None
    salary_max: Optional[float] = None

    source_text: Optional[str] = None


def parse_job(job_text: str) -> Job:
    prompt = f"""
Ты — бот, который структурирует данные вакансии. 
Верни строго JSON формата schema. Никаких комментариев.

Вакансия:
\"\"\"{job_text}\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "JobSchema",
                "schema": Job.model_json_schema()
            }
        }
    )

    raw_json = response.choices[0].message.content
    parsed_job: Job = Job.model_validate_json(raw_json)
    parsed_job.source_text = job_text
    return parsed_job



if __name__ == "__main__":
    if len(sys.argv) >= 2:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            text = f.read()

        result: Job = parse_job(text)

        print(result.model_dump_json(indent=2))

        DATA_PATH = os.path.join("data", "jobs.json")
        os.makedirs("data", exist_ok=True)

        if not os.path.exists(DATA_PATH):
            jobs = []
        else:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                try:
                    jobs = json.load(f)
                except:
                    jobs = []

        if jobs:
            max_id = max(j["id"] for j in jobs if "id" in j)
            result.id = max_id + 1
        else:
            result.id = 1

        jobs.append(result.model_dump())

        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Saved job to {DATA_PATH} (id={result.id})")
    else:
        print("Usage: python extract_job.py vacancy.txt")
