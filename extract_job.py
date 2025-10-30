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


def parse_job(text: str) -> Job:
    schema = {
        "name": "Job",
        "schema": Job.model_json_schema()
    }

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": f"Extract structured job data:\n{text}"}],
        response_format={"type": "json_schema", "json_schema": schema}
    )

    msg = resp.choices[0].message

    if hasattr(msg, "parsed") and msg.parsed:
        data = msg.parsed
    else:
        raw = msg.content
        try:
            data = json.loads(raw)
        except:
            import re
            json_match = re.search(r"\{.*\}", raw, re.S)
            if not json_match:
                raise ValueError("LLM did not return JSON.")
            data = json.loads(json_match.group(0))

    exp_by_area = data.get("exp_min_years_by_area")
    if exp_by_area and isinstance(exp_by_area, dict):

        cleaned = {}
        for k, v in exp_by_area.items():

            if isinstance(v, (int, float)):
                cleaned[k] = float(v)
                continue

            if isinstance(v, str):
                v_clean = v.replace(" ", "").lower()

                if "-" in v_clean:
                    part = v_clean.split("-")[0]
                    try:
                        cleaned[k] = float(part)
                        continue
                    except:
                        pass

                if v_clean.endswith("+"):
                    try:
                        cleaned[k] = float(v_clean[:-1])
                        continue
                    except:
                        pass

                import re
                nums = re.findall(r"\d+\.?\d*", v_clean)
                if nums:
                    cleaned[k] = float(nums[0])
                    continue

            cleaned[k] = 0

        data["exp_min_years_by_area"] = cleaned

    return Job.model_validate(data)





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
