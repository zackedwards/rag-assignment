"""
run_eval.py
Scores the system against the 25 golden questions.

For each question we call the running /query endpoint, then ask Gemini 2.5
Flash to act as a judge and score accuracy and faithfulness from 0 to 5.
Results are written to eval/eval_results.json.

The judge uses structured JSON output so its verdict always parses. The
system's own answer is recorded for every question even if judging fails, so
you can always read what the system said.

The eval is hard capped at 25 questions with no retry loop around questions,
so one run costs well under a dollar.

Usage
    uvicorn serve.api:app --port 8080      # in one terminal
    python -m eval.run_eval                # in another
"""
import asyncio
import json
import os
from pathlib import Path
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8080/query")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
GOLDEN = Path(__file__).parent / "golden.jsonl"
RESULTS = Path(__file__).parent / "eval_results.json"
MAX_QUESTIONS = 25

client = genai.Client(
    vertexai=True,
    project=os.environ["GCP_PROJECT"],
    location=os.environ["GCP_REGION"],
)


class Verdict(BaseModel):
    accuracy: int
    faithfulness: int
    reason: str


JUDGE_PROMPT = """You are grading a retrieval augmented question answering system.

Question:
{question}

Reference answer:
{reference}

System answer:
{system}

Retrieved context the system was allowed to use:
{context}

Score two things from 0 to 5.
accuracy is whether the system answer is factually correct compared to the reference answer.
faithfulness is whether the system answer only makes claims supported by the retrieved context.
Give a one sentence reason."""


def load_golden() -> list[dict]:
    items = []
    with open(GOLDEN) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:MAX_QUESTIONS]


async def ask_system(question: str) -> dict:
    async with httpx.AsyncClient(timeout=120) as http:
        resp = await http.post(API_URL, json={"question": question})
        if resp.status_code != 200:
            raise RuntimeError(f"{resp.status_code} {resp.text[:300]}")
        return resp.json()


async def judge(question: str, reference: str, system_answer: str, context: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        system=system_answer,
        context=context,
    )
    resp = await client.aio.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=500,
            response_mime_type="application/json",
            response_schema=Verdict,
        ),
    )
    if getattr(resp, "parsed", None) is not None:
        v = resp.parsed
        return {"accuracy": v.accuracy, "faithfulness": v.faithfulness, "reason": v.reason}
    # fallback if structured parsing did not populate
    raw = (resp.text or "").strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)
    return {"accuracy": int(data["accuracy"]), "faithfulness": int(data["faithfulness"]), "reason": data.get("reason", "")}


async def main() -> None:
    golden = load_golden()
    per_question = []

    for i, item in enumerate(golden, start=1):
        question = item["question"]
        reference = item.get("answer", item.get("reference_answer", ""))
        record = {"question": question}

        # 1. get the system answer, always record it
        try:
            system = await ask_system(question)
            record["system_answer"] = system.get("answer", "")
            record["citations"] = system.get("citations", [])
        except Exception as e:
            record["api_error"] = str(e)
            per_question.append(record)
            print(f"[{i}/{len(golden)}] api failed, skipping ({e})")
            continue

        # 2. judge it, but a judging failure does not lose the answer
        try:
            context = "\n\n".join(system.get("retrieved_chunks", []))
            scores = await judge(question, reference, record["system_answer"], context)
            record["accuracy"] = scores["accuracy"]
            record["faithfulness"] = scores["faithfulness"]
            record["reason"] = scores["reason"]
            print(f"[{i}/{len(golden)}] acc {scores['accuracy']} faith {scores['faithfulness']}")
        except Exception as e:
            record["judge_error"] = str(e)
            print(f"[{i}/{len(golden)}] answered but judge failed ({e})")

        per_question.append(record)

    scored = [q for q in per_question if "accuracy" in q]
    answered = [q for q in per_question if "system_answer" in q]
    avg_acc = sum(q["accuracy"] for q in scored) / len(scored) if scored else 0
    avg_faith = sum(q["faithfulness"] for q in scored) / len(scored) if scored else 0

    output = {
        "n_questions": len(golden),
        "n_answered": len(answered),
        "n_scored": len(scored),
        "average_accuracy": round(avg_acc, 2),
        "average_faithfulness": round(avg_faith, 2),
        "per_question": per_question,
    }
    RESULTS.write_text(json.dumps(output, indent=2))
    print(f"\nanswered {len(answered)}/{len(golden)}, scored {len(scored)}/{len(golden)}")
    print(f"average accuracy {avg_acc:.2f}, average faithfulness {avg_faith:.2f}")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    asyncio.run(main())
