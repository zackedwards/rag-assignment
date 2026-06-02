"""
run_eval.py
Scores the system against the 25 golden questions.

For each question we call the running /query endpoint, then ask Gemini 2.5
Flash to act as a judge and score accuracy and faithfulness from 0 to 5.
Results are written to eval/eval_results.json.

The eval is hard capped at 25 questions with no retry loop, so one run costs
well under a dollar.

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
accuracy: is the system answer factually correct compared to the reference answer.
faithfulness: does the system answer only make claims supported by the retrieved context.

Reply with JSON only, no markdown, no commentary, in exactly this shape:
{{"accuracy": <int 0-5>, "faithfulness": <int 0-5>, "reason": "<one short sentence>"}}"""


def load_golden() -> list[dict]:
    items = []
    with open(GOLDEN) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:MAX_QUESTIONS]


async def ask_system(question: str) -> dict:
    async with httpx.AsyncClient(timeout=60) as http:
        resp = await http.post(API_URL, json={"question": question})
        resp.raise_for_status()
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
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=300),
    )
    raw = (resp.text or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


async def main() -> None:
    golden = load_golden()
    per_question = []

    for i, item in enumerate(golden, start=1):
        question = item["question"]
        reference = item.get("answer", item.get("reference_answer", ""))
        try:
            system = await ask_system(question)
            context = "\n\n".join(system.get("retrieved_chunks", []))
            scores = await judge(question, reference, system["answer"], context)
            per_question.append(
                {
                    "question": question,
                    "system_answer": system["answer"],
                    "accuracy": scores["accuracy"],
                    "faithfulness": scores["faithfulness"],
                    "reason": scores.get("reason", ""),
                }
            )
            print(f"[{i}/{len(golden)}] acc {scores['accuracy']} faith {scores['faithfulness']}")
        except Exception as e:
            # log and move on, no retry loop per the cost guardrail
            print(f"[{i}/{len(golden)}] failed, skipping ({e})")
            per_question.append({"question": question, "error": str(e)})

    scored = [q for q in per_question if "accuracy" in q]
    avg_acc = sum(q["accuracy"] for q in scored) / len(scored) if scored else 0
    avg_faith = sum(q["faithfulness"] for q in scored) / len(scored) if scored else 0

    output = {
        "n_questions": len(golden),
        "n_scored": len(scored),
        "average_accuracy": round(avg_acc, 2),
        "average_faithfulness": round(avg_faith, 2),
        "per_question": per_question,
    }
    RESULTS.write_text(json.dumps(output, indent=2))
    print(f"\naverage accuracy {avg_acc:.2f}, average faithfulness {avg_faith:.2f}")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    asyncio.run(main())
