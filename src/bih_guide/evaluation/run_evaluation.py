from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42

SYSTEM_INSTRUCTION = (
    "Ti si pouzdan turistički vodič za Bosnu i Hercegovinu. "
    "Odgovaraj na srpskom jeziku, ijekavicom i latinicom. "
    "Koristi samo dostavljene provjerene informacije. "
    "Ne izmišljaj cijene, radna vremena, redove vožnje, datume događaja "
    "ni druge promjenjive podatke."
)

GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.9,
    "max_new_tokens": 300,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Neispravan JSON u {path} na liniji {line_no}: {exc}"
                ) from exc
    return rows


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = read_jsonl(path)

    if len(questions) != 60:
        raise ValueError(
            f"T15 očekuje tačno 60 evaluacionih pitanja/scenarija, "
            f"pronađeno: {len(questions)}"
        )

    question_ids = [str(q.get("question_id", "")).strip() for q in questions]
    if any(not qid for qid in question_ids):
        raise ValueError("Postoji evaluacioni primjer bez question_id.")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("question_id se ponavlja u evaluation_questions.jsonl.")

    for question in questions:
        qid = question["question_id"]
        messages = question.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{qid}: nedostaje neprazna lista messages.")

        for index, message in enumerate(messages, start=1):
            if message.get("role") != "user":
                raise ValueError(
                    f"{qid}: T14 evaluacioni input na poziciji {index} "
                    "mora imati role='user'."
                )
            if not str(message.get("content", "")).strip():
                raise ValueError(f"{qid}: prazna user poruka.")

    return questions


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    """
    Učitava postojeći output radi nastavka nakon prekida.

    Ako je zadnja linija djelimično zapisana zbog prekida Colaba,
    zadržava sve validne prethodne rezultate i uklanja samo tu
    nepotpunu zadnju liniju.
    """
    if not path.exists():
        return []

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    results: list[dict[str, Any]] = []
    valid_lines: list[str] = []

    nonempty = [(i, line) for i, line in enumerate(raw_lines) if line.strip()]
    last_nonempty_index = nonempty[-1][0] if nonempty else -1

    for index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == last_nonempty_index:
                print(
                    "Upozorenje: pronađena je nepotpuna zadnja JSONL linija. "
                    "Biće uklonjena prije nastavka."
                )
                break
            raise ValueError(
                f"Neispravan JSON u postojećem outputu {path} "
                f"na liniji {index + 1}: {exc}"
            ) from exc

        results.append(result)
        valid_lines.append(json.dumps(result, ensure_ascii=False))

    if len(valid_lines) != len(nonempty):
        repaired = "\n".join(valid_lines)
        if repaired:
            repaired += "\n"
        path.write_text(repaired, encoding="utf-8")

    seen: set[str] = set()
    for result in results:
        qid = str(result.get("question_id", "")).strip()
        if not qid:
            raise ValueError("Postojeći baseline rezultat nema question_id.")
        if qid in seen:
            raise ValueError(
                f"Postojeći baseline_results.jsonl ima duplikat question_id={qid}."
            )
        seen.add(qid)

    return results


def validate_existing_config(results: list[dict[str, Any]]) -> None:
    """
    Sprječava da se u isti baseline JSONL pomiješaju različiti modeli
    ili različiti generation parametri.
    """
    for result in results:
        qid = result.get("question_id", "<unknown>")

        if result.get("model_name") != MODEL_NAME:
            raise ValueError(
                f"{qid}: postojeći rezultat koristi drugi model: "
                f"{result.get('model_name')!r}"
            )

        if result.get("seed") != SEED:
            raise ValueError(
                f"{qid}: postojeći rezultat koristi drugi seed: "
                f"{result.get('seed')!r}"
            )

        if result.get("generation_config") != GENERATION_CONFIG:
            raise ValueError(
                f"{qid}: postojeći rezultat koristi drugi generation_config."
            )


def load_base_model():
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU nije dostupan. T15 baseline treba pokrenuti u Colab GPU runtime-u."
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Učitavam osnovni model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


def generate_response(
    tokenizer,
    model,
    conversation: list[dict[str, str]],
) -> str:
    import torch
    from transformers import set_seed

    # Reset prije SVAKOG generisanja čini svaki odgovor reproducibilnim
    # i nezavisnim od toga da li je skripta nastavljena nakon prekida.
    set_seed(SEED)

    prompt_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            do_sample=GENERATION_CONFIG["do_sample"],
            temperature=GENERATION_CONFIG["temperature"],
            top_p=GENERATION_CONFIG["top_p"],
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = model_inputs["input_ids"].shape[1]
    new_token_ids = generated_ids[0, prompt_length:]

    response = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
    ).strip()

    return response


def evaluate_question(
    question: dict[str, Any],
    tokenizer,
    model,
) -> dict[str, Any]:
    """
    Single-turn:
      system -> user -> assistant

    Multi-turn:
      system -> user -> assistant -> user -> assistant -> ...

    `expected_points` i `must_not_include` se namjerno NIKADA ne šalju
    modelu; oni su samo za kasnije ocjenjivanje.
    """
    conversation: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        }
    ]

    final_response = ""

    for input_message in question["messages"]:
        user_message = {
            "role": "user",
            "content": str(input_message["content"]).strip(),
        }
        conversation.append(user_message)

        final_response = generate_response(
            tokenizer=tokenizer,
            model=model,
            conversation=conversation,
        )

        conversation.append({
            "role": "assistant",
            "content": final_response,
        })

    return {
        "question_id": question["question_id"],
        "category": question.get("category"),
        # Originalni T14 prompt/scenario bez rubric podataka.
        "prompt": question["messages"],
        # Za multi-turn ovo je posljednji assistant odgovor.
        "response": final_response,
        "model_name": MODEL_NAME,
        "seed": SEED,
        "generation_config": dict(GENERATION_CONFIG),
        # T15 zahtijeva da se za multi-turn sačuva cijeli razgovor.
        # Čuvamo conversation za sve primjere radi jedinstvene šeme.
        "conversation": conversation,
    }


def append_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def validate_final_results(
    questions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    results = read_jsonl(output_path)

    if len(results) != 60:
        raise AssertionError(
            f"Očekivano 60 baseline rezultata, pronađeno: {len(results)}"
        )

    expected_ids = {q["question_id"] for q in questions}
    result_ids = [r.get("question_id") for r in results]

    if len(result_ids) != len(set(result_ids)):
        raise AssertionError("Duplikat question_id u baseline rezultatima.")

    if set(result_ids) != expected_ids:
        missing = sorted(expected_ids - set(result_ids))
        extra = sorted(set(result_ids) - expected_ids)
        raise AssertionError(
            f"question_id skup nije isti kao T14. missing={missing}, extra={extra}"
        )

    required_fields = {
        "question_id",
        "prompt",
        "response",
        "model_name",
        "seed",
        "generation_config",
        "conversation",
    }

    for result in results:
        missing_fields = required_fields - set(result)
        if missing_fields:
            raise AssertionError(
                f"{result.get('question_id')}: nedostaju polja "
                f"{sorted(missing_fields)}"
            )

        if not str(result["response"]).strip():
            raise AssertionError(
                f"{result['question_id']}: prazan response."
            )

        if result["model_name"] != MODEL_NAME:
            raise AssertionError(
                f"{result['question_id']}: pogrešan model_name."
            )

        if result["seed"] != SEED:
            raise AssertionError(
                f"{result['question_id']}: pogrešan seed."
            )

        if result["generation_config"] != GENERATION_CONFIG:
            raise AssertionError(
                f"{result['question_id']}: pogrešan generation_config."
            )

    print("Završna T15 validacija: OK")
    print("Broj rezultata: 60")
    print(f"Model: {MODEL_NAME}")
    print(f"Seed: {SEED}")
    print(f"Generation config: {GENERATION_CONFIG}")


def run_evaluation(
    questions_path: Path,
    output_path: Path,
) -> None:
    questions = load_questions(questions_path)

    existing_results = load_existing_results(output_path)
    validate_existing_config(existing_results)

    completed_ids = {
        str(result["question_id"])
        for result in existing_results
    }

    print(f"Evaluacionih scenarija: {len(questions)}")
    print(f"Već završeno: {len(completed_ids)}")
    print(f"Preostalo: {len(questions) - len(completed_ids)}")

    if len(completed_ids) == len(questions):
        print("Svih 60 pitanja je već evaluirano. Model se neće ponovo učitavati.")
        validate_final_results(questions, output_path)
        return

    tokenizer, model = load_base_model()

    for index, question in enumerate(questions, start=1):
        qid = question["question_id"]

        if qid in completed_ids:
            print(f"[{index:02d}/60] {qid}: preskočeno - već postoji")
            continue

        print(f"[{index:02d}/60] {qid}: generisanje...")

        result = evaluate_question(
            question=question,
            tokenizer=tokenizer,
            model=model,
        )
        append_result(output_path, result)

        # Odmah dodaj u set da jedan proces nikada ne napiše isti ID dva puta.
        completed_ids.add(qid)

        turns = len(question["messages"])
        print(
            f"[{index:02d}/60] {qid}: sačuvano "
            f"({turns} user turn{'a' if turns != 1 else ''})"
        )

    validate_final_results(questions, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "T15: baseline evaluacija Qwen/Qwen2.5-1.5B-Instruct "
            "na fiksnom T14 skupu od 60 pitanja."
        )
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/evaluation_questions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/baseline_results.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        questions_path=args.questions,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
