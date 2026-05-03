import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def validate_model_source(model_source):
    path = Path(model_source)
    if not path.exists():
        return

    expected_files = {
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    present_files = {item.name for item in path.iterdir() if item.is_file()}
    if not present_files.intersection(expected_files):
        raise FileNotFoundError(
            f"{model_source!r} exists, but it does not contain saved model files. "
            "Train the model first with `python main.py`, or pass a Hugging Face "
            "model id such as `--model_dir Qwen/Qwen3-0.6B`."
        )


def load_tokenizer(model_source, trust_remote_code):
    try:
        return AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=trust_remote_code,
        )
    except ValueError:
        return AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )


def load_benchmark(path):
    records = []
    with open(path, encoding="utf-8") as benchmark_file:
        for line_number, line in enumerate(benchmark_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def normalize(text):
    return " ".join(text.lower().split())


def score_answer(answer, record):
    answer_text = normalize(answer)
    concept_results = []

    for group in record.get("must_include_any", []):
        matched = [term for term in group if normalize(term) in answer_text]
        concept_results.append(
            {
                "terms": group,
                "matched": matched,
                "passed": bool(matched),
            }
        )

    avoided_hits = [
        term for term in record.get("avoid", []) if normalize(term) in answer_text
    ]
    passed_concepts = sum(1 for result in concept_results if result["passed"])
    total_concepts = len(concept_results)
    concept_score = passed_concepts / total_concepts if total_concepts else 1.0
    penalty = 0.15 * len(avoided_hits)
    score = max(0.0, concept_score - penalty)

    return {
        "score": round(score, 3),
        "passed_concepts": passed_concepts,
        "total_concepts": total_concepts,
        "concept_results": concept_results,
        "avoid_hits": avoided_hits,
    }


def build_prompt(question):
    return (
        "You are a careful South African agricultural assistant. "
        "Give practical, locally grounded advice. Be honest about uncertainty. "
        "If facts depend on location, soil tests, weather data, cultivar, pest or "
        "disease diagnosis, or current regulations, say so clearly. Recommend "
        "checking local extension officers, qualified agronomists, laboratory "
        "soil or disease tests, current weather data, and official South African "
        "sources when advice could affect yield, safety, chemical use, or money.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answer(model, tokenizer, question, max_new_tokens):
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if generated.startswith(prompt):
        return generated[len(prompt) :].strip()
    return generated.strip()


def evaluate_model(args):
    records = load_benchmark(args.benchmark)
    validate_model_source(args.model_dir)
    tokenizer = load_tokenizer(args.model_dir, args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        device_map="auto" if args.device_map_auto else None,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    results = []
    for record in records:
        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=record["question"],
            max_new_tokens=args.max_new_tokens,
        )
        scored = score_answer(answer, record)
        results.append(
            {
                "id": record["id"],
                "category": record["category"],
                "province": record.get("province"),
                "question": record["question"],
                "answer": answer,
                "reference_answer": record["reference_answer"],
                "score": scored["score"],
                "passed_concepts": scored["passed_concepts"],
                "total_concepts": scored["total_concepts"],
                "concept_results": scored["concept_results"],
                "avoid_hits": scored["avoid_hits"],
                "source_urls": record.get("source_urls", []),
            }
        )
    return results


def write_results(path, results):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        for result in results:
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")


def print_summary(results):
    if not results:
        print("No benchmark questions found.")
        return

    average = sum(result["score"] for result in results) / len(results)
    print(f"Questions: {len(results)}")
    print(f"Average score: {average:.3f}")
    print()
    print("Lowest scoring items:")
    for result in sorted(results, key=lambda item: item["score"])[:5]:
        print(
            f"- {result['id']}: {result['score']:.3f} "
            f"({result['passed_concepts']}/{result['total_concepts']} concepts)"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the South Africa agriculture benchmark."
    )
    parser.add_argument("--model_dir", default="Lefree-01-model")
    parser.add_argument(
        "--benchmark", default="benchmarks/south_africa_agri_eval.jsonl"
    )
    parser.add_argument("--output", default="eval_results/sa_agri_results.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow model-specific tokenizer/model code when required.",
    )
    parser.add_argument(
        "--device_map_auto",
        action="store_true",
        help="Use transformers device_map='auto' when accelerate is installed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = evaluate_model(args)
    write_results(args.output, results)
    print_summary(results)
    print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
