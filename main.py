import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_and_tokenize_dataset
from train import train_model
from training_arg import build_training_args


def load_env_file(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args():
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Fine-tune a causal language model on your text dataset."
    )
    parser.add_argument("--model_name", default=os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B"))
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--data_file", default="data")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--output_dir", default="Lefree-01-model")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    tokenized_dataset = load_and_tokenize_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        data_file=args.data_file,
        text_column=args.text_column,
        validation_split=args.validation_split,
        max_length=args.max_length,
        seed=args.seed,
    )
    training_args = build_training_args(args)
    train_model(model, tokenizer, tokenized_dataset, training_args)


if __name__ == "__main__":
    main()
