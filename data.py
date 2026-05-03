from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset


SUPPORTED_FILE_SUFFIXES = {".csv", ".json", ".jsonl", ".txt", ".text", ".xlsx", ".xls"}


def _infer_loader(data_file):
    suffix = Path(data_file).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".txt", ".text"}:
        return "text"
    if suffix in {".xlsx", ".xls"}:
        return "excel"
    raise ValueError(
        f"Could not infer dataset loader for {data_file!r}. "
        "Use a .csv, .json/.jsonl, .txt, or .xlsx/.xls file."
    )


def _resolve_data_files(data_file):
    path = Path(data_file)
    if path.is_dir():
        files = [
            str(file)
            for file in sorted(path.iterdir())
            if file.is_file() and file.suffix.lower() in SUPPORTED_FILE_SUFFIXES
        ]
        if not files:
            raise ValueError(f"No supported data files were found in {data_file!r}.")
        return files
    return [str(path)]


def _row_to_text(row):
    parts = []
    for column, value in row.items():
        if pd.isna(value):
            continue
        value = str(value).strip()
        if value:
            parts.append(f"{column}: {value}")
    return "; ".join(parts)


def _add_text_column(dataset, text_column):
    if text_column in dataset.column_names:
        return dataset

    if text_column != "text":
        raise ValueError(
            f"Text column {text_column!r} was not found. "
            f"Available columns: {', '.join(dataset.column_names)}"
        )

    return dataset.map(
        lambda row: {"text": _row_to_text(row)},
        desc="Building text records from tabular rows",
    ).filter(lambda row: bool(row["text"].strip()))


def _load_excel_files(data_files):
    frames = []
    for data_file in data_files:
        try:
            sheets = pd.read_excel(data_file, sheet_name=None, dtype=str)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel files requires openpyxl. Install it with "
                "`pip install openpyxl`, then run training again."
            ) from exc

        for sheet_name, frame in sheets.items():
            frame = frame.dropna(how="all")
            if frame.empty:
                continue
            frame.insert(0, "source_sheet", sheet_name)
            frame.insert(0, "source_file", Path(data_file).name)
            frames.append(frame)

    if not frames:
        raise ValueError("The Excel file(s) did not contain any non-empty rows.")

    return Dataset.from_pandas(pd.concat(frames, ignore_index=True), preserve_index=False)


def load_and_tokenize_dataset(
    tokenizer,
    dataset_name=None,
    data_file=None,
    text_column="text",
    validation_split=0.1,
    max_length=512,
    seed=42,
):
    if not dataset_name and not data_file:
        raise ValueError("Pass either --dataset_name or --data_file.")

    if dataset_name:
        raw_dataset = load_dataset(dataset_name)
        if "train" not in raw_dataset:
            raw_dataset = raw_dataset["train"].train_test_split(
                test_size=validation_split,
                seed=seed,
            )
    else:
        data_files = _resolve_data_files(data_file)
        loaders = {_infer_loader(file) for file in data_files}
        if len(loaders) != 1:
            raise ValueError(
                "Pass files of one data type at a time. "
                f"Found: {', '.join(sorted(loaders))}"
            )

        loader = loaders.pop()
        if loader == "excel":
            dataset = _load_excel_files(data_files)
            dataset = _add_text_column(dataset, text_column)
            raw_dataset = dataset.train_test_split(
                test_size=validation_split,
                seed=seed,
            )
        else:
            raw_dataset = load_dataset(loader, data_files=data_files)
            dataset = _add_text_column(raw_dataset["train"], text_column)
            raw_dataset = dataset.train_test_split(
                test_size=validation_split,
                seed=seed,
            )

    if "test" not in raw_dataset:
        raw_dataset = raw_dataset["train"].train_test_split(
            test_size=validation_split,
            seed=seed,
        )

    sample_columns = raw_dataset["train"].column_names

    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
        )

    return raw_dataset.map(
        tokenize,
        batched=True,
        remove_columns=sample_columns,
    )
