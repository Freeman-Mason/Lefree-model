from transformers import DataCollatorForLanguageModeling, Trainer


def train_model(model, tokenizer, tokenized_dataset, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
    return trainer
