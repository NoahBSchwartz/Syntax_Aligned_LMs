from datasets import load_dataset, concatenate_datasets
from random import randrange
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import torch
import evaluate
import nltk
import numpy as np
import wandb
import os
import re
from typing import Optional, Dict, Sequence
import random
import copy
from dataclasses import dataclass, field
import torch.distributed
from nltk.tokenize import sent_tokenize
import sqlglot
from sqlglot import parse_one, exp
from sqlglot.dialects import Dialect

nltk.download("punkt")

# Load model to GPU
model_id = "google/flan-t5-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model.to(device)
print("Initial model stored on:", next(model.parameters()).device)

# Load Dataset
dataset_id = "NoahBSchwartz/llm-synth-finetune-3"
dataset = load_dataset(dataset_id)
print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")
print(f"Validation dataset size: {len(dataset['validation'])}")

# Add tokens to vocabulary
tokenizer = AutoTokenizer.from_pretrained(model_id)
vocabulary = tokenizer.get_vocab().items()
print(len(tokenizer))
new_tokens = [
    "WHERE",
    "FROM",
    "SELECT",
    "LENGTH",
    "LIMIT",
    "ORDER",
    "BY",
    "DESC",
    "NOT",
    "JOIN",
    "DISTINCT",
    "GROUP",
    "HAVING",
    "SUM",
    "AS",
    "AVG",
    "MAX",
    "MIN",
    "IN",
    "ASC",
    "CAST",
    "BETWEEN",
    "INTERSECT",
    "OF",
    "UNION",
    "EXCEPT",
    "COUNT",
    "YEAR",
    "TYPE",
    "AND",
    "OR",
    "NOT",
    "ON",
    "COUNT(*)",
    "COUNT",
]
for word in new_tokens:
    if word not in vocabulary:
        tokenizer.add_tokens(word)
model.resize_token_embeddings(len(tokenizer))


# Preprocess Natural Language Dataset
def process_prompt(example):
    if "prompt" in example and isinstance(example["prompt"], str):
        example["prompt"] = re.sub(
            r"DSQL is a domain-specific language similar to SQL but with different ordering and syntax to make it closer to natural language\. ",
            "",
            example["prompt"],
        )
        example["prompt"] = example["prompt"].replace("DSQL", "SQL")
        example["prompt"] = example["prompt"].lower()
    return example


def process_sql(example):
    ast = parse_one(example["sql"], read="sqlite")
    example["sql"] = ast.sql(dialect="sqlite", pretty=True)
    words = str(example["sql"]).split()
    processed_words = []
    for i, word in enumerate(words):
        if any(word.strip() == token.strip() for token in new_tokens):
            processed_words.append(word)
        else:
            subword_match = False
            for token in new_tokens:
                if token.strip() in word and "(" in word:
                    word = word.replace(token.strip(), "")
                    word = word.lower()
                    processed_words.append(token + word)
                    subword_match = True
                    break
            if not subword_match:
                processed_words.append(word.lower())
    example["sql"] = " " + " ".join(processed_words)
    example["sql"] = re.sub(r"\((?!\*)", "( ", example["sql"])
    return example


for split in dataset.keys():
    dataset[split] = dataset[split].select(range(1, len(dataset[split])))
    dataset[split] = dataset[split].map(process_prompt)
    dataset[split] = dataset[split].map(process_sql)

# Get max lengths so we know where to truncate
tokenized_inputs = concatenate_datasets(
    [dataset["train"], dataset["test"], dataset["validation"]]
).map(
    lambda x: tokenizer(x["prompt"], truncation=True),
    batched=True,
    remove_columns=["prompt", "sql", "dsql"],
)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
tokenized_targets = concatenate_datasets(
    [dataset["train"], dataset["test"], dataset["validation"]]
).map(
    lambda x: tokenizer(x["sql"], truncation=True),
    batched=True,
    remove_columns=["prompt", "sql", "dsql"],
)
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])


# Tokenize
def preprocess_function(sample, padding="max_length"):
    inputs = [item for item in sample["prompt"]]
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=padding, truncation=True
    )
    labels = tokenizer(
        text_target=sample["sql"],
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(
    preprocess_function, batched=True, remove_columns=["prompt", "sql", "dsql"]
)
test_dataset = tokenized_dataset["train"]
first_item = test_dataset[randrange(len(test_dataset))]
print("\nDecoded input:")
print(tokenizer.decode(first_item["input_ids"]))
print("\nDecoded label:")
print(tokenizer.decode([l for l in first_item["labels"] if l != -100]))

# Setup Metrics
metric = evaluate.load("rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


# Train
torch.cuda.empty_cache()
os.environ["WANDB_PROJECT"] = "flan_t5_base_sql"
wandb.init(project="flan_t5_base_sql", name="flan_t5_base_sql")
label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
)
output_dir = "flan-t5-base-sql"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    predict_with_generate=True,
    fp16=False,
    learning_rate=5e-5,
    num_train_epochs=4,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb",
    run_name="flan_t5_base_sql",
    logging_steps=100,
    logging_dir="./logs",
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.evaluate()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
model_card = trainer.create_model_card()
