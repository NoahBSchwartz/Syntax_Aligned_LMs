#!/usr/bin/env python
# coding: utf-8

""" Main Training code base for the LLM """
# - QWEN-2 7B Instruct LLM
# - Developed by Alibaba group : https://arxiv.org/abs/2309.16609
# - Four variations 0.5B, 1.5B, 7B, and 72B model parameters
# - 72B model has better performance than current open source LLM such as Llama-70B

import peft, accelerate, loralib
import os
import wandb
import torch
import  bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
import json
import os
from datasets import load_dataset
import re
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import wandb
from random_word import RandomWords

dataset_path = "/home/ubuntu/data/spider_data/" # Path to the dataset - Change this to your dataset path

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Load the pre-trained model with quantization configuration
orig_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B",
    quantization_config=quantization_config,
    device_map="auto"
)

# Load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
print("########################\nModel loaded")
print(orig_model) # Looking at the model parameters and architecture

# Freeze model parameters and convert certain parameters to float16
for param in orig_model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
        param.data = param.data.to(torch.float16)

# Enable gradient checkpointing and input gradients
orig_model.gradient_checkpointing_enable()
orig_model.enable_input_require_grads()

# Define a class to cast output to float16
class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float16)

# Convert the last layer to float16
orig_model.lm_head = CastOutputToFloat(orig_model.lm_head)

## Setting the Lora Adapters
def print_trainable_params(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable Param: {trainable_params}")
    print(f"All Params: {all_params}\n% Trainable: {(trainable_params/all_params)*100}")

# Low-Rank Adaptation (LoRA) is a PEFT method that decomposes a large matrix into two smaller low-rank matrices in the attention layers. 
# This drastically reduces the number of parameters that need to be fine-tuned.
loraConfig = peft.LoraConfig(
    r=16, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)
# Apply LoRA configuration to the model
model = peft.get_peft_model(orig_model, loraConfig)
print_trainable_params(model)

# Load training, validation, and test datasets
with open(dataset_path + "train_spider_tsql_cleaned.json") as f:
    train = json.load(f)
with open(dataset_path + "dev_tsql_cleaned.json") as f:
    val = json.load(f)
with open(dataset_path + "test_tsql_cleaned.json") as f:
    test = json.load(f)

# Define regex patterns for extracting questions and system prompts
pattern_q = r"\n### Question: (.*?)\n### Response"
pattern_sp = r"### System Prompt: (.*?)\n### Question:"

# Process datasets to update prompts and extract necessary fields
for x in [train, val, test]:
    for items in x:
        items["prompt"] = items["prompt"]
        items["output"] = items["dsql"]
        items["text"] = ""
        match = re.search(pattern_q, items["prompt"], re.DOTALL)
        if match:
            question = match.group(1)
            items["input"] = str(question)
        match = re.search(pattern_sp, items["prompt"], re.DOTALL)
        if match:
            text = match.group(1)
            items["instruction"] = str(text)
        del items["dsql"], items["prompt"], items["database"]

# Save updated datasets
with open(dataset_path + 'train_updated.json', 'w') as fout:
    json.dump(train, fout)
with open(dataset_path + 'val_updated.json', 'w') as fout:
    json.dump(val, fout)
with open(dataset_path + 'test_updated.json', 'w') as fout:
    json.dump(test, fout)

# Load the updated datasets
data_X = load_dataset("json", data_files={
    'train': dataset_path + "train_updated.json", 'validation': dataset_path + "val_updated.json"
})

# Define the main prompt format
main_prompt = """Below is an instruction that describes a task, paired with an input that provides further context for the given output.
### Instruction:
{}
### Input:
{}
### Output:
{}
"""

EOS_TOKEN = tokenizer.eos_token

# Function to format prompts
def format_prompts(examples):
    inst = examples["instruction"]
    inp = examples["input"]
    outp = examples["output"]
    texts = []
    for inst_, inp_, outp_ in zip(inst, inp, outp):
        text = main_prompt.format(inst_, inp_, outp_) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }

# Apply formatting and tokenization to the dataset
data_X = data_X.map(format_prompts, batched=True)
data_X = data_X.map(lambda samples: tokenizer(samples["text"]), batched=True)

# Delete original datasets to free up memory
del train, val, test

# Generate a random run name for logging
r = RandomWords()
run_name = r.get_random_word()
os.environ["WANDB_PROJECT"] = "qwen-7B-finetune"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Initialize the Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data_X["train"],
    eval_dataset=data_X["validation"],
    args=TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_strategy="epoch",
        num_train_epochs=3,
        warmup_steps=5,
        learning_rate=2e-4,
        weight_decay=1e-2,
        lr_scheduler_type="linear",
        seed=42,
        fp16=True,
        optim="adamw_8bit",
        logging_steps=0.5,
        output_dir='outputs',
        report_to="wandb",
        save_strategy="epoch",
        run_name=f"finetune-lora-{run_name}",
        load_best_model_at_end=True,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False

# Start training
trainer.train()

# Load the test dataset
data_X_test = load_dataset("json", data_files={
    'test': dataset_path + "test_updated.json"
})

# Apply formatting and tokenization to the test dataset
data_X_test = data_X_test.map(format_prompts, batched=True)
data_X_test = data_X_test.map(lambda samples: tokenizer(samples["text"]), batched=True)

# Select a sample of the test dataset for prediction
sample_test = data_X_test["test"].select(range(5))

# Make predictions on the sample test dataset
output_t = trainer.predict(sample_test)
for i in range(len(output_t.label_ids)):
    preds = np.where(output_t.label_ids[i][:-1] != -100, output_t.label_ids[i][:-1], tokenizer.pad_token_id)
    print("###############\nLLM Output:", tokenizer.decode(preds))
    print("###############\nGT ===>",sample_test[i]["output"], "\n\n")

# Finish the W&B run
wandb.finish()
