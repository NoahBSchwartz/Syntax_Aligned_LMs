from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import csv
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
import re
from fuzzywuzzy import fuzz

# Add command-line arguments
parser = argparse.ArgumentParser(description="Run inference with configurable parameters")
parser.add_argument("--top_k", type=int, default=3, help="Top-k sampling parameter")
parser.add_argument("--num_beams", type=int, default=3, help="Number of beams for beam search")
parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
args = parser.parse_args()

# Load Model and Tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B", quantization_config=quantization_config, device_map="auto"
)
peft_model_id = "/home/ubuntu/ck/outputs/checkpoint-20904"
config = PeftConfig.from_pretrained(peft_model_id)
model = PeftModel.from_pretrained(base_model, peft_model_id)

# Format Dataset
main_prompt = """Below is an instruction that describes a task, paired with an input that provides further context for the given output.
### Instruction:
{}
### Input:
{}
### Output:
{}
"""
EOS_TOKEN = tokenizer.eos_token

def format_prompts(examples):
    # Format the dataset examples into the required prompt format
    inst = examples["instruction"]
    inp = examples["input"]
    outp = examples["output"]
    texts = []
    for inst_, inp_, outp_ in zip(inst, inp, outp):
        text = main_prompt.format(inst_, inp_, outp_) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }

# Load and preprocess the test dataset
data_X_test = load_dataset("json", data_files={"test": "datasets_jan25/test_updated.json"})
data_X_test = data_X_test.map(format_prompts, batched=True)
data_X_test = data_X_test.map(lambda samples: tokenizer(samples["text"]), batched=True)
dataset = data_X_test["test"]
print(dataset)

class NewlineStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs):
        # Stop generation if a newline character is generated
        if input_ids[0][-1] == self.tokenizer.encode("\n")[0]:
            return True
        return False

def generate_response(instruction, input_text, len_op):
    # Generate a response based on the given instruction and input text
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context for the given output.
### Instruction:
{instruction}
### Input:
{input_text}
### Output:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    stopping_criteria = StoppingCriteriaList([NewlineStoppingCriteria(tokenizer)])

    # Generate prediction tokens upto a max_new_tokens length plus the current output length
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=len_op+args.max_new_tokens,
            top_k=args.top_k,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
    
    # Output contains top_k sequences, select the first one (best match) and decode it to get the string representations
    outputs = outputs[0].tolist()[0]
    translated_output = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    translated_output = "".join(translated_output)
    # Find only the predicted result part as the whole prompt is generated
    match = re.search(r"### Output:\n(.*?)\n", translated_output, re.DOTALL) 
    if match:
        selected_output = match.group(1)
        return selected_output.strip()
    else:
        print("No match found")
        return ""


# Save to File
batch_size = 10
outputs = []
total_processed = 0
OUTPUT_FILEPATH = "model_generations.csv"

with open(OUTPUT_FILEPATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["question", "gt", "pred", "similarity"])  # Write header

    for i, example in enumerate(dataset):
        len_op = len(example["output"])
        generated_responses = generate_response(example["instruction"], example["input"], len_op) # Gen. string from LLM

        # Calculate the similarity between the expected and generated responses
        similarity = fuzz.ratio(example["output"], generated_responses) 

        outputs.append([example["input"], example["output"], generated_responses, similarity])
        print(f"Example {i+1}:")
        print("Question: " + example["instruction"] + example["input"])
        print("Expected: " + example["output"])
        print("Generated: " + generated_responses)
        print(f"Similarity: {similarity}%")
        print("\n")
        
        if (i + 1) % batch_size == 0:
            writer.writerows(outputs)
            f.flush()
            total_processed += len(outputs)
            print(
                f"Wrote {len(outputs)} rows to file. Total processed: {total_processed}"
            )
            outputs = []

    if outputs:
        writer.writerows(outputs)
        total_processed += len(outputs)

print(f"Generated and saved {total_processed} rows to {OUTPUT_FILEPATH}")