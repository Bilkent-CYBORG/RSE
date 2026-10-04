import os
import time
import torch
import numpy as np
import json
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Config ----
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct" # model will be inputted here
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = "heegyu/news-category-dataset"


START_YEAR = 2012
END_YEAR = 2022
MAX_PROMPT_LEN = 1024

# Output Files
# Added _HEADLINE_ONLY to filename so you know which input was used
FINAL_OUTPUT_FILE = "wildtime_huffpost_phi3.5_mini_it.json"
PARTIAL_OUTPUT_FILE ="wildtime_huffpost_PARTIAL_CHECKPOINT.json"
SAVE_EVERY = 1000  # Save progress every 1k examples

# Initial massive pool of letters (will be trimmed to 11)
CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]

def get_year(example):
    """Robust year extraction"""
    dt = example["date"]
    if isinstance(dt, str):
        try:
            return datetime.strptime(dt, "%Y-%m-%d").year
        except:
            try:
                return datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S").year
            except:
                return -1
    elif isinstance(dt, (int, float)):
        return datetime.fromtimestamp(dt).year
    else:
        return dt.year

def compute_logprob_of_continuation(model, tokenizer, prompt_ids, cont_ids, device):
    full_ids = torch.cat([prompt_ids, cont_ids], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
    prompt_len = prompt_ids.shape[0]
    total_logprob = 0.0
    for i in range(cont_ids.shape[0]):
        logit_pos = prompt_len - 1 + i
        token_id = int(cont_ids[i])
        if logit_pos < 0 or logit_pos >= log_probs.shape[1]: return -1e9
        lp = log_probs[0, logit_pos, token_id].item()
        total_logprob += lp
    return total_logprob

def tokenize_continuation_for_letter(tokenizer, letter):
    continuation_string = f'{{"answer": "{letter}"}}'
    ids = tokenizer(continuation_string, add_special_tokens=False).input_ids
    return torch.tensor(ids, dtype=torch.long) if ids else torch.tensor([], dtype=torch.long)

# ---- 1. Load Resources ----
print(f"Loading Model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(DEVICE)
model.eval()

print(f"Loading Dataset: {DATASET_NAME}...")
dataset = load_dataset(DATASET_NAME, split="train")


print(f"\n--- Identifying Stable Categories ({START_YEAR}-{END_YEAR}) ---")
print("Scanning dataset to find categories present in EVERY single year...")

categories_by_year = {y: set() for y in range(START_YEAR, END_YEAR + 1)}

# Fast scan of metadata
for ex in dataset:
    y = get_year(ex)
    if START_YEAR <= y <= END_YEAR:
        categories_by_year[y].add(ex['category'])

# Find Intersection
common_categories = categories_by_year[START_YEAR]
for y in range(START_YEAR + 1, END_YEAR + 1):
    common_categories = common_categories.intersection(categories_by_year[y])

STABLE_CATEGORIES = sorted(list(common_categories))
print(f"\n[Success] Found {len(STABLE_CATEGORIES)} Stable Categories (Intersection):")
print(STABLE_CATEGORIES)

if len(STABLE_CATEGORIES) == 0:
    raise ValueError("No categories found that persist across all years! Check dataset version.")

# Create Mappings
CAT_TO_INT = {cat: i for i, cat in enumerate(STABLE_CATEGORIES)}
INT_TO_CAT = {i: cat for i, cat in enumerate(STABLE_CATEGORIES)}

# ---- 3. Filter & Prepare Data (FULL DATASET) ----
print(f"\n--- Filtering Dataset ---")
print(f"Original Count: {len(dataset)}")

def strict_filter(example):
    y = get_year(example)
    cat = example['category']
    # 1. Must be within 2012-2022
    if not (START_YEAR <= y <= END_YEAR):
        return False
    # 2. Must be one of the Stable 9
    if cat not in STABLE_CATEGORIES:
        return False
    return True

filtered_dataset = dataset.filter(strict_filter)
# We shuffle to mix years, but we KEEP ALL EXAMPLES
filtered_dataset = filtered_dataset.shuffle(seed=42)

total_examples = len(filtered_dataset)
print(f"Final Count (Stable 11 Only): {total_examples} examples.")
print("WARNING: This will take a long time. Progress will be saved periodically.")

# Build Choices Text
formatted_choices = "\n".join([f"{CHOICE_LETTERS[i]}. {INT_TO_CAT[i]}" for i in range(len(STABLE_CATEGORIES))])

# ---- 4. Inference Loop (ALL DATA) ----
print(f"\n--- Starting Inference on {total_examples} examples ---")
all_results = []
correct = 0

start_time = time.time()

for idx in range(total_examples):
    example = filtered_dataset[idx]

   
    sentence = example.get("headline", "")
    
    # Metadata
    doc_year = get_year(example)
    gt_label_str = example["category"]
    gt_letter = CHOICE_LETTERS[CAT_TO_INT[gt_label_str]]

    # Prompt
    messages = [
        {
            "role": "system",
            "content": f"You are an expert at news classification. Classify the user's headline into one of {len(STABLE_CATEGORIES)} categories. Select the correct answer. Respond with ONLY a JSON object in this exact format: {{\"answer\": \"A\"}} where the value is the corresponding letter."
        },
        {
            "role": "user",
            "content": f"Headline: {sentence}\n\nChoices:\n{formatted_choices}\n\nProvide your answer in JSON format:"
        },
    ]

    # Tokenize
    try:
        prompt_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
        if prompt_tokens.shape[0] > MAX_PROMPT_LEN - 256:
            prompt_tokens = prompt_tokens[-(MAX_PROMPT_LEN - 256):]
    except Exception as e:
        print(f"Skipping Q{idx}: {e}")
        continue

    # Get Logprobs
    per_letter_logprobs = []
    for letter in CHOICE_LETTERS[:len(STABLE_CATEGORIES)]:
        cont_ids = tokenize_continuation_for_letter(tokenizer, letter)
        lp = compute_logprob_of_continuation(model, tokenizer, prompt_tokens, cont_ids, DEVICE)
        per_letter_logprobs.append(lp)

    # Prediction
    pred_idx = int(np.argmax(per_letter_logprobs))
    pred_letter = CHOICE_LETTERS[pred_idx]
    pred_label = INT_TO_CAT[pred_idx]

    is_correct = (pred_letter == gt_letter)
    if is_correct: correct += 1

    # Probabilities
    probs_np = np.exp(np.array(per_letter_logprobs) - np.max(per_letter_logprobs))
    probs_np /= probs_np.sum()

    # Store Result
    all_results.append({
        "question_index": idx,
        "year": doc_year,
        "text": sentence,
        "ground_truth_label": gt_label_str,
        "prediction_label": pred_label,
        "ground_truth_letter": gt_letter,
        "prediction_letter": pred_letter,
        "is_correct": is_correct,
        "probabilities": {INT_TO_CAT[i]: float(probs_np[i]) for i in range(len(STABLE_CATEGORIES))}
    })

    
    if (idx + 1) % 100 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (total_examples - idx) / 3600
        print(f"[{idx+1}/{total_examples}] Acc: {correct/(idx+1):.2%} | Est. Remaining: {remaining:.2f} hours")

    
    if (idx + 1) % SAVE_EVERY == 0:
        print(f"Saving checkpoint to {PARTIAL_OUTPUT_FILE}...")
        with open(PARTIAL_OUTPUT_FILE, 'w') as f:
            json.dump(all_results, f, indent=4)


elapsed = time.time() - start_time
print(f"\nFinished in {elapsed/3600:.2f} hours.")
print(f"Overall Accuracy: {correct/total_examples*100:.2f}%")

with open(FINAL_OUTPUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"Final results saved to {FINAL_OUTPUT_FILE}")