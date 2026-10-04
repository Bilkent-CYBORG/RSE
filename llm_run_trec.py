import os
import time
import torch
import numpy as np
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---- Config ----
MODEL_NAME = "Qwen/Qwen3-14B" # model will be inputted here
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EXAMPLES = 5452   # all TRAIN set for TREC
MAX_PROMPT_LEN = 1024 # Max *prompt* length, continuations will be added
OUTPUT_FILE = "trec_results_train_qwen3_14b.json"

# ---- TREC Label Mappings ----
# TREC has 6 coarse-grained question types
# Based on label_coarse: 0=DESC, 1=ENTY, 2=ABBR, 3=HUM, 4=NUM, 5=LOC
INT_TO_TEXT_LABEL = [
    "description and abstract concepts",  # 0: DESC
    "entities",                           # 1: ENTY
    "abbreviation",                       # 2: ABBR
    "human beings",                       # 3: HUM
    "numeric values",                     # 4: NUM
    "locations"                           # 5: LOC
]
# Map integer label (0-5) to choice letter (A-F)
CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F"]


def compute_logprob_of_continuation(model, tokenizer, prompt_ids, cont_ids, device):
    """
    Compute log-probability assigned *by the model* to the continuation token sequence `cont_ids`
    given the prompt token sequence `prompt_ids`. Returns the summed log-prob (float).
    """
    # Build full sequence: [prompt tokens] + [continuation tokens]
    full_ids = torch.cat([prompt_ids, cont_ids], dim=0).unsqueeze(0).to(device)  # shape (1, L)
    with torch.no_grad():
        outputs = model(full_ids)  # logits shape: (1, L, vocab_size)
        logits = outputs.logits.float()  # cast to float for stability (if bfloat16 was used)
        log_probs = torch.log_softmax(logits, dim=-1)  # (1, L, V)

    # For token at position t (full_ids[0, t]), the model prediction is from logits at t-1.
    # So to score cont_ids which start at index prompt_len, we gather log_probs at positions prompt_len-1 .. etc.
    prompt_len = prompt_ids.shape[0]
    total_logprob = 0.0
    # If continuation contains N tokens, we need logits positions prompt_len-1 ... prompt_len+N-2
    # BUT: the first continuation token is predicted by logits at position prompt_len-1.
    for i in range(cont_ids.shape[0]):
        logit_pos = prompt_len - 1 + i
        token_id = int(cont_ids[i])
        # We'll guard indexing bounds
        if logit_pos < 0 or logit_pos >= log_probs.shape[1]:
            # fallback: cannot compute — return very low prob
            return -1e9
        lp = log_probs[0, logit_pos, token_id].item()
        total_logprob += lp
    return total_logprob

def tokenize_continuation_for_letter(tokenizer, letter):
    """
    Return token ids for the model's expected JSON response.
    We format the continuation as the exact JSON string requested in the prompt.
    """
    # The system prompt demands this *exact* format: {"answer": "A"}
    continuation_string = f'{{"answer": "{letter}"}}'

    # Tokenize *without* adding special tokens (like BOS/EOS)
    # as this is a continuation, not a new input.
    ids = tokenizer(continuation_string, add_special_tokens=False).input_ids

    if not ids:
        print(f"Warning: Tokenizing continuation '{continuation_string}' resulted in empty list.")
        return torch.tensor([], dtype=torch.long)

    return torch.tensor(ids, dtype=torch.long)

# ---- Load model & tokenizer ----
print("Loading model & tokenizer:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(DEVICE)
model.eval()

# ---- Load TREC Dataset (TRAIN) ----
print("Loading CogComp/trec dataset (train)...")
dataset = load_dataset("SetFit/TREC-QC", split="train")
dataset = dataset.shuffle(seed=42)
n_examples = min(NUM_EXAMPLES, len(dataset))
print(f"Total train examples: {len(dataset)} — will run on {n_examples}")

# ---- Pre-build the choices string (it's the same for all examples) ----
formatted_choices = "\n".join([f"{CHOICE_LETTERS[i]}. {INT_TO_TEXT_LABEL[i]}" for i in range(len(CHOICE_LETTERS))])
print("Using choices:\n" + formatted_choices)

# ---- Main eval loop ----
correct = 0
processed = 0
start_time = time.time()
all_results = []

for idx in range(n_examples):
    example = dataset[idx]

    # ---- Data extraction for TREC ----
    question = example["text"]
    gt_label_int = example["label_coarse"]           # integer, 0-5
    gt_label_str = INT_TO_TEXT_LABEL[gt_label_int]   # string, e.g., "entities"
    gt_letter = CHOICE_LETTERS[gt_label_int]         # string, e.g., "B"

    
    messages = [
        {
            "role": "system",
            "content": "You are an expert at question classification. Classify the user's question into one of six categories: description and abstract concepts, entities, abbreviation, human beings, numeric values, or locations. Select the correct answer. Respond with ONLY a JSON object in this exact format: {\"answer\": \"A\"} where the value is A, B, C, D, E, or F."
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nChoices:\n{formatted_choices}\n\nProvide your answer in JSON format:"
        },
    ]

    # Apply the model's chat template
    try:
        prompt_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )[0] # Get the 1D tensor of token IDs
    except Exception as e:
        print(f"Error applying chat template for Q{idx+1}: {e}. Skipping.")
        continue

    # Truncate if extremely long
    if prompt_tokens.shape[0] > MAX_PROMPT_LEN - 32:
        prompt_tokens = prompt_tokens[-(MAX_PROMPT_LEN - 32):]

    # ---- Compute logprob for all 6 choices ----
    per_letter_logprobs = []
    for letter in CHOICE_LETTERS:
        cont_ids = tokenize_continuation_for_letter(tokenizer, letter)
        lp = compute_logprob_of_continuation(model, tokenizer, prompt_tokens, cont_ids, DEVICE)
        per_letter_logprobs.append(lp)

    # ---- Choose prediction based on 6 choices ----
    pred_index = int(np.argmax(per_letter_logprobs)) # 0-5
    pred_letter = CHOICE_LETTERS[pred_index]         # "A" - "F"
    pred_label_str = INT_TO_TEXT_LABEL[pred_index]   # "abbreviation" - "numeric"

    processed += 1
    is_correct = (pred_letter == gt_letter)
    if is_correct:
        correct += 1

    # Print summary for this question
    per_letter_logprobs_np = np.array(per_letter_logprobs)
    shift = np.max(per_letter_logprobs_np)
    per_letter_probs_np = np.exp(per_letter_logprobs_np - shift)
    per_letter_probs_np = per_letter_probs_np / per_letter_probs_np.sum()

    # Show letter (label)
    prob_summary_parts = []
    for i in range(len(CHOICE_LETTERS)):
        prob_summary_parts.append(f"{CHOICE_LETTERS[i]} ({INT_TO_TEXT_LABEL[i][:4]}):{per_letter_probs_np[i]:.3f}")
    prob_summary = " ".join(prob_summary_parts)

    print(f"Q{idx+1}/{n_examples}: GT={gt_letter} ({gt_label_str}) PRED={pred_letter} ({pred_label_str}) | probs: {prob_summary}")

    
    result_data = {
        "question_index": idx + 1,
        "question_text": question,
        "ground_truth_label": gt_label_str,
        "prediction_label": pred_label_str,
        "ground_truth_letter": gt_letter,
        "prediction_letter": pred_letter,
        "is_correct": is_correct,
        "probabilities": {
            INT_TO_TEXT_LABEL[i]: float(per_letter_probs_np[i]) for i in range(len(INT_TO_TEXT_LABEL))
        },
        "log_probabilities": {
            INT_TO_TEXT_LABEL[i]: float(per_letter_logprobs[i]) for i in range(len(INT_TO_TEXT_LABEL))
        },
    }
    all_results.append(result_data)

   
    if DEVICE == "cpu":
        time.sleep(0.05)


end_time = time.time()
total_time = end_time - start_time
acc = correct / processed * 100 if processed > 0 else 0.0
print("\n" + "=" * 50)
print(f"Evaluated {processed} examples in {total_time:.2f} seconds.")
print(f"Accuracy: {correct}/{processed} = {acc:.2f}%")
print("=" * 50)


try:
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSuccessfully saved {len(all_results)} results to {OUTPUT_FILE}")
except Exception as e:
    print(f"\nError saving results to {OUTPUT_FILE}: {e}")