import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
import os
import random

DATASET_FILE = "selective_copying_dataset.json"
CHARS = ['A', 'B', 'C']
N_SAMPLES = 500
TOTAL_NOISE_TOKENS = 500

# Camouflage: Mix of numbers and all other uppercase letters (excluding A, B, C)
CAMOUFLAGE = list("DEFGHIJKLMNOPQRSTUVWXYZ0123456789")


# Load the shared dataset
with open(DATASET_FILE, "r") as f:
    shared_data = json.load(f)

# --- 2. CONFIGURATION & MODEL LOADING ---
# Remember to change this for each model you test
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
FILENAME = f"results_ultrahard_{MODEL_ID.split('/')[-1]}.json"

print(f"\nLoading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)

target_tokens = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in CHARS]

def get_logits(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        relevant_probs = probs[target_tokens].cpu().float().numpy()
        norm_sum = relevant_probs.sum()
        if norm_sum > 0:
            relevant_probs /= norm_sum
        else:
            relevant_probs = np.ones(3) / 3.0

    return relevant_probs.tolist()

# --- 3. INFERENCE LOOP ---
print(f"Starting inference for {MODEL_ID}...")
all_results = {"easy": [], "hard": []}

for i in range(N_SAMPLES):
    all_results["easy"].append({
        "probs": get_logits(shared_data["easy"][i]["prompt"]),
        "label": shared_data["easy"][i]["label"]
    })
    all_results["hard"].append({
        "probs": get_logits(shared_data["hard"][i]["prompt"]),
        "label": shared_data["hard"][i]["label"]
    })

    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{N_SAMPLES} samples...")

with open(FILENAME, "w") as f:
    json.dump(all_results, f)

print(f"Done! Saved {MODEL_ID} results to {FILENAME}")

# =====================================================================
# --- 4. IMMEDIATE ACCURACY CALCULATION ---
# =====================================================================
print("\n" + "="*60)
print(f"=== {MODEL_ID} PERFORMANCE SUMMARY ===")

# Calculate Easy Accuracy
easy_correct = sum([1 for item in all_results["easy"] if np.argmax(item["probs"]) == item["label"]])
easy_acc = (easy_correct / N_SAMPLES) * 100

# Calculate Hard Accuracy
hard_correct = sum([1 for item in all_results["hard"] if np.argmax(item["probs"]) == item["label"]])
hard_acc = (hard_correct / N_SAMPLES) * 100

print(f"Easy Accuracy (Target at END):    {easy_acc:.1f}%")
print(f"Hard Accuracy (Target in MIDDLE): {hard_acc:.1f}%")
print("="*60 + "\n")

# Clean up memory
del model
torch.cuda.empty_cache()